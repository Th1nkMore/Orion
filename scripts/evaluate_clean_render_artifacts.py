#!/usr/bin/env python3
"""Measure block-like CARLA RGB corruption without using model outputs."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


SCHEMA = "orion.clean_render_artifact_audit.v1"


def _box_sum(array: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(array, ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    width = 2 * radius + 1
    return (
        integral[width:, width:]
        - integral[:-width, width:]
        - integral[width:, :-width]
        + integral[:-width, :-width]
    )


def _rectangular_components(mask: np.ndarray):
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    rows = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        seen[y, x] = True
        area = 0
        min_x = max_x = int(x)
        min_y = max_y = int(y)
        while queue:
            current_y, current_x = queue.popleft()
            area += 1
            min_x = min(min_x, current_x)
            max_x = max(max_x, current_x)
            min_y = min(min_y, current_y)
            max_y = max(max_y, current_y)
            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    if not (delta_x or delta_y):
                        continue
                    next_y, next_x = current_y + delta_y, current_x + delta_x
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and mask[next_y, next_x]
                        and not seen[next_y, next_x]
                    ):
                        seen[next_y, next_x] = True
                        queue.append((next_y, next_x))
        component_width = max_x - min_x + 1
        component_height = max_y - min_y + 1
        rectangularity = area / float(component_width * component_height)
        if (
            20 <= area <= 4000
            and min(component_width, component_height) >= 3
            and max(component_width, component_height) <= 100
            and rectangularity >= 0.5
        ):
            rows.append(
                {
                    "area": area,
                    "width": component_width,
                    "height": component_height,
                    "rectangularity": rectangularity,
                }
            )
    return rows


def evaluate_image(path: Path):
    image = Image.open(path).convert("L")
    gray = np.asarray(image, dtype=np.float32)
    neighborhood = 7 * 7
    mean = _box_sum(gray, 3) / neighborhood
    second_moment = _box_sum(gray * gray, 3) / neighborhood
    local_std = np.sqrt(np.maximum(second_moment - mean * mean, 0.0))
    background = np.asarray(
        image.filter(ImageFilter.GaussianBlur(radius=9)), dtype=np.float32
    )
    candidate = (local_std < 3.0) & (np.abs(gray - background) > 12.0)
    components = _rectangular_components(candidate)
    area = int(sum(row["area"] for row in components))
    return {
        "path": str(path.resolve()),
        "width": image.width,
        "height": image.height,
        "candidate_pixel_fraction": float(candidate.mean()),
        "rectangular_component_count": len(components),
        "rectangular_component_area": area,
        "rectangular_component_area_fraction": area / float(image.width * image.height),
        "largest_components": sorted(
            components, key=lambda row: row["area"], reverse=True
        )[:10],
    }


def summarize(rows):
    result = {"frame_count": len(rows)}
    for field in (
        "candidate_pixel_fraction",
        "rectangular_component_count",
        "rectangular_component_area_fraction",
    ):
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        result[field] = {
            "min": float(values.min()),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
        }
    return result


def evaluate_gate(rows, gate):
    thresholds = gate["per_frame_thresholds"]
    suspicious = [
        row
        for row in rows
        if row["candidate_pixel_fraction"]
        >= float(thresholds["candidate_pixel_fraction_min"])
        and row["rectangular_component_count"]
        >= int(thresholds["rectangular_component_count_min"])
        and row["rectangular_component_area_fraction"]
        >= float(thresholds["rectangular_component_area_fraction_min"])
    ]
    minimum_frames = int(gate["sequence_rule"]["minimum_frames"])
    if len(rows) < minimum_frames:
        raise ValueError("clean-render gate has too few frames")
    fraction = len(suspicious) / float(len(rows))
    failed = fraction >= float(
        gate["sequence_rule"]["suspicious_frame_fraction_reject_at_or_above"]
    )
    return {
        "passed": not failed,
        "suspicious_frame_count": len(suspicious),
        "frame_count": len(rows),
        "suspicious_frame_fraction": fraction,
        "thresholds": thresholds,
        "sequence_rule": gate["sequence_rule"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--glob", action="append", required=True)
    parser.add_argument("--gate-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite clean-render artifact audit")
    paths = sorted(
        {
            path.resolve()
            for pattern in args.glob
            for path in Path("/").glob(pattern.lstrip("/"))
        }
    )
    if not paths:
        raise ValueError("clean-render artifact audit matched no images")
    rows = [evaluate_image(path) for path in paths]
    gate_result = None
    if args.gate_config is not None:
        gate = json.loads(args.gate_config.resolve().read_text())
        if gate.get("schema") != "orion.clean_render_artifact_gate.v1":
            raise ValueError("unexpected clean-render artifact gate schema")
        gate_result = evaluate_gate(rows, gate)
    result = {
        "schema": SCHEMA,
        "status": (
            "passed_clean_render_artifact_gate"
            if gate_result and gate_result["passed"]
            else "failed_clean_render_artifact_gate"
            if gate_result
            else "calibration_metrics_only_no_gate_frozen"
        ),
        "label": args.label,
        "summary": summarize(rows),
        "frames": rows,
        "gate": gate_result,
        "claim_boundary": "Engineering visual-artifact calibration only; no ORION or safety outcome.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    if gate_result is not None:
        print(json.dumps(gate_result, indent=2, sort_keys=True))
        if not gate_result["passed"]:
            raise SystemExit(3)


if __name__ == "__main__":
    main()

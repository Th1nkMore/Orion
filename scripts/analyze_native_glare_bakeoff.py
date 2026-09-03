#!/usr/bin/env python3
"""Pose-matched, outcome-blind visual analysis of native Route151 glare."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


PROFILES = ("clean", "light", "medium", "heavy")
SCHEMA = "orion.native_glare_visual_analysis.v1"


def _load_rows(profile_root: Path) -> List[dict]:
    traces = list(profile_root.rglob("capture_trace.jsonl"))
    if len(traces) != 1:
        raise RuntimeError("expected one trace for %s, found %d" % (profile_root, len(traces)))
    rows = [json.loads(line) for line in traces[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("empty capture trace: %s" % traces[0])
    return rows


def _yaw_error(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _pose_error(reference: dict, candidate: dict) -> Tuple[float, float]:
    left = reference["ego_location"]
    right = candidate["ego_location"]
    distance = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))
    yaw = _yaw_error(float(reference["ego_rotation"][2]), float(candidate["ego_rotation"][2]))
    return distance, yaw


def _select_event_rows(rows: Sequence[dict], count: int = 5) -> List[dict]:
    event_rows = []
    for row in rows:
        walkers = [
            actor for actor in row["nearby_actors"]
            if actor["type_id"].startswith("walker.pedestrian")
        ]
        if walkers:
            event_rows.append((min(float(actor["distance_m"]) for actor in walkers), row))
    if len(event_rows) < 3:
        raise RuntimeError("fewer than three clean frames contain the Route151 pedestrian")
    closest = min(range(len(event_rows)), key=lambda index: event_rows[index][0])
    half = count // 2
    start = max(0, min(closest - half, len(event_rows) - count))
    return [row for _, row in event_rows[start : start + count]]


def _select_progress_rows(
    rows: Sequence[dict], targets: Sequence[float], maximum_error: float = 0.03
) -> List[dict]:
    selected = []
    previous = -1
    for target in targets:
        candidates = [
            (abs(float(row["route_progress"]) - float(target)), index, row)
            for index, row in enumerate(rows)
            if index > previous
        ]
        if not candidates:
            raise RuntimeError("capture ended before target progress %.3f" % target)
        error, index, row = min(candidates, key=lambda value: value[0])
        if error > maximum_error:
            raise RuntimeError("no captured frame within %.3f of progress %.3f" % (maximum_error, target))
        walkers = [
            actor for actor in row["nearby_actors"]
            if actor["type_id"].startswith("walker.pedestrian")
        ]
        if not walkers:
            raise RuntimeError("target progress %.3f has no Route151 pedestrian evidence" % target)
        selected.append(row)
        previous = index
    return selected


def _match_rows(
    references: Sequence[dict], candidates: Sequence[dict],
    maximum_distance_m: float = 2.0, maximum_yaw_degrees: float = 10.0,
) -> List[dict]:
    matches = []
    previous = -1
    for reference in references:
        possible = []
        for index in range(previous + 1, len(candidates)):
            distance, yaw = _pose_error(reference, candidates[index])
            possible.append((distance + yaw / 10.0, index, distance, yaw))
        if not possible:
            break
        _, index, distance, yaw = min(possible)
        if distance > maximum_distance_m or yaw > maximum_yaw_degrees:
            continue
        matches.append({
            "reference": reference,
            "candidate": candidates[index],
            "distance_m": distance,
            "yaw_error_degrees": yaw,
        })
        previous = index
    return matches


def _read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to read %s" % path)
    return image


def _roi(image: np.ndarray, normalized: Sequence[float]) -> np.ndarray:
    height, width = image.shape[:2]
    left, top, right, bottom = normalized
    x0, y0 = int(round(left * width)), int(round(top * height))
    x1, y1 = int(round(right * width)), int(round(bottom * height))
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("invalid normalized ROI")
    return image[y0:y1, x0:x1]


def _contrast(image: np.ndarray) -> float:
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).std())


def _edge_energy(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def _boundary_artifact_score(residual: np.ndarray, normalized: Sequence[float]) -> float:
    gray = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gradient = cv2.Laplacian(gray, cv2.CV_32F)
    gradient = np.abs(gradient)
    height, width = gray.shape
    left, top, right, bottom = normalized
    x0, y0 = int(round(left * width)), int(round(top * height))
    x1, y1 = int(round(right * width)), int(round(bottom * height))
    band = max(2, min(height, width) // 200)
    boundary = np.zeros_like(gray, dtype=bool)
    boundary[max(0, y0 - band):min(height, y0 + band + 1), x0:x1] = True
    boundary[max(0, y1 - band):min(height, y1 + band + 1), x0:x1] = True
    boundary[y0:y1, max(0, x0 - band):min(width, x0 + band + 1)] = True
    boundary[y0:y1, max(0, x1 - band):min(width, x1 + band + 1)] = True
    interior = np.zeros_like(gray, dtype=bool)
    interior[min(y1, y0 + band + 1):max(y0, y1 - band), min(x1, x0 + band + 1):max(x0, x1 - band)] = True
    if not boundary.any() or not interior.any():
        return 0.0
    return float(gradient[boundary].mean() / max(gradient[interior].mean(), 1e-6))


def _frame_metrics(clean: np.ndarray, changed: np.ndarray, roi: Sequence[float]) -> dict:
    if clean.shape != changed.shape:
        raise RuntimeError("pose-matched images have different shapes")
    clean_roi = _roi(clean, roi)
    changed_roi = _roi(changed, roi)
    residual = cv2.absdiff(changed, clean)
    return {
        "mean_absolute_pixel_delta": float(residual.mean()),
        "saturated_pixel_fraction_clean": float((clean.max(axis=2) >= 250).mean()),
        "saturated_pixel_fraction_changed": float((changed.max(axis=2) >= 250).mean()),
        "hazard_roi_contrast_ratio": _contrast(changed_roi) / max(_contrast(clean_roi), 1e-6),
        "hazard_roi_edge_visibility_ratio": _edge_energy(changed_roi) / max(_edge_energy(clean_roi), 1e-6),
        "rectangular_boundary_artifact_score": _boundary_artifact_score(residual, roi),
    }


def _mean(items: Sequence[float]) -> float:
    return float(sum(items) / len(items)) if items else 0.0


def _tile(images: Sequence[np.ndarray], labels: Sequence[str], cell=(640, 360)) -> np.ndarray:
    rendered = []
    for image, label in zip(images, labels):
        frame = cv2.resize(image, cell, interpolation=cv2.INTER_AREA)
        cv2.rectangle(frame, (0, 0), (cell[0], 36), (0, 0, 0), -1)
        cv2.putText(frame, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        rendered.append(frame)
    return np.concatenate(rendered, axis=1)


def _save_gif(frames: Sequence[np.ndarray], path: Path, duration_ms: int = 350) -> None:
    if not frames:
        raise RuntimeError("cannot save empty GIF")
    rgb = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    rgb[0].save(str(path), save_all=True, append_images=rgb[1:], duration=duration_ms, loop=0)


def analyze(root: Path, protocol: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite %s" % output)
    output.mkdir(parents=True)
    spec = json.loads(protocol.read_text(encoding="utf-8"))
    roi = spec["existing_clean_source"]["hazard_roi_normalized"]
    rows = {profile: _load_rows(root / "captures" / profile) for profile in PROFILES}
    targets = spec["existing_clean_source"].get("native_capture_target_route_progress")
    references = (
        _select_progress_rows(rows["clean"], targets)
        if targets else _select_event_rows(rows["clean"], count=5)
    )
    match_by_profile: Dict[str, List[dict]] = {
        "clean": [{"reference": row, "candidate": row, "distance_m": 0.0, "yaw_error_degrees": 0.0} for row in references]
    }
    for profile in PROFILES[1:]:
        match_by_profile[profile] = _match_rows(references, rows[profile])
        if len(match_by_profile[profile]) != len(references):
            raise RuntimeError("%s did not pose-match every event frame" % profile)

    metrics = {}
    front_gif_frames = []
    bev_gif_frames = []
    contact_rows = []
    residual_sequences = {profile: [] for profile in PROFILES[1:]}
    for index, reference in enumerate(references):
        clean_front = _read_image(reference["front"])
        front_images = []
        bev_images = []
        for profile in PROFILES:
            match = match_by_profile[profile][index]
            front = _read_image(match["candidate"]["front"])
            bev = _read_image(match["candidate"]["bev"])
            front_images.append(front)
            bev_images.append(bev)
            if profile != "clean":
                residual_sequences[profile].append(
                    cv2.resize(cv2.absdiff(front, clean_front), (320, 180), interpolation=cv2.INTER_AREA)
                )
                row_metrics = _frame_metrics(clean_front, front, roi)
                row_metrics.update({
                    "clean_capture_index": reference["capture_index"],
                    "profile_capture_index": match["candidate"]["capture_index"],
                    "pose_distance_m": match["distance_m"],
                    "pose_yaw_error_degrees": match["yaw_error_degrees"],
                })
                metrics.setdefault(profile, []).append(row_metrics)
        tiled_front = _tile(front_images, PROFILES, cell=(480, 270))
        tiled_bev = _tile(bev_images, PROFILES, cell=(320, 320))
        front_gif_frames.append(tiled_front)
        bev_gif_frames.append(tiled_bev)
        contact_rows.append(tiled_front)

    summaries = {}
    for profile in PROFILES[1:]:
        rows_metrics = metrics[profile]
        residuals = residual_sequences[profile]
        temporal_jerk = _mean([
            float(cv2.absdiff(left, right).mean())
            for left, right in zip(residuals[:-1], residuals[1:])
        ])
        summaries[profile] = {
            key: _mean([row[key] for row in rows_metrics])
            for key in (
                "mean_absolute_pixel_delta",
                "saturated_pixel_fraction_clean",
                "saturated_pixel_fraction_changed",
                "hazard_roi_contrast_ratio",
                "hazard_roi_edge_visibility_ratio",
                "rectangular_boundary_artifact_score",
                "pose_distance_m",
                "pose_yaw_error_degrees",
            )
        }
        summaries[profile]["pose_matched_residual_temporal_jerk"] = temporal_jerk

    impact = [summaries[profile]["mean_absolute_pixel_delta"] for profile in PROFILES[1:]]
    native_effect_detected = summaries["heavy"]["mean_absolute_pixel_delta"] >= 0.5
    impact_monotonic = all(left < right for left, right in zip(impact[:-1], impact[1:]))
    contact_sheet = np.concatenate(contact_rows, axis=0)
    contact_path = output / "route151_native_glare_contact_sheet.png"
    front_gif_path = output / "route151_native_glare_front.gif"
    bev_gif_path = output / "route151_native_glare_bev.gif"
    cv2.imwrite(str(contact_path), contact_sheet)
    _save_gif(front_gif_frames, front_gif_path)
    _save_gif(bev_gif_frames, bev_gif_path)

    comparison_stage = spec.get("comparison_image_stage", {
        "stage": "raw CARLA RGB sensor frame saved losslessly as PNG before ORION JPEG/preprocessing",
        "required_for_all_methods": True,
        "legacy_model_input_preview_is_not_eligible": True,
    })
    payload = {
        "schema": SCHEMA,
        "comparison_image_stage": comparison_stage,
        "profiles": list(PROFILES),
        "matched_event_frame_count": len(references),
        "hazard_roi_normalized": roi,
        "per_frame": metrics,
        "summary": summaries,
        "gates": {
            "native_effect_detected": native_effect_detected,
            "candidate_impact_strictly_monotonic": impact_monotonic,
            "eligible_to_freeze_native_severity": bool(native_effect_detected and impact_monotonic),
        },
        "visuals": {
            "contact_sheet": str(contact_path.resolve()),
            "front_gif": str(front_gif_path.resolve()),
            "bev_gif": str(bev_gif_path.resolve()),
        },
        "outcome_fields_read": [],
        "claim_boundary": "outcome-blind visual mechanism bake-off; no safety, UQ, or model claim",
    }
    report = output / "native_glare_visual_analysis.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["gates"], sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root.resolve(), args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

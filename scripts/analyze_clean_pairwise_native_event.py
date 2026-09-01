#!/usr/bin/env python3
"""Analyze all-view observation evidence around an independent native event."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.render_closedloop_observation_uq_heatmap import (
    ABSOLUTE_SCALE_FLOOR,
    BASELINE_END_SECONDS,
    BASELINE_START_SECONDS,
    CAMERA_ORDER,
    RELATIVE_SCALE_FLOOR,
    Z_CENTER,
    _grid_from_row,
    calibrate_spatial_grid,
    fit_spatial_baseline,
)
from scripts.summarize_closedloop_safety import find_control_trace, load_records


SCHEMA_VERSION = "orion.clean_pairwise_native_event_analysis.v1"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise RuntimeError("requested event window contains no observation rows")
    return {
        "frames": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def analyze(
    records: list[dict[str, Any]],
    *,
    event_time_seconds: float,
    lead_seconds: float = 5.0,
    approach_seconds: float = 2.0,
    post_seconds: float = 2.0,
) -> dict[str, Any]:
    if not math.isfinite(event_time_seconds) or event_time_seconds < 0:
        raise ValueError("event_time_seconds must be finite and non-negative")
    if min(lead_seconds, approach_seconds, post_seconds) < 0:
        raise ValueError("window durations must be non-negative")
    if lead_seconds < approach_seconds:
        raise ValueError("lead_seconds must be at least approach_seconds")

    usable = []
    for row in records:
        observation = row.get("observation_uq")
        if not isinstance(observation, dict):
            raise RuntimeError("one or more frames lack observation_uq")
        order = tuple(observation.get("camera_order") or ())
        if order != CAMERA_ORDER:
            raise RuntimeError("camera_order differs from the frozen six-view order")
        scores = np.asarray(
            (observation.get("aggregate") or {}).get("view_raw_scores"),
            dtype=np.float64,
        )
        if scores.shape != (len(CAMERA_ORDER),) or not np.isfinite(scores).all():
            raise RuntimeError("view_raw_scores must contain six finite values")
        grids = []
        for camera_name in CAMERA_ORDER:
            grid = _grid_from_row(row, camera_name)
            if grid is None:
                raise RuntimeError("one or more frames lack an all-view pooled grid")
            grids.append(grid)
        usable.append(
            (
                float(row["sim_time_seconds"]),
                int(row["step"]),
                float(row["route_progress"]),
                scores,
                np.stack(grids),
            )
        )
    if not usable:
        raise RuntimeError("control trace is empty")

    times = np.asarray([item[0] for item in usable], dtype=np.float64)
    scores = np.stack([item[3] for item in usable])
    grids = np.stack([item[4] for item in usable])
    baseline_mask = (
        (times >= BASELINE_START_SECONDS) & (times < BASELINE_END_SECONDS)
    )
    if int(baseline_mask.sum()) < 40:
        raise RuntimeError("fewer than 40 frames in the fixed 1-4 second baseline")

    scalar_median = np.median(scores[baseline_mask], axis=0)
    scalar_mad = np.median(
        np.abs(scores[baseline_mask] - scalar_median[None]), axis=0
    )
    scalar_scale = np.maximum.reduce(
        (
            1.4826 * scalar_mad,
            RELATIVE_SCALE_FLOOR * np.abs(scalar_median),
            np.full_like(scalar_median, ABSOLUTE_SCALE_FLOOR),
        )
    )
    scalar_calibrated = _sigmoid(
        (scores - scalar_median[None]) / scalar_scale[None] - Z_CENTER
    )

    masks = {
        "baseline": baseline_mask,
        "lead": (
            (times >= event_time_seconds - lead_seconds)
            & (times < event_time_seconds - approach_seconds)
        ),
        "approach": (
            (times >= event_time_seconds - approach_seconds)
            & (times < event_time_seconds)
        ),
        "post_event": (
            (times >= event_time_seconds)
            & (times < event_time_seconds + post_seconds)
        ),
    }
    if not all(bool(mask.any()) for mask in masks.values()):
        missing = [name for name, mask in masks.items() if not bool(mask.any())]
        raise RuntimeError(f"event analysis windows contain no frames: {missing}")

    camera_reports = {}
    approach_uplifts = {}
    for view_index, camera_name in enumerate(CAMERA_ORDER):
        spatial_median, spatial_scale = fit_spatial_baseline(
            grids[baseline_mask, view_index]
        )
        spatial_calibrated = np.stack(
            [
                calibrate_spatial_grid(grid, spatial_median, spatial_scale)
                for grid in grids[:, view_index]
            ]
        )
        window_reports = {}
        for name, mask in masks.items():
            raw = _summary(scores[mask, view_index])
            calibrated = _summary(scalar_calibrated[mask, view_index])
            spatial = spatial_calibrated[mask]
            window_reports[name] = {
                "raw": raw,
                "scalar_calibrated": calibrated,
                "spatial_frame_mean": _summary(spatial.mean(axis=(1, 2))),
                "spatial_frame_maximum": _summary(spatial.max(axis=(1, 2))),
                "spatial_window_maximum_cell": float(spatial.max()),
            }
        approach_uplift = (
            window_reports["approach"]["raw"]["mean"]
            - window_reports["baseline"]["raw"]["mean"]
        )
        approach_uplifts[camera_name] = approach_uplift
        trigger_indices = np.flatnonzero(
            (times >= BASELINE_END_SECONDS)
            & (times < event_time_seconds)
            & (scalar_calibrated[:, view_index] >= 0.5)
        )
        first_trigger = None
        if trigger_indices.size:
            index = int(trigger_indices[0])
            first_trigger = {
                "step": usable[index][1],
                "sim_time_seconds": usable[index][0],
                "route_progress": usable[index][2],
                "calibrated_score": float(scalar_calibrated[index, view_index]),
                "lead_to_event_seconds": float(
                    event_time_seconds - usable[index][0]
                ),
            }
        camera_reports[camera_name] = {
            "baseline_robust_median": float(scalar_median[view_index]),
            "baseline_robust_mad": float(scalar_mad[view_index]),
            "baseline_robust_scale": float(scalar_scale[view_index]),
            "approach_raw_mean_uplift_from_baseline": float(approach_uplift),
            "first_pre_event_calibrated_trigger": first_trigger,
            "windows": window_reports,
        }

    ranking = sorted(
        approach_uplifts,
        key=lambda name: approach_uplifts[name],
        reverse=True,
    )
    return {
        "event_time_seconds": float(event_time_seconds),
        "windows": {
            "lead_seconds": float(lead_seconds),
            "approach_seconds": float(approach_seconds),
            "post_seconds": float(post_seconds),
            "baseline_seconds": [BASELINE_START_SECONDS, BASELINE_END_SECONDS],
            "baseline_frames": int(baseline_mask.sum()),
        },
        "camera_reports": camera_reports,
        "approach_uplift_view_ranking": ranking,
        "largest_approach_uplift_camera": ranking[0],
        "calibration": {
            "method": "fixed pre-event median/MAD per view and per grid position",
            "relative_scale_floor": RELATIVE_SCALE_FLOOR,
            "absolute_scale_floor": ABSOLUTE_SCALE_FLOOR,
            "z_center": Z_CENTER,
            "future_frames_used": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--event-time-seconds", type=float, required=True)
    parser.add_argument("--lead-seconds", type=float, default=5.0)
    parser.add_argument("--approach-seconds", type=float, default=2.0)
    parser.add_argument("--post-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite clean pairwise event analysis")
    trace_path = find_control_trace(args.run_dir)
    report = {
        "schema": SCHEMA_VERSION,
        "run_dir": str(args.run_dir.resolve()),
        "trace_path": str(trace_path.resolve()),
        "observation_uq": analyze(
            load_records(trace_path),
            event_time_seconds=args.event_time_seconds,
            lead_seconds=args.lead_seconds,
            approach_seconds=args.approach_seconds,
            post_seconds=args.post_seconds,
        ),
        "claim_boundary": (
            "The event time is supplied independently from evaluator/safety "
            "evidence. View or spatial uplift is task-agnostic observation "
            "evidence, not proof of task relevance, causal responsibility, "
            "or VLM understanding."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Quantify global Density-UQ activation burden on a clean closed-loop trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Callable

from scripts.summarize_closedloop_safety import find_control_trace, load_records


SCHEMA_VERSION = "orion.clean_global_uq_activation.v1"
BASELINE_START_SECONDS = 1.0
BASELINE_END_SECONDS = 4.0
NO_NEAR_TERM_CONFLICT_TTC_SECONDS = 5.0


def _sample_durations(records: list[dict[str, Any]]) -> list[float]:
    times = [float(row["sim_time_seconds"]) for row in records]
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    default = statistics.median(deltas) if deltas else 0.05
    return [
        times[index + 1] - current
        if index + 1 < len(times) and times[index + 1] > current
        else default
        for index, current in enumerate(times)
    ]


def _first(records: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]):
    for row in records:
        if predicate(row):
            return {
                "step": int(row["step"]),
                "sim_time_seconds": float(row["sim_time_seconds"]),
                "route_progress": float(row["route_progress"]),
                "speed_mps": float(row["speed"]),
                "score": float(row["density_uq_score"]),
            }
    return None


def _longest_exposure(
    records: list[dict[str, Any]],
    durations: list[float],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    best_duration = current_duration = 0.0
    best_start = best_end = current_start = None
    for row, duration in zip(records, durations):
        if predicate(row):
            if current_start is None:
                current_start = float(row["sim_time_seconds"])
            current_duration += duration
            current_end = float(row["sim_time_seconds"]) + duration
            if current_duration > best_duration:
                best_duration = current_duration
                best_start, best_end = current_start, current_end
        else:
            current_duration = 0.0
            current_start = None
    return {
        "duration_seconds": best_duration,
        "start_seconds": best_start,
        "end_seconds": best_end,
    }


def analyze(
    records: list[dict[str, Any]],
    *,
    threshold: float = 0.4,
    robust_z_threshold: float = 4.0,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0,1]")
    if robust_z_threshold < 0:
        raise ValueError("robust_z_threshold must be non-negative")
    if any(bool(row.get("corruption_active")) for row in records):
        raise ValueError("clean activation analysis rejects active corruption frames")
    non_off = {
        (row.get("risk") or {}).get("mode")
        for row in records
        if (row.get("risk") or {}).get("mode") not in (None, "off")
    }
    if non_off:
        raise ValueError(f"clean activation analysis requires risk mode off: {non_off}")

    baseline = [
        float(row["density_uq_score"])
        for row in records
        if row.get("density_uq_score") is not None
        and BASELINE_START_SECONDS
        <= float(row["sim_time_seconds"])
        < BASELINE_END_SECONDS
    ]
    if not baseline:
        raise RuntimeError("trace has no fixed 1-4 second Density-UQ baseline")
    median = statistics.median(baseline)
    mad = statistics.median(abs(value - median) for value in baseline)
    scale = max(1.4826 * mad, 0.05 * abs(median), 0.001)
    usable = [
        row
        for row in records
        if row.get("density_uq_score") is not None
        and float(row["sim_time_seconds"]) >= BASELINE_END_SECONDS
    ]
    if not usable:
        raise RuntimeError("trace has no Density-UQ frames after the baseline")
    durations = _sample_durations(usable)
    score_trigger = lambda row: float(row["density_uq_score"]) >= threshold
    robust_trigger = lambda row: (
        float(row["density_uq_score"]) - median
    ) / scale >= robust_z_threshold

    def no_near_term_conflict(row: dict[str, Any]) -> bool:
        safety = row.get("closedloop_safety") or {}
        if safety.get("available") is not True:
            return False
        ttc = safety.get("min_obb_collision_ttc_seconds")
        return ttc is None or float(ttc) > NO_NEAR_TERM_CONFLICT_TTC_SECONDS

    total_duration = sum(durations)
    threshold_duration = sum(
        duration for row, duration in zip(usable, durations) if score_trigger(row)
    )
    robust_duration = sum(
        duration for row, duration in zip(usable, durations) if robust_trigger(row)
    )
    no_conflict_duration = sum(
        duration
        for row, duration in zip(usable, durations)
        if score_trigger(row) and no_near_term_conflict(row)
    )
    stopped_duration = sum(
        duration
        for row, duration in zip(usable, durations)
        if score_trigger(row) and abs(float(row["speed"])) < 0.25
    )
    moving_duration = sum(
        duration
        for row, duration in zip(usable, durations)
        if score_trigger(row) and float(row["speed"]) >= 1.0
    )
    values = [float(row["density_uq_score"]) for row in usable]
    return {
        "threshold": threshold,
        "robust_z_threshold": robust_z_threshold,
        "baseline": {
            "start_seconds": BASELINE_START_SECONDS,
            "end_seconds": BASELINE_END_SECONDS,
            "frames": len(baseline),
            "median": median,
            "mad": mad,
            "robust_scale": scale,
        },
        "post_baseline": {
            "frames": len(usable),
            "duration_seconds": total_duration,
            "minimum_score": min(values),
            "maximum_score": max(values),
            "mean_score": sum(values) / len(values),
            "threshold_exposure_seconds": threshold_duration,
            "threshold_exposure_fraction": threshold_duration / total_duration,
            "robust_z_exposure_seconds": robust_duration,
            "robust_z_exposure_fraction": robust_duration / total_duration,
            "threshold_exposure_while_stopped_seconds": stopped_duration,
            "threshold_exposure_while_moving_seconds": moving_duration,
            "threshold_exposure_without_near_term_obb_conflict_seconds": (
                no_conflict_duration
            ),
            "threshold_exposure_without_near_term_obb_conflict_fraction": (
                no_conflict_duration / threshold_duration
                if threshold_duration else None
            ),
        },
        "first_threshold_trigger": _first(usable, score_trigger),
        "first_robust_z_trigger": _first(usable, robust_trigger),
        "longest_continuous_threshold_exposure": _longest_exposure(
            usable, durations, score_trigger
        ),
        "last_frame": {
            "step": int(usable[-1]["step"]),
            "sim_time_seconds": float(usable[-1]["sim_time_seconds"]),
            "route_progress": float(usable[-1]["route_progress"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--robust-z-threshold", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite clean UQ activation report")
    trace_path = find_control_trace(args.run_dir)
    report = {
        "schema": SCHEMA_VERSION,
        "run_dir": str(args.run_dir.resolve()),
        "trace_path": str(trace_path.resolve()),
        "density_uq": analyze(
            load_records(trace_path),
            threshold=args.threshold,
            robust_z_threshold=args.robust_z_threshold,
        ),
        "claim_boundary": (
            "Threshold activity on a clean observation trace measures the "
            "operational activation burden and limited specificity of this "
            "global anomaly score. A clean frame may still contain genuine "
            "driving risk, so threshold activity alone is not labeled a false "
            "positive. Lack of a near-term OBB conflict is an independent, "
            "limited diagnostic rather than semantic uncertainty ground truth."
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

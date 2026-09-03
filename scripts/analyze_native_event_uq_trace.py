#!/usr/bin/env python3
"""Summarize a global UQ trace around one independently identified event."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

from scripts.summarize_closedloop_safety import find_control_trace, load_records


SCHEMA_VERSION = "orion.native_event_global_uq_trace.v2"
BASELINE_START_SECONDS = 1.0
BASELINE_END_SECONDS = 4.0


def _window(
    records: list[dict[str, Any]],
    start_seconds: float,
    end_seconds: float,
    *,
    threshold: float,
) -> dict[str, Any]:
    values = [
        float(row["density_uq_score"])
        for row in records
        if row.get("density_uq_score") is not None
        and start_seconds <= float(row["sim_time_seconds"]) < end_seconds
    ]
    if not values:
        raise RuntimeError("requested UQ window contains no density scores")
    return {
        "start_seconds": float(start_seconds),
        "end_seconds": float(end_seconds),
        "frames": len(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "frames_at_or_above_threshold": sum(value >= threshold for value in values),
        "fraction_at_or_above_threshold": (
            sum(value >= threshold for value in values) / len(values)
        ),
    }


def analyze(
    records: list[dict[str, Any]],
    *,
    event_time_seconds: float,
    threshold: float = 0.4,
    lead_seconds: float = 5.0,
    approach_seconds: float = 2.0,
    post_seconds: float = 2.0,
) -> dict[str, Any]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0,1]")
    if min(event_time_seconds, lead_seconds, approach_seconds, post_seconds) < 0:
        raise ValueError("event and window times must be non-negative")
    if lead_seconds < approach_seconds:
        raise ValueError("lead_seconds must be at least approach_seconds")
    baseline_values = [
        float(row["density_uq_score"])
        for row in records
        if row.get("density_uq_score") is not None
        and BASELINE_START_SECONDS
        <= float(row["sim_time_seconds"])
        < BASELINE_END_SECONDS
    ]
    if not baseline_values:
        raise RuntimeError("trace has no fixed 1-4 second UQ baseline")
    median = statistics.median(baseline_values)
    mad = statistics.median(abs(value - median) for value in baseline_values)
    scale = max(1.4826 * mad, 0.05 * abs(median), 0.001)
    def trigger_row(row: dict[str, Any]) -> dict[str, Any]:
        score = float(row["density_uq_score"])
        return {
            "step": int(row["step"]),
            "sim_time_seconds": float(row["sim_time_seconds"]),
            "route_progress": float(row["route_progress"]),
            "score": score,
            "robust_z": (score - median) / scale,
        }

    first_trigger_any = next(
        (
            trigger_row(row)
            for row in records
            if row.get("density_uq_score") is not None
            and float(row["density_uq_score"]) >= threshold
        ),
        None,
    )
    first_trigger_post_baseline = next(
        (
            trigger_row(row)
            for row in records
            if row.get("density_uq_score") is not None
            and float(row["sim_time_seconds"]) >= BASELINE_END_SECONDS
            and float(row["density_uq_score"]) >= threshold
        ),
        None,
    )
    first_robust_trigger_post_baseline = next(
        (
            trigger_row(row)
            for row in records
            if row.get("density_uq_score") is not None
            and float(row["sim_time_seconds"]) >= BASELINE_END_SECONDS
            and (
                float(row["density_uq_score"]) - median
            ) / scale >= 4.0
        ),
        None,
    )
    result = {
        "event_time_seconds": float(event_time_seconds),
        "threshold": float(threshold),
        "baseline": {
            **_window(
                records,
                BASELINE_START_SECONDS,
                BASELINE_END_SECONDS,
                threshold=threshold,
            ),
            "robust_median": median,
            "robust_mad": mad,
            "robust_scale": scale,
        },
        "lead": _window(
            records,
            event_time_seconds - lead_seconds,
            event_time_seconds - approach_seconds,
            threshold=threshold,
        ),
        "approach": _window(
            records,
            event_time_seconds - approach_seconds,
            event_time_seconds,
            threshold=threshold,
        ),
        "post_event": _window(
            records,
            event_time_seconds,
            event_time_seconds + post_seconds,
            threshold=threshold,
        ),
        "first_threshold_trigger_any_time": first_trigger_any,
        "first_threshold_trigger_post_baseline": first_trigger_post_baseline,
        "first_robust_z4_trigger_post_baseline": (
            first_robust_trigger_post_baseline
        ),
        "post_baseline_threshold_trigger_lead_seconds": (
            event_time_seconds
            - first_trigger_post_baseline["sim_time_seconds"]
            if first_trigger_post_baseline is not None else None
        ),
        "post_baseline_robust_z4_trigger_lead_seconds": (
            event_time_seconds
            - first_robust_trigger_post_baseline["sim_time_seconds"]
            if first_robust_trigger_post_baseline is not None else None
        ),
    }
    false_trigger_end = event_time_seconds - lead_seconds
    if false_trigger_end > BASELINE_END_SECONDS:
        result["pre_lead_false_trigger_window"] = _window(
            records,
            BASELINE_END_SECONDS,
            false_trigger_end,
            threshold=threshold,
        )
    else:
        result["pre_lead_false_trigger_window"] = None
    for name in ("lead", "approach", "post_event"):
        result[name]["maximum_robust_z"] = (
            result[name]["maximum"] - median
        ) / scale
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--event-time-seconds", type=float, required=True)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--lead-seconds", type=float, default=5.0)
    parser.add_argument("--approach-seconds", type=float, default=2.0)
    parser.add_argument("--post-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite native event UQ report")
    trace_path = find_control_trace(args.run_dir)
    report = {
        "schema": SCHEMA_VERSION,
        "run_dir": str(args.run_dir.resolve()),
        "trace_path": str(trace_path.resolve()),
        "density_uq": analyze(
            load_records(trace_path),
            event_time_seconds=args.event_time_seconds,
            threshold=args.threshold,
            lead_seconds=args.lead_seconds,
            approach_seconds=args.approach_seconds,
            post_seconds=args.post_seconds,
        ),
        "claim_boundary": (
            "A global score that rises before a collision may be an operational "
            "scene-anomaly trigger. This analysis does not establish spatial "
            "localization, task relevance, causal responsibility, or semantic "
            "understanding of uncertainty."
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

#!/usr/bin/env python3
"""Record the preregistered clean-route low-speed fast screen."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from scripts.summarize_closedloop_safety import find_control_trace, load_records


def longest_low_speed_interval(
    records: list[dict[str, Any]],
    *,
    speed_threshold_mps: float = 0.25,
    ignore_before_seconds: float = 2.0,
) -> dict[str, Any]:
    if not records:
        raise ValueError("control trace is empty")
    times = [float(row["sim_time_seconds"]) for row in records]
    positive_deltas = [right - left for left, right in zip(times, times[1:]) if right > left]
    frame_period = statistics.median(positive_deltas) if positive_deltas else 0.05
    intervals = []
    start_index = None
    for index, row in enumerate(records):
        low = (
            float(row["sim_time_seconds"]) >= ignore_before_seconds
            and abs(float(row["speed"])) < speed_threshold_mps
        )
        if low and start_index is None:
            start_index = index
        if not low and start_index is not None:
            intervals.append((start_index, index - 1))
            start_index = None
    if start_index is not None:
        intervals.append((start_index, len(records) - 1))
    if not intervals:
        return {
            "duration_seconds": 0.0,
            "start_step": None,
            "end_step": None,
            "start_time_seconds": None,
            "end_time_seconds": None,
            "start_progress": None,
            "end_progress": None,
        }
    start, end = max(
        intervals,
        key=lambda interval: (
            float(records[interval[1]]["sim_time_seconds"])
            - float(records[interval[0]]["sim_time_seconds"])
            + frame_period
        ),
    )
    first = records[start]
    last = records[end]
    return {
        "duration_seconds": (
            float(last["sim_time_seconds"])
            - float(first["sim_time_seconds"])
            + frame_period
        ),
        "start_step": int(first["step"]),
        "end_step": int(last["step"]),
        "start_time_seconds": float(first["sim_time_seconds"]),
        "end_time_seconds": float(last["sim_time_seconds"]),
        "start_progress": float(first["route_progress"]),
        "end_progress": float(last["route_progress"]),
    }


def evaluate(
    run_dir: Path,
    *,
    maximum_stop_seconds: float = 8.0,
    speed_threshold_mps: float = 0.25,
    ignore_before_seconds: float = 2.0,
) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("pilot_condition") != "clean_off":
        raise RuntimeError("liveness fast screen is restricted to clean_off")
    if manifest.get("orion_closedloop_risk_mode") != "off":
        raise RuntimeError("liveness fast screen requires risk mode off")
    trace_path = find_control_trace(run_dir)
    records = load_records(trace_path)
    interval = longest_low_speed_interval(
        records,
        speed_threshold_mps=speed_threshold_mps,
        ignore_before_seconds=ignore_before_seconds,
    )
    triggered = float(interval["duration_seconds"]) > maximum_stop_seconds
    return {
        "schema": "orion.closedloop_clean_liveness_fast_screen.v1",
        "run_dir": str(run_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "trace_path": str(trace_path.resolve()),
        "trace_frames_observed": len(records),
        "last_observed": {
            "step": int(records[-1]["step"]),
            "sim_time_seconds": float(records[-1]["sim_time_seconds"]),
            "route_progress": float(records[-1]["route_progress"]),
            "speed_mps": float(records[-1]["speed"]),
        },
        "rule": {
            "speed_threshold_mps": speed_threshold_mps,
            "maximum_stop_seconds": maximum_stop_seconds,
            "ignore_before_seconds": ignore_before_seconds,
            "comparison": "trigger when longest interval is strictly greater than maximum",
        },
        "longest_low_speed_interval": interval,
        "fast_screen_triggered": triggered,
        "stage_b_qualified": not triggered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-stop-seconds", type=float, default=8.0)
    parser.add_argument("--speed-threshold-mps", type=float, default=0.25)
    parser.add_argument("--ignore-before-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite liveness fast-screen report")
    report = evaluate(
        args.run_dir,
        maximum_stop_seconds=args.maximum_stop_seconds,
        speed_threshold_mps=args.speed_threshold_mps,
        ignore_before_seconds=args.ignore_before_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "fast_screen_triggered": report["fast_screen_triggered"],
        "stage_b_qualified": report["stage_b_qualified"],
    }, indent=2))
    return 1 if report["fast_screen_triggered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate one Epic clean baseline with actor-grounded safety telemetry."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from scripts.summarize_closedloop_safety import (
    find_control_trace,
    load_records,
    summarize_records,
)


EXPECTED_SAFETY_SCHEMA = "orion.closedloop_dynamic_actor_safety.v2"


def _check(metric, value, expected):
    return {
        "metric": metric,
        "value": value,
        "expected": expected,
        "passed": value == expected,
    }


def _load_eval(run_dir: Path):
    paths = sorted(run_dir.glob("eval*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one evaluator JSON, found {len(paths)}")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    records = payload.get("_checkpoint", {}).get("records", [])
    if len(records) != 1:
        raise RuntimeError("expected one terminal evaluator record")
    return paths[0], payload, records[0]


def _infraction_count(record, names):
    infractions = record.get("infractions", {})
    return sum(
        len(infractions.get(name, []))
        for name in names
        if isinstance(infractions.get(name, []), list)
    )


def _longest_stop_seconds(records, threshold=0.25):
    times = [float(row["sim_time_seconds"]) for row in records]
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    frame_period = float(statistics.median(deltas)) if deltas else 0.05
    longest = current = 0.0
    for row in records:
        if float(row["speed"]) < threshold:
            current += frame_period
            longest = max(longest, current)
        else:
            current = 0.0
    return longest


def evaluate(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eval_path, eval_payload, terminal = _load_eval(run_dir)
    trace_path = find_control_trace(run_dir)
    records = load_records(trace_path)
    scores = terminal.get("scores", {})
    safety_rows = [row.get("closedloop_safety") for row in records]
    steps = [int(row["step"]) for row in records]
    times = [float(row["sim_time_seconds"]) for row in records]
    contiguous_steps = steps == list(range(steps[0], steps[0] + len(steps)))
    monotonic_times = all(right > left for left, right in zip(times, times[1:]))
    safety_available = all(
        safety is not None and safety.get("available") is True
        for safety in safety_rows
    )
    schemas = sorted({
        safety.get("schema")
        for safety in safety_rows
        if safety is not None
    })
    hard_infractions = _infraction_count(terminal, (
        "collisions_layout",
        "collisions_pedestrian",
        "collisions_vehicle",
        "red_light",
        "stop_infraction",
        "route_dev",
        "vehicle_blocked",
        "scenario_timeouts",
        "route_timeout",
    ))
    min_speed_infractions = terminal.get("infractions", {}).get(
        "min_speed_infractions", []
    )
    if not isinstance(min_speed_infractions, list):
        min_speed_infractions = []
    checks = [
        _check("manifest.pilot_condition", manifest.get("pilot_condition"), "clean_off"),
        _check("manifest.carla_quality_level", manifest.get("carla_quality_level"), "Epic"),
        _check("manifest.risk_mode", manifest.get("orion_closedloop_risk_mode"), "off"),
        _check("manifest.uq_mode", manifest.get("orion_closedloop_uq_mode"), "none"),
        _check("manifest.safety_telemetry", manifest.get("orion_closedloop_safety_telemetry"), "1"),
        _check("evaluator.eligible", bool(eval_payload.get("eligible")), True),
        _check("evaluator.status", terminal.get("status"), "Completed"),
        _check("evaluator.score_route", float(scores.get("score_route", -1)), 100.0),
        _check("evaluator.score_penalty", float(scores.get("score_penalty", -1)), 1.0),
        _check("evaluator.hard_infraction_count", hard_infractions, 0),
        _check("trace.steps_contiguous", contiguous_steps, True),
        _check("trace.times_monotonic", monotonic_times, True),
        _check("trace.safety_all_frames_available", safety_available, True),
        _check("trace.safety_schemas", schemas, [EXPECTED_SAFETY_SCHEMA]),
        {
            "metric": "trace.longest_stop_below_0_25_mps_seconds",
            "value": _longest_stop_seconds(records),
            "expected": "<=8.0",
            "passed": _longest_stop_seconds(records) <= 8.0,
        },
    ]
    return {
        "schema": "orion.closedloop_clean_safety_gate.v1",
        "run_dir": str(run_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "eval_path": str(eval_path.resolve()),
        "trace_path": str(trace_path.resolve()),
        "trace_frames": len(records),
        "checks": checks,
        "official_diagnostics": {
            "min_speed_infraction_count": len(min_speed_infractions),
            "min_speed_infractions": min_speed_infractions,
            "gate_role": (
                "reported as an efficiency diagnostic; not counted as a hard "
                "collision or traffic-rule failure"
            ),
        },
        "summary": summarize_records(records),
        "gate_passed": all(check["passed"] for check in checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite clean safety gate report")
    report = evaluate(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "gate_passed": report["gate_passed"],
    }, indent=2))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Apply the frozen p3 trace gate to one governor-off learned signal run."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "orion.pairwise-learned-closedloop-trace-report/v1"


def _load_records(run_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    paths = sorted(run_dir.glob("records_*/**/control_trace.jsonl"))
    if len(paths) != 1:
        raise RuntimeError("expected exactly one control trace below the run directory")
    records = [
        json.loads(line)
        for line in paths[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError("control trace is empty")
    return paths[0], records


def _longest_true_seconds(records, predicate) -> float:
    longest = 0.0
    start = None
    last = None
    step = 0.05
    times = [float(row["sim_time_seconds"]) for row in records]
    deltas = [right - left for left, right in zip(times, times[1:]) if right > left]
    if deltas:
        step = float(statistics.median(deltas))
    for row in records:
        current = float(row["sim_time_seconds"])
        if predicate(row):
            if start is None:
                start = current
            last = current
            longest = max(longest, last - start + step)
        else:
            start = None
            last = None
    return longest


def _check(metric, value, threshold, comparison):
    if comparison == ">=":
        passed = value >= threshold
    elif comparison == "<=":
        passed = value <= threshold
    elif comparison == ">":
        passed = value > threshold
    else:
        raise ValueError("unsupported comparison")
    return {
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "comparison": comparison,
        "passed": bool(passed),
    }


def evaluate(run_dir: Path, preregistration: Path) -> dict[str, Any]:
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    gate = prereg["trace_gate"]
    trace_path, records = _load_records(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("pilot_condition") != "front_corrupt_transient_pairwise_trace":
        raise RuntimeError("run condition is not the frozen pairwise trace")
    if manifest.get("orion_closedloop_risk_mode") != "off":
        raise RuntimeError("trace diagnostic must keep the governor off")
    if (
        manifest.get("orion_observation_uq_checkpoint_sha256")
        != prereg["signal_checkpoint"]["sha256"]
    ):
        raise RuntimeError("trace checkpoint hash differs from preregistration")

    observation_rows = [row for row in records if row.get("observation_uq")]
    if len(observation_rows) != len(records):
        raise RuntimeError("one or more control frames lack observation-UQ output")
    if any(row["risk"]["mode"] != "off" for row in records):
        raise RuntimeError("governor changed during the trace diagnostic")
    if any(row["risk"]["intensity"] != 0.0 for row in records):
        raise RuntimeError("governor intervened during the trace diagnostic")

    event_indices = [index for index, row in enumerate(records) if row["corruption_active"]]
    if not event_indices:
        raise RuntimeError("trace never reached the frozen dropout event")
    first_event = event_indices[0]
    last_event = event_indices[-1]
    if event_indices != list(range(first_event, last_event + 1)):
        raise RuntimeError("dropout event is not one contiguous interval")
    event = records[first_event : last_event + 1]
    before = records[:first_event]
    after = records[last_event + 1 :]
    if not after:
        raise RuntimeError("trace ended before post-event recovery could be measured")

    threshold = float(gate["trigger_threshold"])
    event_start = float(event[0]["sim_time_seconds"])
    event_end = float(after[0]["sim_time_seconds"])
    detection_rows = [row for row in event if float(row["raw_uq_score"]) >= threshold]
    detection_latency = (
        float(detection_rows[0]["sim_time_seconds"]) - event_start
        if detection_rows else float("inf")
    )
    recovery_rows = [row for row in after if float(row["raw_uq_score"]) < threshold]
    recovery_latency = (
        float(recovery_rows[0]["sim_time_seconds"]) - event_end
        if recovery_rows else float("inf")
    )
    event_scores = [float(row["raw_uq_score"]) for row in event]
    event_coverage = sum(score >= threshold for score in event_scores) / len(event_scores)
    event_median = float(statistics.median(event_scores))
    frozen_before_event = [
        row for row in before
        if row["observation_uq"]["calibration"]["baseline_frozen"]
    ]
    false_trigger_seconds = _longest_true_seconds(
        frozen_before_event,
        lambda row: float(row["raw_uq_score"]) >= threshold,
    )
    baseline_count = max(
        int(row["observation_uq"]["calibration"]["baseline_count"])
        for row in records
    )

    baseline_rows = [
        row for row in before
        if 1.0 <= float(row["sim_time_seconds"]) < 4.0
    ]
    if not baseline_rows:
        raise RuntimeError("trace lacks the frozen baseline interval")
    baseline_views = list(zip(*[
        row["observation_uq"]["aggregate"]["view_raw_scores"]
        for row in baseline_rows
    ]))
    event_views = list(zip(*[
        row["observation_uq"]["aggregate"]["view_raw_scores"]
        for row in event
    ]))
    uplifts = [
        float(
            sum(event_values) / len(event_values)
            - sum(baseline_values) / len(baseline_values)
        )
        for event_values, baseline_values in zip(event_views, baseline_views)
    ]
    front_index = int(
        records[0]["observation_uq"]["front_view_index"]
    )
    nonfront = [value for index, value in enumerate(uplifts) if index != front_index]
    front_selective = bool(nonfront) and uplifts[front_index] > max(nonfront)

    checks = [
        _check(
            "baseline_frames",
            baseline_count,
            int(gate["minimum_baseline_frames"]),
            ">=",
        ),
        _check(
            "pre_event_sustained_false_trigger_seconds",
            false_trigger_seconds,
            float(gate["pre_event_sustained_false_trigger_max_seconds"]),
            "<=",
        ),
        _check(
            "detection_latency_seconds",
            detection_latency,
            float(gate["maximum_detection_latency_seconds"]),
            "<=",
        ),
        _check(
            "event_trigger_coverage",
            event_coverage,
            float(gate["minimum_event_trigger_coverage"]),
            ">=",
        ),
        _check(
            "event_median_score",
            event_median,
            float(gate["minimum_event_median_score"]),
            ">=",
        ),
        _check(
            "post_event_recovery_seconds",
            recovery_latency,
            float(gate["maximum_post_event_recovery_seconds"]),
            "<=",
        ),
        {
            "metric": "front_raw_uplift_exceeds_every_nonfront_uplift",
            "value": uplifts[front_index],
            "largest_nonfront_uplift": max(nonfront),
            "comparison": ">",
            "passed": front_selective,
        },
    ]
    passed = all(row["passed"] for row in checks)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir.resolve()),
        "trace_path": str(trace_path.resolve()),
        "trace_steps": len(records),
        "event": {
            "start_seconds": event_start,
            "end_seconds": event_end,
            "frames": len(event),
        },
        "view_raw_uplifts": uplifts,
        "checks": checks,
        "gate_passed": passed,
        "scope_attestation": {
            "governor_mode": "off",
            "control_intervention": False,
            "known_corruption_state_used_for_scoring_or_calibration": False,
            "known_corruption_state_used_for_evaluation_only": True,
            "spatial_uq_gate_reclassified_as_passed": False,
        },
        "decision": (
            "submit_single_pairwise_controlled_stop"
            if passed
            else "stop_before_pairwise_learned_control"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite pairwise trace report")
    report = evaluate(args.run_dir, args.preregistration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "gate_passed": report["gate_passed"],
        "decision": report["decision"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

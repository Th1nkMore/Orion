#!/usr/bin/env python3
"""Summarize geometry-backed safety and efficiency from closed-loop traces."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "orion.closedloop-safety-summary.v1"
TTC_THRESHOLDS_SECONDS = (1.0, 2.0, 3.0, 5.0)


def find_control_trace(run_dir: Path) -> Path:
    paths = sorted(run_dir.glob("records_*/**/control_trace.jsonl"))
    if len(paths) != 1:
        raise RuntimeError(
            f"expected exactly one control trace below {run_dir}, found {len(paths)}"
        )
    return paths[0]


def load_records(trace_path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"empty control trace: {trace_path}")
    return records


def _sample_durations(records: list[dict[str, Any]]) -> list[float]:
    times = [float(row["sim_time_seconds"]) for row in records]
    positive_deltas = [
        right - left
        for left, right in zip(times, times[1:])
        if right > left
    ]
    default_step = float(statistics.median(positive_deltas)) if positive_deltas else 0.05
    durations = []
    for index, current in enumerate(times):
        if index + 1 < len(times) and times[index + 1] > current:
            durations.append(times[index + 1] - current)
        else:
            durations.append(default_step)
    return durations


def _minimum_present(values):
    present = [float(value) for value in values if value is not None]
    return min(present) if present else None


def _category_frame_metric(safety, category: str, key: str):
    if safety is None or safety.get("available") is not True:
        return None
    values = [
        actor.get(key)
        for actor in safety.get("actors", [])
        if actor.get("category") == category and actor.get(key) is not None
    ]
    return min(float(value) for value in values) if values else None


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    durations = _sample_durations(records)
    safety_rows = [row.get("closedloop_safety") for row in records]
    available = [
        safety
        for safety in safety_rows
        if safety is not None and safety.get("available") is True
    ]
    if not available:
        raise RuntimeError(
            "trace has no available closedloop_safety telemetry; rerun with the "
            "geometry logger enabled"
        )
    obb_ttc = [
        safety.get("min_obb_collision_ttc_seconds")
        if safety is not None and safety.get("available") is True else None
        for safety in safety_rows
    ]
    exposure = {
        str(threshold): sum(
            duration
            for duration, value in zip(durations, obb_ttc)
            if value is not None and float(value) <= threshold
        )
        for threshold in TTC_THRESHOLDS_SECONDS
    }
    speeds = [float(row["speed"]) for row in records]
    total_duration = sum(durations)
    time_weighted_mean_speed = (
        sum(speed * duration for speed, duration in zip(speeds, durations))
        / total_duration
    )
    first_progress = float(records[0]["route_progress"])
    last_progress = float(records[-1]["route_progress"])

    critical = None
    critical_candidates = [
        (float(value), index)
        for index, value in enumerate(obb_ttc)
        if value is not None
    ]
    if critical_candidates:
        _, index = min(critical_candidates)
        critical = {
            "step": int(records[index]["step"]),
            "sim_time_seconds": float(records[index]["sim_time_seconds"]),
            "actor": safety_rows[index].get("critical_actor"),
        }

    by_category = {}
    for category in ("walker", "vehicle"):
        category_ttc = [
            _category_frame_metric(
                safety, category, "obb_collision_ttc_seconds"
            )
            for safety in safety_rows
        ]
        category_gap = [
            _category_frame_metric(
                safety, category, "obb_separating_axis_gap_m"
            )
            for safety in safety_rows
        ]
        by_category[category] = {
            "frames_with_actor_record": sum(
                any(
                    actor.get("category") == category
                    for actor in safety.get("actors", [])
                )
                for safety in available
            ),
            "min_obb_ttc_seconds": _minimum_present(category_ttc),
            "low_ttc_exposure_seconds": {
                str(threshold): sum(
                    duration
                    for duration, value in zip(durations, category_ttc)
                    if value is not None and float(value) <= threshold
                )
                for threshold in TTC_THRESHOLDS_SECONDS
            },
            "min_obb_separating_axis_gap_m": _minimum_present(category_gap),
        }

    return {
        "frames": len(records),
        "duration_seconds": total_duration,
        "safety_telemetry_available_frames": len(available),
        "safety_telemetry_coverage": len(available) / len(records),
        "safety": {
            "primary_metric": "constant_velocity_fixed_orientation_obb_ttc",
            "min_obb_ttc_seconds": _minimum_present(obb_ttc),
            "low_ttc_exposure_seconds": exposure,
            "min_obb_separating_axis_gap_m": _minimum_present(
                safety.get("min_obb_separating_axis_gap_m") for safety in available
            ),
            "min_disc_clearance_m_diagnostic": _minimum_present(
                safety.get("min_disc_clearance_m") for safety in available
            ),
            "critical_frame": critical,
            "by_category": by_category,
        },
        "efficiency": {
            "time_weighted_mean_speed_mps": time_weighted_mean_speed,
            "stopped_below_0_25_mps_seconds": sum(
                duration
                for duration, speed in zip(durations, speeds)
                if speed < 0.25
            ),
            "slow_below_1_mps_seconds": sum(
                duration
                for duration, speed in zip(durations, speeds)
                if speed < 1.0
            ),
            "route_progress_delta": last_progress - first_progress,
        },
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    trace_path = find_control_trace(run_dir)
    return {
        "run_dir": str(run_dir.resolve()),
        "trace_path": str(trace_path.resolve()),
        "summary": summarize_records(load_records(trace_path)),
    }


def _select_step_interval(records, start_step: int, end_step: int):
    selected = [
        row for row in records
        if start_step <= int(row["step"]) <= end_step
    ]
    expected = end_step - start_step + 1
    if len(selected) != expected:
        raise RuntimeError(
            f"trace interval {start_step}:{end_step} has {len(selected)} "
            f"rows, expected {expected}"
        )
    return selected


def _paired_prefix_alignment(
    clean_records: list[dict[str, Any]],
    degraded_records: list[dict[str, Any]],
    event_start_step: int,
):
    """Quantify replay agreement before the intervention becomes active."""

    first_step = max(
        int(clean_records[0]["step"]), int(degraded_records[0]["step"])
    )
    final_step = int(event_start_step) - 1
    if final_step < first_step:
        raise RuntimeError("paired trace has no pre-event alignment prefix")
    clean = _select_step_interval(clean_records, first_step, final_step)
    degraded = _select_step_interval(degraded_records, first_step, final_step)
    step_sequences_equal = [int(row["step"]) for row in clean] == [
        int(row["step"]) for row in degraded
    ]
    if not step_sequences_equal:
        raise RuntimeError("clean and degraded pre-event steps do not align")

    fields = ["route_progress", "speed"]
    if all("steer" in row for row in clean + degraded):
        fields.append("steer")
    absolute_deltas = {
        field: [
            abs(float(clean_row[field]) - float(degraded_row[field]))
            for clean_row, degraded_row in zip(clean, degraded)
        ]
        for field in fields
    }
    clean_by_step = {int(row["step"]): row for row in clean_records}
    degraded_by_step = {int(row["step"]): row for row in degraded_records}
    if event_start_step not in clean_by_step or event_start_step not in degraded_by_step:
        raise RuntimeError("event start step is absent from a paired trace")
    return {
        "start_step": first_step,
        "end_step": final_step,
        "frames": len(clean),
        "control_step_sequences_equal": step_sequences_equal,
        "degraded_corruption_absent": all(
            not row.get("corruption_active") for row in degraded
        ),
        "absolute_delta": {
            field: {
                "mean": sum(values) / len(values),
                "max": max(values),
                "last": values[-1],
            }
            for field, values in absolute_deltas.items()
        },
        "event_start_route_progress": {
            "clean": float(clean_by_step[event_start_step]["route_progress"]),
            "degraded": float(degraded_by_step[event_start_step]["route_progress"]),
        },
        "interpretation": (
            "diagnostic replay-alignment evidence only; no post-hoc pass "
            "threshold is applied"
        ),
    }


def build_paired_event_report(
    clean_records: list[dict[str, Any]],
    degraded_records: list[dict[str, Any]],
    recovery_seconds: float = 3.0,
):
    active = [row for row in degraded_records if row.get("corruption_active")]
    if not active:
        raise RuntimeError("degraded trace has no active corruption event")
    active_steps = [int(row["step"]) for row in active]
    first_step, last_step = active_steps[0], active_steps[-1]
    if active_steps != list(range(first_step, last_step + 1)):
        raise RuntimeError("degraded corruption event is not one contiguous interval")
    degraded_durations = _sample_durations(degraded_records)
    frame_period = float(statistics.median(degraded_durations))
    recovery_frames = int(round(float(recovery_seconds) / frame_period))
    final_common_step = min(
        int(clean_records[-1]["step"]), int(degraded_records[-1]["step"])
    )
    response_end_step = min(last_step + recovery_frames, final_common_step)

    def interval(start, end):
        clean_summary = summarize_records(
            _select_step_interval(clean_records, start, end)
        )
        degraded_summary = summarize_records(
            _select_step_interval(degraded_records, start, end)
        )
        clean_wrapper = {"summary": clean_summary}
        degraded_wrapper = {"summary": degraded_summary}
        return {
            "start_step": start,
            "end_step": end,
            "clean": clean_summary,
            "degraded": degraded_summary,
            "comparison": compare_summaries(clean_wrapper, degraded_wrapper),
        }

    return {
        "alignment": "control_step",
        "frame_period_seconds": frame_period,
        "event_start_sim_time_seconds": float(active[0]["sim_time_seconds"]),
        "event_end_sim_time_seconds": float(active[-1]["sim_time_seconds"])
        + frame_period,
        "active_event": interval(first_step, last_step),
        "event_plus_recovery": interval(first_step, response_end_step),
        "pre_event_alignment": _paired_prefix_alignment(
            clean_records, degraded_records, first_step
        ),
        "recovery_seconds_requested": float(recovery_seconds),
    }


def _difference(left, right):
    if left is None or right is None:
        return None
    return float(right) - float(left)


def compare_summaries(clean: dict[str, Any], degraded: dict[str, Any]):
    clean_summary = clean["summary"]
    degraded_summary = degraded["summary"]
    clean_safety = clean_summary["safety"]
    degraded_safety = degraded_summary["safety"]
    clean_efficiency = clean_summary["efficiency"]
    degraded_efficiency = degraded_summary["efficiency"]
    by_category = {}
    for category in sorted(clean_safety.get("by_category", {})):
        clean_category = clean_safety["by_category"][category]
        degraded_category = degraded_safety["by_category"][category]
        by_category[category] = {
            "min_obb_ttc_seconds": _difference(
                clean_category["min_obb_ttc_seconds"],
                degraded_category["min_obb_ttc_seconds"],
            ),
            "min_obb_separating_axis_gap_m": _difference(
                clean_category["min_obb_separating_axis_gap_m"],
                degraded_category["min_obb_separating_axis_gap_m"],
            ),
            "low_ttc_exposure_seconds": {
                threshold: _difference(
                    clean_category["low_ttc_exposure_seconds"][threshold],
                    degraded_category["low_ttc_exposure_seconds"][threshold],
                )
                for threshold in clean_category["low_ttc_exposure_seconds"]
            },
        }
    return {
        "delta_definition": "degraded_minus_clean",
        "min_obb_ttc_seconds": _difference(
            clean_safety["min_obb_ttc_seconds"],
            degraded_safety["min_obb_ttc_seconds"],
        ),
        "low_ttc_exposure_seconds": {
            threshold: _difference(
                clean_safety["low_ttc_exposure_seconds"][threshold],
                degraded_safety["low_ttc_exposure_seconds"][threshold],
            )
            for threshold in clean_safety["low_ttc_exposure_seconds"]
        },
        "min_obb_separating_axis_gap_m": _difference(
            clean_safety["min_obb_separating_axis_gap_m"],
            degraded_safety["min_obb_separating_axis_gap_m"],
        ),
        "by_category": by_category,
        "stopped_below_0_25_mps_seconds": _difference(
            clean_efficiency["stopped_below_0_25_mps_seconds"],
            degraded_efficiency["stopped_below_0_25_mps_seconds"],
        ),
        "time_weighted_mean_speed_mps": _difference(
            clean_efficiency["time_weighted_mean_speed_mps"],
            degraded_efficiency["time_weighted_mean_speed_mps"],
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--degraded-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite closed-loop safety summary")
    clean_trace = find_control_trace(args.clean_run)
    degraded_trace = find_control_trace(args.degraded_run)
    clean_records = load_records(clean_trace)
    degraded_records = load_records(degraded_trace)
    clean = {
        "run_dir": str(args.clean_run.resolve()),
        "trace_path": str(clean_trace.resolve()),
        "summary": summarize_records(clean_records),
    }
    degraded = {
        "run_dir": str(args.degraded_run.resolve()),
        "trace_path": str(degraded_trace.resolve()),
        "summary": summarize_records(degraded_records),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "clean": clean,
        "degraded": degraded,
        "comparison": compare_summaries(clean, degraded),
        "paired_event": build_paired_event_report(
            clean_records, degraded_records
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

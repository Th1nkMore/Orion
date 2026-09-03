#!/usr/bin/env python3
"""Classify one preregistered native closed-loop difficulty replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_clean_liveness_screen import longest_low_speed_interval
from scripts.summarize_closedloop_safety import (
    find_control_trace,
    load_records,
    summarize_records,
)


SCHEMA_VERSION = "orion.closedloop_native_collision_discovery_result.v1"
EXPECTED_SAFETY_SCHEMA = "orion.closedloop_dynamic_actor_safety.v2"
COLLISION_KEYS = (
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
)
SERIOUS_INFRACTION_KEYS = (
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "route_dev",
    "scenario_timeouts",
    "route_timeout",
)
MAXIMUM_CONTINUOUS_STOP_SECONDS = 8.0
MINIMUM_TTC_THRESHOLD_SECONDS = 1.0
TTC_EXPOSURE_THRESHOLD_SECONDS = 2.0
MINIMUM_TTC_EXPOSURE_SECONDS = 0.5
MAXIMUM_GAP_THRESHOLD_M = 0.5


def _count_entries(infractions: dict[str, Any], keys: tuple[str, ...]) -> int:
    return sum(
        len(infractions.get(key, []))
        for key in keys
        if isinstance(infractions.get(key, []), list)
    )


def _load_terminal(run_dir: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    paths = sorted(run_dir.glob("eval*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one evaluator JSON, found {len(paths)}")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    records = payload.get("_checkpoint", {}).get("records", [])
    if len(records) != 1:
        raise RuntimeError("expected one terminal evaluator record")
    return paths[0], payload, records[0]


def severe_ttc_surrogate(summary: dict[str, Any]) -> dict[str, Any]:
    safety = summary["safety"]
    min_ttc = safety.get("min_obb_ttc_seconds")
    exposure = safety.get("low_ttc_exposure_seconds", {}).get(
        str(TTC_EXPOSURE_THRESHOLD_SECONDS)
    )
    min_gap = safety.get("min_obb_separating_axis_gap_m")
    checks = {
        "minimum_obb_ttc_seconds": {
            "value": min_ttc,
            "required": f"<={MINIMUM_TTC_THRESHOLD_SECONDS}",
            "passed": min_ttc is not None
            and float(min_ttc) <= MINIMUM_TTC_THRESHOLD_SECONDS,
        },
        "ttc_lte_2_exposure_seconds": {
            "value": exposure,
            "required": f">={MINIMUM_TTC_EXPOSURE_SECONDS}",
            "passed": exposure is not None
            and float(exposure) >= MINIMUM_TTC_EXPOSURE_SECONDS,
        },
        "minimum_obb_gap_m": {
            "value": min_gap,
            "required": f"<={MAXIMUM_GAP_THRESHOLD_M}",
            "passed": min_gap is not None
            and float(min_gap) <= MAXIMUM_GAP_THRESHOLD_M,
        },
    }
    return {
        "logic": "all_required",
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }


def first_geometry_contact(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first observed zero-gap OBB frame, if any."""

    for row in records:
        safety = row.get("closedloop_safety") or {}
        gap = safety.get("min_obb_separating_axis_gap_m")
        if gap is None or float(gap) > 1e-6:
            continue
        actor = safety.get("critical_actor") or {}
        return {
            "step": int(row["step"]),
            "sim_time_seconds": float(row["sim_time_seconds"]),
            "route_progress": float(row["route_progress"]),
            "speed_mps": float(row["speed"]),
            "actor_id": actor.get("actor_id"),
            "actor_type": actor.get("type_id"),
            "actor_category": actor.get("category"),
            "minimum_obb_ttc_seconds": safety.get(
                "min_obb_collision_ttc_seconds"
            ),
            "minimum_obb_gap_m": gap,
        }
    return None


def evaluate_cancelled_partial(
    run_dir: Path,
    *,
    slurm_job_id: str,
    slurm_state: str,
    invalid_reason: str = "clean_liveness_fast_screen_triggered",
) -> dict[str, Any]:
    """Record diagnostics from a cancelled nonterminal replay."""

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trace_path = find_control_trace(run_dir)
    records = load_records(trace_path)
    summary = summarize_records(records)
    stop_interval = longest_low_speed_interval(records)
    contact = first_geometry_contact(records)
    ttc_surrogate = severe_ttc_surrogate(summary)
    steps = [int(row["step"]) for row in records]
    contiguous = steps == list(range(steps[0], steps[0] + len(steps)))
    safety_rows = [row.get("closedloop_safety") for row in records]
    safety_coverage = sum(
        row is not None and row.get("available") is True for row in safety_rows
    ) / len(safety_rows)
    liveness_triggered = (
        float(stop_interval["duration_seconds"])
        > MAXIMUM_CONTINUOUS_STOP_SECONDS
    )
    observed_serious_geometry = contact is not None or bool(
        ttc_surrogate["passed"]
    )
    if invalid_reason == "clean_liveness_fast_screen_triggered":
        result_suffix = "liveness_invalid"
    else:
        result_suffix = "runtime_environment_invalid"
    return {
        "schema": SCHEMA_VERSION,
        "run_dir": str(run_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "trace_path": str(trace_path.resolve()),
        "slurm": {
            "job_id": str(slurm_job_id),
            "state": str(slurm_state),
        },
        "official_endpoint_available": False,
        "runtime": {
            "valid": False,
            "invalid_reason": invalid_reason,
            "condition_clean_off": manifest.get("pilot_condition") == "clean_off",
            "risk_mode_off": manifest.get("orion_closedloop_risk_mode") == "off",
            "uq_mode_none": manifest.get("orion_closedloop_uq_mode") == "none",
            "trace_steps_contiguous": contiguous,
            "trace_frames": len(records),
            "safety_telemetry_coverage": safety_coverage,
            "longest_low_speed_interval": stop_interval,
            "maximum_allowed_stop_seconds": MAXIMUM_CONTINUOUS_STOP_SECONDS,
            "liveness_triggered": liveness_triggered,
        },
        "continuous_safety": summary,
        "first_geometry_contact": contact,
        "severe_ttc_surrogate": ttc_surrogate,
        "observed_serious_geometry_in_partial_trace": observed_serious_geometry,
        "result_kind": (
            f"partial_contact_or_severe_ttc_observed_but_{result_suffix}"
            if observed_serious_geometry
            else f"partial_no_serious_event_and_{result_suffix}"
        ),
        "retain_as_mechanism_development_case": False,
        "claim_boundary": (
            "A partial trace may establish that contact-like geometry or severe "
            "TTC occurred, but cancellation before a terminal evaluator record "
            "means it is not an official completed collision result and is not "
            "eligible for the retained development-case set."
        ),
    }


def evaluate(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eval_path, eval_payload, terminal = _load_terminal(run_dir)
    trace_path = find_control_trace(run_dir)
    records = load_records(trace_path)
    summary = summarize_records(records)
    scores = terminal.get("scores", {})
    infractions = terminal.get("infractions", {})
    if not isinstance(infractions, dict):
        infractions = {}

    steps = [int(row["step"]) for row in records]
    contiguous = steps == list(range(steps[0], steps[0] + len(steps)))
    safety_rows = [row.get("closedloop_safety") for row in records]
    safety_coverage = sum(
        row is not None and row.get("available") is True for row in safety_rows
    ) / len(safety_rows)
    safety_schemas = sorted(
        {
            row.get("schema")
            for row in safety_rows
            if row is not None and row.get("available") is True
        }
    )
    stop_interval = longest_low_speed_interval(records)
    contact = first_geometry_contact(records)

    runtime_checks = {
        "condition_clean_off": manifest.get("pilot_condition") == "clean_off",
        "risk_mode_off": manifest.get("orion_closedloop_risk_mode") == "off",
        "uq_mode_none": manifest.get("orion_closedloop_uq_mode") == "none",
        "evaluator_eligible": bool(eval_payload.get("eligible")) is True,
        "terminal_status_recorded": bool(str(terminal.get("status", "")).strip()),
        "route_completion_recorded": scores.get("score_route") is not None,
        "trace_steps_contiguous": contiguous,
        "safety_telemetry_full_coverage": safety_coverage == 1.0,
        "safety_schema_v2": safety_schemas == [EXPECTED_SAFETY_SCHEMA],
    }
    runtime_valid = all(runtime_checks.values())
    collision_count = _count_entries(infractions, COLLISION_KEYS)
    serious_infraction_count = _count_entries(
        infractions, SERIOUS_INFRACTION_KEYS
    )
    ttc_surrogate = severe_ttc_surrogate(summary)

    if not runtime_valid:
        result_kind = "runtime_or_clean_liveness_invalid"
    elif collision_count:
        result_kind = "collision_reproduced"
    elif serious_infraction_count:
        result_kind = "serious_infraction_reproduced"
    elif ttc_surrogate["passed"]:
        result_kind = "severe_ttc_surrogate_reproduced"
    else:
        result_kind = "no_preregistered_serious_event"

    retained = runtime_valid and (
        collision_count > 0
        or serious_infraction_count > 0
        or bool(ttc_surrogate["passed"])
    )
    return {
        "schema": SCHEMA_VERSION,
        "run_dir": str(run_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "eval_path": str(eval_path.resolve()),
        "trace_path": str(trace_path.resolve()),
        "runtime": {
            "valid": runtime_valid,
            "checks": runtime_checks,
            "trace_frames": len(records),
            "safety_telemetry_coverage": safety_coverage,
            "safety_schemas": safety_schemas,
            "longest_low_speed_interval": stop_interval,
            "low_speed_role": (
                "outcome diagnostic only under amendment 2026-08-28T23:26:53+08:00; "
                "not an automatic runtime exclusion"
            ),
            "post_contact_liveness_exception_applied": bool(
                contact is not None
                and float(stop_interval["duration_seconds"])
                > MAXIMUM_CONTINUOUS_STOP_SECONDS
            ),
        },
        "endpoint": {
            "status": terminal.get("status"),
            "scores": scores,
            "collision_count": collision_count,
            "collision_entries": {
                key: infractions.get(key, []) for key in COLLISION_KEYS
            },
            "serious_infraction_count": serious_infraction_count,
            "serious_infraction_entries": {
                key: infractions.get(key, []) for key in SERIOUS_INFRACTION_KEYS
            },
            "min_speed_infractions": infractions.get(
                "min_speed_infractions", []
            ),
        },
        "continuous_safety": summary,
        "first_geometry_contact": contact,
        "severe_ttc_surrogate": ttc_surrogate,
        "result_kind": result_kind,
        "retain_as_mechanism_development_case": retained,
        "claim_boundary": (
            "Development/event-discovery evidence selected from published ORION "
            "outcomes; not held-out confirmation and not evidence that observation "
            "uncertainty caused the event."
        ),
        "protocol_amendment": (
            "native_collision_discovery_amendments_v1: low speed, legal waiting, "
            "post-contact blocking, timeout, and incomplete route progress are "
            "model/outcome diagnostics rather than automatic runtime exclusions"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial-cancelled-job-id")
    parser.add_argument("--partial-slurm-state", default="CANCELLED")
    parser.add_argument(
        "--partial-invalid-reason",
        default="clean_liveness_fast_screen_triggered",
        choices=(
            "clean_liveness_fast_screen_triggered",
            "runtime_environment_failure",
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite native collision discovery report")
    if args.partial_cancelled_job_id:
        report = evaluate_cancelled_partial(
            args.run_dir,
            slurm_job_id=args.partial_cancelled_job_id,
            slurm_state=args.partial_slurm_state,
            invalid_reason=args.partial_invalid_reason,
        )
    else:
        report = evaluate(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "runtime_valid": report["runtime"]["valid"],
                "result_kind": report["result_kind"],
                "retained": report["retain_as_mechanism_development_case"],
            },
            indent=2,
        )
    )
    return 0 if report["runtime"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

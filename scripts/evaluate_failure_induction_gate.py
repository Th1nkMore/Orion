#!/usr/bin/env python3
"""Apply the frozen hard/surrogate failure-induction gate to one pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.summarize_closedloop_safety import (
    build_paired_event_report,
    find_control_trace,
    load_records,
)


HARD_INFRACTION_KEYS = (
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "route_dev",
    "vehicle_blocked",
    "scenario_timeouts",
    "route_timeout",
)


def _load_one_json(root: Path, pattern: str) -> tuple[Path, dict[str, Any]]:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {pattern!r} below {root}, found {len(paths)}")
    return paths[0], json.loads(paths[0].read_text(encoding="utf-8"))


def _terminal_record(evaluation: dict[str, Any]) -> dict[str, Any]:
    records = evaluation.get("_checkpoint", {}).get("records", [])
    if len(records) != 1:
        raise RuntimeError("expected one evaluator terminal record")
    return records[0]


def _infraction_count(record: dict[str, Any], key: str) -> int:
    value = record.get("infractions", {}).get(key, [])
    if isinstance(value, list):
        return len(value)
    return int(bool(value))


def hard_endpoint_comparison(
    clean_eval: dict[str, Any], degraded_eval: dict[str, Any]
) -> dict[str, Any]:
    clean = _terminal_record(clean_eval)
    degraded = _terminal_record(degraded_eval)
    clean_scores = clean.get("scores", {})
    degraded_scores = degraded.get("scores", {})
    checks = {
        "eligibility_or_completion_worse": (
            bool(clean_eval.get("eligible"))
            and clean.get("status") == "Completed"
            and (
                not bool(degraded_eval.get("eligible"))
                or degraded.get("status") != "Completed"
            )
        ),
        "route_completion_lower": (
            float(degraded_scores.get("score_route", 0.0))
            < float(clean_scores.get("score_route", 0.0))
        ),
        "score_penalty_lower": (
            float(degraded_scores.get("score_penalty", 0.0))
            < float(clean_scores.get("score_penalty", 0.0))
        ),
    }
    infraction_deltas = {
        key: _infraction_count(degraded, key) - _infraction_count(clean, key)
        for key in HARD_INFRACTION_KEYS
    }
    checks["hard_infraction_count_higher"] = any(
        delta > 0 for delta in infraction_deltas.values()
    )
    return {
        "degraded": any(checks.values()),
        "checks": checks,
        "infraction_count_delta_degraded_minus_clean": infraction_deltas,
        "clean": {
            "eligible": bool(clean_eval.get("eligible")),
            "status": clean.get("status"),
            "score_route": float(clean_scores.get("score_route", 0.0)),
            "score_penalty": float(clean_scores.get("score_penalty", 0.0)),
        },
        "degraded_run": {
            "eligible": bool(degraded_eval.get("eligible")),
            "status": degraded.get("status"),
            "score_route": float(degraded_scores.get("score_route", 0.0)),
            "score_penalty": float(degraded_scores.get("score_penalty", 0.0)),
        },
    }


def surrogate_comparison(
    paired_event: dict[str, Any],
    *,
    actor_category: str,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    interval = paired_event["event_plus_recovery"]
    clean = interval["clean"]["safety"]["by_category"][actor_category]
    degraded = interval["degraded"]["safety"]["by_category"][actor_category]
    clean_ttc = clean["min_obb_ttc_seconds"]
    degraded_ttc = degraded["min_obb_ttc_seconds"]
    finite_ttc = clean_ttc is not None and degraded_ttc is not None
    required_drop = None
    observed_drop = None
    if finite_ttc:
        required_drop = max(
            float(thresholds["minimum_ttc_drop_seconds_floor"]),
            float(thresholds["minimum_ttc_drop_fraction_of_clean"])
            * float(clean_ttc),
        )
        observed_drop = float(clean_ttc) - float(degraded_ttc)
    clean_exposure = float(clean["low_ttc_exposure_seconds"]["2.0"])
    degraded_exposure = float(degraded["low_ttc_exposure_seconds"]["2.0"])
    exposure_increase = degraded_exposure - clean_exposure
    checks = {
        "finite_category_specific_minimum_ttc": finite_ttc,
        "minimum_ttc_drop_large_enough": (
            finite_ttc and observed_drop >= required_drop
        ),
        "ttc_le_2_exposure_increase_large_enough": (
            exposure_increase
            >= float(thresholds["minimum_ttc_le_2_exposure_increase_seconds"])
        ),
    }
    return {
        "degraded": all(checks.values()),
        "actor_category": actor_category,
        "checks": checks,
        "clean_min_obb_ttc_seconds": clean_ttc,
        "degraded_min_obb_ttc_seconds": degraded_ttc,
        "observed_ttc_drop_seconds": observed_drop,
        "required_ttc_drop_seconds": required_drop,
        "clean_ttc_le_2_exposure_seconds": clean_exposure,
        "degraded_ttc_le_2_exposure_seconds": degraded_exposure,
        "observed_ttc_le_2_exposure_increase_seconds": exposure_increase,
        "required_ttc_le_2_exposure_increase_seconds": float(
            thresholds["minimum_ttc_le_2_exposure_increase_seconds"]
        ),
        "min_gap_delta_degraded_minus_clean_diagnostic": interval[
            "comparison"
        ]["by_category"][actor_category]["min_obb_separating_axis_gap_m"],
    }


def _safety_valid(records: list[dict[str, Any]], schema: str) -> bool:
    return bool(records) and all(
        row.get("closedloop_safety", {}).get("available") is True
        and row.get("closedloop_safety", {}).get("schema") == schema
        for row in records
    )


def evaluate(
    *,
    spec_path: Path,
    route_index: str,
    clean_run: Path,
    degraded_run: Path,
) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    actor_category = spec["route_actor_category"].get(str(route_index))
    if actor_category is None:
        raise RuntimeError(f"route {route_index} has no frozen actor category")
    clean_manifest_path, clean_manifest = _load_one_json(clean_run, "manifest.json")
    degraded_manifest_path, degraded_manifest = _load_one_json(
        degraded_run, "manifest.json"
    )
    clean_eval_path, clean_eval = _load_one_json(clean_run, "eval*.json")
    degraded_eval_path, degraded_eval = _load_one_json(degraded_run, "eval*.json")
    clean_gate_path, clean_gate = _load_one_json(clean_run, "clean_safety_gate.json")
    clean_trace_path = find_control_trace(clean_run)
    degraded_trace_path = find_control_trace(degraded_run)
    clean_records = load_records(clean_trace_path)
    degraded_records = load_records(degraded_trace_path)
    required_schema = spec["validity_requirements"]["safety_schema"]
    validity_checks = {
        "clean_gate_passed": clean_gate.get("gate_passed") is True,
        "route_indices_match": (
            str(clean_manifest.get("pilot_route_index")) == str(route_index)
            and str(degraded_manifest.get("pilot_route_index")) == str(route_index)
        ),
        "epic_quality": (
            clean_manifest.get("carla_quality_level") == "Epic"
            and degraded_manifest.get("carla_quality_level") == "Epic"
        ),
        "risk_mode_off": (
            clean_manifest.get("orion_closedloop_risk_mode") == "off"
            and degraded_manifest.get("orion_closedloop_risk_mode") == "off"
        ),
        "uq_mode_none": (
            clean_manifest.get("orion_closedloop_uq_mode") == "none"
            and degraded_manifest.get("orion_closedloop_uq_mode") == "none"
        ),
        "clean_v2_telemetry_complete": _safety_valid(clean_records, required_schema),
        "degraded_v2_telemetry_complete": _safety_valid(
            degraded_records, required_schema
        ),
    }
    paired_event = None
    paired_error = None
    try:
        paired_event = build_paired_event_report(
            clean_records,
            degraded_records,
            recovery_seconds=float(
                spec["surrogate_safety_margin_degradation"]["thresholds"][
                    "paired_recovery_seconds"
                ]
            ),
        )
    except (RuntimeError, ValueError, KeyError) as error:
        paired_error = str(error)
    validity_checks["paired_event_valid"] = paired_event is not None
    valid = all(validity_checks.values())
    hard = hard_endpoint_comparison(clean_eval, degraded_eval)
    surrogate = None
    if paired_event is not None:
        surrogate = surrogate_comparison(
            paired_event,
            actor_category=actor_category,
            thresholds=spec["surrogate_safety_margin_degradation"]["thresholds"],
        )
    hard_pass = hard["degraded"]
    surrogate_pass = surrogate is not None and surrogate["degraded"]
    failure_induction_pass = valid and (hard_pass or surrogate_pass)
    if not valid:
        evidence_tier = "invalid"
    elif hard_pass:
        evidence_tier = "hard_failure_induction"
    elif surrogate_pass:
        evidence_tier = "near_miss_surrogate_failure_induction"
    else:
        evidence_tier = "failure_induction_not_demonstrated"
    return {
        "schema": "orion.closedloop_failure_induction_decision.v1",
        "spec_path": str(spec_path.resolve()),
        "route_index": str(route_index),
        "actor_category": actor_category,
        "clean_run": str(clean_run.resolve()),
        "degraded_run": str(degraded_run.resolve()),
        "source_files": {
            "clean_manifest": str(clean_manifest_path.resolve()),
            "degraded_manifest": str(degraded_manifest_path.resolve()),
            "clean_eval": str(clean_eval_path.resolve()),
            "degraded_eval": str(degraded_eval_path.resolve()),
            "clean_gate": str(clean_gate_path.resolve()),
            "clean_trace": str(clean_trace_path.resolve()),
            "degraded_trace": str(degraded_trace_path.resolve()),
        },
        "validity": {
            "valid": valid,
            "checks": validity_checks,
            "paired_error": paired_error,
        },
        "hard_endpoint": hard,
        "surrogate_safety_margin": surrogate,
        "paired_event": paired_event,
        "decision": {
            "failure_induction_pass": failure_induction_pass,
            "evidence_tier": evidence_tier,
            "counts_toward_final_hard_case_target": valid and hard_pass,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--route-index", required=True)
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--degraded-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite failure-induction decision")
    report = evaluate(
        spec_path=args.spec,
        route_index=args.route_index,
        clean_run=args.clean_run,
        degraded_run=args.degraded_run,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "valid": report["validity"]["valid"],
        "decision": report["decision"],
    }, indent=2))
    return 0 if report["validity"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

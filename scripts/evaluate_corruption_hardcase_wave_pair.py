#!/usr/bin/env python3
"""Evaluate one prospective clean/corrupt development hard-case pair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.evaluate_clean_liveness_screen import longest_low_speed_interval
from scripts.summarize_closedloop_safety import (
    build_paired_event_report,
    find_control_trace,
    load_records,
    summarize_records,
)


SCHEMA = "orion.corruption_hardcase_wave_pair_decision.v1"
HARD_INFRACTION_KEYS = (
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "route_dev",
    "vehicle_blocked",
    "scenario_timeouts",
    "route_timeout",
)
ROUTE_ACTOR_CATEGORY = {
    "151": "walker",
    "158": "vehicle",
    "180": "vehicle",
    "194": "vehicle",
}
CONDITIONS = {
    "front_stale_200ms": {
        "pilot_condition": "front_stale_transient_off",
        "scope": "event_plus_3s_recovery",
    },
    "waterdrop_medium": {
        "pilot_condition": "lens_waterdrop_paired_template_transient_off",
        "scope": "event_plus_3s_recovery",
    },
    "native_motion_blur_medium": {
        "pilot_condition": "native_motion_blur_off",
        "scope": "whole_route",
    },
}


def _load_one_json(root: Path, pattern: str) -> tuple[Path, dict[str, Any]]:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(
            "expected exactly one %r below %s, found %d"
            % (pattern, root, len(paths))
        )
    return paths[0], json.loads(paths[0].read_text(encoding="utf-8"))


def _terminal_record(evaluation: dict[str, Any]) -> dict[str, Any]:
    records = evaluation.get("_checkpoint", {}).get("records", [])
    if len(records) != 1:
        raise RuntimeError("expected exactly one evaluator terminal record")
    return records[0]


def _infraction_count(record: dict[str, Any], key: str) -> int:
    value = record.get("infractions", {}).get(key, [])
    if isinstance(value, list):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(bool(value))


def hard_endpoint_comparison(
    clean_eval: dict[str, Any], degraded_eval: dict[str, Any]
) -> dict[str, Any]:
    clean = _terminal_record(clean_eval)
    degraded = _terminal_record(degraded_eval)
    clean_route = float(clean.get("scores", {}).get("score_route", 0.0))
    degraded_route = float(degraded.get("scores", {}).get("score_route", 0.0))
    completion_drop = clean_route - degraded_route
    infraction_deltas = {
        key: _infraction_count(degraded, key) - _infraction_count(clean, key)
        for key in HARD_INFRACTION_KEYS
    }
    checks = {
        "new_hard_infraction": any(value > 0 for value in infraction_deltas.values()),
        "route_completion_drop_at_least_10pp": completion_drop >= 10.0,
    }
    return {
        "degraded": any(checks.values()),
        "checks": checks,
        "clean_route_completion_percent": clean_route,
        "degraded_route_completion_percent": degraded_route,
        "route_completion_drop_percentage_points": completion_drop,
        "infraction_count_delta_degraded_minus_clean": infraction_deltas,
    }


def continuous_margin_comparison(
    clean: dict[str, Any], degraded: dict[str, Any]
) -> dict[str, Any]:
    clean_ttc = clean.get("min_obb_ttc_seconds")
    degraded_ttc = degraded.get("min_obb_ttc_seconds")
    finite_ttc = clean_ttc is not None and degraded_ttc is not None
    ttc_drop = (
        float(clean_ttc) - float(degraded_ttc) if finite_ttc else None
    )
    required_ttc_drop = (
        max(0.30, 0.20 * float(clean_ttc)) if finite_ttc else None
    )

    clean_gap = clean.get("min_obb_separating_axis_gap_m")
    degraded_gap = degraded.get("min_obb_separating_axis_gap_m")
    finite_gap = clean_gap is not None and degraded_gap is not None
    gap_drop = (
        float(clean_gap) - float(degraded_gap) if finite_gap else None
    )
    required_gap_drop = (
        max(0.50, 0.20 * float(clean_gap)) if finite_gap else None
    )
    checks = {
        "ttc_drop_gate": bool(
            finite_ttc and ttc_drop is not None and ttc_drop >= required_ttc_drop
        ),
        "gap_drop_gate": bool(
            finite_gap and gap_drop is not None and gap_drop >= required_gap_drop
        ),
    }
    return {
        "degraded": any(checks.values()),
        "checks": checks,
        "clean_min_obb_ttc_seconds": clean_ttc,
        "degraded_min_obb_ttc_seconds": degraded_ttc,
        "observed_ttc_drop_seconds": ttc_drop,
        "required_ttc_drop_seconds": required_ttc_drop,
        "clean_min_obb_gap_m": clean_gap,
        "degraded_min_obb_gap_m": degraded_gap,
        "observed_gap_drop_m": gap_drop,
        "required_gap_drop_m": required_gap_drop,
    }


def _full_telemetry(records: list[dict[str, Any]]) -> bool:
    return bool(records) and all(
        row.get("closedloop_safety", {}).get("available") is True
        and row.get("closedloop_safety", {}).get("schema")
        == "orion.closedloop_dynamic_actor_safety.v2"
        for row in records
    )


def _contiguous_active(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active = [row for row in records if row.get("corruption_active")]
    if not active:
        return []
    steps = [int(row["step"]) for row in active]
    if steps != list(range(steps[0], steps[-1] + 1)):
        raise RuntimeError("corruption event is not one contiguous interval")
    return active


def _visual_approval_matches(metadata: dict[str, Any], condition: str) -> bool:
    approval = metadata.get("visual_approval") or {}
    return (
        approval.get("decision_status") == "approved"
        and approval.get("condition") == condition
    )


def exact_condition_checks(
    *,
    condition: str,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    readback: dict[str, Any] | None,
) -> dict[str, bool]:
    active = _contiguous_active(records)
    if condition == "front_stale_200ms":
        metadata = [row.get("corruption_metadata") or {} for row in active]
        return {
            "active_event_present": bool(active),
            "active_event_duration_is_5s": len(active) in {100, 101},
            "requested_delay_is_200ms": bool(metadata) and all(
                row.get("requested_delay_ms") == 200 for row in metadata
            ),
            "effective_delay_is_200ms": bool(metadata) and all(
                math.isclose(
                    float(row.get("effective_delay_ms", float("nan"))),
                    200.0,
                    rel_tol=0.0,
                    abs_tol=1e-3,
                )
                for row in metadata
            ),
            "front_view_only": bool(metadata) and all(
                row.get("view_indices") == [0] for row in metadata
            ),
            "visual_approval_exact": bool(metadata) and all(
                _visual_approval_matches(row, "delay_ms:200") for row in metadata
            ),
        }
    if condition == "waterdrop_medium":
        metadata = [row.get("corruption_metadata") or {} for row in active]
        return {
            "active_event_present": bool(active),
            "active_event_duration_is_5s": len(active) in {100, 101},
            "manifest_profile_medium": (
                manifest.get("orion_paired_waterdrop_profile") == "medium"
            ),
            "metadata_profile_medium": bool(metadata) and all(
                row.get("profile") == "medium" for row in metadata
            ),
            "pre_pipeline_1600x900": bool(metadata) and all(
                row.get("application_stage")
                == "pre_pipeline_1600x900_front_rgb"
                and row.get("resolution") == [1600, 900]
                for row in metadata
            ),
            "front_view_only": bool(metadata) and all(
                row.get("view_indices") == [0] for row in metadata
            ),
            "visual_approval_exact": bool(metadata) and all(
                _visual_approval_matches(row, "profile:medium") for row in metadata
            ),
        }
    if condition == "native_motion_blur_medium":
        cameras = (readback or {}).get("cameras", {})
        front = cameras.get("CAM_FRONT", {}).get("attributes", {})
        bev = cameras.get("bev", {}).get("attributes", {})
        return {
            "manifest_profile_medium": (
                manifest.get("orion_native_motion_blur_profile") == "medium"
            ),
            "readback_status_verified": (readback or {}).get("status") == "verified",
            "readback_profile_medium": (
                (readback or {}).get("native_motion_blur_profile") == "medium"
            ),
            "front_motion_blur_exact": (
                front.get("motion_blur_intensity") == "0.7"
                and front.get("motion_blur_max_distortion") == "0.45"
                and front.get("motion_blur_min_object_screen_size") == "0.05"
            ),
            "bev_motion_blur_disabled": (
                bev.get("motion_blur_intensity") == "0.0"
                and bev.get("motion_blur_max_distortion") == "0.0"
                and bev.get("motion_blur_min_object_screen_size") == "0.0"
            ),
            "trace_has_no_synthetic_event": not active,
        }
    raise ValueError("unknown condition %r" % condition)


def evaluate_pair(
    *, route_index: str, condition: str, clean_run: Path, degraded_run: Path
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError("unsupported condition %r" % condition)
    actor_category = ROUTE_ACTOR_CATEGORY.get(str(route_index))
    if actor_category is None:
        raise ValueError("route %s has no frozen hazard actor category" % route_index)
    clean_manifest_path, clean_manifest = _load_one_json(clean_run, "manifest.json")
    degraded_manifest_path, degraded_manifest = _load_one_json(
        degraded_run, "manifest.json"
    )
    clean_eval_path, clean_eval = _load_one_json(clean_run, "eval*.json")
    degraded_eval_path, degraded_eval = _load_one_json(degraded_run, "eval*.json")
    clean_trace_path = find_control_trace(clean_run)
    degraded_trace_path = find_control_trace(degraded_run)
    clean_records = load_records(clean_trace_path)
    degraded_records = load_records(degraded_trace_path)
    clean_terminal = _terminal_record(clean_eval)
    degraded_terminal = _terminal_record(degraded_eval)
    longest_stop = longest_low_speed_interval(clean_records)

    readback_path = degraded_run / "render_condition_readback.json"
    readback = (
        json.loads(readback_path.read_text(encoding="utf-8"))
        if readback_path.is_file()
        else None
    )
    exact_checks = exact_condition_checks(
        condition=condition,
        manifest=degraded_manifest,
        records=degraded_records,
        readback=readback,
    )
    locks = (
        "orion_closedloop_uq_mode",
        "orion_closedloop_conditioning",
        "orion_closedloop_risk_mode",
        "orion_planning_response_mode",
    )
    validity_checks = {
        "route_indices_match": (
            str(clean_manifest.get("pilot_route_index")) == str(route_index)
            and str(degraded_manifest.get("pilot_route_index")) == str(route_index)
        ),
        "conditions_match": (
            clean_manifest.get("pilot_condition") == "clean_off"
            and degraded_manifest.get("pilot_condition")
            == CONDITIONS[condition]["pilot_condition"]
        ),
        "route_xml_hash_matches": (
            clean_manifest.get("route_sha256") == degraded_manifest.get("route_sha256")
        ),
        "same_code_hashes": (
            clean_manifest.get("source_sha256")
            == degraded_manifest.get("source_sha256")
        ),
        "architecture_locks": all(
            clean_manifest.get(key) == degraded_manifest.get(key)
            and clean_manifest.get(key) in {"none", "off"}
            for key in locks
        )
        and clean_manifest.get("orion_enable_legacy_density_uq") == "0"
        and degraded_manifest.get("orion_enable_legacy_density_uq") == "0",
        "clean_finished": (
            clean_eval.get("entry_status") == "Finished"
            and clean_terminal.get("status") == "Completed"
        ),
        "degraded_finished": (
            degraded_eval.get("entry_status") == "Finished"
            and degraded_terminal.get("status") == "Completed"
        ),
        "clean_no_hard_infraction": all(
            _infraction_count(clean_terminal, key) == 0
            for key in HARD_INFRACTION_KEYS
        ),
        "clean_liveness_fast_screen_passed": (
            float(longest_stop["duration_seconds"]) <= 8.0
        ),
        "clean_safety_telemetry_complete": _full_telemetry(clean_records),
        "degraded_safety_telemetry_complete": _full_telemetry(degraded_records),
        "exact_corruption_condition_realized": all(exact_checks.values()),
    }

    hard = hard_endpoint_comparison(clean_eval, degraded_eval)
    paired_event = None
    if CONDITIONS[condition]["scope"] == "event_plus_3s_recovery":
        paired_event = build_paired_event_report(
            clean_records, degraded_records, recovery_seconds=3.0
        )
        clean_margin = paired_event["event_plus_recovery"]["clean"]["safety"][
            "by_category"
        ][actor_category]
        degraded_margin = paired_event["event_plus_recovery"]["degraded"][
            "safety"
        ]["by_category"][actor_category]
    else:
        clean_margin = summarize_records(clean_records)["safety"]["by_category"][
            actor_category
        ]
        degraded_margin = summarize_records(degraded_records)["safety"][
            "by_category"
        ][actor_category]
    continuous = continuous_margin_comparison(clean_margin, degraded_margin)
    valid = all(validity_checks.values())
    positive = valid and (hard["degraded"] or continuous["degraded"])
    if not valid:
        tier = "invalid"
    elif hard["degraded"]:
        tier = "hard_failure_induction"
    elif continuous["degraded"]:
        tier = "continuous_margin_failure_induction"
    else:
        tier = "valid_negative"
    return {
        "schema": SCHEMA,
        "route_index": str(route_index),
        "condition": condition,
        "actor_category": actor_category,
        "comparison_scope": CONDITIONS[condition]["scope"],
        "source_files": {
            "clean_manifest": str(clean_manifest_path.resolve()),
            "degraded_manifest": str(degraded_manifest_path.resolve()),
            "clean_eval": str(clean_eval_path.resolve()),
            "degraded_eval": str(degraded_eval_path.resolve()),
            "clean_trace": str(clean_trace_path.resolve()),
            "degraded_trace": str(degraded_trace_path.resolve()),
            "degraded_render_readback": (
                str(readback_path.resolve()) if readback_path.is_file() else None
            ),
        },
        "validity": {
            "valid": valid,
            "checks": validity_checks,
            "exact_condition_checks": exact_checks,
            "clean_longest_low_speed_interval": longest_stop,
        },
        "hard_endpoint": hard,
        "continuous_safety_margin": continuous,
        "paired_event": paired_event,
        "decision": {
            "positive_case": positive,
            "evidence_tier": tier,
            "retain_for_case_study": positive,
            "eligible_for_heldout_confirmation": positive,
        },
        "claim_boundary": (
            "Development failure-induction screen only. A positive case shows "
            "native ORION degradation under the frozen corruption; it does not "
            "show uncertainty estimation or method benefit."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-index", required=True)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--degraded-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite hard-case pair decision")
    report = evaluate_pair(
        route_index=args.route_index,
        condition=args.condition,
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
    }, indent=2, sort_keys=True))
    return 0 if report["validity"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

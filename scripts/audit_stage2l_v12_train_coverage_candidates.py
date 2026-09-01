#!/usr/bin/env python3
"""Audit whether frozen formal-train candidates can repair Stage2-L R coverage.

This audit is deliberately outcome-blind with respect to the locked test split.
It reads route identities/splits from the frozen plan, reviewed train membership,
and existing train-only technical/geometry dispositions.  It does not inspect
model predictions, uncertainty maps, QA answers, closed-loop outcomes from the
locked test split, or select candidates using held-out model performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable


SCHEMA = "orion.stage2l_v12_train_coverage_candidate_audit.v1"


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference(path: Path) -> Dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _events_by_route(value: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    result = {}
    for event in value.get("events", []):
        route_index = int(event["route_index"])
        if route_index in result:
            raise ValueError("duplicate route index %d" % route_index)
        result[route_index] = event
    return result


def _require_status(value: Dict[str, Any], expected: str, path: Path) -> None:
    if value.get("status") != expected:
        raise ValueError("unexpected status in %s" % path)


def audit(
    *,
    formal_plan_path: Path,
    accepted_bank_path: Path,
    route177_reviewed_path: Path,
    route177_geometry_gate_path: Path,
    route201_hold_path: Path,
    route208_hold_path: Path,
) -> Dict[str, Any]:
    formal_plan = _read_json(formal_plan_path)
    accepted_bank = _read_json(accepted_bank_path)
    route177_reviewed = _read_json(route177_reviewed_path)
    route177_gate = _read_json(route177_geometry_gate_path)
    route201_hold = _read_json(route201_hold_path)
    route208_hold = _read_json(route208_hold_path)

    if formal_plan.get("schema") != "orion.stage2_l.formal_route_plan.v1":
        raise ValueError("formal route plan schema differs")
    if accepted_bank.get("schema") != "orion.stage2_l.formal_event_bank.v1":
        raise ValueError("accepted event bank schema differs")
    _require_status(
        route177_gate,
        "formal_event_qa_ineligible_under_frozen_geometry_gate",
        route177_geometry_gate_path,
    )
    _require_status(
        route201_hold,
        "route201_runtime_retry_exhausted_formal_training_hold",
        route201_hold_path,
    )
    _require_status(
        route208_hold,
        "route208_clean_liveness_fast_screen_triggered_stage_b_ineligible",
        route208_hold_path,
    )

    plan_events = _events_by_route(formal_plan)
    bank_events = _events_by_route(accepted_bank)
    reviewed177 = _events_by_route(route177_reviewed)
    planned_train = sorted(
        route for route, row in plan_events.items() if row["formal_split"] == "train"
    )
    planned_dev = sorted(
        route for route, row in plan_events.items() if row["formal_split"] == "dev"
    )
    planned_test = sorted(
        route for route, row in plan_events.items() if row["formal_split"] == "test"
    )
    accepted_train = sorted(
        route for route, row in bank_events.items() if row["formal_split"] == "train"
    )
    missing_train = sorted(set(planned_train) - set(accepted_train))
    if missing_train != [201, 208]:
        raise ValueError("frozen missing-train set differs")
    if 177 not in accepted_train or 177 not in reviewed177:
        raise ValueError("Route177 reviewed train lineage differs")

    authoritative_splits = {
        route: str(plan_events[route]["formal_split"])
        for route in (177, 201, 208)
    }
    legacy_split_labels = {
        177: str(route177_gate.get("source", {}).get("formal_split", "missing")),
        201: str(route201_hold.get("route", {}).get("formal_split", "missing")),
        208: str(route208_hold.get("route", {}).get("formal_split", "missing")),
    }
    split_discrepancies = [
        {
            "route_index": route,
            "authoritative_formal_plan_split": authoritative_splits[route],
            "legacy_amendment_split_label": legacy_split_labels[route],
            "disposition": (
                "recorded_clerical_lineage_mismatch; formal plan owns split and "
                "the route-specific technical exclusion remains applicable"
            ),
        }
        for route in (177, 201, 208)
        if authoritative_splits[route] != legacy_split_labels[route]
    ]

    route177_valid_keyframes = int(
        route177_gate["validation"]["retained_geometry_valid_keyframes"]
    )
    route177_minimum = int(route177_gate["validation"]["minimum_required_keyframes"])
    candidates = [
        {
            "route_index": 177,
            "formal_split": "train",
            "formal_event_reviewed": True,
            "formal_r_data_eligible": False,
            "eligible_geometry_keyframes": route177_valid_keyframes,
            "minimum_geometry_keyframes": route177_minimum,
            "blocking_gate": "frozen_fixed_offset_geometry_gate",
            "reason": route177_gate["validation"]["failure"],
            "may_directly_add_view_coverage": False,
        },
        {
            "route_index": 201,
            "formal_split": "train",
            "formal_event_reviewed": False,
            "formal_r_data_eligible": False,
            "eligible_geometry_keyframes": 0,
            "minimum_geometry_keyframes": 3,
            "blocking_gate": "runtime_and_sensor_integrity",
            "reason": route201_hold["decision"]["resolution_requires"],
            "may_directly_add_view_coverage": False,
        },
        {
            "route_index": 208,
            "formal_split": "train",
            "formal_event_reviewed": False,
            "formal_r_data_eligible": False,
            "eligible_geometry_keyframes": 0,
            "minimum_geometry_keyframes": 3,
            "blocking_gate": "clean_liveness",
            "reason": "clean baseline failed the frozen liveness screen",
            "may_directly_add_view_coverage": False,
        },
    ]
    if any(row["formal_r_data_eligible"] for row in candidates):
        raise RuntimeError("candidate disposition unexpectedly became eligible")

    return {
        "schema": SCHEMA,
        "status": "frozen_train_candidates_cannot_repair_r_view_coverage",
        "passed": True,
        "gpu_used": False,
        "training_started": False,
        "inputs": {
            "formal_route_plan": _reference(formal_plan_path),
            "accepted_event_bank": _reference(accepted_bank_path),
            "route177_reviewed_shard": _reference(route177_reviewed_path),
            "route177_geometry_gate": _reference(route177_geometry_gate_path),
            "route201_runtime_hold": _reference(route201_hold_path),
            "route208_liveness_hold": _reference(route208_hold_path),
        },
        "formal_identity_inventory": {
            "train_routes": planned_train,
            "dev_routes": planned_dev,
            "locked_test_routes": planned_test,
            "accepted_train_routes": accepted_train,
            "missing_train_routes": missing_train,
        },
        "inspection_boundary": {
            "formal_train_dispositions_read": [177, 201, 208],
            "dev_result_files_read": [],
            "locked_test_result_files_read": [],
            "model_predictions_read_for_selection": False,
            "uncertainty_or_qa_outputs_read_for_selection": False,
            "closed_loop_outcomes_used_to_rank_candidates": False,
        },
        "lineage_discrepancies": split_discrepancies,
        "candidate_dispositions": candidates,
        "coverage_repair": {
            "current_candidates_with_eligible_r_geometry": [],
            "current_candidates_can_fill_cam_back_right": False,
            "current_candidates_can_raise_cam_front_right_independent_events": False,
            "objective_reweighting_is_sufficient": False,
            "reason": (
                "No frozen formal-train remainder has an eligible 3-5-keyframe R package. "
                "A loss can redistribute mass only among support that exists."
            ),
        },
        "next_authorized_work": {
            "gpu_r_only_smoke": False,
            "language_or_stage2p_training": False,
            "action": (
                "Freeze a train-only coverage-repair candidate pool using static route/event "
                "geometry before model outcomes, require accepted 3-5-keyframe support and "
                "nonzero side/rear-view contribution, then rerun the CPU coverage gate."
            ),
            "formal_plan_change_requires_timestamped_amendment": True,
            "held_out_routes_may_move_to_train": False,
        },
        "claim_boundary": (
            "Train-only lineage/readiness audit. It proves that the currently frozen "
            "Route177/201/208 assets cannot directly repair R coverage; it does not prove "
            "that another event will supply a particular view, that R will generalize, or "
            "that uncertainty, language, planning, or safety improves."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-plan", type=Path, required=True)
    parser.add_argument("--accepted-bank", type=Path, required=True)
    parser.add_argument("--route177-reviewed", type=Path, required=True)
    parser.add_argument("--route177-geometry-gate", type=Path, required=True)
    parser.add_argument("--route201-hold", type=Path, required=True)
    parser.add_argument("--route208-hold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite coverage candidate audit")
    value = audit(
        formal_plan_path=args.formal_plan,
        accepted_bank_path=args.accepted_bank,
        route177_reviewed_path=args.route177_reviewed,
        route177_geometry_gate_path=args.route177_geometry_gate,
        route201_hold_path=args.route201_hold,
        route208_hold_path=args.route208_hold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": value["status"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

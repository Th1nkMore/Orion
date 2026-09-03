#!/usr/bin/env python3
"""Freeze the 24-event Stage2-L formal route identities before new replays."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.scenario_factory_lib import sha256_file
from scripts.select_closedloop_heldout_routes import (
    load_baseline_results,
    parse_route,
)


CONFIG_SCHEMA = "orion.stage2_l.formal_route_expansion_config.v1"
PLAN_SCHEMA = "orion.stage2_l.formal_route_plan.v1"
PILOT_BANK_SCHEMA = "orion.stage2_l.pilot_event_bank.v1"
PILOT_DATASET_SCHEMA = "orion.stage2_l.pilot_dataset.v1"


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _route_candidate(
    *,
    routes_dir: Path,
    route_index: int,
    scenario_type: str,
    baseline: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    path = routes_dir / ("bench2drive220_%d_orion_traj.xml" % route_index)
    if not path.is_file():
        raise FileNotFoundError(path)
    matches = [
        row for row in parse_route(path, baseline)
        if row["scenario_type"] == scenario_type
    ]
    if len(matches) != 1:
        raise ValueError(
            "route %d must contain exactly one %s scenario" %
            (route_index, scenario_type)
        )
    return matches[0]


def freeze_formal_route_plan(
    *,
    config: Mapping[str, Any],
    pilot_bank: Mapping[str, Any],
    pilot_dataset: Mapping[str, Any],
    routes_dir: Path,
    baseline: Mapping[str, Mapping[str, Any]],
    failure_amendment: Mapping[str, Any],
) -> Dict[str, Any]:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported formal expansion config")
    if (
        pilot_bank.get("schema") != PILOT_BANK_SCHEMA
        or pilot_bank.get("status") != "frozen_before_stage2l_pilot_training"
    ):
        raise ValueError("pilot event bank is not frozen")
    if (
        pilot_dataset.get("schema") != PILOT_DATASET_SCHEMA
        or int(pilot_dataset.get("event_count", -1)) != 8
        or int(pilot_dataset.get("qa_record_count", -1)) < 480
    ):
        raise ValueError("assembled pilot dataset is absent or invalid")

    pilot_events = list(pilot_bank.get("events", []))
    if len(pilot_events) != 8:
        raise ValueError("formal expansion must inherit exactly eight pilot events")
    pilot_ids = {str(row["event_id"]) for row in pilot_events}
    if {str(row["event_id"]) for row in pilot_dataset.get("events", [])} != pilot_ids:
        raise ValueError("pilot bank and assembled pilot dataset events differ")

    additions = config.get("additions", {})
    development = list(additions.get("development", []))
    locked_test = list(additions.get("locked_test", []))
    if len(development) != 12 or len(locked_test) != 4:
        raise ValueError("formal expansion requires 12 development and 4 test additions")
    all_new = development + locked_test
    route_indices = [int(row["route_index"]) for row in all_new]
    pilot_routes = {int(row["route_index"]) for row in pilot_events}
    if len(set(route_indices)) != 16 or pilot_routes.intersection(route_indices):
        raise ValueError("formal expansion routes must be new and unique")

    allowed_failures = {
        int(value)
        for value in failure_amendment.get("allowed_development_failure_routes", [])
    }
    if (
        failure_amendment.get("schema") != "orion.scenario_factory.amendment.v1"
        or failure_amendment.get("launch_locks", {}).get(
            "formal_stage2l_training_allowed"
        ) is not False
    ):
        raise ValueError("development failure amendment is absent or over-broad")

    development_candidates = []
    test_candidates = []
    new_event_rows = []
    for planned in development:
        index = int(planned["route_index"])
        candidate = _route_candidate(
            routes_dir=routes_dir,
            route_index=index,
            scenario_type=str(planned["scenario_type"]),
            baseline=baseline,
        )
        role = str(planned["development_selection_role"])
        clean_valid = candidate.get("clean_baseline", {}).get("valid") is True
        if role == "published_clean_valid":
            if not clean_valid:
                raise ValueError("clean-valid development route is not valid: %d" % index)
        elif role == "published_failure_hard_case":
            if clean_valid or index not in allowed_failures:
                raise ValueError("development failure route is not explicitly amended")
        else:
            raise ValueError("unsupported development selection role")
        candidate.update({
            "formal_split": str(planned["formal_split"]),
            "development_selection_role": role,
            "existing_reviewed_event_reuse": bool(
                planned.get("existing_reviewed_event_reuse", False)
            ),
        })
        development_candidates.append(candidate)
        new_event_rows.append({
            "route_index": index,
            "formal_split": candidate["formal_split"],
            "town": candidate["town"],
            "scenario_family": candidate["scenario_type"],
            "split_origin": "development_screen",
            "selection_role": role,
            "replay_required": not candidate["existing_reviewed_event_reuse"],
        })

    for planned in locked_test:
        index = int(planned["route_index"])
        candidate = _route_candidate(
            routes_dir=routes_dir,
            route_index=index,
            scenario_type=str(planned["scenario_type"]),
            baseline={},
        )
        if candidate.get("clean_baseline", {}).get("available") is not False:
            raise ValueError("locked-test candidate unexpectedly contains an outcome")
        candidate.update({
            "formal_split": "test",
            "selection_role": "static_geometry_outcome_blind",
        })
        test_candidates.append(candidate)
        new_event_rows.append({
            "route_index": index,
            "formal_split": "test",
            "town": candidate["town"],
            "scenario_family": candidate["scenario_type"],
            "split_origin": "locked_test",
            "selection_role": "static_geometry_outcome_blind",
            "replay_required": True,
        })

    inherited = [
        {
            "event_id": row["event_id"],
            "route_index": int(row["route_index"]),
            "formal_split": row["pilot_split"],
            "town": row["town"],
            "scenario_family": row["scenario_family"],
            "split_origin": row["split_origin"],
            "selection_role": "inherited_frozen_pilot_event",
            "replay_required": False,
        }
        for row in pilot_events
    ]
    events = inherited + new_event_rows
    split_counts = Counter(str(row["formal_split"]) for row in events)
    counts = {
        "events": len(events),
        "towns": len({str(row["town"]) for row in events}),
        "scenario_families": len({str(row["scenario_family"]) for row in events}),
        "splits": dict(sorted(split_counts.items())),
    }
    expected = config["formal_gate"]
    if (
        counts["events"] != int(expected["events"])
        or counts["towns"] < int(expected["minimum_towns"])
        or counts["scenario_families"] < int(expected["minimum_scenario_families"])
        or split_counts != Counter(expected["split"])
    ):
        raise ValueError("formal route plan does not meet frozen count/diversity gates")
    pilot_qa = int(pilot_dataset["qa_record_count"])
    min_qa = pilot_qa + len(all_new) * 3 * 20
    max_qa = pilot_qa + len(all_new) * 5 * 20
    qa_target = list(map(int, expected["qa_record_range"]))
    if min_qa < qa_target[0] or max_qa > qa_target[1]:
        raise ValueError("fixed keyframe policy falls outside formal QA target")

    return {
        "formal_plan": {
            "schema": PLAN_SCHEMA,
            "status": "route_identities_frozen_before_new_replays",
            "formal_training_ready": False,
            "counts": counts,
            "expected_qa_records_after_geometry_gate": [min_qa, max_qa],
            "events": sorted(events, key=lambda row: (
                str(row["formal_split"]), int(row["route_index"])
            )),
            "locked_test_selection": {
                "published_orion_outcomes_used": False,
                "current_orion_replay_outcomes_used": False,
                "learned_uq_outcomes_used": False,
                "stage2_outcomes_used": False,
                "post_freeze_model_failure_is_not_an_exclusion_reason": True,
            },
            "remaining_gates": [
                "current-environment clean_off replay",
                "runtime and sensor integrity audit",
                "actor-event human review",
                "fixed 3-5 keyframe geometry gate",
                "Stage1/QA/cache construction and QA geometry review",
                "formal corruption-family protocol freeze",
            ],
            "claim_boundary": "Frozen route identities and splits only; no model, UQ, trajectory, closed-loop, or safety result.",
        },
        "development_candidates": {
            "schema": "orion.closedloop_heldout_route_screen.v1",
            "status": "formal_development_candidates_frozen_no_jobs_submitted",
            "clean_baseline_required": False,
            "development_failure_candidates_allowed": True,
            "selection_inputs": {
                "static_route_geometry_used": True,
                "scenario_family_used": True,
                "published_orion_outcomes_used": True,
                "learned_uq_outcomes_used": False,
                "stage2_outcomes_used": False,
            },
            "candidate_count": len(development_candidates),
            "candidates": development_candidates,
        },
        "locked_test_candidates": {
            "schema": "orion.closedloop_heldout_route_screen.v1",
            "status": "formal_locked_test_candidates_frozen_no_jobs_submitted",
            "clean_baseline_required": False,
            "development_failure_candidates_allowed": False,
            "selection_inputs": {
                "static_route_geometry_used": True,
                "scenario_family_used": True,
                "published_orion_outcomes_used": False,
                "learned_uq_outcomes_used": False,
                "stage2_outcomes_used": False,
            },
            "locked_test_selection_eligible": True,
            "candidate_count": len(test_candidates),
            "candidates": test_candidates,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot-bank", type=Path, required=True)
    parser.add_argument("--pilot-dataset", type=Path, required=True)
    parser.add_argument("--routes-dir", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--failure-amendment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite formal route plan")
    result = freeze_formal_route_plan(
        config=_load(args.config.resolve()),
        pilot_bank=_load(args.pilot_bank.resolve()),
        pilot_dataset=_load(args.pilot_dataset.resolve()),
        routes_dir=args.routes_dir.resolve(),
        baseline=load_baseline_results(args.baseline_results.resolve()),
        failure_amendment=_load(args.failure_amendment.resolve()),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev_path = args.output_dir / "development_candidates.json"
    test_path = args.output_dir / "locked_test_candidates.json"
    dev_path.write_text(json.dumps(result["development_candidates"], indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    test_path.write_text(json.dumps(result["locked_test_candidates"], indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    plan = result["formal_plan"]
    plan["candidate_manifests"] = {
        "development": {"path": str(dev_path.resolve()), "sha256": sha256_file(dev_path)},
        "locked_test": {"path": str(test_path.resolve()), "sha256": sha256_file(test_path)},
    }
    plan["provenance"] = {
        "config": {"path": str(args.config.resolve()), "sha256": sha256_file(args.config.resolve())},
        "pilot_bank": {"path": str(args.pilot_bank.resolve()), "sha256": sha256_file(args.pilot_bank.resolve())},
        "pilot_dataset": {"path": str(args.pilot_dataset.resolve()), "sha256": sha256_file(args.pilot_dataset.resolve())},
        "scenario_factory_protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol.resolve())},
        "development_failure_amendment": {"path": str(args.failure_amendment.resolve()), "sha256": sha256_file(args.failure_amendment.resolve())},
        "published_orion_baseline_for_development_only": {"path": str(args.baseline_results.resolve()), "sha256": sha256_file(args.baseline_results.resolve())},
    }
    plan_path = args.output_dir / "formal_route_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(plan_path.resolve()), "counts": plan["counts"], "expected_qa_records": plan["expected_qa_records_after_geometry_gate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

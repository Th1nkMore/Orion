#!/usr/bin/env python3
"""Assemble reviewed shards against the frozen 24-event Stage2-L plan."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.scenario_factory_lib import sha256_file


PLAN_SCHEMA = "orion.stage2_l.formal_route_plan.v1"
FORMAL_SHARD_SCHEMA = "orion.stage2_l.formal_reviewed_wave.v1"
PILOT_BANK_SCHEMA = "orion.stage2_l.pilot_event_bank.v1"
OUTPUT_SCHEMA = "orion.stage2_l.formal_event_bank.v1"


def _read(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def assemble_formal_event_bank(
    *, formal_plan_path: Path, event_bank_paths: Sequence[Path]
) -> Dict[str, Any]:
    plan = _read(formal_plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported formal route plan")
    if not event_bank_paths:
        raise ValueError("at least one reviewed source bank is required")
    planned_by_route = {
        int(row["route_index"]): row for row in plan.get("events", [])
    }
    if len(planned_by_route) != len(plan.get("events", [])):
        raise ValueError("formal plan contains duplicate routes")

    accepted = []
    seen_events = set()
    seen_routes = set()
    sources = []
    for raw_path in event_bank_paths:
        path = raw_path.resolve()
        bank = _read(path)
        schema = bank.get("schema")
        if schema not in (FORMAL_SHARD_SCHEMA, PILOT_BANK_SCHEMA):
            raise ValueError("unsupported reviewed-bank schema: %s" % schema)
        for source_event in bank.get("events", []):
            row = dict(source_event)
            route_index = int(row["route_index"])
            event_id = str(row["event_id"])
            if route_index in seen_routes or event_id in seen_events:
                raise ValueError("formal reviewed sources duplicate route or event")
            if route_index not in planned_by_route:
                raise ValueError("reviewed route is absent from formal plan")
            planned = planned_by_route[route_index]
            source_split = row.get("formal_split", row.get("pilot_split"))
            if str(source_split) != str(planned["formal_split"]):
                raise ValueError("reviewed source changed frozen formal split")
            if (
                row.get("human_review", {}).get("decision") != "accept"
                or row.get("qa_input_ready") is not True
                or row.get("runtime_valid") is not True
                or row.get("actor_grounded_event") is not True
            ):
                raise ValueError("formal source contains an unaccepted event")
            if (
                str(row["town"]) != str(planned["town"])
                or str(row["scenario_family"]) != str(planned["scenario_family"])
                or (
                    planned.get("event_id") is not None
                    and event_id != str(planned["event_id"])
                )
            ):
                raise ValueError("reviewed event differs from frozen identity")
            row["formal_split"] = str(planned["formal_split"])
            row["formal_plan_selection_role"] = str(planned["selection_role"])
            row.pop("pilot_split", None)
            accepted.append(row)
            seen_routes.add(route_index)
            seen_events.add(event_id)
        sources.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "schema": schema,
            "event_count": len(bank.get("events", [])),
        })

    accepted.sort(key=lambda row: (
        str(row["formal_split"]), int(row["route_index"])
    ))
    expected_split = Counter(
        str(row["formal_split"]) for row in planned_by_route.values()
    )
    actual_split = Counter(str(row["formal_split"]) for row in accepted)
    missing_routes = sorted(set(planned_by_route) - seen_routes)
    missing_by_split = {
        split: sorted(
            route
            for route in missing_routes
            if str(planned_by_route[route]["formal_split"]) == split
        )
        for split in ("train", "dev", "test")
    }
    complete = (
        not missing_routes
        and len(accepted) == len(planned_by_route)
        and actual_split == expected_split
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "status": (
            "formal_event_bank_complete_reviewed"
            if complete
            else "formal_event_bank_incomplete_reviewed_subset"
        ),
        "formal_training_ready": False,
        "counts": {
            "accepted_events": len(accepted),
            "planned_events": len(planned_by_route),
            "towns": len({str(row["town"]) for row in accepted}),
            "scenario_families": len(
                {str(row["scenario_family"]) for row in accepted}
            ),
            "formal_splits": {
                split: int(actual_split.get(split, 0))
                for split in ("train", "dev", "test")
            },
        },
        "checks": {
            "all_present_events_human_review_accepted": True,
            "all_present_events_runtime_and_qa_ready": True,
            "route_and_event_ids_disjoint": True,
            "frozen_splits_preserved": True,
            "all_24_planned_routes_present": complete,
            "formal_training_still_locked": True,
        },
        "missing_routes": missing_routes,
        "missing_routes_by_split": missing_by_split,
        "events": accepted,
        "provenance": {
            "formal_route_plan": {
                "path": str(formal_plan_path.resolve()),
                "sha256": sha256_file(formal_plan_path),
            },
            "reviewed_source_banks": sources,
        },
        "remaining_gates_after_complete_bank": [
            "fixed 3-5 keyframe geometry gate",
            "Stage1/QA/cache construction and QA geometry review",
            "formal corruption-family protocol freeze",
            "Stage2-L training protocol and release-gate freeze",
        ],
        "claim_boundary": (
            "Mechanical assembly of hash-bound, human-reviewed events under "
            "frozen route splits. Even a complete bank does not by itself "
            "authorize training or establish UQ, model, planning, or safety evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-route-plan", type=Path, required=True)
    parser.add_argument("--event-bank", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite formal event bank")
    result = assemble_formal_event_bank(
        formal_plan_path=args.formal_route_plan.resolve(),
        event_bank_paths=[path.resolve() for path in args.event_bank],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": result["status"],
        "counts": result["counts"],
        "missing_routes": result["missing_routes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

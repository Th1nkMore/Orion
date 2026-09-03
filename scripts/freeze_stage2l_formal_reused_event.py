#!/usr/bin/env python3
"""Bind one already-reviewed development event into the frozen formal plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.scenario_factory_lib import sha256_file


PLAN_SCHEMA = "orion.stage2_l.formal_route_plan.v1"
QUEUE_SCHEMA = "orion.scenario_event_review_queue.v1"
DECISIONS_SCHEMA = "orion.scenario_event_review_decisions.v1"
PACKAGE_SCHEMA = "orion.scenario_event_package.v1"
OUTPUT_SCHEMA = "orion.stage2_l.formal_reviewed_wave.v1"
CHECKS = {
    "visual_stream_integrity",
    "actor_event_semantics",
    "front_bev_temporal_alignment",
    "no_actor_disappearance_or_spawn_artifact",
}


def _load(path: Path, schema: str) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError("unexpected schema for %s" % path)
    return value


def freeze_reused_event(
    *,
    formal_plan_path: Path,
    review_queue_path: Path,
    review_decisions_path: Path,
    event_package_path: Path,
    route_index: int,
) -> Dict[str, Any]:
    plan = _load(formal_plan_path, PLAN_SCHEMA)
    queue = _load(review_queue_path, QUEUE_SCHEMA)
    decisions = _load(review_decisions_path, DECISIONS_SCHEMA)
    package = _load(event_package_path, PACKAGE_SCHEMA)
    if decisions.get("review_queue", {}).get("sha256") != sha256_file(
        review_queue_path
    ):
        raise ValueError("review decisions are not hash-bound to the queue")

    planned = [
        row for row in plan["events"] if int(row["route_index"]) == route_index
    ]
    if len(planned) != 1:
        raise ValueError("reused route is absent or duplicated in formal plan")
    planned = planned[0]
    if planned.get("replay_required") is not False:
        raise ValueError("formal plan does not authorize event reuse")
    if planned.get("formal_split") == "test":
        raise ValueError("locked-test events cannot reuse development outcomes")

    queued = [
        row
        for row in queue["review_order"]
        if int(row["route_index"]) == route_index
    ]
    if len(queued) != 1:
        raise ValueError("review queue must contain exactly one reused event")
    queued = queued[0]
    event_id = str(queued["event_id"])
    decision = [
        row for row in decisions["decisions"] if str(row["event_id"]) == event_id
    ]
    if len(decision) != 1:
        raise ValueError("review decision must exactly cover reused event")
    decision = decision[0]

    package_sha = sha256_file(event_package_path)
    if (
        queued.get("event_package", {}).get("sha256") != package_sha
        or decision.get("event_package_sha256") != package_sha
    ):
        raise ValueError("reused event package hash is not consistently bound")
    checks = decision.get("checks", {})
    if (
        decision.get("decision") != "accept"
        or set(checks) != CHECKS
        or any(value != "pass" for value in checks.values())
        or queued.get("runtime_valid") is not True
        or queued.get("qa_input_ready") is not True
        or queued.get("actor_grounded_event") is not True
    ):
        raise ValueError("reused event lacks accepted integrity evidence")
    if (
        int(package["route"]["route_index"]) != route_index
        or str(queued["town"]) != str(planned["town"])
        or str(queued["scenario_family"]) != str(planned["scenario_family"])
        or str(queued["split_origin"]) != str(planned["split_origin"])
    ):
        raise ValueError("reused event differs from frozen route identity")

    event = dict(queued)
    event["formal_split"] = str(planned["formal_split"])
    event["formal_plan_selection_role"] = str(planned["selection_role"])
    event["human_review"] = {
        "decision": "accept",
        "checks": dict(checks),
        "notes": decision.get("notes", ""),
        "rejection_basis": None,
    }
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "reviewed_formal_wave_shard",
        "formal_training_ready": False,
        "run_id": str(package["route"]["run_id"]),
        "counts": {
            "accepted_events": 1,
            "rejected_events": 0,
            "towns": 1,
            "scenario_families": 1,
            "formal_splits": {
                "train": int(planned["formal_split"] == "train"),
                "dev": int(planned["formal_split"] == "dev"),
                "test": 0,
            },
        },
        "events": [event],
        "rejected_events": [],
        "provenance": {
            "reuse_authorized_by_formal_plan": True,
            "formal_route_plan": {
                "path": str(formal_plan_path.resolve()),
                "sha256": sha256_file(formal_plan_path),
            },
            "event_package": {
                "path": str(event_package_path.resolve()),
                "sha256": package_sha,
            },
            "review_queue": {
                "path": str(review_queue_path.resolve()),
                "sha256": sha256_file(review_queue_path),
            },
            "review_decisions": {
                "path": str(review_decisions_path.resolve()),
                "sha256": sha256_file(review_decisions_path),
            },
        },
        "claim_boundary": (
            "Hash-bound reuse of one prereviewed development event under its "
            "frozen formal split; event integrity only, with no UQ, model, "
            "trajectory, closed-loop, or safety claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-route-plan", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--review-decisions", type=Path, required=True)
    parser.add_argument("--event-package", type=Path, required=True)
    parser.add_argument("--route-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite formal reused-event shard")
    result = freeze_reused_event(
        formal_plan_path=args.formal_route_plan.resolve(),
        review_queue_path=args.review_queue.resolve(),
        review_decisions_path=args.review_decisions.resolve(),
        event_package_path=args.event_package.resolve(),
        route_index=args.route_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve()), "counts": result["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze one reviewed formal-route wave without reassigning frozen splits."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict

from scripts.scenario_factory_lib import sha256_file


PLAN_SCHEMA = "orion.stage2_l.formal_route_plan.v1"
BATCH_SCHEMA = "orion.scenario_factory.batch.v1"
QUEUE_SCHEMA = "orion.scenario_event_review_queue.v1"
DECISIONS_SCHEMA = "orion.scenario_event_review_decisions.v1"
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


def freeze_wave(
    *, formal_plan_path: Path, batch_manifest_path: Path,
    review_queue_path: Path, review_decisions_path: Path,
) -> Dict[str, Any]:
    plan = _load(formal_plan_path, PLAN_SCHEMA)
    batch = _load(batch_manifest_path, BATCH_SCHEMA)
    queue = _load(review_queue_path, QUEUE_SCHEMA)
    decisions = _load(review_decisions_path, DECISIONS_SCHEMA)
    if decisions.get("review_queue", {}).get("sha256") != sha256_file(review_queue_path):
        raise ValueError("review decisions are not hash-bound to the queue")

    plan_by_route = {int(row["route_index"]): row for row in plan["events"]}
    batch_by_route = {int(row["route_index"]): row for row in batch["routes"]}
    queue_by_event = {str(row["event_id"]): row for row in queue["review_order"]}
    decision_by_event = {
        str(row["event_id"]): row for row in decisions["decisions"]
    }
    if set(queue_by_event) != set(decision_by_event):
        raise ValueError("review decisions do not exactly cover the queue")
    queue_routes = {int(row["route_index"]) for row in queue_by_event.values()}
    if queue_routes != set(batch_by_route):
        raise ValueError("review queue and batch routes differ")

    accepted = []
    rejected = []
    for event_id, row in sorted(
        queue_by_event.items(), key=lambda item: int(item[1]["route_index"])
    ):
        route_index = int(row["route_index"])
        if route_index not in plan_by_route:
            raise ValueError("reviewed route is absent from the formal plan")
        planned = plan_by_route[route_index]
        batched = batch_by_route[route_index]
        if str(planned["formal_split"]) != str(batched["formal_split"]):
            raise ValueError("batch changed a frozen formal split")
        if str(planned["split_origin"]) != str(row["split_origin"]):
            raise ValueError("review queue changed a frozen split origin")
        decision = decision_by_event[event_id]
        if decision.get("event_package_sha256") != row["event_package"]["sha256"]:
            raise ValueError("review decision event package hash differs")
        checks = decision.get("checks", {})
        if set(checks) != CHECKS or any(
            value not in ("pass", "fail") for value in checks.values()
        ):
            raise ValueError("review checks are incomplete")
        value = decision.get("decision")
        if value not in ("accept", "reject"):
            raise ValueError("pending or invalid review decision remains")
        if value == "accept" and (
            not all(result == "pass" for result in checks.values())
            or row.get("qa_input_ready") is not True
        ):
            raise ValueError("accepted event failed integrity or QA readiness")
        if planned["formal_split"] == "test" and value == "reject" and all(
            result == "pass" for result in checks.values()
        ):
            raise ValueError("locked test event may only be rejected technically")

        frozen = dict(row)
        frozen["formal_split"] = str(planned["formal_split"])
        frozen["formal_plan_selection_role"] = str(planned["selection_role"])
        frozen["human_review"] = {
            "decision": value,
            "checks": dict(checks),
            "notes": decision.get("notes", ""),
            "rejection_basis": decision.get("rejection_basis"),
        }
        (accepted if value == "accept" else rejected).append(frozen)

    split_counts = Counter(row["formal_split"] for row in accepted)
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "reviewed_formal_wave_shard",
        "formal_training_ready": False,
        "run_id": str(batch["run_id"]),
        "counts": {
            "accepted_events": len(accepted),
            "rejected_events": len(rejected),
            "towns": len({row["town"] for row in accepted}),
            "scenario_families": len(
                {row["scenario_family"] for row in accepted}
            ),
            "formal_splits": {
                split: int(split_counts.get(split, 0))
                for split in ("train", "dev", "test")
            },
        },
        "events": accepted,
        "rejected_events": rejected,
        "provenance": {
            "formal_route_plan": {
                "path": str(formal_plan_path.resolve()),
                "sha256": sha256_file(formal_plan_path),
            },
            "batch_manifest": {
                "path": str(batch_manifest_path.resolve()),
                "sha256": sha256_file(batch_manifest_path),
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
            "Human-reviewed formal-route shard with preregistered splits. "
            "It establishes event integrity only, not UQ, model, trajectory, "
            "closed-loop, or safety performance."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-route-plan", type=Path, required=True)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--review-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite reviewed formal wave")
    result = freeze_wave(
        formal_plan_path=args.formal_route_plan.resolve(),
        batch_manifest_path=args.batch_manifest.resolve(),
        review_queue_path=args.review_queue.resolve(),
        review_decisions_path=args.review_decisions.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "counts": result["counts"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

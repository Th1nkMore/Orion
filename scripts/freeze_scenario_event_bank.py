#!/usr/bin/env python3
"""Freeze reviewed scenario events into route-disjoint Stage2-L data splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

try:
    from scripts.build_scenario_review_queue import (
        DECISIONS_SCHEMA,
        HUMAN_CHECKS,
        QUEUE_SCHEMA,
    )
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from build_scenario_review_queue import DECISIONS_SCHEMA, HUMAN_CHECKS, QUEUE_SCHEMA
    from scenario_factory_lib import sha256_file


BANK_SCHEMA = "orion.scenario_event_bank.v1"


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object: %s" % path)
    return payload


def _resolve_reference(reference: Mapping[str, Any], base: Path, name: str) -> Path:
    path = Path(str(reference.get("path", "")))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file():
        raise FileNotFoundError("%s is missing: %s" % (name, path))
    if sha256_file(path) != reference.get("sha256"):
        raise ValueError("%s SHA-256 mismatch" % name)
    return path


def _hash_rank(seed: str, event_id: str) -> str:
    return hashlib.sha256((seed + "\0" + event_id).encode("utf-8")).hexdigest()


def _select_development_split(
    rows: Sequence[Dict[str, Any]], dev_count: int, seed: str
) -> Dict[str, str]:
    if dev_count < 0 or dev_count > len(rows):
        raise ValueError("dev count exceeds accepted development events")
    remaining = list(rows)
    selected: List[Dict[str, Any]] = []
    seen_towns = set()
    seen_families = set()
    while remaining and len(selected) < dev_count:
        row = min(
            remaining,
            key=lambda item: (
                -int(item.get("scenario_family") not in seen_families),
                -int(item.get("town") not in seen_towns),
                _hash_rank(seed, str(item["event_id"])),
            ),
        )
        remaining.remove(row)
        selected.append(row)
        seen_towns.add(row.get("town"))
        seen_families.add(row.get("scenario_family"))
    dev_ids = {row["event_id"] for row in selected}
    return {
        row["event_id"]: ("dev" if row["event_id"] in dev_ids else "train")
        for row in rows
    }


def _validate_locked_test_lineage(row: Mapping[str, Any]) -> None:
    package_ref = row["event_package"]
    package_path = _resolve_reference(package_ref, Path("/"), "event package")
    package = _load_json(package_path)
    batch_ref = package.get("source_files", {}).get("batch_manifest")
    if not isinstance(batch_ref, dict):
        raise ValueError("locked-test event lacks batch-manifest lineage")
    batch_path = _resolve_reference(batch_ref, package_path.parent, "batch manifest")
    batch = _load_json(batch_path)
    audit = batch.get("audit", {})
    if batch.get("split") != "locked_test":
        raise ValueError("locked-test event points to a non-locked batch")
    if audit.get("eligible_for_locked_test_claim") is not True:
        raise ValueError("locked-test batch is not claim-eligible")
    forbidden = (
        audit.get("selection_uses_published_orion_outcomes"),
        audit.get("selection_uses_learned_uq_outcomes"),
        audit.get("selection_uses_stage2_outcomes"),
    )
    if any(value is not False for value in forbidden):
        raise ValueError("locked-test batch has outcome-informed selection lineage")


def freeze_event_bank(
    *,
    queue_path: Path,
    decisions_path: Path,
    stage2_config_path: Path,
    split_seed: str,
) -> Dict[str, Any]:
    queue = _load_json(queue_path)
    decisions = _load_json(decisions_path)
    config = _load_json(stage2_config_path)
    if queue.get("schema") != QUEUE_SCHEMA:
        raise ValueError("unsupported review-queue schema")
    if decisions.get("schema") != DECISIONS_SCHEMA:
        raise ValueError("unsupported review-decisions schema")
    _resolve_reference(decisions["review_queue"], decisions_path.parent, "review queue")
    if decisions.get("reviewer") in (None, "") or decisions.get("reviewed_at") in (None, ""):
        raise ValueError("reviewer and reviewed_at must be recorded before freezing")

    queue_by_event = {row["event_id"]: row for row in queue.get("review_order", [])}
    decision_by_event: Dict[str, Mapping[str, Any]] = {}
    for decision in decisions.get("decisions", []):
        event_id = str(decision.get("event_id", ""))
        if event_id in decision_by_event:
            raise ValueError("duplicate review decision for %s" % event_id)
        if event_id not in queue_by_event:
            raise ValueError("decision references an event outside the review queue")
        if decision.get("event_package_sha256") != queue_by_event[event_id]["event_package"]["sha256"]:
            raise ValueError("review decision event-package hash mismatch")
        decision_by_event[event_id] = decision
    if set(decision_by_event) != set(queue_by_event):
        raise ValueError("every queued event must have exactly one decision")

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    pending: List[str] = []
    for event_id, row in queue_by_event.items():
        decision = decision_by_event[event_id]
        value = decision.get("decision")
        if value == "pending":
            pending.append(event_id)
            continue
        if value not in ("accept", "reject"):
            raise ValueError("invalid decision for %s" % event_id)
        checks = decision.get("checks", {})
        if set(checks) != set(HUMAN_CHECKS):
            raise ValueError("review checks are incomplete for %s" % event_id)
        if any(result not in ("pass", "fail") for result in checks.values()):
            raise ValueError("review checks must be pass/fail for %s" % event_id)
        all_pass = all(result == "pass" for result in checks.values())
        if value == "accept" and not all_pass:
            raise ValueError("accepted event has a failed integrity check: %s" % event_id)
        if value == "reject" and all_pass:
            if row.get("split_origin") == "locked_test":
                raise ValueError("locked-test event may only be rejected for technical integrity")
            if decision.get("rejection_basis") not in (
                "duplicate_or_redundant_development_event",
                "not_useful_for_development",
            ):
                raise ValueError("development rejection with passing checks lacks a valid basis")
        record = dict(row)
        record["human_review"] = {
            "decision": value,
            "checks": dict(checks),
            "rejection_basis": decision.get("rejection_basis"),
            "notes": decision.get("notes", ""),
        }
        (accepted if value == "accept" else rejected).append(record)
    if pending:
        raise ValueError("pending review decisions remain: %s" % ", ".join(sorted(pending)))

    route_ids = [int(row["route_index"]) for row in accepted]
    if len(route_ids) != len(set(route_ids)):
        raise ValueError("accepted bank currently permits only one event per route")
    locked = [row for row in accepted if row.get("split_origin") == "locked_test"]
    development = [
        row for row in accepted if row.get("split_origin") == "development_screen"
    ]
    unknown = [
        row for row in accepted
        if row.get("split_origin") not in ("development_screen", "locked_test")
    ]
    if unknown:
        raise ValueError("accepted events contain unsupported split origins")
    for row in locked:
        _validate_locked_test_lineage(row)

    if isinstance(config.get("data_gates"), Mapping):
        target = config["data_gates"].get("formal_training_target")
    else:
        target = None
    if target is None and isinstance(config.get("event_review_and_freezing"), Mapping):
        target = config["event_review_and_freezing"].get("formal_target")
    if not isinstance(target, Mapping):
        raise ValueError(
            "stage2 config must define data_gates.formal_training_target or "
            "event_review_and_freezing.formal_target"
        )
    minimum_events = target.get("minimum_independent_events", target.get("events"))
    minimum_towns = target.get("minimum_towns", target.get("towns"))
    minimum_families = target.get(
        "minimum_scenario_families", target.get("scenario_families")
    )
    if any(value is None for value in (minimum_events, minimum_towns, minimum_families)):
        raise ValueError("formal target is missing event/town/scenario-family gates")
    split_target = target["route_event_split"]
    assignments = _select_development_split(
        development, int(split_target["dev"]), split_seed
    ) if len(development) >= int(split_target["dev"]) else {
        row["event_id"]: "train" for row in development
    }
    for row in accepted:
        row["stage2_split"] = (
            "test" if row.get("split_origin") == "locked_test" else assignments[row["event_id"]]
        )
    accepted.sort(key=lambda row: (row["stage2_split"], int(row["route_index"])))

    split_counts = {
        split: sum(row["stage2_split"] == split for row in accepted)
        for split in ("train", "dev", "test")
    }
    counts = {
        "accepted_events": len(accepted),
        "rejected_events": len(rejected),
        "towns": len({row.get("town") for row in accepted}),
        "scenario_families": len({row.get("scenario_family") for row in accepted}),
        "splits": split_counts,
    }
    checks = {
        "minimum_independent_events": counts["accepted_events"] >= int(minimum_events),
        "minimum_towns": counts["towns"] >= int(minimum_towns),
        "minimum_scenario_families": counts["scenario_families"] >= int(minimum_families),
        "minimum_train_events": split_counts["train"] >= int(split_target["train"]),
        "minimum_dev_events": split_counts["dev"] >= int(split_target["dev"]),
        "minimum_locked_test_events": split_counts["test"] >= int(split_target["test"]),
        "route_disjoint": len(route_ids) == len(set(route_ids)),
        "locked_test_lineage_valid": True,
    }
    formal_ready = all(checks.values())
    return {
        "schema": BANK_SCHEMA,
        "status": "formal_event_bank_ready" if formal_ready else "reviewed_pilot_bank_below_formal_gate",
        "formal_training_ready": formal_ready,
        "split_seed": split_seed,
        "counts": counts,
        "checks": checks,
        "events": accepted,
        "rejected_events": rejected,
        "provenance": {
            "review_queue": {"path": str(queue_path.resolve()), "sha256": sha256_file(queue_path)},
            "review_decisions": {"path": str(decisions_path.resolve()), "sha256": sha256_file(decisions_path)},
            "stage2_config": {"path": str(stage2_config_path.resolve()), "sha256": sha256_file(stage2_config_path)},
        },
        "claim_boundary": (
            "A frozen event bank establishes data integrity and split governance only. "
            "Stage1, Stage2-L, and closed-loop claims require their own evaluations."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--review-decisions", type=Path, required=True)
    parser.add_argument("--stage2-config", type=Path, required=True)
    parser.add_argument("--split-seed", default="orion-stage2l-split-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite event bank")
    bank = freeze_event_bank(
        queue_path=args.review_queue.resolve(),
        decisions_path=args.review_decisions.resolve(),
        stage2_config_path=args.stage2_config.resolve(),
        split_seed=args.split_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bank, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve()), "formal_training_ready": bank["formal_training_ready"], "counts": bank["counts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

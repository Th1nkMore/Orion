#!/usr/bin/env python3
"""Freeze human-reviewed Stage2-L QA geometry decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    from scripts.build_stage2l_qa_geometry_review_queue import (
        DECISIONS_SCHEMA,
        HUMAN_CHECKS,
        QUEUE_SCHEMA,
    )
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from build_stage2l_qa_geometry_review_queue import (
        DECISIONS_SCHEMA,
        HUMAN_CHECKS,
        QUEUE_SCHEMA,
    )
    from scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2_l.qa_geometry_review_bank.v1"


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _resolve(reference: Mapping[str, Any], base: Path, name: str) -> Path:
    path = Path(str(reference.get("path", "")))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file() or sha256_file(path) != reference.get("sha256"):
        raise ValueError("%s is absent or has a SHA-256 mismatch" % name)
    return path


def freeze_review(*, queue_path: Path, decisions_path: Path) -> Dict[str, Any]:
    queue = _load(queue_path)
    decisions = _load(decisions_path)
    if queue.get("schema") != QUEUE_SCHEMA or decisions.get("schema") != DECISIONS_SCHEMA:
        raise ValueError("unsupported QA geometry review schema")
    resolved_queue = _resolve(decisions["review_queue"], decisions_path.parent, "QA geometry review queue")
    if resolved_queue != queue_path:
        raise ValueError("decisions reference a different QA geometry review queue")
    if not decisions.get("reviewer") or not decisions.get("reviewed_at"):
        raise ValueError("reviewer and reviewed_at are required")
    queued = {row["event_id"]: row for row in queue.get("review_order", [])}
    decided = {row["event_id"]: row for row in decisions.get("decisions", [])}
    if len(decided) != len(decisions.get("decisions", [])) or set(decided) != set(queued):
        raise ValueError("every queued QA event must have exactly one decision")
    accepted = []
    rejected = []
    for event_id, item in queued.items():
        decision = decided[event_id]
        if decision.get("factory_report_sha256") != item["factory_report"]["sha256"]:
            raise ValueError("QA geometry decision factory-report hash mismatch")
        checks = decision.get("checks", {})
        if set(checks) != set(HUMAN_CHECKS) or any(value not in ("pass", "fail") for value in checks.values()):
            raise ValueError("QA geometry review checks must be complete pass/fail values")
        value = decision.get("decision")
        if value not in ("accept", "reject"):
            raise ValueError("pending or invalid QA geometry decision remains")
        all_pass = all(result == "pass" for result in checks.values())
        if value == "accept" and not all_pass:
            raise ValueError("accepted QA geometry event has a failed check")
        if value == "reject" and all_pass and not decision.get("rejection_basis"):
            raise ValueError("rejected QA geometry event lacks a rejection basis")
        row = {
            "event_id": event_id,
            "factory_report": item["factory_report"],
            "decision": value,
            "checks": checks,
            "rejection_basis": decision.get("rejection_basis"),
            "notes": decision.get("notes", ""),
        }
        (accepted if value == "accept" else rejected).append(row)
    return {
        "schema": SCHEMA,
        "status": "frozen_human_qa_geometry_review",
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted": accepted,
        "rejected": rejected,
        "reviewer": decisions["reviewer"],
        "reviewed_at": decisions["reviewed_at"],
        "provenance": {
            "review_queue": {"path": str(queue_path.resolve()), "sha256": sha256_file(queue_path)},
            "review_decisions": {"path": str(decisions_path.resolve()), "sha256": sha256_file(decisions_path)},
        },
        "claim_boundary": "Human QA-geometry integrity review only; not Stage1, VLM, trajectory, closed-loop, or safety evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--review-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen QA geometry review bank")
    result = freeze_review(queue_path=args.review_queue.resolve(), decisions_path=args.review_decisions.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "accepted_count": result["accepted_count"], "rejected_count": result["rejected_count"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

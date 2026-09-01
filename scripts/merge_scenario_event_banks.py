#!/usr/bin/env python3
"""Merge hash-audited reviewed development event banks before pilot selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

try:
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from scenario_factory_lib import sha256_file


SCHEMA = "orion.scenario_event_bank.v1"


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def merge_event_banks(paths: Sequence[Path]) -> Dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two reviewed event banks are required")
    events = []
    rejected = []
    sources = []
    event_ids = set()
    route_ids = set()
    for raw_path in paths:
        path = raw_path.resolve()
        bank = _load(path)
        if bank.get("schema") != SCHEMA:
            raise ValueError("unsupported event-bank schema: %s" % path)
        for source_row in bank.get("events", []):
            row = dict(source_row)
            event_id = str(row.get("event_id", ""))
            route_id = int(row.get("route_index", -1))
            if not event_id or event_id in event_ids:
                raise ValueError("event id is empty or duplicated: %s" % event_id)
            if route_id < 0 or route_id in route_ids:
                raise ValueError("route id is invalid or duplicated: %s" % route_id)
            if row.get("human_review", {}).get("decision") != "accept":
                raise ValueError("merged source contains a non-accepted event")
            if row.get("split_origin") != "development_screen":
                raise ValueError("pilot merge accepts development-screen events only")
            row.pop("stage2_split", None)
            row.pop("pilot_split", None)
            events.append(row)
            event_ids.add(event_id)
            route_ids.add(route_id)
        rejected.extend(bank.get("rejected_events", []))
        sources.append({"path": str(path), "sha256": sha256_file(path)})
    events.sort(key=lambda row: int(row["route_index"]))
    return {
        "schema": SCHEMA,
        "status": "merged_reviewed_event_bank_below_formal_gate",
        "formal_training_ready": False,
        "counts": {
            "accepted_events": len(events),
            "rejected_events": len(rejected),
            "towns": len({row["town"] for row in events}),
            "scenario_families": len({row["scenario_family"] for row in events}),
            "splits": {"train": 0, "dev": 0, "test": 0},
        },
        "checks": {
            "route_disjoint": len(route_ids) == len(events),
            "all_events_human_review_accepted": True,
            "development_screen_only": True,
        },
        "events": events,
        "rejected_events": rejected,
        "provenance": {"source_event_banks": sources},
        "claim_boundary": (
            "Mechanical merge of reviewed development events before explicit pilot "
            "selection; it does not assign splits or establish model evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-bank", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite merged event bank")
    result = merge_event_banks(args.event_bank)
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

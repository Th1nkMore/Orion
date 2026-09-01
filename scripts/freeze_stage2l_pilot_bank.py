#!/usr/bin/env python3
"""Freeze an eight-event, route-disjoint 6/2 Stage2-L pilot view."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

try:
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2_l.pilot_event_bank.v1"
EVENT_BANK_SCHEMA = "orion.scenario_event_bank.v1"
SCHEDULE_SCHEMA = "orion.stage2_l.schedule.v1"
SPLIT_SEED = "orion-stage2l-pilot-split-v1"


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _hash_rank(event_id: str) -> str:
    return hashlib.sha256((SPLIT_SEED + "\0" + event_id).encode("utf-8")).hexdigest()


def _dev_ids(events: Sequence[Mapping[str, Any]], count: int) -> set:
    remaining = list(events)
    selected: List[Mapping[str, Any]] = []
    towns = set()
    families = set()
    while len(selected) < count:
        row = min(
            remaining,
            key=lambda item: (
                -int(item["scenario_family"] not in families),
                -int(item["town"] not in towns),
                _hash_rank(str(item["event_id"])),
            ),
        )
        remaining.remove(row)
        selected.append(row)
        towns.add(row["town"])
        families.add(row["scenario_family"])
    return {str(row["event_id"]) for row in selected}


def freeze_pilot_bank(
    *, event_bank_path: Path, schedule_path: Path, event_ids: Sequence[str]
) -> Dict[str, Any]:
    bank = _load(event_bank_path)
    schedule = _load(schedule_path)
    if bank.get("schema") != EVENT_BANK_SCHEMA:
        raise ValueError("unsupported reviewed event-bank schema")
    if schedule.get("schema") != SCHEDULE_SCHEMA:
        raise ValueError("unsupported Stage2-L schedule schema")
    base_ref = schedule.get("base_scenario_factory_protocol", {})
    base_path = Path(str(base_ref.get("path", "")))
    if not base_path.is_absolute():
        base_path = schedule_path.parents[2] / base_path
    if not base_path.is_file() or sha256_file(base_path) != base_ref.get("sha256"):
        raise ValueError("Stage2-L schedule base-protocol provenance mismatch")

    gate = schedule["pilot_gate"]
    required = int(gate["minimum_independent_events"])
    source = {str(row["event_id"]): dict(row) for row in bank.get("events", [])}
    if len(source) != len(bank.get("events", [])):
        raise ValueError("reviewed event bank duplicates event ids")
    if event_ids:
        if len(event_ids) != required or len(set(event_ids)) != required:
            raise ValueError("pilot selection must explicitly contain exactly %d events" % required)
        missing = sorted(set(event_ids) - set(source))
        if missing:
            raise ValueError("pilot selection references unknown events: %s" % missing)
        selected = [source[event_id] for event_id in event_ids]
    else:
        if len(source) != required:
            raise ValueError("explicit --event-id values are required unless the bank has exactly %d events" % required)
        selected = list(source.values())

    if any(row.get("split_origin") != "development_screen" for row in selected):
        raise ValueError("Stage2-L pilot may not consume locked-test events")
    routes = [int(row["route_index"]) for row in selected]
    if len(routes) != len(set(routes)):
        raise ValueError("Stage2-L pilot events are not route-disjoint")
    towns = {row["town"] for row in selected}
    families = {row["scenario_family"] for row in selected}
    if len(towns) < int(gate["minimum_towns"]):
        raise ValueError("Stage2-L pilot town gate is not met")
    if len(families) < int(gate["minimum_scenario_families"]):
        raise ValueError("Stage2-L pilot scenario-family gate is not met")

    split = gate["event_level_split"]
    if int(split["train"]) + int(split["dev"]) != required or int(split["test"]) != 0:
        raise ValueError("Stage2-L pilot schedule must define an eight-event train/dev split")
    dev_ids = _dev_ids(selected, int(split["dev"]))
    for row in selected:
        row["pilot_split"] = "dev" if row["event_id"] in dev_ids else "train"
    selected.sort(key=lambda row: (row["pilot_split"], int(row["route_index"])))
    split_counts = {
        name: sum(row["pilot_split"] == name for row in selected)
        for name in ("train", "dev")
    }
    if split_counts != {"train": int(split["train"]), "dev": int(split["dev"])}:
        raise RuntimeError("Stage2-L pilot split count mismatch")
    return {
        "schema": SCHEMA,
        "status": "frozen_before_stage2l_pilot_training",
        "formal_training_ready": False,
        "split_seed": SPLIT_SEED,
        "counts": {
            "events": len(selected),
            "towns": len(towns),
            "scenario_families": len(families),
            "splits": split_counts,
        },
        "events": selected,
        "provenance": {
            "reviewed_event_bank": {
                "path": str(event_bank_path.resolve()),
                "sha256": sha256_file(event_bank_path),
            },
            "stage2l_schedule": {
                "path": str(schedule_path.resolve()),
                "sha256": sha256_file(schedule_path),
            },
        },
        "claim_boundary": "Frozen 6/2 engineering pilot split only; no formal generalization, trajectory, closed-loop, or safety claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-bank", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--event-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen Stage2-L pilot bank")
    result = freeze_pilot_bank(
        event_bank_path=args.event_bank.resolve(),
        schedule_path=args.schedule.resolve(),
        event_ids=args.event_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve()), "counts": result["counts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

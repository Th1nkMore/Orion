#!/usr/bin/env python3
"""Freeze a Stage2-L pilot from reviewed formal-wave shards.

Unlike the legacy pilot freezer, this entry point never invents a new split.
It preserves the train/dev assignments frozen in the formal route plan and
fills each quota in formal-plan order.  Consequently collision, TTC, UQ, and
Stage2 outcomes cannot influence which technically accepted event is selected.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

try:
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2_l.formal_pilot_event_bank.v1"
FORMAL_PLAN_SCHEMA = "orion.stage2_l.formal_route_plan.v1"
REVIEWED_WAVE_SCHEMA = "orion.stage2_l.formal_reviewed_wave.v1"
SCHEDULE_SCHEMA = "orion.stage2_l.schedule.v2"
GEOMETRY_PREFLIGHT_SCHEMA = "orion.stage2l_event_geometry_preflight.v1"
ALLOWED_PILOT_SPLITS = ("train", "dev")


def _load(path: Path, schema: str) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError("unexpected schema for %s" % path)
    return value


def _resolve_project_reference(schedule_path: Path, raw_path: str) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path.resolve()
    return (schedule_path.parents[2] / path).resolve()


def _validate_schedule_provenance(
    *, schedule: Mapping[str, Any], schedule_path: Path, formal_plan_path: Path
) -> None:
    base = schedule.get("base_scenario_factory_protocol", {})
    base_path = _resolve_project_reference(schedule_path, str(base.get("path", "")))
    if not base_path.is_file() or sha256_file(base_path) != base.get("sha256"):
        raise ValueError("Stage2-L schedule base-protocol provenance mismatch")
    plan = schedule.get("formal_route_plan", {})
    referenced_plan = _resolve_project_reference(
        schedule_path, str(plan.get("path", ""))
    )
    if referenced_plan != formal_plan_path.resolve():
        raise ValueError("schedule references a different formal route plan")
    if sha256_file(formal_plan_path) != plan.get("sha256"):
        raise ValueError("Stage2-L schedule formal-plan provenance mismatch")
    if schedule.get("pilot_gate", {}).get("preserve_formal_route_plan_splits") is not True:
        raise ValueError("schedule does not require preserved formal splits")


def freeze_formal_pilot_bank(
    *,
    formal_plan_path: Path,
    schedule_path: Path,
    reviewed_wave_paths: Sequence[Path],
    geometry_preflight_paths: Sequence[Path],
) -> Dict[str, Any]:
    if not reviewed_wave_paths:
        raise ValueError("at least one reviewed formal wave is required")
    formal_plan_path = formal_plan_path.resolve()
    schedule_path = schedule_path.resolve()
    plan = _load(formal_plan_path, FORMAL_PLAN_SCHEMA)
    schedule = _load(schedule_path, SCHEDULE_SCHEMA)
    _validate_schedule_provenance(
        schedule=schedule,
        schedule_path=schedule_path,
        formal_plan_path=formal_plan_path,
    )

    planned_events = list(plan.get("events", []))
    plan_by_route = {int(row["route_index"]): row for row in planned_events}
    if len(plan_by_route) != len(planned_events):
        raise ValueError("formal route plan duplicates a route")
    plan_rank = {
        int(row["route_index"]): rank for rank, row in enumerate(planned_events)
    }
    plan_hash = sha256_file(formal_plan_path)

    accepted_by_route: Dict[int, Dict[str, Any]] = {}
    event_ids = set()
    sources = []
    for raw_path in reviewed_wave_paths:
        path = raw_path.resolve()
        wave = _load(path, REVIEWED_WAVE_SCHEMA)
        wave_plan = wave.get("provenance", {}).get("formal_route_plan", {})
        if wave_plan.get("sha256") != plan_hash:
            raise ValueError("reviewed wave is bound to a different formal plan")
        for source_row in wave.get("events", []):
            row = dict(source_row)
            route = int(row.get("route_index", -1))
            event_id = str(row.get("event_id", ""))
            if route not in plan_by_route:
                raise ValueError("reviewed event route is absent from formal plan")
            if route in accepted_by_route or not event_id or event_id in event_ids:
                raise ValueError("reviewed waves duplicate a route or event id")
            planned = plan_by_route[route]
            if row.get("human_review", {}).get("decision") != "accept":
                raise ValueError("reviewed wave contains a non-accepted event")
            if row.get("qa_input_ready") is not True:
                raise ValueError("accepted event is not QA-input ready")
            if row.get("split_origin") != "development_screen":
                raise ValueError("pilot may consume development-screen events only")
            if row.get("formal_split") != planned.get("formal_split"):
                raise ValueError("reviewed wave changed a frozen formal split")
            if row.get("town") != planned.get("town"):
                raise ValueError("reviewed event town differs from formal plan")
            if row.get("scenario_family") != planned.get("scenario_family"):
                raise ValueError("reviewed event family differs from formal plan")
            if row.get("formal_split") not in ALLOWED_PILOT_SPLITS:
                raise ValueError("locked test events may not enter the pilot")
            accepted_by_route[route] = row
            event_ids.add(event_id)
        sources.append({"path": str(path), "sha256": sha256_file(path)})

    geometry_by_event: Dict[str, Dict[str, Any]] = {}
    geometry_reference_by_event: Dict[str, Dict[str, str]] = {}
    geometry_sources = []
    for raw_path in geometry_preflight_paths:
        path = raw_path.resolve()
        preflight = _load(path, GEOMETRY_PREFLIGHT_SCHEMA)
        event_id = str(preflight.get("event_id", ""))
        if not event_id or event_id in geometry_by_event:
            raise ValueError("geometry preflights duplicate or omit an event id")
        geometry_by_event[event_id] = preflight
        reference = {"path": str(path), "sha256": sha256_file(path)}
        geometry_reference_by_event[event_id] = reference
        geometry_sources.append(reference)
    if set(geometry_by_event) != event_ids:
        raise ValueError("geometry preflights do not exactly cover reviewed events")
    for row in accepted_by_route.values():
        preflight = geometry_by_event[str(row["event_id"])]
        if (
            preflight.get("provenance", {}).get("event_package", {}).get("sha256")
            != row.get("event_package", {}).get("sha256")
        ):
            raise ValueError("geometry preflight event-package hash mismatch")
        if int(preflight.get("minimum_retained_keyframes", -1)) != int(
            schedule["pilot_gate"]["geometry_eligibility"][
                "minimum_retained_keyframes"
            ]
        ):
            raise ValueError("geometry preflight uses a different retention gate")

    gate = schedule["pilot_gate"]
    split_gate = gate["event_level_split"]
    if int(split_gate.get("test", -1)) != 0:
        raise ValueError("Stage2-L pilot must not allocate test events")
    required_by_split = {
        split: int(split_gate[split]) for split in ALLOWED_PILOT_SPLITS
    }
    required_total = int(gate["minimum_independent_events"])
    if sum(required_by_split.values()) != required_total:
        raise ValueError("pilot split quotas do not sum to the event gate")

    technically_excluded = []
    eligible_rows = []
    for row in accepted_by_route.values():
        preflight = geometry_by_event[str(row["event_id"])]
        if (
            preflight.get("status") == "eligible_before_stage1_extraction"
            and preflight.get("eligible") is True
            and int(preflight.get("retained_keyframe_count", 0))
            >= int(preflight["minimum_retained_keyframes"])
        ):
            value = dict(row)
            value["geometry_preflight"] = geometry_reference_by_event[
                str(row["event_id"])
            ]
            eligible_rows.append(value)
        else:
            technically_excluded.append({
                "event_id": str(row["event_id"]),
                "route_index": int(row["route_index"]),
                "reason": "fewer_than_three_fixed_keyframes_have_visible_task_relevance_support",
            })
    ordered = sorted(
        eligible_rows, key=lambda row: plan_rank[int(row["route_index"])]
    )
    selected: List[Dict[str, Any]] = []
    reserve: List[Dict[str, Any]] = []
    remaining = dict(required_by_split)
    for source_row in ordered:
        row = dict(source_row)
        split = str(row["formal_split"])
        if remaining[split] > 0:
            row["pilot_split"] = split
            row["pilot_selection_basis"] = "formal_plan_order_within_split"
            selected.append(row)
            remaining[split] -= 1
        else:
            row["pilot_role"] = "reviewed_reserve"
            reserve.append(row)
    if any(remaining.values()):
        raise ValueError("reviewed formal waves do not satisfy pilot split quotas")

    selected.sort(key=lambda row: plan_rank[int(row["route_index"])] )
    towns = {str(row["town"]) for row in selected}
    families = {str(row["scenario_family"]) for row in selected}
    routes = {int(row["route_index"]) for row in selected}
    if len(selected) != required_total or len(routes) != required_total:
        raise RuntimeError("pilot selection is not event- and route-disjoint")
    if len(towns) < int(gate["minimum_towns"]):
        raise ValueError("formal pilot town gate is not met")
    if len(families) < int(gate["minimum_scenario_families"]):
        raise ValueError("formal pilot scenario-family gate is not met")
    split_counts = Counter(str(row["pilot_split"]) for row in selected)
    if any(split_counts[split] != required_by_split[split] for split in ALLOWED_PILOT_SPLITS):
        raise RuntimeError("formal pilot split count mismatch")

    return {
        "schema": SCHEMA,
        "status": "frozen_bank_training_still_locked",
        "formal_training_ready": False,
        "pilot_training_ready": False,
        "selection_policy": {
            "name": "formal_plan_order_within_preserved_split_v1",
            "uses_collision_or_ttc": False,
            "uses_uq_or_stage2_outcomes": False,
            "reassigns_frozen_splits": False,
        },
        "counts": {
            "events": len(selected),
            "towns": len(towns),
            "scenario_families": len(families),
            "splits": {
                split: int(split_counts[split]) for split in ALLOWED_PILOT_SPLITS
            },
            "reviewed_reserves": len(reserve),
            "technical_geometry_exclusions": len(technically_excluded),
        },
        "checks": {
            "route_disjoint": len(routes) == len(selected),
            "all_human_review_accepted": True,
            "all_qa_input_ready": True,
            "development_screen_only": True,
            "formal_splits_preserved": True,
            "locked_test_untouched": True,
        },
        "events": selected,
        "reviewed_reserve_events": reserve,
        "technical_geometry_exclusions": technically_excluded,
        "provenance": {
            "formal_route_plan": {
                "path": str(formal_plan_path),
                "sha256": plan_hash,
            },
            "stage2l_schedule": {
                "path": str(schedule_path),
                "sha256": sha256_file(schedule_path),
            },
            "reviewed_formal_waves": sources,
            "geometry_preflights": geometry_sources,
        },
        "claim_boundary": (
            "Frozen eight-event engineering pilot bank with preserved formal "
            "splits. Training remains locked and this artifact provides no model, "
            "trajectory, closed-loop, or safety evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-route-plan", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--reviewed-wave", type=Path, action="append", required=True)
    parser.add_argument("--geometry-preflight", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen formal pilot bank")
    result = freeze_formal_pilot_bank(
        formal_plan_path=args.formal_route_plan,
        schedule_path=args.schedule,
        reviewed_wave_paths=args.reviewed_wave,
        geometry_preflight_paths=args.geometry_preflight,
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

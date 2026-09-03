#!/usr/bin/env python3
"""Preview the deterministic first-8 Stage2-L selection without freezing it."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

try:
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from scenario_factory_lib import sha256_file


PLAN_SCHEMA = "orion.stage2_l.formal_route_plan.v1"
REVIEWED_WAVE_SCHEMA = "orion.stage2_l.formal_reviewed_wave.v1"
BATCH_SCREEN_SCHEMA = "orion.scenario_factory.batch_screen_report.v1"
GEOMETRY_SCHEMA = "orion.stage2l_event_geometry_preflight.v1"
SCHEMA = "orion.stage2_l.formal_pilot_selection_preview.v1"


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def preview_selection(
    *,
    formal_plan_path: Path,
    schedule_path: Path,
    reviewed_wave_paths: Sequence[Path],
    pending_batch_path: Path,
    geometry_preflight_paths: Sequence[Path],
) -> Dict[str, Any]:
    plan = _load(formal_plan_path)
    schedule = _load(schedule_path)
    pending = _load(pending_batch_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported formal route plan")
    if pending.get("schema") != BATCH_SCREEN_SCHEMA:
        raise ValueError("unsupported pending batch-screen report")
    if schedule.get("schema") != "orion.stage2_l.schedule.v2":
        raise ValueError("unsupported Stage2-L schedule")
    plan_by_route = {int(row["route_index"]): row for row in plan.get("events", [])}
    plan_rank = {route: index for index, route in enumerate(plan_by_route)}
    accepted: Dict[int, Dict[str, Any]] = {}
    for path in reviewed_wave_paths:
        wave = _load(path)
        if wave.get("schema") != REVIEWED_WAVE_SCHEMA:
            raise ValueError("unsupported reviewed wave")
        for row in wave.get("events", []):
            route = int(row["route_index"])
            if route in accepted:
                raise ValueError("reviewed waves duplicate a route")
            if row.get("human_review", {}).get("decision") != "accept":
                raise ValueError("reviewed wave is not accepted")
            if route not in plan_by_route:
                raise ValueError("reviewed route is absent from formal plan")
            planned = plan_by_route[route]
            if row.get("formal_split") != planned.get("formal_split"):
                raise ValueError("reviewed wave changed a formal split")
            accepted[route] = {
                "route_index": route,
                "event_id": str(row["event_id"]),
                "town": str(row["town"]),
                "scenario_family": str(row["scenario_family"]),
                "formal_split": str(row["formal_split"]),
                "source_status": "reviewed_accepted",
            }

    pending_by_route = {}
    for row in pending.get("routes", []):
        route = int(row["route_index"])
        if route in accepted or route not in plan_by_route:
            raise ValueError("pending batch route duplicates or is absent from plan")
        planned = plan_by_route[route]
        if row.get("qa_input_ready") is not True or row.get("runtime_valid") is not True:
            raise ValueError("pending candidate is not runtime/QA ready")
        if row.get("town") != planned.get("town"):
            raise ValueError("pending candidate changed the planned town")
        if row.get("scenario_type") != planned.get("scenario_family"):
            raise ValueError("pending candidate changed the planned scenario family")
        pending_by_route[route] = {
            "route_index": route,
            "event_id": "pending:%d" % route,
            "town": str(row["town"]),
            "scenario_family": str(row["scenario_type"]),
            "formal_split": str(planned["formal_split"]),
            "source_status": "pending_event_review",
        }

    geometry_by_route = {}
    for path in geometry_preflight_paths:
        report = _load(path)
        if report.get("schema") != GEOMETRY_SCHEMA:
            raise ValueError("unsupported geometry preflight")
        event_id = str(report.get("event_id", ""))
        try:
            route = int(event_id.split("_", 1)[0].replace("route", ""))
        except (IndexError, ValueError) as error:
            raise ValueError("geometry event id lacks route index") from error
        if route in geometry_by_route:
            raise ValueError("geometry preflights duplicate a route")
        if report.get("eligible") is not True or int(report.get("retained_keyframe_count", 0)) < 3:
            raise ValueError("geometry candidate does not meet the three-keyframe gate")
        geometry_by_route[route] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "event_id": event_id,
            "retained_keyframes": int(report["retained_keyframe_count"]),
        }

    candidates = []
    for route, row in {**accepted, **pending_by_route}.items():
        if route not in geometry_by_route:
            raise ValueError("candidate lacks a geometry preflight")
        value = dict(row)
        value["geometry_preflight"] = geometry_by_route[route]
        value["formal_plan_rank"] = plan_rank[route]
        candidates.append(value)
    candidates.sort(key=lambda row: row["formal_plan_rank"])

    quotas = schedule["pilot_gate"]["event_level_split"]
    selected = []
    remaining = {split: int(quotas[split]) for split in ("train", "dev")}
    for row in candidates:
        split = row["formal_split"]
        if remaining[split] > 0:
            value = dict(row)
            value["preview_role"] = "selected_if_source_accepted"
            selected.append(value)
            remaining[split] -= 1
        else:
            value = dict(row)
            value["preview_role"] = "reserve_if_source_accepted"
            selected.append(value)
    selected_first8 = [row for row in selected if row["preview_role"] == "selected_if_source_accepted"]
    selected_first8.sort(key=lambda row: row["formal_plan_rank"])
    split_counts = Counter(row["formal_split"] for row in selected_first8)
    return {
        "schema": SCHEMA,
        "status": "candidate_preview_human_acceptance_pending",
        "frozen": False,
        "training_allowed": False,
        "selection_policy": {
            "name": "formal_plan_order_within_preserved_split_v1",
            "uses_collision_or_ttc": False,
            "uses_uq_or_stage2_outcomes": False,
            "uses_review_outcome_for_order": False,
        },
        "candidate_count": len(candidates),
        "selected_if_pending_candidates_accepted": selected_first8,
        "selected_counts_if_pending_candidates_accepted": {
            "events": len(selected_first8),
            "splits": dict(sorted(split_counts.items())),
            "towns": len({row["town"] for row in selected_first8}),
            "scenario_families": len({row["scenario_family"] for row in selected_first8}),
        },
        "reserve_if_pending_candidates_accepted": [
            row for row in selected if row["preview_role"] == "reserve_if_source_accepted"
        ],
        "provenance": {
            "formal_route_plan": {"path": str(formal_plan_path.resolve()), "sha256": sha256_file(formal_plan_path)},
            "schedule": {"path": str(schedule_path.resolve()), "sha256": sha256_file(schedule_path)},
            "reviewed_waves": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in reviewed_wave_paths],
            "pending_batch_screen": {"path": str(pending_batch_path.resolve()), "sha256": sha256_file(pending_batch_path)},
        },
        "claim_boundary": "Deterministic candidate selection preview only; no frozen bank, training, model, planning, closed-loop, or safety evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-route-plan", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--reviewed-wave", type=Path, action="append", required=True)
    parser.add_argument("--pending-batch-screen", type=Path, required=True)
    parser.add_argument("--geometry-preflight", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite selection preview")
    report = preview_selection(
        formal_plan_path=args.formal_route_plan.resolve(),
        schedule_path=args.schedule.resolve(),
        reviewed_wave_paths=[path.resolve() for path in args.reviewed_wave],
        pending_batch_path=args.pending_batch_screen.resolve(),
        geometry_preflight_paths=[path.resolve() for path in args.geometry_preflight],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["selected_counts_if_pending_candidates_accepted"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

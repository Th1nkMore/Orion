#!/usr/bin/env python3
"""Separate existing raw R coverage into route- and actor-grounded support."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.scan_stage2l_v12_existing_raw_coverage_inventory import (
    _aligned_frames,
    _job_id,
    _read_json,
    _reference,
    strict_clean_off_reason,
)
from uq_estimator.task_relevance_geometry import (
    CAMERA_ORDER,
    TaskRelevanceGeometryError,
    build_task_relevance_map,
)


SCHEMA = "orion.stage2l_v12_existing_raw_actor_support_inventory.v1"
PROTOCOL_SCHEMA = (
    "orion.stage2l_v12_existing_raw_actor_support_inventory_amendment.v1"
)
COMPONENTS = (
    "union_relevance",
    "route_support",
    "actor_support",
    "route_only",
    "actor_only",
    "route_and_actor",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_flags(geometry: Any, view_index: int) -> Dict[str, bool]:
    """Return mutually inspectable support flags for one view and frame."""

    route = bool((geometry.route_corridor[view_index] > 0.0).any())
    actor = bool((geometry.relevant_actor_support[view_index] > 0.0).any())
    union = bool((geometry.relevance[view_index] > 0.0).any())
    if union != (route or actor):
        raise ValueError("union relevance differs from route-or-actor support")
    return {
        "union_relevance": union,
        "route_support": route,
        "actor_support": actor,
        "route_only": route and not actor,
        "actor_only": actor and not route,
        "route_and_actor": route and actor,
    }


def _empty_component_frames() -> Dict[str, Dict[str, list[int]]]:
    return {
        component: {view: [] for view in CAMERA_ORDER}
        for component in COMPONENTS
    }


def scan_scenario_components(scenario_root: Path) -> Dict[str, Any]:
    meta_root, frames = _aligned_frames(scenario_root)
    component_frames = _empty_component_frames()
    global_support_modes = Counter()
    errors = Counter()
    relevant_actor_ids = set()
    for frame in frames:
        meta = _read_json(meta_root / ("%04d.json" % frame))
        try:
            geometry = build_task_relevance_map(
                meta["plan"], meta["closedloop_safety"], patch_hw=(40, 40)
            )
        except (TaskRelevanceGeometryError, KeyError) as error:
            errors["%s: %s" % (type(error).__name__, error)] += 1
            continue
        global_support_modes[geometry.provenance["support_mode"]] += 1
        relevant_actor_ids.update(map(int, geometry.relevant_actor_ids))
        for view_index, view in enumerate(CAMERA_ORDER):
            for component, positive in component_flags(geometry, view_index).items():
                if positive:
                    component_frames[component][view].append(frame)

    component_counts = {
        component: {
            view: len(component_frames[component][view]) for view in CAMERA_ORDER
        }
        for component in COMPONENTS
    }
    return {
        "scenario_root": str(scenario_root.resolve()),
        "aligned_frame_count": len(frames),
        "geometry_valid_frame_count": int(sum(global_support_modes.values())),
        "component_positive_frame_count": component_counts,
        "component_positive_frames": component_frames,
        "support_mode_counts": dict(sorted(global_support_modes.items())),
        "geometry_error_counts": dict(sorted(errors.items())),
        "relevant_actor_ids": sorted(relevant_actor_ids),
    }


def choose_per_route(rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    chosen: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        actor_counts = row["component_positive_frame_count"]["actor_support"]
        key = (
            sum(int(count > 0) for count in actor_counts.values()),
            sum(actor_counts.values()),
            row["geometry_valid_frame_count"],
            row["aligned_frame_count"],
            -row["job_id"],
        )
        current = chosen.get(int(row["route_index"]))
        if current is None:
            chosen[int(row["route_index"])] = row
            continue
        current_actor = current["component_positive_frame_count"]["actor_support"]
        current_key = (
            sum(int(count > 0) for count in current_actor.values()),
            sum(current_actor.values()),
            current["geometry_valid_frame_count"],
            current["aligned_frame_count"],
            -current["job_id"],
        )
        if key > current_key:
            chosen[int(row["route_index"])] = row
    return list(chosen.values())


def summarize_routes(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    route_counts = {
        component: {view: 0 for view in CAMERA_ORDER} for component in COMPONENTS
    }
    frame_view_counts = {
        component: {view: 0 for view in CAMERA_ORDER} for component in COMPONENTS
    }
    formal_train_route_counts = {
        component: {view: 0 for view in CAMERA_ORDER} for component in COMPONENTS
    }
    nonformal_route_counts = {
        component: {view: 0 for view in CAMERA_ORDER} for component in COMPONENTS
    }
    for row in rows:
        split_bucket = (
            formal_train_route_counts
            if row.get("formal_split") == "train"
            else nonformal_route_counts
        )
        counts = row["component_positive_frame_count"]
        for component in COMPONENTS:
            for view in CAMERA_ORDER:
                value = int(counts[component][view])
                frame_view_counts[component][view] += value
                if value > 0:
                    route_counts[component][view] += 1
                    split_bucket[component][view] += 1
    return {
        "independent_route_count_by_component_and_view": route_counts,
        "frame_view_count_by_component_and_view": frame_view_counts,
        "formal_train_route_count_by_component_and_view": formal_train_route_counts,
        "nonformal_route_count_by_component_and_view": nonformal_route_counts,
        "zero_actor_support_views": [
            view
            for view in CAMERA_ORDER
            if route_counts["actor_support"][view] == 0
        ],
        "zero_union_support_views": [
            view
            for view in CAMERA_ORDER
            if route_counts["union_relevance"][view] == 0
        ],
    }


def scan(
    *,
    protocol_path: Path,
    base_protocol_path: Path,
    base_result_path: Path,
    formal_plan_path: Path,
    accepted_bank_path: Path,
    results_root: Path,
) -> Dict[str, Any]:
    protocol = _read_json(protocol_path)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("actor-support inventory protocol schema differs")
    predecessors = protocol["authoritative_predecessors"]
    for name, path in (
        ("base_protocol", base_protocol_path),
        ("base_result", base_result_path),
    ):
        if predecessors[name]["sha256"] != _sha256(path):
            raise ValueError("protocol predecessor hash differs: %s" % name)
    base_protocol = _read_json(base_protocol_path)
    refs = base_protocol["authoritative_inputs"]
    for name, path in (
        ("formal_route_plan", formal_plan_path),
        ("accepted_event_bank", accepted_bank_path),
    ):
        if refs[name]["sha256"] != _sha256(path):
            raise ValueError("base protocol input hash differs: %s" % name)
    if str(results_root.resolve()) != refs["raw_results_root"]:
        raise ValueError("raw results root differs from base protocol")

    formal_plan = _read_json(formal_plan_path)
    accepted_bank = _read_json(accepted_bank_path)
    formal_split = {
        int(row["route_index"]): row["formal_split"] for row in formal_plan["events"]
    }
    accepted_routes = {int(row["route_index"]) for row in accepted_bank["events"]}
    skip_reasons = Counter()
    inspected_manifests = 0
    strict_runs = 0
    scenario_rows: list[Dict[str, Any]] = []
    for manifest_path in sorted(results_root.rglob("manifest.json")):
        inspected_manifests += 1
        try:
            manifest = _read_json(manifest_path)
        except Exception:
            skip_reasons["manifest_unreadable"] += 1
            continue
        reason = strict_clean_off_reason(manifest)
        if reason is not None:
            skip_reasons["strict_filter_%s" % reason] += 1
            continue
        route = int(manifest["pilot_route_index"])
        if formal_split.get(route) in ("dev", "test"):
            skip_reasons["formal_heldout_identity"] += 1
            continue
        strict_runs += 1
        run_dir = manifest_path.parent
        roots = sorted(
            path
            for path in (run_dir / "records_orion_traj_0").glob("RouteScenario_*")
            if path.is_dir()
        )
        if not roots:
            skip_reasons["no_scenario_root"] += 1
            continue
        for scenario_root in roots:
            row = scan_scenario_components(scenario_root)
            if row["aligned_frame_count"] == 0:
                skip_reasons["no_aligned_frames"] += 1
                continue
            row.update(
                {
                    "route_index": route,
                    "job_id": _job_id(run_dir, manifest),
                    "run_id": manifest.get("pilot_run_id"),
                    "run_dir": str(run_dir.resolve()),
                    "manifest": _reference(manifest_path),
                    "formal_split": formal_split.get(route),
                    "accepted_bank_identity": route in accepted_routes,
                    "new_independent_event_if_selected": route not in formal_split,
                }
            )
            scenario_rows.append(row)

    per_route = choose_per_route(scenario_rows)
    per_route.sort(key=lambda row: row["route_index"])
    summary = summarize_routes(per_route)
    actor_routes = {
        view: [
            int(row["route_index"])
            for row in per_route
            if row["component_positive_frame_count"]["actor_support"][view] > 0
        ]
        for view in CAMERA_ORDER
    }
    return {
        "schema": SCHEMA,
        "status": "existing_raw_support_components_frozen",
        "diagnostic_completed": True,
        "formal_bank_modified": False,
        "gpu_used": False,
        "orion_forward_run": False,
        "training_started": False,
        "inputs": {
            "protocol": _reference(protocol_path),
            "base_protocol": _reference(base_protocol_path),
            "base_result": _reference(base_result_path),
            "formal_plan": _reference(formal_plan_path),
            "accepted_bank": _reference(accepted_bank_path),
            "raw_results_root": str(results_root.resolve()),
        },
        "inventory": {
            "manifest_count": inspected_manifests,
            "strict_clean_off_nonheldout_run_count": strict_runs,
            "geometry_scenario_count": len(scenario_rows),
            "deduplicated_route_count": len(per_route),
            "skip_reason_counts": dict(sorted(skip_reasons.items())),
        },
        "summary": summary,
        "actor_support_route_indices_by_view": actor_routes,
        "all_geometry_runs": scenario_rows,
        "deduplicated_routes": per_route,
        "inspection_boundary": {
            "run_manifests_and_raw_frame_meta_read": True,
            "camera_pixels_read": False,
            "evaluator_checkpoint_or_control_trace_read": False,
            "dev_or_locked_test_route_identity_scanned": False,
            "model_uq_qa_or_outcome_used": False,
        },
        "release_locks": {
            "new_carla_collection": False,
            "gpu_r_only_smoke": False,
            "language_bridge": False,
            "stage2p": False,
        },
        "claim_boundary": "Descriptive CPU inventory separating route and actor geometry. Positive support is weak privileged supervision, not proof of semantic R, U, language, planning or safety.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--formal-plan", type=Path, required=True)
    parser.add_argument("--accepted-bank", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite actor-support inventory")
    value = scan(
        protocol_path=args.protocol,
        base_protocol_path=args.base_protocol,
        base_result_path=args.base_result,
        formal_plan_path=args.formal_plan,
        accepted_bank_path=args.accepted_bank,
        results_root=args.results_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": value["status"],
                "actor_support_route_indices_by_view": value[
                    "actor_support_route_indices_by_view"
                ],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

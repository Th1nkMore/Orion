#!/usr/bin/env python3
"""Scan strict clean-off raw assets for missing back-right R support."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uq_estimator.task_relevance_geometry import (
    CAMERA_ORDER,
    TaskRelevanceGeometryError,
    build_task_relevance_map,
)


SCHEMA = "orion.stage2l_v12_existing_raw_coverage_inventory.v1"
CAMERA_DIRECTORY_BY_VIEW = {
    "CAM_FRONT": "rgb_front_model_input",
    "CAM_FRONT_LEFT": "rgb_front_left",
    "CAM_FRONT_RIGHT": "rgb_front_right",
    "CAM_BACK": "rgb_back",
    "CAM_BACK_LEFT": "rgb_back_left",
    "CAM_BACK_RIGHT": "rgb_back_right",
}


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference(path: Path) -> Dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _empty(value: Any) -> bool:
    return value in (None, "", "none", "None")


def strict_clean_off_reason(manifest: Mapping[str, Any]) -> Optional[str]:
    checks = (
        (manifest.get("pilot_condition") == "clean_off", "condition"),
        (manifest.get("pilot_variant") == "hazard", "variant"),
        (manifest.get("orion_closedloop_conditioning") in (None, "none"), "conditioning"),
        (manifest.get("orion_closedloop_uq_mode") in (None, "none"), "uq_mode"),
        (str(manifest.get("orion_enable_legacy_density_uq", "0")) == "0", "legacy_density"),
        (_empty(manifest.get("orion_observation_uq_checkpoint")), "observation_uq"),
        (_empty(manifest.get("orion_stage1_spatial_uq_checkpoint")), "stage1_uq"),
        (manifest.get("orion_stage2_spatial_uq_source") in (None, "disabled"), "stage2_source"),
        (_empty(manifest.get("orion_stage2_task_checkpoint")), "stage2_task"),
        (manifest.get("orion_closedloop_risk_mode") in (None, "off"), "risk_mode"),
        (manifest.get("orion_planning_response_mode") in (None, "off"), "planning"),
        (_empty(manifest.get("orion_closedloop_corruption")), "corruption"),
        (manifest.get("orion_native_glare_profile") in (None, "none"), "native_glare"),
        (manifest.get("orion_native_motion_blur_profile") in (None, "none"), "native_motion_blur"),
        (str(manifest.get("orion_closedloop_safety_telemetry", "0")) == "1", "safety_telemetry"),
    )
    for passed, name in checks:
        if not passed:
            return name
    try:
        int(manifest["pilot_route_index"])
    except (KeyError, TypeError, ValueError):
        return "route_index"
    return None


def _job_id(run_dir: Path, manifest: Mapping[str, Any]) -> int:
    try:
        return int(manifest.get("slurm_job_id"))
    except (TypeError, ValueError):
        match = re.search(r"-(\d+)$", run_dir.name)
        return int(match.group(1)) if match else -1


def _aligned_frames(scenario_root: Path) -> tuple[Path, list[int]]:
    meta_root = scenario_root / "meta"
    streams = [{int(path.stem) for path in meta_root.glob("*.json")}]
    for view in CAMERA_ORDER:
        streams.append(
            {
                int(path.stem)
                for path in (scenario_root / CAMERA_DIRECTORY_BY_VIEW[view]).glob("*.png")
            }
        )
    return meta_root, sorted(set.intersection(*streams))


def _scan_scenario(scenario_root: Path) -> Dict[str, Any]:
    meta_root, frames = _aligned_frames(scenario_root)
    per_view = {view: [] for view in CAMERA_ORDER}
    modes = Counter()
    errors = Counter()
    for frame in frames:
        meta = _read_json(meta_root / ("%04d.json" % frame))
        try:
            geometry = build_task_relevance_map(
                meta["plan"], meta["closedloop_safety"], patch_hw=(40, 40)
            )
        except (TaskRelevanceGeometryError, KeyError) as error:
            errors["%s: %s" % (type(error).__name__, error)] += 1
            continue
        modes[geometry.provenance["support_mode"]] += 1
        for index, view in enumerate(CAMERA_ORDER):
            if bool((geometry.relevance[index] > 0.0).any()):
                per_view[view].append(frame)
    return {
        "scenario_root": str(scenario_root.resolve()),
        "aligned_frame_count": len(frames),
        "geometry_valid_frame_count": int(sum(modes.values())),
        "per_view_positive_frame_count": {
            view: len(per_view[view]) for view in CAMERA_ORDER
        },
        "per_view_positive_frames": per_view,
        "support_mode_counts": dict(sorted(modes.items())),
        "geometry_error_counts": dict(sorted(errors.items())),
    }


def choose_per_route(rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    chosen: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        route = int(row["route_index"])
        key = (
            row["per_view_positive_frame_count"]["CAM_BACK_RIGHT"],
            row["aligned_frame_count"],
            -row["job_id"],
        )
        current = chosen.get(route)
        if current is None:
            chosen[route] = row
            continue
        current_key = (
            current["per_view_positive_frame_count"]["CAM_BACK_RIGHT"],
            current["aligned_frame_count"],
            -current["job_id"],
        )
        if key > current_key:
            chosen[route] = row
    return list(chosen.values())


def scan(
    *,
    protocol_path: Path,
    formal_plan_path: Path,
    accepted_bank_path: Path,
    results_root: Path,
) -> Dict[str, Any]:
    protocol = _read_json(protocol_path)
    formal_plan = _read_json(formal_plan_path)
    accepted_bank = _read_json(accepted_bank_path)
    if protocol.get("schema") != "orion.stage2l_v12_existing_raw_coverage_inventory_protocol.v1":
        raise ValueError("inventory protocol schema differs")
    refs = protocol["authoritative_inputs"]
    for name, path in (
        ("formal_route_plan", formal_plan_path),
        ("accepted_event_bank", accepted_bank_path),
    ):
        if refs[name]["sha256"] != _sha256(path):
            raise ValueError("protocol input hash differs: %s" % name)
    if str(results_root.resolve()) != refs["raw_results_root"]:
        raise ValueError("raw results root differs from protocol")

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
            row = _scan_scenario(scenario_root)
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

    minimum_br = int(protocol["geometry"]["minimum_cam_back_right_positive_frames"])
    minimum_valid = int(protocol["geometry"]["minimum_geometry_valid_frames"])
    per_route = choose_per_route(scenario_rows)
    candidates = [
        row
        for row in per_route
        if row["per_view_positive_frame_count"]["CAM_BACK_RIGHT"] >= minimum_br
        and row["geometry_valid_frame_count"] >= minimum_valid
    ]
    candidates.sort(
        key=lambda row: (
            0 if row["new_independent_event_if_selected"] else 1,
            -row["per_view_positive_frame_count"]["CAM_BACK_RIGHT"],
            -row["aligned_frame_count"],
            row["route_index"],
        )
    )
    return {
        "schema": SCHEMA,
        "status": (
            "existing_raw_back_right_candidates_found"
            if candidates
            else "no_existing_raw_back_right_candidate"
        ),
        "diagnostic_completed": True,
        "formal_bank_modified": False,
        "gpu_used": False,
        "orion_forward_run": False,
        "training_started": False,
        "inputs": {
            "protocol": _reference(protocol_path),
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
        "minimum_back_right_positive_frames": minimum_br,
        "all_geometry_runs": scenario_rows,
        "deduplicated_routes": sorted(per_route, key=lambda row: row["route_index"]),
        "ranked_candidates": candidates,
        "inspection_boundary": {
            "run_manifests_and_raw_frame_meta_read": True,
            "evaluator_checkpoint_or_control_trace_read": False,
            "dev_or_locked_test_route_identity_scanned": False,
            "model_uq_qa_or_outcome_used": False,
        },
        "next_action": (
            "freeze the top train-only geometry candidates for runtime and human visual review"
            if candidates
            else "freeze a new explicit off-axis rear-right custom train-only scenario before CARLA"
        ),
        "launch_locks": {
            "new_carla_collection": False,
            "gpu_r_only_smoke": False,
            "language_bridge": False,
            "stage2p": False,
        },
        "claim_boundary": "CPU inventory of strict clean-off raw geometry only. Candidate ranking reads no model or route outcomes; hits remain unaccepted train-only candidates pending runtime, visual, Stage1 and QA review.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--formal-plan", type=Path, required=True)
    parser.add_argument("--accepted-bank", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite raw coverage inventory")
    value = scan(
        protocol_path=args.protocol,
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
                "candidate_routes": [
                    row["route_index"] for row in value["ranked_candidates"]
                ],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify the Route197 map- and dynamics-aware privileged yield oracle."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_native_collision_discovery import (
    COLLISION_KEYS,
    SERIOUS_INFRACTION_KEYS,
    _count_entries,
    _load_terminal,
)
from scripts.summarize_closedloop_safety import (
    find_control_trace,
    load_records,
    summarize_records,
)
from uq_estimator.dynamic_yield_expert import (
    BrakingAwareYieldStateMachine,
    DynamicYieldExpertConfig,
    build_dynamics_aware_yield_trajectory,
    compute_junction_yield_geometry,
    suppress_unbounded_conflict,
)
from uq_estimator.privileged_yield_labels import (
    TrajectoryConflictConfig,
    evaluate_trajectory_conflicts,
    trajectory_residual,
)


SCHEMA_VERSION = "orion.route197_dynamics_aware_yield_oracle_result.v4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same(left: Any, right: Any, tolerance: float = 1e-5) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            same(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            same(left[key], right[key], tolerance) for key in left
        )
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return left == right


def configs_from_prereg(prereg: dict):
    frozen = prereg["frozen_planning_response"]
    conflict = TrajectoryConflictConfig(
        interpolation_step_seconds=frozen["interpolation_step_seconds"],
        safety_margin_m=frozen["safety_margin_m"],
        clearance_seconds=frozen["clearance_seconds"],
        release_seconds=frozen["release_seconds"],
    )
    expert = DynamicYieldExpertConfig(
        certified_deceleration_mps2=frozen["certified_deceleration_mps2"],
        reaction_seconds=frozen["reaction_seconds"],
        junction_front_clearance_m=frozen["junction_front_clearance_m"],
        clearance_seconds=frozen["clearance_seconds"],
        release_seconds=frozen["release_seconds"],
        prepare_creep_speed_mps=frozen["prepare_creep_speed_mps"],
        release_creep_speed_mps=frozen["release_creep_speed_mps"],
        release_creep_distance_m=frozen["release_creep_distance_m"],
    )
    return conflict, expert


def evaluate(
    run_dir: Path,
    preregistration: Path,
    clean_report_path: Path,
    geometry_path: Path,
) -> dict:
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    clean_report = json.loads(clean_report_path.read_text(encoding="utf-8"))
    geometry_payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    geometry_by_step = {
        int(record["step"]): record for record in geometry_payload["records"]
    }
    trace_path = find_control_trace(run_dir)
    records = load_records(trace_path)
    safety_summary = summarize_records(records)
    eval_path, eval_payload, terminal = _load_terminal(run_dir)
    scores = terminal.get("scores") or {}
    infractions = terminal.get("infractions") or {}
    if not isinstance(infractions, dict):
        infractions = {}
    collision_count = _count_entries(infractions, COLLISION_KEYS)
    serious_infraction_count = _count_entries(infractions, SERIOUS_INFRACTION_KEYS)
    min_speed_infraction_count = _count_entries(
        infractions, ("min_speed_infractions",)
    )
    hashes = prereg["frozen_hashes"]
    manifest_hashes = manifest.get("source_sha256") or {}
    frozen = prereg["frozen_planning_response"]
    conflict_config, expert_config = configs_from_prereg(prereg)

    manifest_checks = {
        "condition_dynamics_aware_oracle": (
            manifest.get("pilot_condition")
            == "native_dynamics_aware_yield_oracle"
        ),
        "route197": manifest.get("pilot_route_index") == "197",
        "corruption_none": not bool(manifest.get("orion_closedloop_corruption")),
        "uq_mode_none": manifest.get("orion_closedloop_uq_mode") == "none",
        "legacy_density_disabled": (
            manifest.get("orion_enable_legacy_density_uq") == "0"
        ),
        "observation_adapter_absent": not bool(
            manifest.get("orion_observation_uq_checkpoint")
        ),
        "scalar_risk_governor_off": (
            manifest.get("orion_closedloop_risk_mode") == "off"
        ),
        "planning_response_mode_frozen": (
            manifest.get("orion_planning_response_mode")
            == "privileged_dynamics_aware_yield"
        ),
        "interpolation_step_frozen": same(
            manifest.get("orion_planning_interpolation_step_seconds"),
            frozen["interpolation_step_seconds"],
        ),
        "safety_margin_frozen": same(
            manifest.get("orion_planning_safety_margin_m"),
            frozen["safety_margin_m"],
        ),
        "certified_deceleration_frozen": same(
            manifest.get("orion_planning_certified_deceleration_mps2"),
            frozen["certified_deceleration_mps2"],
        ),
        "reaction_seconds_frozen": same(
            manifest.get("orion_planning_reaction_seconds"),
            frozen["reaction_seconds"],
        ),
        "junction_clearance_frozen": same(
            manifest.get("orion_planning_junction_front_clearance_m"),
            frozen["junction_front_clearance_m"],
        ),
        "map_resolution_frozen": same(
            manifest.get("orion_planning_map_resolution_m"),
            frozen["map_resolution_m"],
        ),
        "clearance_seconds_frozen": same(
            manifest.get("orion_planning_clearance_seconds"),
            frozen["clearance_seconds"],
        ),
        "release_seconds_frozen": same(
            manifest.get("orion_planning_release_seconds"),
            frozen["release_seconds"],
        ),
        "prepare_creep_frozen": same(
            manifest.get("orion_planning_prepare_creep_speed_mps"),
            frozen["prepare_creep_speed_mps"],
        ),
        "release_creep_speed_frozen": same(
            manifest.get("orion_planning_release_creep_speed_mps"),
            frozen["release_creep_speed_mps"],
        ),
        "release_creep_distance_frozen": same(
            manifest.get("orion_planning_release_creep_distance_m"),
            frozen["release_creep_distance_m"],
        ),
        "route_hash_frozen": (
            manifest.get("route_sha256") == hashes["route_197_hazard.xml"]
        ),
        "clean_report_hash_frozen": (
            sha256(clean_report_path) == prereg["clean_reference"]["report_sha256"]
        ),
        "offline_audit_hash_frozen": (
            sha256(Path(prereg["offline_gate"]["report_path"]))
            == prereg["offline_gate"]["report_sha256"]
        ),
        "offline_audit_authorized_single_run": (
            json.loads(Path(prereg["offline_gate"]["report_path"]).read_text())
            .get("offline_gate_pass") is True
        ),
    }
    for relative_path, expected_hash in hashes.items():
        if relative_path == "route_197_hazard.xml":
            continue
        manifest_checks[f"source_hash:{relative_path}"] = (
            manifest_hashes.get(relative_path) == expected_hash
        )

    labeler = BrakingAwareYieldStateMachine(expert_config)
    state_counts: Counter[str] = Counter()
    all_raw_conflicts_match = True
    all_effective_conflicts_match = True
    all_map_scope_matches = True
    all_geometry_matches = True
    all_labels_match = True
    all_targets_match = True
    all_residuals_match = True
    all_risk_passthrough = True
    all_density_absent = True
    all_adapter_absent = True
    steps = []
    transition_count = 0
    previous_state = None
    first_hold = None
    release_to_hold_count = 0
    scoped_conflict_count = 0
    unbounded_conflict_passthrough_count = 0

    for row in records:
        step = int(row["step"])
        steps.append(step)
        all_density_absent &= row.get("density_uq_score") is None
        all_adapter_absent &= row.get("observation_uq") is None
        risk = row.get("risk") or {}
        all_risk_passthrough &= (
            risk.get("mode") == "off"
            and same(risk.get("throttle"), risk.get("base_throttle"))
            and same(risk.get("brake"), risk.get("base_brake"))
            and risk.get("applied_score") is None
        )
        response = row.get("planning_response") or {}
        if response.get("mode") != "privileged_dynamics_aware_yield":
            all_raw_conflicts_match = False
            continue
        base_plan = response.get("base_plan_cumulative_m")
        raw_conflict = evaluate_trajectory_conflicts(
            base_plan,
            row.get("closedloop_safety"),
            config=conflict_config,
        )
        geometry_record = geometry_by_step.get(step)
        if geometry_record is None:
            all_map_scope_matches = False
            continue
        state_before = labeler.state
        queried = bool(raw_conflict.has_conflict or state_before != "go")
        ego_is_junction = bool(
            (geometry_record.get("ego_map_waypoint") or {}).get("is_junction")
        )
        entry_distance = geometry_record.get("junction_entry_path_distance_m")
        junction_scoped = bool(
            raw_conflict.has_conflict
            and (
                state_before != "go"
                or entry_distance is not None
                or ego_is_junction
            )
        )
        effective_conflict = (
            raw_conflict
            if junction_scoped
            else suppress_unbounded_conflict(raw_conflict)
        )
        if junction_scoped:
            scoped_conflict_count += 1
        elif raw_conflict.has_conflict:
            unbounded_conflict_passthrough_count += 1
        if entry_distance is not None:
            geometry_entry_distance = entry_distance
        elif state_before != "go":
            geometry_entry_distance = 0.0
        else:
            geometry_entry_distance = 1_000_000.0
        ego_extent = float(row["closedloop_safety"]["ego"]["extent_xy_m"][0])
        expected_geometry = compute_junction_yield_geometry(
            geometry_entry_distance,
            ego_extent,
            row["speed"],
            config=expert_config,
        )
        expected_label = labeler.update(
            effective_conflict,
            expected_geometry,
            float(row["sim_time_seconds"]),
        )
        expected_target = build_dynamics_aware_yield_trajectory(
            base_plan,
            expected_label,
            row["speed"],
            config=expert_config,
        )
        expected_residual = trajectory_residual(base_plan, expected_target)
        expected_entry = None
        if queried:
            expected_entry = {
                "distance_m": entry_distance,
                "world_xy": geometry_record.get("junction_entry_world_xy"),
                "ego_is_junction": ego_is_junction,
            }
        expected_scope = {
            "queried": queried,
            "junction_scoped_conflict": junction_scoped,
            "entry": expected_entry,
        }
        all_raw_conflicts_match &= same(
            response.get("raw_conflict"), raw_conflict.to_dict()
        )
        all_effective_conflicts_match &= same(
            response.get("effective_conflict"), effective_conflict.to_dict()
        )
        all_map_scope_matches &= same(
            response.get("junction_scope"), expected_scope, tolerance=2e-3
        )
        all_geometry_matches &= same(
            response.get("yield_geometry"), expected_geometry.to_dict(),
            tolerance=2e-3,
        )
        all_labels_match &= same(
            response.get("yield_label"), expected_label.to_dict(),
            tolerance=2e-3,
        )
        all_targets_match &= same(
            response.get("target_plan_cumulative_m"), expected_target,
            tolerance=2e-3,
        )
        all_residuals_match &= same(
            response.get("trajectory_residual_m"), expected_residual,
            tolerance=2e-3,
        )
        state = expected_label.state
        state_counts[state] += 1
        timestamp = float(row["sim_time_seconds"])
        if state == "hold" and first_hold is None:
            first_hold = timestamp
        if previous_state is not None and state != previous_state:
            transition_count += 1
        if previous_state == "release" and state == "hold":
            release_to_hold_count += 1
        previous_state = state

    trace_checks = {
        "trace_nonempty": bool(records),
        "trace_steps_contiguous": bool(steps) and steps == list(
            range(steps[0], steps[0] + len(steps))
        ),
        "geometry_trace_hash_matches": (
            geometry_payload.get("trace_sha256") == sha256(trace_path)
        ),
        "geometry_record_count_matches": len(geometry_by_step) == len(records),
        "density_score_absent_every_frame": all_density_absent,
        "observation_adapter_absent_every_frame": all_adapter_absent,
        "scalar_risk_governor_exact_passthrough": all_risk_passthrough,
        "raw_conflicts_exactly_recomputed": all_raw_conflicts_match,
        "effective_conflicts_exactly_recomputed": all_effective_conflicts_match,
        "map_scope_matches_independent_xodr_export": all_map_scope_matches,
        "braking_geometry_exactly_recomputed": all_geometry_matches,
        "yield_state_exactly_recomputed": all_labels_match,
        "planning_targets_exactly_recomputed": all_targets_match,
        "trajectory_residuals_exactly_recomputed": all_residuals_match,
        "map_scoped_conflict_observed": scoped_conflict_count > 0,
        "hold_observed": state_counts["hold"] > 0,
        "trajectory_intervention_observed": any(
            any(
                abs(float(value)) > 1e-6
                for point in (
                    (row.get("planning_response") or {}).get(
                        "trajectory_residual_m", []
                    )
                )
                for value in point
            )
            for row in records
        ),
    }
    endpoint_checks = {
        "runtime_valid_terminal_endpoint": (
            bool(eval_payload.get("eligible"))
            and bool(str(terminal.get("status", "")).strip())
        ),
        "official_collision_count_zero": collision_count == 0,
        "route_completion_100": same(scores.get("score_route"), 100.0),
        "new_serious_infraction_count_zero": serious_infraction_count == 0,
    }
    primary_success = (
        all(manifest_checks.values())
        and all(trace_checks.values())
        and all(endpoint_checks.values())
    )
    return {
        "schema": SCHEMA_VERSION,
        "run_dir": str(run_dir.resolve()),
        "preregistration": str(preregistration.resolve()),
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": sha256(trace_path),
        "geometry_path": str(geometry_path.resolve()),
        "geometry_sha256": sha256(geometry_path),
        "eval_path": str(eval_path.resolve()),
        "manifest_checks": manifest_checks,
        "trace_checks": trace_checks,
        "endpoint_checks": endpoint_checks,
        "planning_response": {
            "state_counts": dict(state_counts),
            "transition_count": transition_count,
            "release_to_hold_count": release_to_hold_count,
            "first_hold_time_seconds": first_hold,
            "map_scoped_conflict_frame_count": scoped_conflict_count,
            "unbounded_conflict_passthrough_frame_count": (
                unbounded_conflict_passthrough_count
            ),
        },
        "endpoint": {
            "status": terminal.get("status"),
            "scores": scores,
            "official_collision_count": collision_count,
            "collision_entries": {
                key: infractions.get(key, []) for key in COLLISION_KEYS
            },
            "serious_infraction_count": serious_infraction_count,
            "min_speed_infraction_count": min_speed_infraction_count,
        },
        "continuous_safety": safety_summary,
        "clean_reference": {
            "official_collision_count": clean_report["endpoint"]["collision_count"],
            "scores": clean_report["endpoint"]["scores"],
            "continuous_safety": clean_report["continuous_safety"],
        },
        "primary_success": primary_success,
        "decision": (
            "planning_mechanism_upper_bound_supported"
            if primary_success
            else "do_not_train_or_control_with_learned_uq_yet"
        ),
        "claim_boundary": prereg["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--clean-report", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite dynamics-aware oracle report")
    report = evaluate(
        args.run_dir,
        args.preregistration,
        args.clean_report,
        args.geometry,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "primary_success": report["primary_success"],
        "decision": report["decision"],
    }, indent=2, sort_keys=True))
    return 0 if report["primary_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

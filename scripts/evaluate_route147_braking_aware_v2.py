#!/usr/bin/env python3
"""Evaluate the frozen Route147 clean reference against one v2 oracle run."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from scripts.evaluate_clean_liveness_screen import longest_low_speed_interval
from scripts.evaluate_route147_bounded_crossing_pair import (
    _risk_passthrough,
    _terminal_summary,
    same,
    sha256,
)
from scripts.summarize_closedloop_safety import (
    find_control_trace,
    load_records,
    summarize_records,
)
from uq_estimator.bounded_crossing_expert import (
    BoundedCrossingExpertConfig,
    build_braking_aware_crossing_trajectory,
)
from uq_estimator.privileged_yield_labels import (
    DynamicYieldLabeler,
    TrajectoryConflictConfig,
    evaluate_trajectory_conflicts,
    select_actor_categories,
    trajectory_residual,
)


SCHEMA_VERSION = "orion.route147_braking_aware_v2_result.v1"


def _common_trace_checks(records: list[dict], *, oracle: bool) -> dict:
    steps = [int(row["step"]) for row in records]
    checks = {
        "nonempty": bool(records),
        "steps_contiguous": bool(steps) and steps == list(
            range(steps[0], steps[0] + len(steps))
        ),
        "density_absent_every_frame": all(
            row.get("density_uq_score") is None for row in records
        ),
        "new_adapter_absent_every_frame": all(
            row.get("observation_uq") is None for row in records
        ),
        "scalar_risk_passthrough_every_frame": all(
            _risk_passthrough(row) for row in records
        ),
        "corruption_absent_every_frame": all(
            not row.get("corruption_active") for row in records
        ),
    }
    if not oracle:
        checks["planning_response_absent_every_frame"] = all(
            row.get("planning_response") is None for row in records
        )
    return checks


def _oracle_trace_contract(
    records: list[dict],
    *,
    conflict_config: TrajectoryConflictConfig,
    expert_config: BoundedCrossingExpertConfig,
    immediate_brake_ratio_threshold: float,
) -> dict:
    checks = _common_trace_checks(records, oracle=True)
    labeler = DynamicYieldLabeler(conflict_config)
    state_counts: Counter[str] = Counter()
    response_modes_match = True
    filter_match = True
    conflicts_match = True
    labels_match = True
    targets_match = True
    profiles_match = True
    residuals_match = True
    go_identity = True
    event_targets_bounded = True
    intervention_frames = 0
    first_hold_time = None
    first_release_time = None
    first_hold_profile = None

    for row in records:
        response = row.get("planning_response") or {}
        response_modes_match &= (
            response.get("mode") == "privileged_braking_aware_crossing"
        )
        filter_match &= response.get("actor_categories") == ["walker"]
        base = response.get("base_plan_cumulative_m")
        if not base:
            conflicts_match = labels_match = targets_match = False
            profiles_match = residuals_match = False
            continue
        planning_safety = select_actor_categories(
            row.get("closedloop_safety") or {}, ("walker",)
        )
        conflict = evaluate_trajectory_conflicts(
            base, planning_safety, config=conflict_config
        )
        label = labeler.update(conflict, float(row["sim_time_seconds"]))
        target, profile = build_braking_aware_crossing_trajectory(
            base, label, float(row["speed"]), config=expert_config
        )
        residual = trajectory_residual(base, target)
        conflicts_match &= same(response.get("conflict"), conflict.to_dict())
        labels_match &= same(response.get("yield_label"), label.to_dict())
        targets_match &= same(response.get("target_plan_cumulative_m"), target)
        profiles_match &= same(response.get("braking_profile"), profile.to_dict())
        residuals_match &= same(response.get("trajectory_residual_m"), residual)
        state_counts[label.state] += 1
        if label.state == "go":
            go_identity &= same(base, target)
        else:
            intervention_frames += 1
            if profile.safe_stop_path_distance_m is not None:
                event_targets_bounded &= all(
                    (float(point[0]) ** 2 + float(point[1]) ** 2) ** 0.5
                    <= profile.safe_stop_path_distance_m + 1e-6
                    for point in target
                )
        timestamp = float(row["sim_time_seconds"])
        if label.state == "hold" and first_hold_time is None:
            first_hold_time = timestamp
            first_hold_profile = profile.to_dict()
        if label.state == "release" and first_release_time is None:
            first_release_time = timestamp

    first_hold_brakes = bool(
        first_hold_profile
        and first_hold_profile.get("immediate_brake_ratio") is not None
        and float(first_hold_profile["immediate_brake_ratio"])
        >= immediate_brake_ratio_threshold
    )
    checks.update({
        "planning_response_mode_frozen_every_frame": response_modes_match,
        "walker_filter_frozen_every_frame": filter_match,
        "conflict_recomputed_every_frame": conflicts_match,
        "yield_label_recomputed_every_frame": labels_match,
        "target_trajectory_recomputed_every_frame": targets_match,
        "braking_profile_recomputed_every_frame": profiles_match,
        "trajectory_residual_recomputed_every_frame": residuals_match,
        "go_state_exactly_preserves_orion_plan": go_identity,
        "event_trajectory_respects_safe_stop_distance": event_targets_bounded,
        "bounded_crossing_intervention_observed": intervention_frames > 0,
        "hold_and_release_observed": (
            state_counts["hold"] > 0 and state_counts["release"] > 0
        ),
        "first_hold_commands_immediate_braking": first_hold_brakes,
    })
    return {
        "checks": checks,
        "state_counts": dict(state_counts),
        "intervention_frames": intervention_frames,
        "first_hold_time_seconds": first_hold_time,
        "first_release_time_seconds": first_release_time,
        "first_hold_braking_profile": first_hold_profile,
    }


def _oracle_manifest_checks(manifest: dict, prereg: dict) -> dict:
    checks = {
        "route147": manifest.get("pilot_route_index") == "147",
        "hazard_variant": manifest.get("pilot_variant") == "hazard",
        "condition_frozen": (
            manifest.get("pilot_condition")
            == "native_braking_aware_crossing_oracle"
        ),
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
            == "privileged_braking_aware_crossing"
        ),
        "actor_categories_frozen": (
            manifest.get("orion_planning_actor_categories") == "walker"
        ),
        "effective_conditioning_frozen": (
            manifest.get("orion_effective_conditioning")
            == "privileged_planning_response:privileged_braking_aware_crossing"
        ),
        "route_hash_frozen": (
            manifest.get("route_sha256")
            == prereg["frozen_hashes"]["route_147_hazard.xml"]
        ),
    }
    source_hashes = manifest.get("source_sha256") or {}
    for relative, expected in prereg["frozen_hashes"].items():
        if relative == "route_147_hazard.xml":
            continue
        checks[f"source_hash:{relative}"] = source_hashes.get(relative) == expected
    frozen = prereg["frozen_planning_response"]
    manifest_parameters = {
        "orion_planning_interpolation_step_seconds": "interpolation_step_seconds",
        "orion_planning_safety_margin_m": "safety_margin_m",
        "orion_planning_imminent_horizon_seconds": "imminent_horizon_seconds",
        "orion_planning_certified_deceleration_mps2": "certified_deceleration_mps2",
        "orion_planning_clearance_seconds": "clearance_seconds",
        "orion_planning_release_seconds": "release_seconds",
        "orion_planning_prepare_creep_speed_mps": "prepare_creep_speed_mps",
        "orion_planning_release_creep_speed_mps": "release_creep_speed_mps",
        "orion_planning_stop_buffer_m": "stop_buffer_m",
        "orion_planning_release_creep_distance_m": "release_creep_distance_m",
    }
    for manifest_key, frozen_key in manifest_parameters.items():
        checks[f"parameter:{frozen_key}"] = same(
            manifest.get(manifest_key), frozen[frozen_key]
        )
    return checks


def _frozen_clean_checks(clean_run_dir: Path, prereg: dict) -> dict:
    observed = {
        "control_trace.jsonl": find_control_trace(clean_run_dir),
        "manifest.json": clean_run_dir / "manifest.json",
        "eval_orion_traj_0.json": clean_run_dir / "eval_orion_traj_0.json",
    }
    expected = prereg["frozen_clean_reference"]["artifacts"]
    return {
        f"sha256:{name}": path.is_file() and sha256(path) == expected[name]
        for name, path in observed.items()
    }


def evaluate(clean_run_dir: Path, oracle_run_dir: Path, prereg_path: Path) -> dict:
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    clean_records = load_records(find_control_trace(clean_run_dir))
    oracle_records = load_records(find_control_trace(oracle_run_dir))
    clean_summary = summarize_records(clean_records)
    oracle_summary = summarize_records(oracle_records)
    clean_terminal = _terminal_summary(clean_run_dir)
    oracle_terminal = _terminal_summary(oracle_run_dir)
    frozen = prereg["frozen_planning_response"]
    conflict_config = TrajectoryConflictConfig(
        interpolation_step_seconds=frozen["interpolation_step_seconds"],
        safety_margin_m=frozen["safety_margin_m"],
        imminent_horizon_seconds=frozen["imminent_horizon_seconds"],
        clearance_seconds=frozen["clearance_seconds"],
        release_seconds=frozen["release_seconds"],
        stop_buffer_m=frozen["stop_buffer_m"],
        release_creep_distance_m=frozen["release_creep_distance_m"],
    )
    expert_config = BoundedCrossingExpertConfig(
        certified_deceleration_mps2=frozen["certified_deceleration_mps2"],
        prepare_creep_speed_mps=frozen["prepare_creep_speed_mps"],
        release_creep_speed_mps=frozen["release_creep_speed_mps"],
        release_creep_distance_m=frozen["release_creep_distance_m"],
    )
    clean_artifact_checks = _frozen_clean_checks(clean_run_dir, prereg)
    oracle_manifest = json.loads(
        (oracle_run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    oracle_manifest_checks = _oracle_manifest_checks(oracle_manifest, prereg)
    clean_trace = {"checks": _common_trace_checks(clean_records, oracle=False)}
    oracle_trace = _oracle_trace_contract(
        oracle_records,
        conflict_config=conflict_config,
        expert_config=expert_config,
        immediate_brake_ratio_threshold=(
            prereg["mechanism_gate"]["minimum_first_hold_brake_ratio"]
        ),
    )

    clean_walker = clean_summary["safety"]["by_category"]["walker"]
    oracle_walker = oracle_summary["safety"]["by_category"]["walker"]
    clean_vehicle = clean_summary["safety"]["by_category"]["vehicle"]
    oracle_vehicle = oracle_summary["safety"]["by_category"]["vehicle"]
    clean_ttc = clean_walker["min_obb_ttc_seconds"]
    oracle_ttc = oracle_walker["min_obb_ttc_seconds"]
    clean_exposure = clean_walker["low_ttc_exposure_seconds"]["1.0"]
    oracle_exposure = oracle_walker["low_ttc_exposure_seconds"]["1.0"]
    exposure_reduction = float(clean_exposure) - float(oracle_exposure)
    ttc_gain = (
        None if oracle_ttc is None or clean_ttc is None
        else float(oracle_ttc) - float(clean_ttc)
    )
    clean_stop = longest_low_speed_interval(clean_records)
    oracle_stop = longest_low_speed_interval(oracle_records)
    duration_delta = (
        float(oracle_summary["duration_seconds"])
        - float(clean_summary["duration_seconds"])
    )
    vehicle_exposure_delta = (
        float(oracle_vehicle["low_ttc_exposure_seconds"]["1.0"])
        - float(clean_vehicle["low_ttc_exposure_seconds"]["1.0"])
    )
    thresholds = prereg["success_thresholds"]
    validity = prereg["validity_thresholds"]
    endpoint_checks = {
        "both_completed": (
            clean_terminal["status"] == "Completed"
            and oracle_terminal["status"] == "Completed"
        ),
        "both_full_route_completion": (
            same(clean_terminal["score_route"], 100.0)
            and same(oracle_terminal["score_route"], 100.0)
        ),
        "both_zero_collisions": (
            clean_terminal["collision_count"] == 0
            and oracle_terminal["collision_count"] == 0
        ),
        "both_zero_serious_infractions": (
            clean_terminal["serious_infraction_count"] == 0
            and oracle_terminal["serious_infraction_count"] == 0
        ),
    }
    outcome_checks = {
        "clean_reproduces_walker_near_miss": (
            clean_ttc is not None
            and float(clean_ttc) <= validity["maximum_clean_walker_ttc_seconds"]
            and float(clean_exposure) >= validity[
                "minimum_clean_walker_ttc_lte_1_exposure_seconds"
            ]
        ),
        "walker_ttc_improved_or_horizon_censored": (
            oracle_ttc is None
            or (
                ttc_gain is not None
                and ttc_gain >= thresholds["minimum_walker_ttc_gain_seconds"]
            )
        ),
        "walker_ttc_lte_1_exposure_reduced": (
            exposure_reduction >= thresholds[
                "minimum_walker_ttc_lte_1_exposure_reduction_seconds"
            ]
        ),
        "no_material_vehicle_ttc_regression": (
            vehicle_exposure_delta <= thresholds[
                "maximum_vehicle_ttc_lte_1_exposure_increase_seconds"
            ]
        ),
        "bounded_route_duration_cost": (
            duration_delta <= thresholds["maximum_duration_increase_seconds"]
        ),
        "bounded_continuous_stop": (
            float(oracle_stop["duration_seconds"])
            <= thresholds["maximum_continuous_stop_seconds"]
        ),
    }
    all_contract_checks = (
        all(clean_artifact_checks.values())
        and all(oracle_manifest_checks.values())
        and all(clean_trace["checks"].values())
        and all(oracle_trace["checks"].values())
        and all(endpoint_checks.values())
    )
    primary_success = all_contract_checks and all(outcome_checks.values())
    return {
        "schema": SCHEMA_VERSION,
        "primary_success": primary_success,
        "stage2_eligible": primary_success,
        "decision": (
            "unlock_new_spatial_adapter_stage2_task_training"
            if primary_success
            else "keep_stage2_locked_and_redesign_mechanism_or_route"
        ),
        "claim_boundary": (
            "This run tests only whether an accurate privileged crossing signal "
            "can improve ORION closed-loop safety with the frozen v1 success "
            "thresholds. Density and the learned spatial adapter are absent."
        ),
        "clean_artifact_checks": clean_artifact_checks,
        "oracle_manifest_checks": oracle_manifest_checks,
        "trace_contract": {"clean": clean_trace, "oracle": oracle_trace},
        "endpoint_checks": endpoint_checks,
        "outcome_checks": outcome_checks,
        "outcomes": {
            "clean_walker": clean_walker,
            "oracle_walker": oracle_walker,
            "walker_minimum_ttc_gain_seconds": ttc_gain,
            "walker_ttc_lte_1_exposure_reduction_seconds": exposure_reduction,
            "vehicle_ttc_lte_1_exposure_delta_seconds": vehicle_exposure_delta,
            "route_duration_delta_seconds": duration_delta,
            "clean_longest_stop": clean_stop,
            "oracle_longest_stop": oracle_stop,
        },
        "official": {"clean": clean_terminal, "oracle": oracle_terminal},
        "summaries": {"clean": clean_summary, "oracle": oracle_summary},
        "artifacts": {
            "preregistration_path": str(prereg_path),
            "preregistration_sha256": sha256(prereg_path),
            "clean_trace_path": str(find_control_trace(clean_run_dir)),
            "oracle_trace_path": str(find_control_trace(oracle_run_dir)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-run-dir", required=True, type=Path)
    parser.add_argument("--oracle-run-dir", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        args.clean_run_dir, args.oracle_run_dir, args.preregistration
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate the preregistered Route147 clean/planning-oracle pair."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from scripts.evaluate_clean_liveness_screen import longest_low_speed_interval
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
from uq_estimator.privileged_yield_labels import (
    DynamicYieldLabeler,
    TrajectoryConflictConfig,
    build_safe_yield_trajectory,
    evaluate_trajectory_conflicts,
    select_actor_categories,
    trajectory_residual,
)


SCHEMA_VERSION = "orion.route147_bounded_crossing_pair_result.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
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


def _terminal_summary(run_dir: Path) -> dict:
    eval_path, _, terminal = _load_terminal(run_dir)
    scores = terminal.get("scores") or {}
    infractions = terminal.get("infractions") or {}
    if not isinstance(infractions, dict):
        infractions = {}
    return {
        "eval_path": str(eval_path),
        "status": terminal.get("status"),
        "score_route": scores.get("score_route"),
        "score_penalty": scores.get("score_penalty"),
        "collision_count": _count_entries(infractions, COLLISION_KEYS),
        "serious_infraction_count": _count_entries(
            infractions, SERIOUS_INFRACTION_KEYS
        ),
    }


def _risk_passthrough(row: dict) -> bool:
    risk = row.get("risk") or {}
    return (
        risk.get("mode") == "off"
        and risk.get("applied_score") is None
        and same(risk.get("throttle"), risk.get("base_throttle"))
        and same(risk.get("brake"), risk.get("base_brake"))
    )


def _trace_contract(records: list[dict], *, oracle: bool, config) -> dict:
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
        return {"checks": checks, "state_counts": {}}

    labeler = DynamicYieldLabeler(config)
    state_counts: Counter[str] = Counter()
    responses_match = True
    target_match = True
    residual_match = True
    filter_match = True
    go_is_exact_passthrough = True
    intervention_frames = 0
    first_hold_time = None
    first_release_time = None
    for row in records:
        response = row.get("planning_response") or {}
        if response.get("mode") != "privileged_bounded_crossing":
            responses_match = False
            continue
        filter_match &= response.get("actor_categories") == ["walker"]
        base = response.get("base_plan_cumulative_m")
        planning_safety = select_actor_categories(
            row.get("closedloop_safety") or {}, ("walker",)
        )
        conflict = evaluate_trajectory_conflicts(
            base, planning_safety, config=config
        )
        label = labeler.update(conflict, float(row["sim_time_seconds"]))
        target = build_safe_yield_trajectory(base, label)
        residual = trajectory_residual(base, target)
        responses_match &= same(response.get("conflict"), conflict.to_dict())
        responses_match &= same(response.get("yield_label"), label.to_dict())
        target_match &= same(response.get("target_plan_cumulative_m"), target)
        residual_match &= same(response.get("trajectory_residual_m"), residual)
        state_counts[label.state] += 1
        if label.state != "go":
            intervention_frames += 1
        else:
            go_is_exact_passthrough &= same(base, target)
        timestamp = float(row["sim_time_seconds"])
        if label.state == "hold" and first_hold_time is None:
            first_hold_time = timestamp
        if label.state == "release" and first_release_time is None:
            first_release_time = timestamp
    checks.update({
        "planning_response_recomputed_every_frame": responses_match,
        "walker_filter_frozen_every_frame": filter_match,
        "target_trajectory_recomputed_every_frame": target_match,
        "trajectory_residual_recomputed_every_frame": residual_match,
        "go_state_exactly_preserves_orion_plan": go_is_exact_passthrough,
        "bounded_crossing_intervention_observed": intervention_frames > 0,
        "hold_and_release_observed": (
            state_counts["hold"] > 0 and state_counts["release"] > 0
        ),
    })
    return {
        "checks": checks,
        "state_counts": dict(state_counts),
        "intervention_frames": intervention_frames,
        "first_hold_time_seconds": first_hold_time,
        "first_release_time_seconds": first_release_time,
    }


def _manifest_checks(
    manifest: dict,
    prereg: dict,
    *,
    condition: str,
    effective_conditioning: str,
) -> dict:
    checks = {
        "route147": manifest.get("pilot_route_index") == "147",
        "hazard_variant": manifest.get("pilot_variant") == "hazard",
        "condition_frozen": manifest.get("pilot_condition") == condition,
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
        "effective_conditioning_frozen": (
            manifest.get("orion_effective_conditioning")
            == effective_conditioning
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
    return checks


def evaluate(clean_run_dir: Path, oracle_run_dir: Path, prereg_path: Path) -> dict:
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    clean_manifest = json.loads(
        (clean_run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    oracle_manifest = json.loads(
        (oracle_run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    clean_records = load_records(find_control_trace(clean_run_dir))
    oracle_records = load_records(find_control_trace(oracle_run_dir))
    clean_summary = summarize_records(clean_records)
    oracle_summary = summarize_records(oracle_records)
    clean_terminal = _terminal_summary(clean_run_dir)
    oracle_terminal = _terminal_summary(oracle_run_dir)
    frozen = prereg["frozen_planning_response"]
    config = TrajectoryConflictConfig(
        interpolation_step_seconds=frozen["interpolation_step_seconds"],
        safety_margin_m=frozen["safety_margin_m"],
        imminent_horizon_seconds=frozen["imminent_horizon_seconds"],
        clearance_seconds=frozen["clearance_seconds"],
        release_seconds=frozen["release_seconds"],
        stop_buffer_m=frozen["stop_buffer_m"],
        release_creep_distance_m=frozen["release_creep_distance_m"],
    )
    clean_manifest_checks = _manifest_checks(
        clean_manifest,
        prereg,
        condition="clean_off",
        effective_conditioning="none",
    )
    oracle_manifest_checks = _manifest_checks(
        oracle_manifest,
        prereg,
        condition="native_bounded_crossing_oracle",
        effective_conditioning=(
            "privileged_planning_response:privileged_bounded_crossing"
        ),
    )
    oracle_manifest_checks.update({
        "planning_response_mode_frozen": (
            oracle_manifest.get("orion_planning_response_mode")
            == "privileged_bounded_crossing"
        ),
        "actor_categories_frozen": (
            oracle_manifest.get("orion_planning_actor_categories") == "walker"
        ),
    })
    clean_trace = _trace_contract(clean_records, oracle=False, config=config)
    oracle_trace = _trace_contract(oracle_records, oracle=True, config=config)

    clean_walker = clean_summary["safety"]["by_category"]["walker"]
    oracle_walker = oracle_summary["safety"]["by_category"]["walker"]
    clean_vehicle = clean_summary["safety"]["by_category"]["vehicle"]
    oracle_vehicle = oracle_summary["safety"]["by_category"]["vehicle"]
    clean_ttc = clean_walker["min_obb_ttc_seconds"]
    oracle_ttc = oracle_walker["min_obb_ttc_seconds"]
    clean_exposure = clean_walker["low_ttc_exposure_seconds"]["1.0"]
    oracle_exposure = oracle_walker["low_ttc_exposure_seconds"]["1.0"]
    exposure_reduction = float(clean_exposure) - float(oracle_exposure)
    ttc_improved_or_censored = (
        oracle_ttc is None
        or (
            clean_ttc is not None
            and float(oracle_ttc) - float(clean_ttc)
            >= prereg["success_thresholds"]["minimum_walker_ttc_gain_seconds"]
        )
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
            and float(clean_ttc)
            <= prereg["validity_thresholds"]["maximum_clean_walker_ttc_seconds"]
            and float(clean_exposure)
            >= prereg["validity_thresholds"][
                "minimum_clean_walker_ttc_lte_1_exposure_seconds"
            ]
        ),
        "walker_ttc_improved_or_horizon_censored": ttc_improved_or_censored,
        "walker_ttc_lte_1_exposure_reduced": (
            exposure_reduction
            >= prereg["success_thresholds"][
                "minimum_walker_ttc_lte_1_exposure_reduction_seconds"
            ]
        ),
        "no_material_vehicle_ttc_regression": (
            vehicle_exposure_delta
            <= prereg["success_thresholds"][
                "maximum_vehicle_ttc_lte_1_exposure_increase_seconds"
            ]
        ),
        "bounded_route_duration_cost": (
            duration_delta
            <= prereg["success_thresholds"]["maximum_duration_increase_seconds"]
        ),
        "bounded_continuous_stop": (
            float(oracle_stop["duration_seconds"])
            <= prereg["success_thresholds"]["maximum_continuous_stop_seconds"]
        ),
    }
    all_contract_checks = (
        all(clean_manifest_checks.values())
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
            "unlock_spatial_adapter_stage2_task_training"
            if primary_success
            else "keep_stage2_locked_and_inspect_route147_pair"
        ),
        "claim_boundary": (
            "A passing pair validates the bounded planning response only. It "
            "does not establish learned-adapter efficacy; Density and the new "
            "adapter are absent from both conditions."
        ),
        "manifest_checks": {
            "clean": clean_manifest_checks,
            "oracle": oracle_manifest_checks,
        },
        "trace_contract": {"clean": clean_trace, "oracle": oracle_trace},
        "endpoint_checks": endpoint_checks,
        "outcome_checks": outcome_checks,
        "outcomes": {
            "clean_walker": clean_walker,
            "oracle_walker": oracle_walker,
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

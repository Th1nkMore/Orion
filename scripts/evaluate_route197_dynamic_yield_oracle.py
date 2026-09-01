#!/usr/bin/env python3
"""Verify the Route197 planning-level privileged dynamic-yield oracle."""

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
from uq_estimator.privileged_yield_labels import (
    DynamicYieldLabeler,
    TrajectoryConflictConfig,
    build_safe_yield_trajectory,
    evaluate_trajectory_conflicts,
    trajectory_residual,
)


SCHEMA_VERSION = "orion.route197_dynamic_yield_oracle_result.v3.1"


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


def config_from_prereg(prereg: dict) -> TrajectoryConflictConfig:
    frozen = prereg["frozen_planning_response"]
    return TrajectoryConflictConfig(
        interpolation_step_seconds=frozen["interpolation_step_seconds"],
        safety_margin_m=frozen["safety_margin_m"],
        imminent_horizon_seconds=frozen["imminent_horizon_seconds"],
        clearance_seconds=frozen["clearance_seconds"],
        release_seconds=frozen["release_seconds"],
        stop_buffer_m=frozen["stop_buffer_m"],
        release_creep_distance_m=frozen["release_creep_distance_m"],
    )


def evaluate(
    run_dir: Path,
    preregistration: Path,
    clean_report_path: Path,
) -> dict:
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    clean_report = json.loads(clean_report_path.read_text(encoding="utf-8"))
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
    config = config_from_prereg(prereg)

    manifest_checks = {
        "condition_dynamic_yield_oracle": (
            manifest.get("pilot_condition") == "native_dynamic_yield_oracle"
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
            == "privileged_dynamic_yield"
        ),
        "interpolation_step_frozen": same(
            manifest.get("orion_planning_interpolation_step_seconds"),
            frozen["interpolation_step_seconds"],
        ),
        "safety_margin_frozen": same(
            manifest.get("orion_planning_safety_margin_m"),
            frozen["safety_margin_m"],
        ),
        "imminent_horizon_frozen": same(
            manifest.get("orion_planning_imminent_horizon_seconds"),
            frozen["imminent_horizon_seconds"],
        ),
        "clearance_seconds_frozen": same(
            manifest.get("orion_planning_clearance_seconds"),
            frozen["clearance_seconds"],
        ),
        "release_seconds_frozen": same(
            manifest.get("orion_planning_release_seconds"),
            frozen["release_seconds"],
        ),
        "stop_buffer_frozen": same(
            manifest.get("orion_planning_stop_buffer_m"),
            frozen["stop_buffer_m"],
        ),
        "release_creep_frozen": same(
            manifest.get("orion_planning_release_creep_distance_m"),
            frozen["release_creep_distance_m"],
        ),
        "route_hash_frozen": (
            manifest.get("route_sha256") == hashes["route_197_hazard.xml"]
        ),
        "clean_report_hash_frozen": (
            sha256(clean_report_path) == prereg["clean_reference"]["report_sha256"]
        ),
    }
    for relative_path, expected_hash in hashes.items():
        if relative_path == "route_197_hazard.xml":
            continue
        manifest_checks[f"source_hash:{relative_path}"] = (
            manifest_hashes.get(relative_path) == expected_hash
        )

    labeler = DynamicYieldLabeler(config)
    state_counts: Counter[str] = Counter()
    all_recomputed = True
    all_targets_match = True
    all_residuals_match = True
    all_risk_passthrough = True
    all_density_absent = True
    all_adapter_absent = True
    steps = []
    transition_count = 0
    previous_state = None
    first_prepare = None
    first_hold = None
    release_to_hold_count = 0
    for row in records:
        steps.append(int(row["step"]))
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
        if response.get("mode") != "privileged_dynamic_yield":
            all_recomputed = False
            continue
        base_plan = response.get("base_plan_cumulative_m")
        expected_conflict = evaluate_trajectory_conflicts(
            base_plan,
            row.get("closedloop_safety"),
            config=config,
        )
        expected_label = labeler.update(
            expected_conflict, float(row["sim_time_seconds"])
        )
        expected_target = build_safe_yield_trajectory(base_plan, expected_label)
        expected_residual = trajectory_residual(base_plan, expected_target)
        all_recomputed &= same(response.get("conflict"), expected_conflict.to_dict())
        all_recomputed &= same(response.get("yield_label"), expected_label.to_dict())
        all_targets_match &= same(
            response.get("target_plan_cumulative_m"), expected_target
        )
        all_residuals_match &= same(
            response.get("trajectory_residual_m"), expected_residual
        )
        state = expected_label.state
        state_counts[state] += 1
        timestamp = float(row["sim_time_seconds"])
        if state == "prepare_yield" and first_prepare is None:
            first_prepare = timestamp
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
        "density_score_absent_every_frame": all_density_absent,
        "observation_adapter_absent_every_frame": all_adapter_absent,
        "scalar_risk_governor_exact_passthrough": all_risk_passthrough,
        "planning_conflict_and_state_exactly_recomputed": all_recomputed,
        "planning_targets_exactly_recomputed": all_targets_match,
        "trajectory_residuals_exactly_recomputed": all_residuals_match,
        "prepare_or_hold_observed": (
            state_counts["prepare_yield"] + state_counts["hold"] > 0
        ),
        "trajectory_intervention_observed": any(
            any(abs(float(value)) > 1e-6 for point in (
                (row.get("planning_response") or {}).get(
                    "trajectory_residual_m", []
                )
            ) for value in point)
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
        "eval_path": str(eval_path.resolve()),
        "manifest_checks": manifest_checks,
        "trace_checks": trace_checks,
        "endpoint_checks": endpoint_checks,
        "planning_response": {
            "state_counts": dict(state_counts),
            "transition_count": transition_count,
            "release_to_hold_count": release_to_hold_count,
            "first_prepare_time_seconds": first_prepare,
            "first_hold_time_seconds": first_hold,
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
        "verifier_amendment": {
            "kind": "post_run_numeric_manifest_normalization_only",
            "scientific_outcome_changed": False,
            "reason": "The frozen v3 verifier compared numeric manifest strings to preregistered JSON numbers without numeric coercion. v3.1 corrects only that representation check and adds the already-recorded min-speed count as a secondary endpoint.",
        },
        "claim_boundary": prereg["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--clean-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite dynamic-yield oracle report")
    report = evaluate(args.run_dir, args.preregistration, args.clean_report)
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

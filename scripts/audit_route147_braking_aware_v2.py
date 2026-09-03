#!/usr/bin/env python3
"""Offline gate for one Route147 braking-aware v2 oracle run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from uq_estimator.bounded_crossing_expert import (
    BoundedCrossingExpertConfig,
    build_braking_aware_crossing_trajectory,
)
from uq_estimator.dynamic_yield_expert import pid_desired_speed_proxy
from uq_estimator.privileged_yield_labels import YieldLabel


SCHEMA_VERSION = "orion.route147_braking_aware_v2_offline_gate.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonlines(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit(
    *,
    v1_result_path: Path,
    v1_oracle_trace_path: Path,
    immediate_brake_ratio_threshold: float = 1.1,
) -> dict:
    v1_result = json.loads(v1_result_path.read_text(encoding="utf-8"))
    rows = load_jsonlines(v1_oracle_trace_path)
    if not rows:
        raise ValueError("v1 oracle trace is empty")
    event_rows = [
        row for row in rows
        if (row.get("planning_response") or {}).get("yield_label", {}).get("state")
        not in (None, "go")
    ]
    if not event_rows:
        raise ValueError("v1 oracle trace has no bounded crossing event")
    config = BoundedCrossingExpertConfig()
    diagnostics = []
    all_go_identity = True
    release_speed_commands = []
    for row in rows:
        response = row.get("planning_response") or {}
        label_payload = response.get("yield_label")
        if label_payload is None:
            raise ValueError("v1 oracle response is missing a yield label")
        label = YieldLabel(**label_payload)
        base = response["base_plan_cumulative_m"]
        target, profile = build_braking_aware_crossing_trajectory(
            base, label, row["speed"], config=config
        )
        if label.state == "go":
            all_go_identity &= target == tuple(map(tuple, base))
        if label.state == "release":
            release_speed_commands.append(profile.target_pid_desired_speed_mps)
        if label.state != "go":
            old_target = response["target_plan_cumulative_m"]
            diagnostics.append({
                "step": int(row["step"]),
                "sim_time_seconds": float(row["sim_time_seconds"]),
                "state": label.state,
                "speed_mps": float(row["speed"]),
                "safe_stop_path_distance_m": label.stop_path_distance_m,
                "v1_target_pid_desired_speed_mps": pid_desired_speed_proxy(old_target),
                "v2_target_pid_desired_speed_mps": profile.target_pid_desired_speed_mps,
                "v2_immediate_brake_ratio": profile.immediate_brake_ratio,
                "v1_first_two_waypoints": old_target[:2],
                "v2_first_two_waypoints": target[:2],
                "v2_profile": profile.to_dict(),
                "v2_max_path_distance_m": max(
                    (point[0] * point[0] + point[1] * point[1]) ** 0.5
                    for point in target
                ),
            })

    first_hold = next(item for item in diagnostics if item["state"] == "hold")
    v1_failed_outcomes = sorted(
        key for key, value in v1_result["outcome_checks"].items() if not value
    )
    v1_other_outcomes_pass = all(
        value for key, value in v1_result["outcome_checks"].items()
        if key != "walker_ttc_improved_or_horizon_censored"
    )
    stop_distance = float(first_hold["safe_stop_path_distance_m"])
    checks = {
        "v1_was_valid_but_primary_failed": (
            v1_result.get("primary_success") is False
            and v1_result.get("stage2_eligible") is False
        ),
        "v1_only_failed_minimum_ttc_gain": (
            v1_failed_outcomes == ["walker_ttc_improved_or_horizon_censored"]
            and v1_other_outcomes_pass
        ),
        "v1_all_endpoint_contracts_passed": all(
            v1_result["endpoint_checks"].values()
        ),
        "v1_all_trace_contracts_passed": all(
            value
            for condition in v1_result["trace_contract"].values()
            for value in condition["checks"].values()
        ),
        "v1_first_hold_did_not_command_immediate_braking": (
            first_hold["speed_mps"]
            / first_hold["v1_target_pid_desired_speed_mps"]
            < immediate_brake_ratio_threshold
        ),
        "v2_first_hold_commands_immediate_braking": (
            first_hold["v2_immediate_brake_ratio"]
            >= immediate_brake_ratio_threshold
        ),
        "v2_changes_both_near_waypoints": (
            first_hold["v2_first_two_waypoints"][0]
            != first_hold["v1_first_two_waypoints"][0]
            and first_hold["v2_first_two_waypoints"][1]
            != first_hold["v1_first_two_waypoints"][1]
        ),
        "v2_stops_before_frozen_safe_distance": (
            first_hold["v2_max_path_distance_m"] <= stop_distance + 1e-6
        ),
        "v2_go_is_exact_orion_identity": all_go_identity,
        "v2_release_is_half_mps_creep": (
            release_speed_commands
            and all(abs(value - 0.5) <= 1e-4 for value in release_speed_commands)
        ),
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA_VERSION,
        "offline_gate_pass": passed,
        "decision": (
            "eligible_for_one_preregistered_route147_braking_aware_v2_oracle"
            if passed else "do_not_submit_route147_braking_aware_v2"
        ),
        "scientific_role": (
            "mechanism repair only; retains v1 thresholds and existing clean "
            "reference, and does not evaluate the learned adapter"
        ),
        "checks": checks,
        "thresholds": {
            "immediate_brake_ratio": immediate_brake_ratio_threshold,
            "certified_deceleration_mps2": config.certified_deceleration_mps2,
            "prepare_creep_speed_mps": config.prepare_creep_speed_mps,
            "release_creep_speed_mps": config.release_creep_speed_mps,
            "release_creep_distance_m": config.release_creep_distance_m,
        },
        "v1_failure": {
            "failed_outcome_checks": v1_failed_outcomes,
            "minimum_walker_ttc_seconds": {
                "clean": v1_result["outcomes"]["clean_walker"]["min_obb_ttc_seconds"],
                "oracle_v1": v1_result["outcomes"]["oracle_walker"]["min_obb_ttc_seconds"],
            },
            "walker_ttc_lte_1_exposure_reduction_seconds": v1_result[
                "outcomes"
            ]["walker_ttc_lte_1_exposure_reduction_seconds"],
        },
        "first_hold_counterfactual": first_hold,
        "event_record_count": len(event_rows),
        "artifacts": {
            "v1_result_path": str(v1_result_path),
            "v1_result_sha256": sha256(v1_result_path),
            "v1_oracle_trace_path": str(v1_oracle_trace_path),
            "v1_oracle_trace_sha256": sha256(v1_oracle_trace_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-result", required=True, type=Path)
    parser.add_argument("--v1-oracle-trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(
        v1_result_path=args.v1_result,
        v1_oracle_trace_path=args.v1_oracle_trace,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

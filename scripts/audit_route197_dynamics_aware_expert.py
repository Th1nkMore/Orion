#!/usr/bin/env python3
"""Gate a single Route197 dynamics-aware privileged-oracle rerun.

This is deliberately an offline mechanism audit.  It may authorize one new
closed-loop oracle run, but cannot establish collision avoidance because the
recorded ego/actor states diverge as soon as the counterfactual target is used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uq_estimator.dynamic_yield_expert import (
    BrakingAwareYieldStateMachine,
    DynamicYieldExpertConfig,
    build_dynamics_aware_yield_trajectory,
    compute_junction_yield_geometry,
    conservative_stopping_distance,
    pid_desired_speed_proxy,
    suppress_unbounded_conflict,
)
from uq_estimator.privileged_yield_labels import TrajectoryConflictResult


SCHEMA_VERSION = "orion.route197_dynamics_aware_expert_audit.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError("control trace is empty")
    if [row["step"] for row in rows] != list(range(len(rows))):
        raise ValueError("control trace steps must be contiguous from zero")
    return rows


def conflict_from_trace(row: dict) -> TrajectoryConflictResult:
    source = row["planning_response"]["conflict"]
    return TrajectoryConflictResult(
        per_horizon_conflict=tuple(bool(value) for value in source["per_horizon_conflict"]),
        per_horizon_min_gap_m=tuple(source["per_horizon_min_gap_m"]),
        per_horizon_actor_ids=tuple(
            tuple(int(actor_id) for actor_id in actor_ids)
            for actor_ids in source["per_horizon_actor_ids"]
        ),
        earliest_conflict_seconds=source["earliest_conflict_seconds"],
        minimum_gap_m=source["minimum_gap_m"],
        critical_actor_id=source["critical_actor_id"],
        conflict_path_distance_m=source["conflict_path_distance_m"],
        base_plan_world_xy=tuple(tuple(point) for point in source["base_plan_world_xy"]),
    )


def xy_distance(first, second) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def pid_requests_brake(speed_mps: float, desired_speed_mps: float) -> bool:
    return desired_speed_mps < 0.05 or (
        desired_speed_mps > 0 and speed_mps / desired_speed_mps > 1.1
    )


def observed_brake_prefix(rows: list[dict], first_index: int) -> dict:
    """Measure the already-recorded continuous full-brake response prefix."""

    if rows[first_index]["risk"]["brake"] < 0.99:
        return {
            "available": False,
            "reason": "original controller was not already at full brake",
        }
    last_index = first_index
    while (
        last_index + 1 < len(rows)
        and rows[last_index]["risk"]["brake"] >= 0.99
    ):
        last_index += 1
        if rows[last_index]["risk"]["brake"] < 0.99:
            break
    # Control at row i affects the transition to row i+1.  Include the first
    # snapshot after the final full-brake command.
    end_index = min(last_index, len(rows) - 1)
    first = rows[first_index]
    end = rows[end_index]
    start_xy = first["closedloop_safety"]["ego"]["position_xy"]
    distance = 0.0
    for left, right in zip(rows[first_index:end_index], rows[first_index + 1:end_index + 1]):
        distance += xy_distance(
            left["closedloop_safety"]["ego"]["position_xy"],
            right["closedloop_safety"]["ego"]["position_xy"],
        )
    return {
        "available": True,
        "start_step": first["step"],
        "end_step": end["step"],
        "start_time_seconds": first["sim_time_seconds"],
        "end_time_seconds": end["sim_time_seconds"],
        "duration_seconds": end["sim_time_seconds"] - first["sim_time_seconds"],
        "start_speed_mps": first["speed"],
        "end_speed_mps": end["speed"],
        "observed_distance_m": distance,
        "start_xy": start_xy,
        "end_xy": end["closedloop_safety"]["ego"]["position_xy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--failed-mechanism-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_trace(args.trace)
    geometry_payload = json.loads(args.geometry.read_text())
    geometry_by_step = {
        int(record["step"]): record for record in geometry_payload["records"]
    }
    failed_report = json.loads(args.failed_mechanism_report.read_text())
    trace_sha = sha256_file(args.trace)
    config = DynamicYieldExpertConfig()
    state_machine = BrakingAwareYieldStateMachine(config)
    timeline = []
    first_actionable_conflict_index = None
    first_ego_junction_index = None
    unbounded_raw_conflict_steps = []

    for index, row in enumerate(rows):
        geometry_record = geometry_by_step.get(int(row["step"]))
        if geometry_record is None:
            raise ValueError(f"missing map geometry for trace step {row['step']}")
        ego_waypoint = geometry_record.get("ego_map_waypoint") or {}
        if ego_waypoint.get("is_junction") and first_ego_junction_index is None:
            first_ego_junction_index = index
        raw_conflict = conflict_from_trace(row)
        entry_distance = geometry_record.get("junction_entry_path_distance_m")
        junction_scoped_conflict = bool(
            raw_conflict.has_conflict
            and entry_distance is not None
            and not ego_waypoint.get("is_junction")
        )
        if raw_conflict.has_conflict and not junction_scoped_conflict:
            unbounded_raw_conflict_steps.append(int(row["step"]))
        conflict = (
            raw_conflict
            if junction_scoped_conflict
            else suppress_unbounded_conflict(raw_conflict)
        )
        if junction_scoped_conflict and first_actionable_conflict_index is None:
            first_actionable_conflict_index = index
        if entry_distance is None:
            # Geometry is immaterial while there is no conflict.  A distant
            # dummy boundary keeps the state-machine input finite.
            entry_distance = 1_000_000.0
        ego_extent = float(row["closedloop_safety"]["ego"]["extent_xy_m"][0])
        geometry = compute_junction_yield_geometry(
            entry_distance,
            ego_extent,
            row["speed"],
            config=config,
        )
        label = state_machine.update(
            conflict,
            geometry,
            row["sim_time_seconds"],
        )
        base = row["planning_response"]["base_plan_cumulative_m"]
        target = build_dynamics_aware_yield_trajectory(
            base, label, row["speed"], config=config
        )
        base_desired = pid_desired_speed_proxy(base)
        target_desired = pid_desired_speed_proxy(target)
        timeline.append({
            "step": int(row["step"]),
            "sim_time_seconds": float(row["sim_time_seconds"]),
            "route_progress": float(row["route_progress"]),
            "speed_mps": float(row["speed"]),
            "ego_is_junction": bool(ego_waypoint.get("is_junction")),
            "junction_entry_path_distance_m": geometry_record.get(
                "junction_entry_path_distance_m"
            ),
            "safe_center_stop_distance_m": geometry.safe_center_stop_distance_m,
            "conservative_stopping_distance_m": geometry.conservative_stopping_distance_m,
            "braking_boundary_reached": geometry.brake_required,
            "raw_conflict_present": raw_conflict.has_conflict,
            "junction_scoped_conflict": junction_scoped_conflict,
            "critical_actor_id": raw_conflict.critical_actor_id,
            "expert_state": label.state,
            "expert_reason": label.reason,
            "base_pid_desired_speed_mps": base_desired,
            "target_pid_desired_speed_mps": target_desired,
            "target_pid_requests_brake": pid_requests_brake(
                row["speed"], target_desired
            ),
            "first_two_waypoints_changed": target[:2] != tuple(
                tuple(point) for point in base[:2]
            ),
            "target_plan_cumulative_m": target,
        })

    if first_actionable_conflict_index is None:
        raise ValueError("trace contains no map-bounded junction conflict")
    if first_ego_junction_index is None:
        raise ValueError("map audit never observes the ego in a junction")
    first = timeline[first_actionable_conflict_index]
    preentry_conflicts = [
        record
        for record in timeline[
            first_actionable_conflict_index:first_ego_junction_index
        ]
        if record["junction_scoped_conflict"]
    ]
    out_of_scope_before_actionable = [
        record
        for record in timeline[:first_actionable_conflict_index]
        if record["raw_conflict_present"]
        and not record["junction_scoped_conflict"]
    ]
    brake_prefix = observed_brake_prefix(rows, first_actionable_conflict_index)
    if brake_prefix.get("available"):
        tail_distance = conservative_stopping_distance(
            brake_prefix["end_speed_mps"],
            deceleration_mps2=config.certified_deceleration_mps2,
            reaction_seconds=config.reaction_seconds,
        )
        projected_total = brake_prefix["observed_distance_m"] + tail_distance
        projected_margin = first["safe_center_stop_distance_m"] - projected_total
        brake_prefix.update({
            "conservative_tail_stopping_distance_m": tail_distance,
            "projected_total_stop_distance_m": projected_total,
            "projected_safe_line_margin_m": projected_margin,
        })
    checks = {
        "geometry_schema_valid": (
            geometry_payload.get("schema_version")
            == "orion.carla_junction_geometry.v1"
        ),
        "geometry_trace_hash_matches": geometry_payload.get("trace_sha256") == trace_sha,
        "geometry_record_count_matches": len(geometry_by_step) == len(rows),
        "failed_v3_report_is_not_relabelled_success": (
            failed_report.get("primary_success") is False
            and failed_report.get("decision")
            == "do_not_train_or_control_with_learned_uq_yet"
        ),
        "first_actionable_conflict_precedes_junction_entry": (
            first_actionable_conflict_index < first_ego_junction_index
        ),
        "unbounded_early_conflicts_preserve_native_plan": all(
            record["expert_state"] == "go"
            and not record["first_two_waypoints_changed"]
            for record in out_of_scope_before_actionable
        ),
        "first_conflict_reaches_braking_boundary": first["braking_boundary_reached"],
        "first_target_changes_first_two_waypoints": first["first_two_waypoints_changed"],
        "first_target_requests_pid_brake": first["target_pid_requests_brake"],
        "all_preentry_conflict_targets_request_pid_brake": bool(preentry_conflicts)
        and all(record["target_pid_requests_brake"] for record in preentry_conflicts),
        "observed_brake_prefix_available": brake_prefix.get("available") is True,
        "projected_stop_before_safe_line": (
            brake_prefix.get("projected_safe_line_margin_m", -math.inf) > 0
        ),
        "legacy_density_absent_every_frame": all(
            row.get("density_uq_score") is None for row in rows
        ),
        "learned_adapter_absent_from_oracle": all(
            row.get("observation_uq") is None for row in rows
        ),
    }
    gate_pass = all(checks.values())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "Post-hoc offline mechanism audit only. Recorded future states become "
            "invalid after counterfactual intervention; a passed gate authorizes "
            "one preregistered privileged-oracle rerun but does not prove collision "
            "avoidance, Stage-1 adapter quality, or learned task response."
        ),
        "trace_path": str(args.trace),
        "trace_sha256": trace_sha,
        "geometry_path": str(args.geometry),
        "geometry_sha256": sha256_file(args.geometry),
        "failed_mechanism_report_path": str(args.failed_mechanism_report),
        "failed_mechanism_report_sha256": sha256_file(args.failed_mechanism_report),
        "config": asdict(config),
        "first_actionable_conflict": first,
        "first_ego_junction": timeline[first_ego_junction_index],
        "preentry_conflict_frame_count": len(preentry_conflicts),
        "unbounded_raw_conflict_steps": unbounded_raw_conflict_steps,
        "out_of_scope_conflict_count_before_actionable": len(
            out_of_scope_before_actionable
        ),
        "observed_brake_prefix_plus_conservative_tail": brake_prefix,
        "checks": checks,
        "offline_gate_pass": gate_pass,
        "decision": (
            "eligible_for_one_preregistered_dynamics_aware_privileged_oracle_run"
            if gate_pass
            else "do_not_submit_new_closed_loop_run"
        ),
        "timeline_until_observed_junction_entry": timeline[: first_ego_junction_index + 1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "offline_gate_pass": gate_pass,
        "decision": payload["decision"],
        "first_conflict_time_seconds": first["sim_time_seconds"],
        "projected_safe_line_margin_m": brake_prefix.get(
            "projected_safe_line_margin_m"
        ),
    }, sort_keys=True))
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

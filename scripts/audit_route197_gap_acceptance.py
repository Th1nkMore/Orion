#!/usr/bin/env python3
"""Audit whether Route197 supports a conservative gap-acceptance merge.

The audit uses the resource-stopped v4 trace only.  It estimates the observed
ActorFlow lane and vehicle center-crossing times, intersects each stationary
ORION base plan with that lane, and applies front-clearance plus rear catch-up
constraints.  It cannot prove a counterfactual closed-loop outcome.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA_VERSION = "orion.route197_gap_acceptance_audit.v1"
HORIZONS_SECONDS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


@dataclass(frozen=True)
class GapAuditConfig:
    stationary_speed_threshold_mps: float = 0.25
    minimum_flow_speed_mps: float = 8.0
    horizontal_velocity_ratio: float = 4.0
    maximum_flow_lateral_distance_m: float = 20.0
    crossing_fit_radius_m: float = 30.0
    longitudinal_safety_margin_m: float = 0.75
    minimum_front_residual_m: float = 1.0
    minimum_rear_residual_m: float = 1.0
    certified_ego_acceleration_mps2: float = 3.0
    maximum_acceptable_wait_seconds: float = 8.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError("trace is empty")
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(steps[0], steps[0] + len(steps))):
        raise ValueError("trace steps are not contiguous")
    return rows


def path_lane_crossing(row: dict, lane_y: float) -> dict | None:
    ego = row["closedloop_safety"]["ego"]
    response = row.get("planning_response") or {}
    raw = response.get("raw_conflict") or response.get("conflict") or {}
    world_plan = raw.get("base_plan_world_xy")
    if not isinstance(world_plan, list) or len(world_plan) != len(HORIZONS_SECONDS):
        return None
    points = [ego["position_xy"]] + world_plan
    times = [0.0] + list(HORIZONS_SECONDS)
    arc_before = 0.0
    for first, second, start_time, end_time in zip(
        points, points[1:], times, times[1:]
    ):
        delta_y = float(second[1]) - float(first[1])
        segment_length = math.hypot(
            float(second[0]) - float(first[0]), delta_y
        )
        if abs(delta_y) <= 1e-9:
            arc_before += segment_length
            continue
        fraction = (lane_y - float(first[1])) / delta_y
        if not 0.0 <= fraction <= 1.0:
            arc_before += segment_length
            continue
        duration = end_time - start_time
        angle = math.atan2(delta_y, float(second[0]) - float(first[0]))
        ego_extent = ego["extent_xy_m"]
        projected_extent = (
            abs(math.cos(angle)) * float(ego_extent[0])
            + abs(math.sin(angle)) * float(ego_extent[1])
        )
        return {
            "world_xy": [
                float(first[0])
                + fraction * (float(second[0]) - float(first[0])),
                lane_y,
            ],
            "relative_time_seconds": start_time + fraction * duration,
            "segment_speed_mps": segment_length / duration,
            "arc_distance_m": arc_before + fraction * segment_length,
            "ego_projected_half_extent_m": projected_extent,
        }
    return None


def _flow_observations(rows: list[dict], config: GapAuditConfig):
    stationary = [
        row
        for row in rows
        if float(row["speed"]) < config.stationary_speed_threshold_mps
        and float(row["sim_time_seconds"]) >= 10.0
    ]
    if not stationary:
        raise ValueError("no post-event stationary frames")
    stopped_y = statistics.median(
        float(row["closedloop_safety"]["ego"]["position_xy"][1])
        for row in stationary
    )
    observations = []
    tracks = defaultdict(list)
    for row in rows:
        timestamp = float(row["sim_time_seconds"])
        for actor in row["closedloop_safety"].get("actors", []):
            if actor.get("category") != "vehicle":
                continue
            velocity_x, velocity_y = map(float, actor["velocity_xy"])
            speed = math.hypot(velocity_x, velocity_y)
            position_x, position_y = map(float, actor["position_xy"])
            if speed < config.minimum_flow_speed_mps:
                continue
            if abs(velocity_x) < config.horizontal_velocity_ratio * abs(velocity_y):
                continue
            if abs(position_y - stopped_y) > config.maximum_flow_lateral_distance_m:
                continue
            observations.append((position_x, position_y, velocity_x, velocity_y))
            tracks[int(actor["actor_id"])].append((timestamp, actor))
    if not observations or len(tracks) < 2:
        raise ValueError("could not identify the Route197 ActorFlow")
    lane_y = statistics.median(value[1] for value in observations)
    direction = 1 if statistics.median(value[2] for value in observations) > 0 else -1
    return stationary, lane_y, direction, tracks


def estimate_actor_crossings(
    tracks: dict,
    merge_x: float,
    direction: int,
    config: GapAuditConfig,
) -> list[dict]:
    crossings = []
    for actor_id, observations in tracks.items():
        estimates = []
        for timestamp, actor in observations:
            position_x = float(actor["position_xy"][0])
            velocity_x = float(actor["velocity_xy"][0])
            if direction * velocity_x <= 1.0:
                continue
            if abs(position_x - merge_x) > config.crossing_fit_radius_m:
                continue
            estimates.append(timestamp + (merge_x - position_x) / velocity_x)
        if not estimates:
            continue
        closest = min(
            observations,
            key=lambda item: abs(float(item[1]["position_xy"][0]) - merge_x),
        )[1]
        crossings.append({
            "actor_id": int(actor_id),
            "center_crossing_time_seconds": statistics.median(estimates),
            "crossing_estimate_spread_seconds": max(estimates) - min(estimates),
            "flow_speed_mps": statistics.median(
                abs(float(actor["velocity_xy"][0]))
                for unused, actor in observations
            ),
            "longitudinal_half_extent_m": float(closest["extent_xy_m"][0]),
            "type_id": closest.get("type_id"),
            "observation_count": len(observations),
        })
    crossings.sort(key=lambda row: row["center_crossing_time_seconds"])
    if len(crossings) < 2:
        raise ValueError("fewer than two flow actors cross the merge point")
    return crossings


def candidate_gap_metrics(
    row: dict,
    crossing: dict,
    actors: list[dict],
    config: GapAuditConfig,
    commanded_merge_speed_floor_mps: float = 0.0,
) -> dict | None:
    absolute_merge_time = (
        float(row["sim_time_seconds"]) + crossing["relative_time_seconds"]
    )
    before = [
        actor
        for actor in actors
        if actor["center_crossing_time_seconds"] < absolute_merge_time
    ]
    after = [
        actor
        for actor in actors
        if actor["center_crossing_time_seconds"] >= absolute_merge_time
    ]
    if not before or not after:
        return None
    preceding = before[-1]
    following = after[0]
    ego_extent = crossing["ego_projected_half_extent_m"]
    front_center_distance = (
        absolute_merge_time - preceding["center_crossing_time_seconds"]
    ) * preceding["flow_speed_mps"]
    rear_center_distance = (
        following["center_crossing_time_seconds"] - absolute_merge_time
    ) * following["flow_speed_mps"]
    front_clearance = front_center_distance - (
        preceding["longitudinal_half_extent_m"]
        + ego_extent
        + config.longitudinal_safety_margin_m
    )
    rear_clearance = rear_center_distance - (
        following["longitudinal_half_extent_m"]
        + ego_extent
        + config.longitudinal_safety_margin_m
    )
    native_merge_speed = float(crossing["segment_speed_mps"])
    evaluated_merge_speed = max(
        native_merge_speed, float(commanded_merge_speed_floor_mps)
    )
    following_closing_speed = max(
        0.0, following["flow_speed_mps"] - evaluated_merge_speed
    )
    rear_catchup_distance = following_closing_speed ** 2 / (
        2.0 * config.certified_ego_acceleration_mps2
    )
    rear_dynamic_residual = rear_clearance - rear_catchup_distance
    raw_conflict = (
        (row.get("planning_response") or {})
        .get("raw_conflict", {})
        .get("earliest_conflict_seconds")
        is not None
    )
    accepted = bool(
        not raw_conflict
        and front_clearance >= config.minimum_front_residual_m
        and rear_dynamic_residual >= config.minimum_rear_residual_m
    )
    return {
        "release_step": int(row["step"]),
        "release_time_seconds": float(row["sim_time_seconds"]),
        "absolute_merge_time_seconds": absolute_merge_time,
        "relative_merge_time_seconds": float(crossing["relative_time_seconds"]),
        "merge_world_xy": crossing["world_xy"],
        "merge_arc_distance_m": float(crossing["arc_distance_m"]),
        "native_merge_segment_speed_mps": native_merge_speed,
        "commanded_merge_speed_floor_mps": float(commanded_merge_speed_floor_mps),
        "evaluated_merge_speed_mps": evaluated_merge_speed,
        "preceding_actor_id": preceding["actor_id"],
        "following_actor_id": following["actor_id"],
        "front_clearance_residual_m": front_clearance,
        "rear_clearance_before_catchup_m": rear_clearance,
        "rear_catchup_distance_m": rear_catchup_distance,
        "rear_dynamic_residual_m": rear_dynamic_residual,
        "raw_three_second_obb_conflict_present": raw_conflict,
        "accepted": accepted,
    }


def _accepted_candidates(
    stationary: list[dict],
    lane_y: float,
    actors: list[dict],
    config: GapAuditConfig,
    speed_floor: float,
) -> list[dict]:
    candidates = []
    for row in stationary:
        crossing = path_lane_crossing(row, lane_y)
        if crossing is None:
            continue
        metrics = candidate_gap_metrics(
            row, crossing, actors, config, speed_floor
        )
        if metrics is not None and metrics["accepted"]:
            candidates.append(metrics)
    return candidates


def audit(
    trace_path: Path,
    geometry_path: Path,
    deadlock_report_path: Path,
    config: GapAuditConfig,
) -> dict:
    rows = load_trace(trace_path)
    geometry = json.loads(geometry_path.read_text())
    deadlock = json.loads(deadlock_report_path.read_text())
    trace_sha = sha256_file(trace_path)
    if geometry.get("trace_sha256") != trace_sha:
        raise ValueError("geometry does not describe this trace")
    if deadlock.get("classification") != "resource_stopped_structural_deadlock":
        raise ValueError("input is not the frozen structural-deadlock report")
    if deadlock.get("stage2_eligible") is not False:
        raise ValueError("deadlock report must remain Stage2-ineligible")

    stationary, lane_y, direction, tracks = _flow_observations(rows, config)
    frozen_stationary = deadlock["trace"]["longest_stationary_interval"]
    frozen_start_step = int(frozen_stationary["start_step"])
    frozen_end_step = int(frozen_stationary["end_step"])
    stationary = [
        row
        for row in stationary
        if frozen_start_step <= int(row["step"]) <= frozen_end_step
    ]
    if not stationary:
        raise ValueError("deadlock report's longest stationary interval is absent")
    crossings = [path_lane_crossing(row, lane_y) for row in stationary]
    crossings = [crossing for crossing in crossings if crossing is not None]
    if not crossings:
        raise ValueError("stationary ORION plans never reach the flow lane")
    merge_x = statistics.median(
        float(crossing["world_xy"][0]) for crossing in crossings
    )
    actor_crossings = estimate_actor_crossings(
        tracks, merge_x, direction, config
    )
    actor_times = [row["center_crossing_time_seconds"] for row in actor_crossings]
    observed_headways = [
        right - left for left, right in zip(actor_times, actor_times[1:])
    ]
    stationary_start = float(stationary[0]["sim_time_seconds"])
    native_candidates = _accepted_candidates(
        stationary, lane_y, actor_crossings, config, 0.0
    )
    native_observed_candidates = [
        candidate
        for candidate in native_candidates
        if candidate["absolute_merge_time_seconds"]
        <= float(rows[-1]["sim_time_seconds"])
    ]
    native_utility_candidates = [
        candidate
        for candidate in native_observed_candidates
        if candidate["release_time_seconds"] - stationary_start
        <= config.maximum_acceptable_wait_seconds
    ]

    sensitivity = []
    for speed_floor in (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0):
        candidates = _accepted_candidates(
            stationary, lane_y, actor_crossings, config, speed_floor
        )
        observed = [
            candidate
            for candidate in candidates
            if candidate["absolute_merge_time_seconds"]
            <= float(rows[-1]["sim_time_seconds"])
        ]
        useful = [
            candidate
            for candidate in observed
            if candidate["release_time_seconds"] - stationary_start
            <= config.maximum_acceptable_wait_seconds
        ]
        sensitivity.append({
            "commanded_merge_speed_floor_mps": speed_floor,
            "accepted_candidate_count": len(candidates),
            "observed_candidate_count": len(observed),
            "utility_candidate_count": len(useful),
            "first_utility_candidate": useful[0] if useful else None,
        })

    native_gate = bool(native_utility_candidates)
    checks = {
        "source_deadlock_is_nonterminal_and_stage2_ineligible": (
            deadlock.get("primary_success") is False
            and deadlock.get("stage2_eligible") is False
        ),
        "legacy_density_absent": deadlock["signal_and_control_contract"][
            "legacy_density_score_absent_every_frame"
        ],
        "new_adapter_absent_from_privileged_oracle": deadlock[
            "signal_and_control_contract"
        ]["new_observation_adapter_absent_every_frame"],
        "flow_actor_sequence_observed": len(actor_crossings) >= 10,
        "stationary_native_plan_crosses_flow_lane": bool(crossings),
        "native_gap_candidate_within_recorded_trace": bool(
            native_observed_candidates
        ),
        "native_gap_candidate_within_utility_wait": native_gate,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "Post-hoc kinematic audit of a resource-stopped trace. Actor crossing "
            "times and stationary ORION plans are observational evidence only; "
            "future actor reactions and counterfactual ego control are not recorded."
        ),
        "trace_path": str(trace_path),
        "trace_sha256": trace_sha,
        "geometry_path": str(geometry_path),
        "geometry_sha256": sha256_file(geometry_path),
        "deadlock_report_path": str(deadlock_report_path),
        "deadlock_report_sha256": sha256_file(deadlock_report_path),
        "config": asdict(config),
        "inferred_flow": {
            "lane_center_y": lane_y,
            "flow_direction_x_sign": direction,
            "merge_point_x_median": merge_x,
            "actor_crossing_count": len(actor_crossings),
            "actor_crossings": actor_crossings,
            "observed_headway_seconds": observed_headways,
            "minimum_observed_headway_seconds": min(observed_headways),
            "median_observed_headway_seconds": statistics.median(
                observed_headways
            ),
            "maximum_observed_headway_seconds": max(observed_headways),
        },
        "stationary_plan": {
            "start_time_seconds": stationary_start,
            "end_time_seconds": float(stationary[-1]["sim_time_seconds"]),
            "frame_count": len(stationary),
            "plan_reaches_flow_lane_frame_count": len(crossings),
            "median_relative_merge_time_seconds": statistics.median(
                crossing["relative_time_seconds"] for crossing in crossings
            ),
            "median_native_merge_segment_speed_mps": statistics.median(
                crossing["segment_speed_mps"] for crossing in crossings
            ),
            "median_merge_arc_distance_m": statistics.median(
                crossing["arc_distance_m"] for crossing in crossings
            ),
        },
        "native_orion_gap_gate": {
            "accepted_candidate_count": len(native_candidates),
            "accepted_within_recorded_trace_count": len(
                native_observed_candidates
            ),
            "accepted_within_utility_wait_count": len(native_utility_candidates),
            "first_candidate": native_candidates[0] if native_candidates else None,
            "first_observed_candidate": (
                native_observed_candidates[0]
                if native_observed_candidates else None
            ),
            "first_utility_candidate": (
                native_utility_candidates[0] if native_utility_candidates else None
            ),
            "offline_gate_pass": native_gate,
        },
        "assertive_merge_speed_sensitivity": sensitivity,
        "checks": checks,
        "primary_success": False,
        "stage2_eligible": False,
        "decision": (
            "eligible_for_one_gap_acceptance_oracle"
            if native_gate
            else "route197_not_authorized_as_conservative_primary_case"
        ),
        "next_action": (
            "Select a finite, bounded hazard where braking/yielding has a valid "
            "utility-preserving oracle; retain Route197 as a hard merge-planning "
            "negative case. An assertive merge controller would require its own "
            "research question and preregistration."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--deadlock-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite gap-acceptance audit")
    report = audit(
        args.trace,
        args.geometry,
        args.deadlock_report,
        GapAuditConfig(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "offline_gate_pass": report["native_orion_gap_gate"][
            "offline_gate_pass"
        ],
        "decision": report["decision"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

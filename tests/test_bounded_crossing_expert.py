import json
from pathlib import Path

import pytest

from uq_estimator.bounded_crossing_expert import (
    BoundedCrossingExpertConfig,
    build_braking_aware_crossing_trajectory,
)
from uq_estimator.privileged_yield_labels import YieldLabel


ROOT = Path(__file__).resolve().parents[1]
V1_TRACE = ROOT / (
    "results/closedloop_scenario_bank/route147_bounded_crossing_pair_v1/"
    "oracle_1087791/control_trace.jsonl"
)


def label(state, stop_distance=None):
    return YieldLabel(
        state=state,
        state_index={"go": 0, "prepare_yield": 1, "hold": 2, "release": 3}[state],
        conflict_present=state in {"prepare_yield", "hold"},
        imminent_conflict=state == "hold",
        clearance_elapsed_seconds=0.0,
        release_elapsed_seconds=0.0,
        stop_path_distance_m=stop_distance,
        reason="test",
    )


def plan():
    return [[0.0, value] for value in (2.5, 6.0, 10.0, 14.0, 18.0, 22.0)]


def test_hold_retimes_first_waypoints_and_commands_immediate_braking():
    target, profile = build_braking_aware_crossing_trajectory(
        plan(), label("hold", 7.4), 4.66
    )
    assert target[0][1] < plan()[0][1]
    assert target[1][1] < plan()[1][1]
    assert max(point[1] for point in target) <= 7.4
    assert profile.applied_deceleration_mps2 == pytest.approx(3.0)
    assert profile.immediate_brake_ratio > 1.1


def test_go_is_exact_identity_and_release_is_half_mps_creep():
    go, go_profile = build_braking_aware_crossing_trajectory(
        plan(), label("go"), 4.0
    )
    assert go == tuple(map(tuple, plan()))
    assert go_profile.target_pid_desired_speed_mps == pytest.approx(
        go_profile.base_pid_desired_speed_mps
    )
    release, release_profile = build_braking_aware_crossing_trajectory(
        plan(), label("release", 1.0), 0.0
    )
    assert [point[1] for point in release] == pytest.approx(
        [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
    )
    assert release_profile.target_pid_desired_speed_mps == pytest.approx(0.5)


def test_real_v1_first_hold_would_switch_from_throttle_to_brake_proxy():
    rows = [json.loads(line) for line in V1_TRACE.read_text().splitlines()]
    first = next(
        row for row in rows
        if (row.get("planning_response") or {}).get("yield_label", {}).get("state")
        == "hold"
    )
    response = first["planning_response"]
    yield_label = YieldLabel(**response["yield_label"])
    target, profile = build_braking_aware_crossing_trajectory(
        response["base_plan_cumulative_m"], yield_label, first["speed"]
    )
    assert first["risk"]["base_throttle"] == pytest.approx(0.75)
    assert first["risk"]["base_brake"] == pytest.approx(0.0)
    assert profile.base_pid_desired_speed_mps > first["speed"]
    assert profile.target_pid_desired_speed_mps < first["speed"] / 1.1
    assert target[0][1] < response["target_plan_cumulative_m"][0][1]
    assert target[1][1] < response["target_plan_cumulative_m"][1][1]

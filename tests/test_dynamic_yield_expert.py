import pytest

from uq_estimator.dynamic_yield_expert import (
    BrakingAwareYieldStateMachine,
    DynamicYieldExpertConfig,
    build_dynamics_aware_yield_trajectory,
    compute_junction_yield_geometry,
    conservative_stopping_distance,
    first_path_junction_entry,
    PathJunctionEntry,
    pid_desired_speed_proxy,
    resolve_junction_scoped_conflict,
    suppress_unbounded_conflict,
)
from uq_estimator.privileged_yield_labels import TrajectoryConflictResult


def plan():
    return [[0.0, value] for value in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)]


def conflict(present=True, actor=7):
    return TrajectoryConflictResult(
        per_horizon_conflict=(present,) * 6,
        per_horizon_min_gap_m=((0.0 if present else 5.0),) * 6,
        per_horizon_actor_ids=(((actor,) if present else ()),) * 6,
        earliest_conflict_seconds=1.5 if present else None,
        minimum_gap_m=0.0 if present else None,
        critical_actor_id=actor if present else None,
        conflict_path_distance_m=6.0 if present else None,
        base_plan_world_xy=tuple((float(i), 0.0) for i in range(6)),
    )


def test_route197_like_boundary_requires_braking_before_junction_entry():
    config = DynamicYieldExpertConfig()
    geometry = compute_junction_yield_geometry(
        junction_entry_path_distance_m=6.08,
        ego_forward_extent_m=2.446,
        speed_mps=4.12,
        config=config,
    )
    assert geometry.safe_center_stop_distance_m == pytest.approx(3.134)
    assert geometry.conservative_stopping_distance_m == pytest.approx(
        conservative_stopping_distance(
            4.12, deceleration_mps2=3.0, reaction_seconds=0.1
        )
    )
    assert geometry.brake_required


def test_braking_boundary_enters_hold_without_waiting_for_low_ttc():
    config = DynamicYieldExpertConfig()
    state = BrakingAwareYieldStateMachine(config)
    far = compute_junction_yield_geometry(12.0, 2.4, 3.0, config=config)
    assert state.update(conflict(), far, 0.0).state == "prepare_yield"
    near = compute_junction_yield_geometry(6.0, 2.4, 4.2, config=config)
    label = state.update(conflict(), near, 0.5)
    assert label.state == "hold"
    assert label.reason == "braking_boundary_reached"


def test_hold_trajectory_changes_first_two_waypoints_and_triggers_pid_brake():
    config = DynamicYieldExpertConfig()
    state = BrakingAwareYieldStateMachine(config)
    geometry = compute_junction_yield_geometry(6.08, 2.446, 4.12, config=config)
    label = state.update(conflict(), geometry, 0.0)
    target = build_dynamics_aware_yield_trajectory(
        plan(), label, 4.12, config=config
    )
    assert target[0][1] < plan()[0][1]
    assert target[1][1] < plan()[1][1]
    assert max(point[1] for point in target) <= geometry.safe_center_stop_distance_m
    desired = pid_desired_speed_proxy(target)
    assert 4.12 / desired > 1.1  # Existing PID selects full brake.


def test_stopped_hold_is_zero_and_release_is_half_mps_creep():
    config = DynamicYieldExpertConfig(
        clearance_seconds=1.0,
        release_seconds=0.5,
        release_creep_speed_mps=0.5,
        release_creep_distance_m=1.0,
    )
    state = BrakingAwareYieldStateMachine(config)
    geometry = compute_junction_yield_geometry(2.5, 2.0, 0.0, config=config)
    hold = state.update(conflict(), geometry, 0.0)
    assert build_dynamics_aware_yield_trajectory(plan(), hold, 0.0, config=config) == (
        (0.0, 0.0),
    ) * 6
    state.update(conflict(False), geometry, 0.5)
    release = state.update(conflict(False), geometry, 1.0)
    assert release.state == "release"
    target = build_dynamics_aware_yield_trajectory(plan(), release, 0.0, config=config)
    assert [point[1] for point in target] == pytest.approx(
        [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
    )
    assert pid_desired_speed_proxy(target) == pytest.approx(0.5)


def test_new_conflict_during_release_returns_to_hold():
    state = BrakingAwareYieldStateMachine()
    geometry = compute_junction_yield_geometry(2.5, 2.0, 0.0)
    state.update(conflict(), geometry, 0.0)
    state.update(conflict(False), geometry, 0.5)
    assert state.update(conflict(False), geometry, 1.0).state == "release"
    returned = state.update(conflict(True, actor=9), geometry, 1.1)
    assert returned.state == "hold"
    assert returned.critical_actor_id == 9


def test_stopped_prepare_creeps_toward_wait_line_without_crossing_it():
    config = DynamicYieldExpertConfig(prepare_creep_speed_mps=1.0)
    geometry = compute_junction_yield_geometry(8.0, 2.0, 0.0, config=config)
    label = BrakingAwareYieldStateMachine(config).update(
        conflict(), geometry, 0.0
    )
    assert label.state == "prepare_yield"
    target = build_dynamics_aware_yield_trajectory(
        plan(), label, 0.0, config=config
    )
    assert pid_desired_speed_proxy(target) == pytest.approx(1.0)
    assert max(point[1] for point in target) <= geometry.safe_center_stop_distance_m


def test_map_junction_entry_is_refined_and_reports_inside_state():
    entry = first_path_junction_entry(
        [(0.0, 0.0), (0.0, 10.0)],
        lambda xy: xy[1] >= 6.08,
        resolution_m=0.5,
    )
    assert entry.ego_is_junction is False
    assert entry.distance_m == pytest.approx(6.08, abs=1e-4)
    assert entry.world_xy == pytest.approx((0.0, 6.08), abs=1e-4)
    inside = first_path_junction_entry(
        [(0.0, 7.0), (0.0, 10.0)],
        lambda xy: xy[1] >= 6.08,
    )
    assert inside.ego_is_junction is True
    assert inside.distance_m == 0.0


def test_unbounded_conflict_is_suppressed_without_mutating_world_plan():
    original = conflict(True, actor=42)
    suppressed = suppress_unbounded_conflict(original)
    assert original.has_conflict
    assert not suppressed.has_conflict
    assert suppressed.base_plan_world_xy == original.base_plan_world_xy
    assert all(not flag for flag in suppressed.per_horizon_conflict)


def test_junction_scope_preserves_native_go_but_never_drops_latched_hold():
    raw = conflict(True, actor=42)
    unbounded = resolve_junction_scoped_conflict(
        raw, None, expert_state="go"
    )
    assert not unbounded.junction_scoped_conflict
    assert not unbounded.effective_conflict.has_conflict
    assert unbounded.geometry_entry_distance_m == 1_000_000.0
    bounded = resolve_junction_scoped_conflict(
        raw,
        PathJunctionEntry(6.0, (0.0, 6.0), False),
        expert_state="go",
    )
    assert bounded.junction_scoped_conflict
    assert bounded.effective_conflict.has_conflict
    assert bounded.geometry_entry_distance_m == 6.0
    latched = resolve_junction_scoped_conflict(
        raw, None, expert_state="hold"
    )
    assert latched.junction_scoped_conflict
    assert latched.effective_conflict.has_conflict
    assert latched.geometry_entry_distance_m == 0.0

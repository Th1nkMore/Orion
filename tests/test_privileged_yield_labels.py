import pytest

from uq_estimator.privileged_yield_labels import (
    DynamicYieldLabeler,
    TrajectoryConflictConfig,
    TrajectoryConflictResult,
    build_safe_yield_trajectory,
    evaluate_trajectory_conflicts,
    orion_local_plan_to_world,
    select_actor_categories,
    trajectory_residual,
)


def ego(yaw=0.0):
    return {
        "actor_id": 1,
        "position_xy": [0.0, 0.0],
        "velocity_xy": [0.0, 0.0],
        "yaw_degrees": yaw,
        "extent_xy_m": [2.0, 1.0],
    }


def actor(actor_id, position, velocity=(0.0, 0.0), yaw=0.0):
    return {
        "actor_id": actor_id,
        "position_xy": list(position),
        "velocity_xy": list(velocity),
        "yaw_degrees": yaw,
        "extent_xy_m": [2.0, 1.0],
    }


def safety(*actors, yaw=0.0):
    return {"available": True, "ego": ego(yaw), "actors": list(actors)}


def plan():
    # ORION [right, forward]; ego yaw zero means forward is world +X.
    return [[0.0, value] for value in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)]


def conflict_result(earliest=None, distance=None, actor_id=None):
    present = earliest is not None
    return TrajectoryConflictResult(
        per_horizon_conflict=(present,) * 6,
        per_horizon_min_gap_m=(0.0 if present else 5.0,) * 6,
        per_horizon_actor_ids=((actor_id,) if present else (),) * 6,
        earliest_conflict_seconds=earliest,
        minimum_gap_m=0.0 if present else None,
        critical_actor_id=actor_id,
        conflict_path_distance_m=distance,
        base_plan_world_xy=tuple((float(i), 0.0) for i in range(6)),
    )


def test_local_right_forward_conversion_respects_carla_yaw():
    world = orion_local_plan_to_world([[1.0, 2.0]], ego(yaw=90.0))
    # forward=(0,+1), right=(-1,0)
    assert world[0] == pytest.approx((-1.0, 2.0))


def test_crossing_actor_conflicts_with_candidate_but_off_path_actor_does_not():
    config = TrajectoryConflictConfig(safety_margin_m=0.25)
    crossing = actor(7, (4.0, -6.0), velocity=(0.0, 6.0), yaw=90.0)
    result = evaluate_trajectory_conflicts(plan(), safety(crossing), config=config)
    assert result.has_conflict
    assert result.critical_actor_id == 7
    assert result.earliest_conflict_seconds < 2.0
    assert 7 in {item for group in result.per_horizon_actor_ids for item in group}

    off_path = actor(8, (4.0, 9.0), velocity=(0.0, 1.0), yaw=90.0)
    clear = evaluate_trajectory_conflicts(plan(), safety(off_path), config=config)
    assert not clear.has_conflict


def test_dynamic_state_waits_for_clearance_and_rechecks_returning_conflict():
    config = TrajectoryConflictConfig(
        imminent_horizon_seconds=1.0,
        clearance_seconds=1.0,
        release_seconds=0.5,
        stop_buffer_m=2.0,
    )
    labeler = DynamicYieldLabeler(config)
    assert labeler.update(conflict_result(2.0, 8.0, 10), 0.0).state == "prepare_yield"
    hold = labeler.update(conflict_result(0.8, 5.0, 10), 0.5)
    assert hold.state == "hold"
    assert hold.stop_path_distance_m == pytest.approx(3.0)
    assert labeler.update(conflict_result(), 1.0).state == "hold"
    assert labeler.update(conflict_result(), 1.5).state == "release"
    # A new actor during release must restore hold, not obey an expired timer.
    returned = labeler.update(conflict_result(2.0, 7.0, 11), 2.0)
    assert returned.state == "hold"
    assert returned.reason == "conflict_present_after_yield"
    assert labeler.update(conflict_result(), 2.5).state == "hold"
    assert labeler.update(conflict_result(), 3.0).state == "release"
    assert labeler.update(conflict_result(), 3.5).state == "go"


def test_safe_target_stops_then_creeps_before_restoring_base_plan():
    config = TrajectoryConflictConfig(
        stop_buffer_m=2.0,
        release_creep_distance_m=1.0,
    )
    labeler = DynamicYieldLabeler(config)
    hold = labeler.update(conflict_result(0.5, 7.0, 10), 0.0)
    target = build_safe_yield_trajectory(plan(), hold)
    assert target[:2] == ((0.0, 2.0), (0.0, 4.0))
    assert target[2:] == ((0.0, 5.0),) * 4
    residual = trajectory_residual(plan(), target)
    assert residual[0] == (0.0, 0.0)
    assert residual[-1] == (0.0, -7.0)

    labeler.update(conflict_result(), 0.5)
    release = labeler.update(conflict_result(), 1.0)
    assert release.state == "release"
    assert build_safe_yield_trajectory(plan(), release) == ((0.0, 1.0),) * 6
    go = labeler.update(conflict_result(), 1.5)
    assert go.state == "go"
    assert build_safe_yield_trajectory(plan(), go) == tuple(map(tuple, plan()))


def test_telemetry_unavailable_fails_closed():
    with pytest.raises(ValueError, match="unavailable"):
        evaluate_trajectory_conflicts(plan(), {"available": False})


def test_actor_category_selector_is_planning_only_and_does_not_mutate_source():
    source = safety(
        actor(7, (4.0, -6.0), velocity=(0.0, 6.0), yaw=90.0),
        actor(8, (2.0, 0.0)),
    )
    source["actors"][0]["category"] = "walker"
    source["actors"][1]["category"] = "vehicle"
    selected = select_actor_categories(source, ("walker",))
    assert [item["actor_id"] for item in selected["actors"]] == [7]
    assert selected["actor_count_considered"] == 1
    assert selected["planning_actor_categories"] == ["walker"]
    assert len(source["actors"]) == 2


def test_actor_category_selector_rejects_empty_or_duplicate_contracts():
    with pytest.raises(ValueError, match="at least one"):
        select_actor_categories(safety(), ())
    with pytest.raises(ValueError, match="duplicates"):
        select_actor_categories(safety(), ("walker", "walker"))

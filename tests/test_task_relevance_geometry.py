import numpy as np
import pytest

from uq_estimator.task_relevance_geometry import (
    CAMERA_ORDER,
    TaskRelevanceGeometryError,
    build_task_relevance_map,
    project_local_points,
)


def _ego():
    return {
        "actor_id": 1,
        "position_xy": [0.0, 0.0],
        "position_z": 0.75,
        "velocity_xy": [0.0, 0.0],
        "yaw_degrees": 0.0,
        "extent_xy_m": [2.0, 1.0],
        "extent_z_m": 0.75,
    }


def _actor(actor_id, position, velocity=(0.0, 0.0), yaw=0.0):
    x, y = position
    return {
        "actor_id": actor_id,
        "position_xy": [x, y],
        "position_z": 0.8,
        "velocity_xy": list(velocity),
        "yaw_degrees": yaw,
        "extent_xy_m": [2.0, 1.0],
        "extent_z_m": 0.8,
        "relative_longitudinal_m": x,
        "relative_lateral_m": y,
    }


def _safety(*actors):
    return {"available": True, "ego": _ego(), "actors": list(actors)}


def _straight_plan():
    return [[0.0, value] for value in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)]


def test_forward_ground_point_projects_into_front_camera():
    pixels, visible = project_local_points(np.asarray([[0.0, 10.0, -1.84]]))
    front = CAMERA_ORDER.index("CAM_FRONT")
    assert visible[front, 0]
    assert 700.0 < pixels[front, 0, 0] < 900.0
    assert 500.0 < pixels[front, 0, 1] < 800.0


def test_route_corridor_is_spatial_and_independent_of_actor_ttc_fields():
    result = build_task_relevance_map(
        _straight_plan(), _safety(), patch_hw=(10, 10)
    )
    assert result.relevance.shape == (6, 10, 10)
    # The first few ground points lie behind/below the camera near plane; the
    # farther route must nevertheless have substantial visible support.
    assert result.route_point_coverage > 0.5
    assert np.count_nonzero(result.route_corridor) > 0
    assert not result.relevant_actor_ids
    assert result.provenance["uses_recorded_ttc"] is False


def test_crossing_actor_gets_projected_support_but_off_path_actor_does_not():
    crossing = _actor(7, (4.0, -6.0), velocity=(0.0, 6.0), yaw=90.0)
    off_path = _actor(8, (4.0, 9.0), velocity=(0.0, 1.0), yaw=90.0)
    result = build_task_relevance_map(
        _straight_plan(), _safety(crossing, off_path), patch_hw=(20, 20)
    )
    assert result.relevant_actor_ids == (7,)
    assert np.count_nonzero(result.relevant_actor_support) > 0
    assert result.relevance.max() == 1.0


def test_visible_conflict_actor_can_supervise_nearfield_route_blind_spot():
    nearfield_plan = [[0.0, value] for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)]
    blocking_actor = _actor(9, (1.0, 0.0))
    result = build_task_relevance_map(
        nearfield_plan, _safety(blocking_actor), patch_hw=(20, 20)
    )
    assert result.route_point_coverage == 0.0
    assert np.count_nonzero(result.route_corridor) == 0
    assert np.count_nonzero(result.relevant_actor_support) > 0
    assert result.relevant_actor_ids == (9,)
    assert result.provenance["support_mode"] == "visible_conflict_actor_only"
    assert result.provenance["actor_only_fallback_used"] is True


def test_rejects_frame_when_neither_route_nor_conflict_actor_is_visible():
    nearfield_plan = [[0.0, value] for value in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)]
    with pytest.raises(TaskRelevanceGeometryError, match="no visible route"):
        build_task_relevance_map(nearfield_plan, _safety(), patch_hw=(20, 20))

"""Unit tests for the Route197 gap-acceptance audit geometry."""

from scripts.audit_route197_gap_acceptance import (
    GapAuditConfig,
    candidate_gap_metrics,
    path_lane_crossing,
)


def _row():
    return {
        "step": 10,
        "sim_time_seconds": 5.0,
        "speed": 0.0,
        "closedloop_safety": {
            "ego": {
                "position_xy": [0.0, 10.0],
                "extent_xy_m": [2.0, 1.0],
            }
        },
        "planning_response": {
            "raw_conflict": {
                "base_plan_world_xy": [
                    [0.0, 8.0],
                    [0.0, 6.0],
                    [0.0, 4.0],
                    [0.0, 2.0],
                    [1.0, 0.0],
                    [3.0, -1.0],
                ],
                "earliest_conflict_seconds": None,
            }
        },
    }


def test_path_lane_crossing_preserves_time_speed_and_extent():
    crossing = path_lane_crossing(_row(), lane_y=0.0)
    assert crossing is not None
    assert crossing["world_xy"] == [1.0, 0.0]
    assert crossing["relative_time_seconds"] == 2.5
    assert crossing["segment_speed_mps"] > 4.0
    assert crossing["arc_distance_m"] > 10.0
    assert crossing["ego_projected_half_extent_m"] > 1.0


def test_candidate_rejects_rear_catchup_and_accepts_large_gap():
    row = _row()
    crossing = path_lane_crossing(row, lane_y=0.0)
    actors = [
        {
            "actor_id": 1,
            "center_crossing_time_seconds": 6.0,
            "flow_speed_mps": 10.0,
            "longitudinal_half_extent_m": 2.0,
        },
        {
            "actor_id": 2,
            "center_crossing_time_seconds": 8.5,
            "flow_speed_mps": 14.0,
            "longitudinal_half_extent_m": 2.0,
        },
    ]
    rejected = candidate_gap_metrics(
        row, crossing, actors, GapAuditConfig(), 0.0
    )
    assert rejected is not None
    assert rejected["accepted"] is False
    assert rejected["rear_catchup_distance_m"] > 0

    actors[1]["center_crossing_time_seconds"] = 10.0
    accepted = candidate_gap_metrics(
        row, crossing, actors, GapAuditConfig(), 0.0
    )
    assert accepted is not None
    assert accepted["accepted"] is True

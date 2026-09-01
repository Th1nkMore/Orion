from scripts.render_closedloop_event_review import (
    critical_actor_for_row,
    critical_trace_event,
    safety_label,
    validate_region,
)


def _row(step, ttc_values):
    actors = []
    for actor_id, ttc, gap in ttc_values:
        actors.append(
            {
                "actor_id": actor_id,
                "category": "walker" if actor_id == 7 else "vehicle",
                "type_id": "test.actor",
                "obb_collision_ttc_seconds": ttc,
                "obb_separating_axis_gap_m": gap,
            }
        )
    return {
        "step": step,
        "sim_time_seconds": step * 0.05,
        "route_progress": step / 100.0,
        "speed": 4.0,
        "closedloop_safety": {"actors": actors},
    }


def test_row_prefers_finite_ttc_over_smaller_noncollision_gap():
    row = _row(10, [(3, None, 0.2), (7, 1.4, 5.0)])
    assert critical_actor_for_row(row)["actor_id"] == 7
    assert "OBB-TTC=1.40s" in safety_label(row)
    assert "walker#7" in safety_label(row)


def test_trace_event_finds_global_minimum_finite_ttc():
    rows = [
        _row(10, [(7, 2.0, 6.0)]),
        _row(20, [(7, 1.25, 4.0)]),
        _row(30, [(3, None, 0.1)]),
    ]
    row, actor = critical_trace_event(rows)
    assert row["step"] == 20
    assert actor["actor_id"] == 7


def test_row_without_usable_actor_has_explicit_empty_label():
    row = _row(0, [(3, None, None)])
    assert critical_actor_for_row(row) is None
    assert safety_label(row) == "OBB-TTC=--  gap=--  actor=none"


def test_region_validation_uses_top_left_bottom_right_order():
    assert validate_region([0.25, 0.55, 0.95, 1.0], "region") == (
        0.25,
        0.55,
        0.95,
        1.0,
    )

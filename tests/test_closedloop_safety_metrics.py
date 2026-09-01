import importlib.util
import pathlib

import pytest


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "uq_estimator"
    / "closedloop_safety_metrics.py"
)
SPEC = importlib.util.spec_from_file_location(
    "closedloop_safety_metrics_under_test", MODULE_PATH
)
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)
disc_collision_ttc = METRICS.disc_collision_ttc
obb_collision_ttc = METRICS.obb_collision_ttc
pairwise_safety_metrics = METRICS.pairwise_safety_metrics
summarize_dynamic_actor_safety = METRICS.summarize_dynamic_actor_safety
vertical_separating_gap = METRICS.vertical_separating_gap


def _state(actor_id, position, velocity, radius=1.0, category="vehicle"):
    return {
        "actor_id": actor_id,
        "type_id": f"{category}.test",
        "category": category,
        "position_xy": position,
        "velocity_xy": velocity,
        "yaw_degrees": 0.0,
        "extent_xy_m": [radius, radius],
        "radius_m": radius,
    }


def test_disc_collision_ttc_head_on_has_known_first_contact():
    assert disc_collision_ttc((12.0, 0.0), (-5.0, 0.0), 2.0) == pytest.approx(2.0)


def test_vertical_gap_excludes_bench2drive_hidden_pedestrian_placeholder():
    assert vertical_separating_gap(0.5, 0.7, -99.0, 0.9) == pytest.approx(97.9)
    assert vertical_separating_gap(0.5, 0.7, 0.6, 0.9) == 0.0


def test_disc_collision_ttc_rejects_diverging_and_crossing_miss():
    assert disc_collision_ttc((10.0, 0.0), (2.0, 0.0), 2.0) is None
    assert disc_collision_ttc((5.0, -5.0), (0.0, 2.0), 1.0) is None


def test_obb_ttc_rejects_parallel_lateral_miss_that_discs_can_overcover():
    ttc = obb_collision_ttc(
        (12.0, 3.0),
        (-5.0, 0.0),
        (1.0, 0.5),
        0.0,
        (1.0, 0.5),
        0.0,
    )
    assert ttc is None


def test_pairwise_metrics_capture_lateral_crossing_collision_course():
    ego = _state(1, (0.0, 0.0), (0.0, 0.0), radius=0.5)
    walker = _state(
        2, (0.0, -5.0), (0.0, 2.0), radius=0.5, category="walker"
    )
    result = pairwise_safety_metrics(ego, walker, horizon_seconds=10.0)
    assert result["disc_collision_ttc_seconds"] == pytest.approx(2.0)
    assert result["obb_collision_ttc_seconds"] == pytest.approx(2.0)
    assert result["closest_approach_time_seconds"] == pytest.approx(2.5)
    assert result["predicted_min_disc_clearance_m"] == pytest.approx(-1.0)


def test_summary_prioritizes_collision_course_and_preserves_raw_states():
    ego = _state(1, (0.0, 0.0), (5.0, 0.0), radius=1.0)
    crossing = _state(2, (12.0, 0.0), (0.0, 0.0), radius=1.0)
    safe = _state(3, (8.0, 10.0), (5.0, 0.0), radius=1.0)
    result = summarize_dynamic_actor_safety(
        ego, [safe, crossing], horizon_seconds=10.0, max_actor_records=1
    )
    assert result["actor_count_considered"] == 2
    assert result["min_obb_collision_ttc_seconds"] == pytest.approx(2.0)
    assert result["critical_actor"]["actor_id"] == 2
    assert [record["actor_id"] for record in result["actors"]] == [2]

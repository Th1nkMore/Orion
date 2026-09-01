from scripts.evaluate_corruption_hardcase_wave_pair import (
    ROUTE_ACTOR_CATEGORY,
    continuous_margin_comparison,
    hard_endpoint_comparison,
)


def test_route158_hard_brake_uses_vehicle_safety_margin():
    assert ROUTE_ACTOR_CATEGORY["158"] == "vehicle"


def _evaluation(*, route=100.0, collisions=0, red_lights=0):
    return {
        "_checkpoint": {
            "records": [{
                "status": "Completed",
                "scores": {"score_route": route, "score_penalty": 1.0},
                "infractions": {
                    "collisions_layout": [],
                    "collisions_pedestrian": ["hit"] * collisions,
                    "collisions_vehicle": [],
                    "red_light": ["red"] * red_lights,
                    "stop_infraction": [],
                    "outside_route_lanes": [],
                    "route_dev": [],
                    "vehicle_blocked": [],
                    "scenario_timeouts": [],
                    "route_timeout": [],
                },
            }]
        }
    }


def test_hard_endpoint_uses_new_infraction_or_ten_point_completion_drop():
    unchanged = hard_endpoint_comparison(_evaluation(), _evaluation(route=91.0))
    assert not unchanged["degraded"]
    completion = hard_endpoint_comparison(_evaluation(), _evaluation(route=90.0))
    assert completion["degraded"]
    collision = hard_endpoint_comparison(_evaluation(), _evaluation(collisions=1))
    assert collision["degraded"]


def test_continuous_ttc_gate_requires_floor_and_fraction():
    result = continuous_margin_comparison(
        {
            "min_obb_ttc_seconds": 1.2,
            "min_obb_separating_axis_gap_m": 3.0,
        },
        {
            "min_obb_ttc_seconds": 0.89,
            "min_obb_separating_axis_gap_m": 3.0,
        },
    )
    assert result["checks"]["ttc_drop_gate"]
    assert result["required_ttc_drop_seconds"] == 0.30


def test_continuous_gap_gate_requires_floor_and_fraction():
    result = continuous_margin_comparison(
        {
            "min_obb_ttc_seconds": None,
            "min_obb_separating_axis_gap_m": 4.0,
        },
        {
            "min_obb_ttc_seconds": None,
            "min_obb_separating_axis_gap_m": 3.19,
        },
    )
    assert result["checks"]["gap_drop_gate"]
    assert result["required_gap_drop_m"] == 0.8


def test_small_route151_stale_change_is_a_valid_negative_signal():
    result = continuous_margin_comparison(
        {
            "min_obb_ttc_seconds": 1.2073832545637084,
            "min_obb_separating_axis_gap_m": 1.6034800656009054,
        },
        {
            "min_obb_ttc_seconds": 1.1636546362093152,
            "min_obb_separating_axis_gap_m": 1.6604880622928044,
        },
    )
    assert not result["degraded"]
    assert result["observed_ttc_drop_seconds"] < 0.30

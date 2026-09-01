from scripts.evaluate_native_collision_discovery import (
    first_geometry_contact,
    severe_ttc_surrogate,
)


def _summary(min_ttc, exposure, min_gap):
    return {
        "safety": {
            "min_obb_ttc_seconds": min_ttc,
            "low_ttc_exposure_seconds": {"2.0": exposure},
            "min_obb_separating_axis_gap_m": min_gap,
        }
    }


def test_severe_ttc_surrogate_requires_all_frozen_thresholds():
    assert severe_ttc_surrogate(_summary(1.0, 0.5, 0.5))["passed"]
    assert not severe_ttc_surrogate(_summary(1.01, 0.5, 0.5))["passed"]
    assert not severe_ttc_surrogate(_summary(1.0, 0.49, 0.5))["passed"]
    assert not severe_ttc_surrogate(_summary(1.0, 0.5, 0.51))["passed"]


def test_severe_ttc_surrogate_fails_closed_on_missing_geometry():
    result = severe_ttc_surrogate(_summary(None, None, None))
    assert not result["passed"]
    assert not any(check["passed"] for check in result["checks"].values())


def test_first_geometry_contact_returns_first_zero_gap_actor():
    rows = [
        {
            "step": 10,
            "sim_time_seconds": 0.5,
            "route_progress": 0.2,
            "speed": 4.0,
            "closedloop_safety": {"min_obb_separating_axis_gap_m": 0.1},
        },
        {
            "step": 11,
            "sim_time_seconds": 0.55,
            "route_progress": 0.21,
            "speed": 3.5,
            "closedloop_safety": {
                "min_obb_separating_axis_gap_m": 0.0,
                "min_obb_collision_ttc_seconds": 0.0,
                "critical_actor": {
                    "actor_id": 7,
                    "type_id": "vehicle.test",
                    "category": "vehicle",
                },
            },
        },
    ]
    contact = first_geometry_contact(rows)
    assert contact["step"] == 11
    assert contact["actor_id"] == 7
    assert contact["minimum_obb_gap_m"] == 0.0

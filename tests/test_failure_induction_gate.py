import json

from scripts.evaluate_failure_induction_gate import (
    evaluate,
    hard_endpoint_comparison,
    surrogate_comparison,
)


def _evaluation(*, eligible=True, status="Completed", route=100, penalty=1, collisions=0):
    return {
        "eligible": eligible,
        "_checkpoint": {
            "records": [{
                "status": status,
                "scores": {"score_route": route, "score_penalty": penalty},
                "infractions": {
                    "collisions_layout": [],
                    "collisions_pedestrian": ["hit"] * collisions,
                    "collisions_vehicle": [],
                    "red_light": [],
                    "stop_infraction": [],
                    "route_dev": [],
                    "vehicle_blocked": [],
                    "scenario_timeouts": [],
                    "route_timeout": [],
                },
            }]
        },
    }


def test_hard_endpoint_detects_new_collision():
    result = hard_endpoint_comparison(
        _evaluation(), _evaluation(penalty=0.5, collisions=1)
    )
    assert result["degraded"]
    assert result["infraction_count_delta_degraded_minus_clean"][
        "collisions_pedestrian"
    ] == 1


def test_surrogate_requires_ttc_drop_and_exposure_increase():
    paired = {
        "event_plus_recovery": {
            "clean": {"safety": {"by_category": {"walker": {
                "min_obb_ttc_seconds": 1.0,
                "low_ttc_exposure_seconds": {"2.0": 0.5},
            }}}},
            "degraded": {"safety": {"by_category": {"walker": {
                "min_obb_ttc_seconds": 0.7,
                "low_ttc_exposure_seconds": {"2.0": 1.1},
            }}}},
            "comparison": {"by_category": {"walker": {
                "min_obb_separating_axis_gap_m": -0.4,
            }}},
        }
    }
    result = surrogate_comparison(
        paired,
        actor_category="walker",
        thresholds={
            "minimum_ttc_drop_seconds_floor": 0.2,
            "minimum_ttc_drop_fraction_of_clean": 0.2,
            "minimum_ttc_le_2_exposure_increase_seconds": 0.5,
        },
    )
    assert result["degraded"]
    assert result["observed_ttc_drop_seconds"] == 0.30000000000000004


def test_surrogate_rejects_one_frame_ttc_drop_without_exposure():
    paired = {
        "event_plus_recovery": {
            "clean": {"safety": {"by_category": {"walker": {
                "min_obb_ttc_seconds": 1.0,
                "low_ttc_exposure_seconds": {"2.0": 0.5},
            }}}},
            "degraded": {"safety": {"by_category": {"walker": {
                "min_obb_ttc_seconds": 0.5,
                "low_ttc_exposure_seconds": {"2.0": 0.7},
            }}}},
            "comparison": {"by_category": {"walker": {
                "min_obb_separating_axis_gap_m": -1.0,
            }}},
        }
    }
    result = surrogate_comparison(
        paired,
        actor_category="walker",
        thresholds={
            "minimum_ttc_drop_seconds_floor": 0.2,
            "minimum_ttc_drop_fraction_of_clean": 0.2,
            "minimum_ttc_le_2_exposure_increase_seconds": 0.5,
        },
    )
    assert not result["degraded"]
    assert not result["checks"]["ttc_le_2_exposure_increase_large_enough"]


def test_full_gate_produces_valid_surrogate_tier(tmp_path):
    spec = tmp_path / "gate.json"
    spec.write_text(json.dumps({
        "route_actor_category": {"147": "walker"},
        "validity_requirements": {
            "safety_schema": "orion.closedloop_dynamic_actor_safety.v2",
        },
        "surrogate_safety_margin_degradation": {"thresholds": {
            "minimum_ttc_drop_seconds_floor": 0.2,
            "minimum_ttc_drop_fraction_of_clean": 0.2,
            "minimum_ttc_le_2_exposure_increase_seconds": 0.5,
            "paired_recovery_seconds": 0.1,
        }},
    }))
    clean = tmp_path / "clean"
    degraded = tmp_path / "degraded"
    for run, condition in ((clean, "clean_off"), (degraded, "spatial_corrupt_transient_off")):
        trace_dir = run / "records_x" / "route"
        trace_dir.mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps({
            "pilot_route_index": "147",
            "pilot_condition": condition,
            "carla_quality_level": "Epic",
            "orion_closedloop_risk_mode": "off",
            "orion_closedloop_uq_mode": "none",
        }))
        (run / "eval_test.json").write_text(json.dumps(_evaluation()))
        rows = []
        for step in range(20):
            active = condition != "clean_off" and 2 <= step <= 12
            ttc = 0.8 if active else 3.0
            rows.append({
                "step": step,
                "sim_time_seconds": step * 0.05,
                "route_progress": step * 0.01,
                "speed": 2.0,
                "corruption_active": active,
                "closedloop_safety": {
                    "schema": "orion.closedloop_dynamic_actor_safety.v2",
                    "available": True,
                    "min_obb_collision_ttc_seconds": ttc,
                    "min_obb_separating_axis_gap_m": 2.0,
                    "min_disc_clearance_m": 1.0,
                    "critical_actor": {"actor_id": 7},
                    "actors": [{
                        "actor_id": 7,
                        "category": "walker",
                        "obb_collision_ttc_seconds": ttc,
                        "obb_separating_axis_gap_m": 2.0,
                    }],
                },
            })
        (trace_dir / "control_trace.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    (clean / "clean_safety_gate.json").write_text(json.dumps({
        "gate_passed": True,
    }))
    report = evaluate(
        spec_path=spec,
        route_index="147",
        clean_run=clean,
        degraded_run=degraded,
    )
    assert report["validity"]["valid"]
    assert report["decision"]["failure_induction_pass"]
    assert report["decision"]["evidence_tier"] == (
        "near_miss_surrogate_failure_induction"
    )

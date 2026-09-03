import json

from scripts.evaluate_clean_safety_gate import evaluate


def test_clean_safety_gate_accepts_complete_epic_v2_trace(tmp_path):
    run = tmp_path / "run"
    trace_dir = run / "records_x" / "route"
    trace_dir.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "pilot_condition": "clean_off",
        "carla_quality_level": "Epic",
        "orion_closedloop_risk_mode": "off",
        "orion_closedloop_uq_mode": "none",
        "orion_closedloop_safety_telemetry": "1",
    }))
    (run / "eval_test.json").write_text(json.dumps({
        "eligible": True,
        "_checkpoint": {"records": [{
            "status": "Completed",
            "scores": {"score_route": 100, "score_penalty": 1},
            "infractions": {
                "collisions_layout": [], "collisions_pedestrian": [],
                "collisions_vehicle": [], "red_light": [],
                "stop_infraction": [], "route_dev": [],
                "vehicle_blocked": [], "scenario_timeouts": [],
                "route_timeout": [],
                "min_speed_infractions": ["Average speed is 75% of traffic"],
            },
        }]},
    }))
    rows = []
    for step in range(4):
        rows.append({
            "step": step,
            "sim_time_seconds": step * 0.05,
            "route_progress": step * 0.1,
            "speed": 2.0,
            "closedloop_safety": {
                "schema": "orion.closedloop_dynamic_actor_safety.v2",
                "available": True,
                "min_obb_collision_ttc_seconds": None,
                "min_obb_separating_axis_gap_m": None,
                "min_disc_clearance_m": None,
                "critical_actor": None,
                "actors": [],
            },
        })
    (trace_dir / "control_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    report = evaluate(run)
    assert report["gate_passed"]
    assert report["trace_frames"] == 4
    assert report["official_diagnostics"]["min_speed_infraction_count"] == 1
    assert "efficiency diagnostic" in report["official_diagnostics"]["gate_role"]

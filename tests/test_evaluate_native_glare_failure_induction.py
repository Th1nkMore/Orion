import json
from pathlib import Path

from PIL import Image

from scripts.evaluate_native_glare_failure_induction import evaluate


def _terminal_eval():
    return {
        "eligible": True,
        "_checkpoint": {"records": [{
            "status": "Completed",
            "scores": {"score_route": 100.0, "score_penalty": 1.0},
            "infractions": {
                "collisions_layout": [],
                "collisions_pedestrian": [],
                "collisions_vehicle": [],
                "red_light": [],
                "stop_infraction": [],
                "route_dev": [],
                "vehicle_blocked": [],
                "scenario_timeouts": [],
                "route_timeout": [],
            },
        }]},
    }


def _make_run(root: Path, profile: str, pixel: int, degraded: bool = False):
    scenario = root / "records_orion_traj_0" / "RouteScenario_test"
    tensor_dir = scenario / "rgb_front_model_tensor"
    tensor_dir.mkdir(parents=True)
    camera = {
        "enable_postprocess_effects": "true",
        "exposure_mode": "histogram",
        "lens_flare_intensity": "0.75" if degraded else "0.0",
        "bloom_intensity": "1.5" if degraded else "0.0",
    }
    bev = {
        "enable_postprocess_effects": "false",
        "lens_flare_intensity": "0.0",
        "bloom_intensity": "0.0",
    }
    weather = {"sun_altitude_angle": 8.0, "sun_azimuth_angle": 180.0}
    readback = {
        "schema": "orion.closedloop_render_condition_readback.v1",
        "status": "verified",
        "native_glare_profile": profile,
        "cameras": {
            "CAM_FRONT": {"attributes": camera},
            "bev": {"attributes": bev},
        },
        "weather": weather,
    }
    (root / "render_condition_readback.json").write_text(json.dumps(readback))
    (root / "manifest.json").write_text(json.dumps({
        "pilot_route_index": "151",
        "pilot_condition": "clean_off",
        "carla_quality_level": "Epic",
        "orion_closedloop_risk_mode": "off",
        "orion_closedloop_uq_mode": "none",
        "orion_planning_response_mode": "off",
        "orion_closedloop_corruption": "",
        "orion_effective_conditioning": "none",
        "route_sha256": "same-route-hash",
        "base_checkpoint_path": "/assets/checkpoints/Orion.pth",
        "render_condition": {
            "kind": "carla_native_low_sun_glare",
            "native_glare_profile": profile,
            "actual_readback": {
                "status": "verified",
                "schema": "orion.closedloop_render_condition_readback.v1",
                "path": "render_condition_readback.json",
            },
        },
    }))
    (root / "eval_orion_traj_0.json").write_text(json.dumps(_terminal_eval()))
    rows = []
    for step in range(50):
        progress = step / 100.0
        ttc = 0.7 if degraded and 10 <= step <= 40 else 1.0
        rows.append({
            "step": step,
            "sim_time_seconds": step * 0.05,
            "route_progress": progress,
            "speed": 2.0,
            "closedloop_safety": {
                "schema": "orion.closedloop_dynamic_actor_safety.v2",
                "available": True,
                "min_obb_collision_ttc_seconds": ttc,
                "min_obb_separating_axis_gap_m": 1.0,
                "min_disc_clearance_m": 0.5,
                "critical_actor": {"actor_id": 7, "category": "walker"},
                "actors": [{
                    "actor_id": 7,
                    "category": "walker",
                    "obb_collision_ttc_seconds": ttc,
                    "obb_separating_axis_gap_m": 1.0,
                }],
            },
        })
    (scenario / "control_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    for frame in (1, 2, 3):
        Image.new("RGB", (640, 640), color=(pixel, pixel, pixel)).save(
            tensor_dir / ("%04d.png" % frame)
        )


def _specs(tmp_path: Path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "first_pair": {
            "route": 151,
            "event_window": {
                "start_route_progress": 0.1,
                "end_route_progress": 0.3,
                "recovery_seconds": 0.2,
            },
            "exact_model_tensor_gate": {
                "minimum_event_pairs": 1,
                "minimum_median_mean_absolute_delta_8bit": 1.0,
            },
        },
        "claim_boundary": "fixture",
    }))
    render = tmp_path / "render.json"
    render.write_text(json.dumps({
        "methods": {"carla_native_low_sun": {
            "camera_profiles": {
                "clean": {"lens_flare_intensity": 0.0, "bloom_intensity": 0.0},
                "medium": {"lens_flare_intensity": 0.75, "bloom_intensity": 1.5},
            },
            "weather_shared_by_all_profiles": {
                "sun_altitude_angle": 8.0,
                "sun_azimuth_angle": 180.0,
            },
        }},
    }))
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({
        "route_actor_category": {"151": "walker"},
        "surrogate_safety_margin_degradation": {"thresholds": {
            "minimum_ttc_drop_seconds_floor": 0.2,
            "minimum_ttc_drop_fraction_of_clean": 0.2,
            "minimum_ttc_le_2_exposure_increase_seconds": 0.0,
            "paired_recovery_seconds": 0.2,
        }},
    }))
    return protocol, render, gate


def test_native_glare_gate_requires_readback_and_exact_model_tensor_change(tmp_path):
    clean = tmp_path / "clean"
    degraded = tmp_path / "degraded"
    clean.mkdir()
    degraded.mkdir()
    _make_run(clean, "clean", 64)
    _make_run(degraded, "medium", 80, degraded=True)
    (clean / "clean_safety_gate.json").write_text(json.dumps({
        "gate_passed": True,
    }))
    protocol, render, gate = _specs(tmp_path)

    report = evaluate(
        protocol_path=protocol,
        render_protocol_path=render,
        gate_path=gate,
        clean_run=clean,
        degraded_run=degraded,
    )

    assert report["validity"]["valid"]
    assert report["exact_model_tensor"]["passed"]
    assert report["exact_model_tensor"]["median_mean_absolute_delta_8bit"] == 16.0
    assert report["decision"]["failure_induction_pass"]
    assert report["decision"]["evidence_tier"] == (
        "near_miss_surrogate_failure_induction"
    )


def test_native_glare_gate_fails_closed_on_weather_mismatch(tmp_path):
    clean = tmp_path / "clean"
    degraded = tmp_path / "degraded"
    clean.mkdir()
    degraded.mkdir()
    _make_run(clean, "clean", 64)
    _make_run(degraded, "medium", 80, degraded=True)
    (clean / "clean_safety_gate.json").write_text(json.dumps({
        "gate_passed": True,
    }))
    readback = json.loads((degraded / "render_condition_readback.json").read_text())
    readback["weather"]["sun_altitude_angle"] = 9.0
    (degraded / "render_condition_readback.json").write_text(json.dumps(readback))
    protocol, render, gate = _specs(tmp_path)

    report = evaluate(
        protocol_path=protocol,
        render_protocol_path=render,
        gate_path=gate,
        clean_run=clean,
        degraded_run=degraded,
    )

    assert not report["validity"]["valid"]
    assert not report["validity"]["checks"]["degraded_render_readback"]
    assert report["decision"]["evidence_tier"] == "invalid"

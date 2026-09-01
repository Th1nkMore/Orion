import json

from scripts.evaluate_corruption_hardcase_clean_qualification import evaluate


def _write_run(
    tmp_path, *, collision=False, stop_frames=0, phase="q1", wave2=False
):
    config_dir = tmp_path / "configs" / "scenario_factory"
    config_dir.mkdir(parents=True)
    protocol = config_dir / "protocol.json"
    protocol.write_text(json.dumps({
        "schema": "orion.corruption_hardcase_wave1_clean_qualification.v1",
        "selection": {"routes": [158]},
        "qualification_protocol": {
            "q1": {"run_id": "corruption_hardcase_wave1_clean_q1_v1"},
        },
    }))
    selected_protocol = protocol
    run_id = "corruption_hardcase_wave1_clean_q1_v1"
    exact_speedometer = None
    if wave2:
        import hashlib

        base = config_dir / "wave2_prereg.json"
        base.write_text(json.dumps({
            "schema": (
                "orion.corruption_hardcase_wave2_clean_qualification_"
                "preregistration.v1"
            ),
            "selection": {"routes": [158]},
            "qualification_protocol": {
                "q1": {"run_id": "corruption_hardcase_wave2_clean_q1_v1"}
            },
        }))
        activation = config_dir / "wave2_q1_activation.json"
        activation.write_text(json.dumps({
            "schema": "orion.corruption_hardcase_wave2_clean_q1_activation.v1",
            "status": "authorized_after_user_resume",
            "base_prereg": {
                "path": str(base.relative_to(tmp_path)),
                "sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
            },
            "scope": {
                "routes": [158],
                "condition": "clean_off",
                "runs_per_route": 1,
                "run_id": "corruption_hardcase_wave2_clean_q1_v1",
            },
            "authorization": {
                "q1_clean_submission": True,
                "user_resume_recorded": True,
            },
        }))
        selected_protocol = activation
        run_id = "corruption_hardcase_wave2_clean_q1_v1"
        exact_speedometer = "1"
    if phase == "q2":
        import hashlib

        amendment_dir = config_dir / "amendments"
        amendment_dir.mkdir()
        q1_result = amendment_dir / "q1_result.json"
        q1_result.write_text(json.dumps({
            "schema": "orion.corruption_hardcase_wave1_clean_q1_result.v1",
            "decision": {"q2_exact_scope": [158]},
        }))
        activation = config_dir / "q2_activation.json"
        activation.write_text(json.dumps({
            "schema": "orion.corruption_hardcase_wave1_clean_q2_activation.v1",
            "base_protocol": {
                "path": str(protocol.relative_to(tmp_path)),
                "sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
            },
            "q1_result": {
                "path": str(q1_result.relative_to(tmp_path)),
                "sha256": hashlib.sha256(q1_result.read_bytes()).hexdigest(),
            },
            "scope": {
                "routes": [158],
                "condition": "clean_off",
                "runs_per_route": 1,
                "run_id": "corruption_hardcase_wave1_clean_q2_v1",
            },
        }))
        selected_protocol = activation
        run_id = "corruption_hardcase_wave1_clean_q2_v1"
    run = tmp_path / "run"
    trace_dir = run / "records_x" / "route"
    trace_dir.mkdir(parents=True)
    manifest_payload = {
        "pilot_route_index": "158",
        "pilot_run_id": run_id,
        "pilot_condition": "clean_off",
        "orion_closedloop_uq_mode": "none",
        "orion_closedloop_conditioning": "none",
        "orion_closedloop_risk_mode": "off",
        "orion_planning_response_mode": "off",
        "orion_stage2_spatial_uq_source": "disabled",
        "orion_enable_legacy_density_uq": "0",
    }
    if exact_speedometer is not None:
        manifest_payload["orion_exact_frame_speedometer"] = exact_speedometer
        manifest_payload["orion_sensor_queue_diagnostics"] = "1"
    (run / "manifest.json").write_text(json.dumps(manifest_payload))
    infractions = {
        key: [] for key in (
            "collisions_layout", "collisions_pedestrian", "collisions_vehicle",
            "red_light", "stop_infraction", "outside_route_lanes", "route_dev",
            "vehicle_blocked", "scenario_timeouts", "route_timeout",
        )
    }
    if collision:
        infractions["collisions_vehicle"] = ["collision"]
    (run / "eval_test.json").write_text(json.dumps({
        "entry_status": "Finished",
        "_checkpoint": {"records": [{
            "status": "Completed",
            "scores": {"score_route": 100.0},
            "infractions": infractions,
        }]},
    }))
    rows = []
    for step in range(220):
        rows.append({
            "step": step,
            "sim_time_seconds": step * 0.05,
            "route_progress": min(1.0, step / 219.0),
            "speed": 0.0 if step >= 220 - stop_frames else 2.0,
            "closedloop_safety": {
                "schema": "orion.closedloop_dynamic_actor_safety.v2",
                "available": True,
            },
        })
    (trace_dir / "control_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return selected_protocol, run


def test_q1_clean_qualification_passes_complete_run(tmp_path):
    protocol, run = _write_run(tmp_path, stop_frames=20)
    report = evaluate(
        run_dir=run, route_index=158, phase="q1", protocol_path=protocol
    )
    assert report["qualified_for_next_clean_repeat"]
    assert report["failed_checks"] == []


def test_q1_rejects_clean_collision(tmp_path):
    protocol, run = _write_run(tmp_path, collision=True)
    report = evaluate(
        run_dir=run, route_index=158, phase="q1", protocol_path=protocol
    )
    assert not report["qualified_for_next_clean_repeat"]
    assert report["failed_checks"] == ["zero_hard_infractions"]


def test_q1_rejects_stop_longer_than_eight_seconds(tmp_path):
    protocol, run = _write_run(tmp_path, stop_frames=180)
    report = evaluate(
        run_dir=run, route_index=158, phase="q1", protocol_path=protocol
    )
    assert not report["qualified_for_next_clean_repeat"]
    assert report["failed_checks"] == ["liveness_at_most_8s"]


def test_q2_clean_qualification_passes_hash_bound_activation(tmp_path):
    activation, run = _write_run(tmp_path, stop_frames=20, phase="q2")
    report = evaluate(
        run_dir=run, route_index=158, phase="q2", protocol_path=activation
    )
    assert report["passed_clean_qualification"]
    assert not report["qualified_for_next_clean_repeat"]
    assert report["qualified_for_corruption_screen"]
    assert report["failed_checks"] == []


def test_q2_rejects_run_id_from_q1(tmp_path):
    activation, run = _write_run(tmp_path, phase="q2")
    manifest = run / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["pilot_run_id"] = "corruption_hardcase_wave1_clean_q1_v1"
    manifest.write_text(json.dumps(payload))
    report = evaluate(
        run_dir=run, route_index=158, phase="q2", protocol_path=activation
    )
    assert not report["passed_clean_qualification"]
    assert report["failed_checks"] == ["run_id_matches_phase"]


def test_wave2_q1_requires_exact_frame_speedometer(tmp_path):
    activation, run = _write_run(tmp_path, stop_frames=20, wave2=True)
    report = evaluate(
        run_dir=run, route_index=158, phase="q1", protocol_path=activation
    )
    assert report["passed_clean_qualification"]
    assert report["exact_frame_speedometer_required"] is True

    manifest = run / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["orion_exact_frame_speedometer"] = "0"
    manifest.write_text(json.dumps(payload))
    report = evaluate(
        run_dir=run, route_index=158, phase="q1", protocol_path=activation
    )
    assert not report["passed_clean_qualification"]
    assert report["failed_checks"] == ["exact_frame_speedometer_enabled"]

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/scenario_factory/corruption_hardcase_native_motion_blur_revalidation_route151_v1.json"
RUNNER = ROOT / "scripts/run_native_motion_blur_clean_revalidation_route151.sh"
SUBMITTER = ROOT / "scripts/submit_native_motion_blur_clean_revalidation_route151.sh"


def test_protocol_is_minimal_clean_medium_visual_only():
    value = json.loads(PROTOCOL.read_text())
    assert [row["profile"] for row in value["conditions"]] == ["none", "medium"]
    assert value["capture_contract"]["fresh_carla_server_per_profile"] is True
    assert value["capture_contract"]["orion_loaded"] is False
    assert value["artifact_gate"]["required_profiles"] == ["none", "medium"]
    assert all(flag is False for flag in value["locks"].values())


def test_runner_uses_fresh_server_and_fail_closed_artifact_gates():
    source = RUNNER.read_text()
    assert "for profile in none medium" in source
    assert source.count("stop_carla") >= 3
    assert "MOTION_BLUR_CAPTURE_PROFILE=\"${profile}\"" in source
    assert "evaluate_clean_render_artifacts.py" in source
    assert "--gate-config \"${gate_config}\"" in source
    assert "BASE_CHECKPOINT_PATH=\"/dev/null\"" in source


def test_submitter_requests_small_non_orion_job():
    source = SUBMITTER.read_text()
    assert "--cpus-per-task=2" in source
    assert "--mem=16G" in source
    assert "--time=00:25:00" in source
    assert 'echo "ORION_LOAD=0"' in source

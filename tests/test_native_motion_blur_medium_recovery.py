import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "configs/scenario_factory/amendments/20260831_native_motion_blur_medium_recovery_prereg_v2.json"
RUNNER = ROOT / "scripts/run_native_motion_blur_medium_recovery_route151.sh"
SUBMITTER = ROOT / "scripts/submit_native_motion_blur_medium_recovery_route151.sh"
RENDERER = ROOT / "scripts/render_native_motion_blur_clean_revalidation.py"


def test_recovery_is_one_shot_medium_only_and_no_orion():
    value = json.loads(AMENDMENT.read_text())
    assert value["maximum_capture_attempts"] == 1
    assert value["orion_loaded"] is False
    assert value["prospective_condition"]["profile"] == "medium"
    assert value["locks"] == {
        "orion_screen": False,
        "profile_freeze": False,
        "closed_loop": False,
        "safety_claim": False,
    }


def test_runner_reuses_hashed_clean_and_captures_only_medium():
    source = RUNNER.read_text()
    assert "diagnostic_good_none.json" in source
    assert "calibration_audit_sha256" in source
    assert "evaluate_clean_render_artifacts.py" in source
    assert "png_manifest_sha256" in source
    assert "MOTION_BLUR_CAPTURE_PROFILE=medium" in source
    assert "for profile in none medium" not in source
    assert "BASE_CHECKPOINT_PATH=\"/dev/null\"" in source
    assert "MAXIMUM_SUBMISSIONS=1" in source
    assert 'if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]' in source


def test_renderer_supports_explicit_profile_roots():
    source = RENDERER.read_text()
    assert 'parser.add_argument("--none-root", type=Path)' in source
    assert 'parser.add_argument("--medium-root", type=Path)' in source
    assert '"profile_roots": {profile: str(path)' in source
    assert "import cv2" not in source
    assert "laplacian = (" in source


def test_submitter_is_small_bounded_job():
    source = SUBMITTER.read_text()
    assert "--cpus-per-task=2" in source
    assert "--mem=16G" in source
    assert "--time=00:15:00" in source
    assert 'echo "PROFILES=medium_only"' in source
    assert 'echo "MAXIMUM_SUBMISSIONS=1"' in source

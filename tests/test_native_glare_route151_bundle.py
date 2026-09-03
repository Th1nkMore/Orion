from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bundle_runs_frozen_pair_sequentially_without_stage2_or_governor():
    source = (ROOT / "scripts/run_native_glare_route151_bundle.sh").read_text()
    clean = source.index('run_profile clean "${clean_pilot_id}"')
    clean_gate = source.index('--run-dir "${clean_run}"', clean)
    medium = source.index('run_profile medium "${medium_pilot_id}"')
    decision = source.index('--protocol "${protocol}"', medium)
    assert clean < clean_gate < medium < decision
    assert '"original_orion_only": True' in source
    assert '"stage2l_checkpoint_loaded": False' in source
    assert '"density_uq": False' in source
    assert '"governor": False' in source
    assert "ORION_NATIVE_GLARE_PROFILE=${profile}" in source
    assert "ORION_CLOSEDLOOP_SAFETY_TELEMETRY=1" in source


def test_submission_is_one_full_memory_bundle_without_retry():
    source = (ROOT / "scripts/submit_native_glare_route151_bundle.sh").read_text()
    assert "--gres=gpu:1" in source
    assert "--cpus-per-task=2" in source
    assert "--mem=192G" in source
    assert "--time=08:00:00" in source
    assert '"automatic_retry": False' in source
    assert '"maximum_submissions": 1' in source
    assert "route203_native_glare_submission_allowed" not in source


def test_manifest_uses_frozen_compat_python_not_centos_system_python():
    runner = (ROOT / "scripts/run_closedloop_uq_pilot.sh").read_text()
    assert '"${COMPAT_PYTHON_BIN}" - "${OUTPUT_ROOT}/manifest.json"' in runner
    assert 'python3 - "${OUTPUT_ROOT}/manifest.json"' not in runner

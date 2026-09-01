from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v8_submitter_is_fixed_bounded_and_locked():
    source = (ROOT / "scripts/submit_stage2l_v8_route151_smoke.sh").read_text()
    assert "immutable launch amendment is absent" in source
    assert "--max-optimizer-steps 60" in source
    assert "--answer-batch-size 2" in source
    assert "--gres=gpu:1" in source
    assert "--cpus-per-task=2" in source
    assert "--mem=192G" in source
    assert "--time=08:00:00" in source
    assert "--exclude=gpu5" in source
    assert "refusing duplicate active v8 smoke submission" in source
    assert "submission_attestation.json" in source
    assert "route151_v8_objective_data_preflight_v5/preflight.json" in source
    assert "route151_v8_trainer_preflight_v3/preflight.json" in source
    assert "route151_v8_objective_data_preflight_v4/preflight.json" not in source
    assert "route151_v8_trainer_preflight_v2/preflight.json" not in source


def test_v8_attester_records_no_retry_claim_boundary():
    source = (
        ROOT / "scripts/write_stage2l_v8_submission_attestation.py"
    ).read_text()
    assert '"maximum_submissions": 1' in source
    assert '"maximum_optimizer_steps": 60' in source
    assert '"automatic_retry_or_extension": False' in source
    assert '"formal_training": False' in source
    assert '"stage2p_training": False' in source

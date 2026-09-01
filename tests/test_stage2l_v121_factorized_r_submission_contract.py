from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v121_submitter_is_single_hash_attested_r_only_job():
    text = (
        PROJECT_ROOT / "scripts/submit_stage2l_v121_factorized_r_smoke.sh"
    ).read_text()
    assert "s2l_v121_factr" in text
    assert "--cpus-per-task=2" in text
    assert "--mem=192G" in text
    assert "--gres=gpu:1" in text
    assert "--time=05:00:00" in text
    assert "refusing duplicate active v12.1 factorized-R submission" in text
    assert "cancelled unattested v12.1 factorized-R job" in text
    assert "write_stage2l_v121_factorized_r_submission_attestation.py" in text
    assert "--factorized-cpu-report" in text
    assert "--launch-amendment" in text
    assert "scancel" in text
    assert "--array" not in text
    assert "--u-tokenizer" not in text
    assert "--trajectory" not in text


def test_v121_attester_binds_scope_and_release_locks():
    text = (
        PROJECT_ROOT
        / "scripts/write_stage2l_v121_factorized_r_submission_attestation.py"
    ).read_text()
    assert 'maximum_submissions") != 1' in text
    assert 'automatic_retry") is not False' in text
    assert 'maximum_optimizer_steps") != 40' in text
    assert 'bounded_r_only_smoke_allowed") is not True' in text
    for lock in (
        "stage1_uq_input",
        "u_tokenizer",
        "language_training",
        "trajectory_or_control",
        "formal_stage2l",
        "stage2p",
        "closed_loop",
        "locked_test_read",
    ):
        assert lock in text

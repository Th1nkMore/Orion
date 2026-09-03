from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v111_submitter_is_single_hash_attested_replacement():
    text = (
        PROJECT_ROOT / "scripts/submit_stage2l_v111_identifiable_smoke.sh"
    ).read_text()
    assert "stage2l_expanded_coverage_17event_v11_1_consumer_grid_v1" in text
    assert "stage2l_v11_1_bounded_identifiability_smoke_protocol_v1.json" in text
    assert "s2l_v111_ident" in text
    assert "--cpus-per-task=2" in text
    assert "--mem=192G" in text
    assert "--gres=gpu:1" in text
    assert "refusing duplicate active v11.1 submission" in text
    assert "cancelled unattested v11.1 job" in text
    assert "write_stage2l_v111_submission_attestation.py" in text
    assert "--consumer-grid-audit" in text
    assert "--dataset-upgrade-report" in text
    assert "scancel" in text
    assert "--array" not in text


def test_v111_attester_binds_consumer_grid_evidence_and_locks():
    text = (
        PROJECT_ROOT / "scripts/write_stage2l_v111_submission_attestation.py"
    ).read_text()
    assert "consumer_grid_audit_sha256" in text
    assert "dataset_upgrade_report_sha256" in text
    assert 'maximum_submissions") != 1' in text
    assert 'automatic_retry") is not False' in text
    assert 'formal_stage2l_allowed") is not False' in text
    assert 'stage2p_allowed") is not False' in text
    assert 'closed_loop_allowed") is not False' in text

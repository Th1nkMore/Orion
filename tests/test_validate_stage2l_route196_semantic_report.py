import json

import pytest

from scripts.scenario_factory_lib import sha256_file
from scripts.validate_stage2l_route196_semantic_report import validate_report


SOURCE_PATHS = {
    "trainer_sha256": "scripts/train_stage2l_route196_semantic_smoke.py",
    "bridge_trainer_helpers_sha256": "scripts/train_stage2l_route196_bridge_smoke.py",
    "uq_relevance_bridge_sha256": "uq_estimator/uq_relevance_tokenizer.py",
    "two_pass_runtime_sha256": "uq_estimator/stage2l_bridge_runtime.py",
    "semantic_bottleneck_sha256": "uq_estimator/stage2l_semantic_bottleneck.py",
    "semantic_runtime_sha256": "uq_estimator/stage2l_semantic_runtime.py",
    "submit_wrapper_sha256": "scripts/submit_stage2l_route196_semantic_smoke.sh",
}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        path.write_bytes(value)
    return path


def _fixture(tmp_path, *, failed=False):
    root = tmp_path / "project"
    protocol = _write(
        root / "configs/protocol.json",
        {"schema": "orion.stage2l_uq_language_grounding_protocol.v1"},
    )
    source_hashes = {"training_protocol_sha256": sha256_file(protocol)}
    for name, relative in SOURCE_PATHS.items():
        path = _write(root / relative, (name + "\n").encode())
        source_hashes[name] = sha256_file(path)
    amendment_hash = "a" * 64
    source_hashes["launch_amendment_sha256"] = amendment_hash
    data_hashes = {
        "records_sha256": "1" * 64,
        "visual_cache_sha256": "2" * 64,
        "base_orion_checkpoint_sha256": "3" * 64,
    }
    attestation = _write(
        tmp_path / "attestation.json",
        {
            "schema": "orion.stage2l_submission_attestation.v1",
            "authorized_run": {"steps": 240},
            "source_hashes": source_hashes,
            "data_hashes": data_hashes,
        },
    )
    checkpoint = _write(tmp_path / "semantic.pt", b"checkpoint")
    checks = {
        "optimization_reduces_language_nll": True,
        "optimization_reduces_relevance_bce": True,
        "optimization_reduces_structured_stance_ce": True,
        "all_structured_stances_match_targets": True,
        "structured_target_probability_floor": not failed,
        "on_path_risk_exceeds_off_path_by_margin": True,
        "zero_uq_prefers_maintain": True,
        "off_path_prefers_maintain": True,
        "on_path_prefers_conservative": True,
        "all_generated_stances_match_targets": True,
    }
    passed = not failed
    report = _write(
        tmp_path / "report.json",
        {
            "schema": "orion.stage2l_route196_structured_semantic_smoke.v1",
            "status": (
                "engineering_structured_semantic_overfit_pass"
                if passed
                else "engineering_structured_semantic_overfit_failed_gate"
            ),
            "steps": 240,
            "stage2l_pilot_training_ready": False,
            "stage2l_pilot_migration_review_ready": passed,
            "architecture": {
                "ground_truth_stance_enters_forward": False,
                "semantic_token_uses_predicted_distribution": True,
                "trajectory": False,
                "direct_control": False,
                "legacy_density_uq_used": False,
                "hard_governor_used": False,
            },
            "before": {
                "mean_language_nll": 2.0,
                "relevance_bce": 0.5,
                "mean_structured_stance_ce": 1.0,
            },
            "after": {
                "mean_language_nll": 1.0,
                "relevance_bce": 0.1,
                "mean_structured_stance_ce": 0.2,
                "structured_stance_accuracy": 1.0,
                "minimum_structured_target_probability": 0.4 if failed else 0.9,
                "on_minus_off_peak_task_risk": 0.3,
                "generated_stance_accuracy": 1.0,
                "expected_answer_nll": {
                    "zero_uq": 1.0,
                    "off_path_uq": 1.0,
                    "on_path_uq": 1.0,
                },
                "counterfactual_answer_nll": {
                    "zero_uq": 2.0,
                    "off_path_uq": 2.0,
                    "on_path_uq": 2.0,
                },
            },
            "checks": checks,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
            },
            "inputs": {
                "training_protocol": {"sha256": sha256_file(protocol)},
                "launch_amendment": {"sha256": amendment_hash},
                "records": {"sha256": data_hashes["records_sha256"]},
                "visual_cache": {"sha256": data_hashes["visual_cache_sha256"]},
                "base_orion_checkpoint": {
                    "sha256": data_hashes["base_orion_checkpoint_sha256"]
                },
            },
        },
    )
    return root, protocol, attestation, report


@pytest.mark.parametrize(
    "failed,expected_status",
    [(False, "validated_pass"), (True, "validated_failed_gate")],
)
def test_independently_validates_pass_and_honest_failed_gate(
    tmp_path, failed, expected_status
):
    root, protocol, attestation, report = _fixture(tmp_path, failed=failed)
    result = validate_report(
        report_path=report,
        attestation_path=attestation,
        protocol_path=protocol,
        project_root=root,
    )
    assert result["integrity_valid"] is True
    assert result["status"] == expected_status
    assert result["smoke_passed"] is (not failed)
    assert result["failed_checks"] == (
        [] if not failed else ["structured_target_probability_floor"]
    )


def test_rejects_trainer_check_that_disagrees_with_raw_metrics(tmp_path):
    root, protocol, attestation, report = _fixture(tmp_path)
    value = json.loads(report.read_text())
    value["checks"]["all_generated_stances_match_targets"] = False
    report.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="independent recomputation"):
        validate_report(
            report_path=report,
            attestation_path=attestation,
            protocol_path=protocol,
            project_root=root,
        )


def test_rejects_source_changed_after_submission(tmp_path):
    root, protocol, attestation, report = _fixture(tmp_path)
    (root / SOURCE_PATHS["semantic_runtime_sha256"]).write_text("changed")
    with pytest.raises(ValueError, match="attested source hash mismatch"):
        validate_report(
            report_path=report,
            attestation_path=attestation,
            protocol_path=protocol,
            project_root=root,
        )

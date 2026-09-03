import json

import pytest

from scripts.scenario_factory_lib import sha256_file
from scripts.validate_stage2l_v6_route151_smoke import (
    _recompute_checks,
    validate_report,
)


SOURCE_PATHS = {
    "submitter": "scripts/submit_stage2l_v6_route151_smoke.sh",
    "trainer": "scripts/train_stage2l_v6_route151_smoke.py",
    "semantic_bottleneck": "uq_estimator/stage2l_semantic_bottleneck_v2.py",
    "semantic_runtime": "uq_estimator/stage2l_semantic_runtime_v2.py",
    "matched_objective": "uq_estimator/stage2l_matched_objective.py",
    "training_protocol": "configs/scenario_factory/stage2l_training_v6_matched_magnitude_cross_family.json",
    "launch_amendment": "configs/scenario_factory/amendments/20260830_stage2l_v6_route151_matched_smoke_v1.json",
    "trainer_test": "tests/test_train_stage2l_v6_route151_smoke.py",
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
        root / SOURCE_PATHS["training_protocol"],
        {
            "schema": "orion.stage2l_uq_language_grounding_protocol.v2",
            "launch_locks": {"stage2l_pilot_training_allowed": False},
        },
    )
    amendment = _write(
        root / SOURCE_PATHS["launch_amendment"],
        {"schema": "orion.scenario_factory.amendment.v1"},
    )
    source_hashes = {}
    for name, relative in SOURCE_PATHS.items():
        path = root / relative
        if not path.exists():
            _write(path, (name + "\n").encode("utf-8"))
        source_hashes[name] = sha256_file(path)
    source_hashes.update(
        {
            "records": "1" * 64,
            "visual_cache": "2" * 64,
            "base_orion_checkpoint": "3" * 64,
        }
    )
    checkpoint = _write(tmp_path / "smoke.pt", b"diagnostic checkpoint")
    history = [
        {
            "optimizer_step": step,
            "group_id": "group_%d" % ((step - 1) % 5),
            "records_in_optimizer_unit": 20,
            "hard_language_anchors": 18,
            "optimizer_steps_inside_group": 0,
            "gradient_norm_before_clip": 1.0,
            "loss": 1.0,
        }
        for step in range(1, 21)
    ]
    report_value = {
        "schema": "orion.stage2l_v6_matched_group_smoke.v1",
        "engineering_smoke_only": True,
        "formal_training_ready": False,
        "stage2l_pilot_training_ready": False,
        "stage2p_ready": False,
        "event_id": "route151_step218",
        "optimizer_steps": 20,
        "record_equivalent_presentations": 400,
        "language_anchor_presentations": 360,
        "matched_record_audit": {
            "record_count": 100,
            "matched_group_count": 5,
            "optimizer_step_count_per_epoch": 5,
            "optimizer_steps_inside_group": 0,
            "hard_language_record_count": 90,
            "hard_stance_record_count": 15,
            "diagnostic_only_driving_record_count": 10,
            "cross_family_pairwise_comparison_count": 270,
        },
        "before": {
            "first_group_mean_hard_language_nll": 2.0,
            "mean_relevance_bce": 0.8,
            "first_group_cross_family_margin_pass_fraction": 0.1,
        },
        "after": {
            "first_group_mean_hard_language_nll": 1.0,
            "mean_relevance_bce": 0.1,
            "mean_on_minus_off_peak_task_risk": 0.3,
            "hard_stance_accuracy": 0.5 if failed else 1.0,
            "first_group_cross_family_margin_pass_fraction": 0.9,
        },
        "history": history,
        "loss_weights": {"trajectory": 0.0, "direct_control": 0.0},
        "architecture": {
            "raw_k_magnitude_side_channel": True,
            "magnitude_layer_norm_before_classifier": False,
            "one_optimizer_step_per_20_record_group": True,
            "observed_driving_is_diagnostic_only": True,
            "view_shuffled_driving_is_diagnostic_only": True,
            "ground_truth_stance_enters_forward": False,
            "legacy_density_uq_used": False,
            "hard_governor_used": False,
        },
        "provenance": {
            "records": {"sha256": source_hashes["records"]},
            "visual_cache": {"sha256": source_hashes["visual_cache"]},
            "training_protocol": {"sha256": source_hashes["training_protocol"]},
            "launch_amendment": {"sha256": source_hashes["launch_amendment"]},
            "trainer": {"sha256": source_hashes["trainer"]},
            "base_orion_checkpoint": {
                "sha256": source_hashes["base_orion_checkpoint"]
            },
            "checkpoint": {"sha256": sha256_file(checkpoint)},
        },
    }
    report_value["checks"] = _recompute_checks(report_value)
    passed = all(report_value["checks"].values())
    report_value["status"] = (
        "engineering_v6_matched_smoke_pass"
        if passed
        else "engineering_v6_matched_smoke_failed_gate"
    )
    report = _write(tmp_path / "report.json", report_value)
    attestation = _write(
        tmp_path / "submission_attestation.json",
        {
            "schema": "orion.stage2l_smoke_submission_attestation.v1",
            "training_bounds": {
                "maximum_submissions": 1,
                "maximum_optimizer_steps": 20,
                "formal_training": False,
                "stage2p_training": False,
                "trajectory_or_control_loss": False,
            },
            "source_sha256": source_hashes,
        },
    )
    return root, protocol, amendment, checkpoint, report, attestation


@pytest.mark.parametrize(
    "failed,expected",
    [(False, "validated_pass"), (True, "validated_failed_gate")],
)
def test_independently_validates_honest_pass_and_failed_gate(
    tmp_path, failed, expected
):
    root, protocol, amendment, checkpoint, report, attestation = _fixture(
        tmp_path, failed=failed
    )
    result = validate_report(
        report_path=report,
        checkpoint_path=checkpoint,
        attestation_path=attestation,
        protocol_path=protocol,
        amendment_path=amendment,
        project_root=root,
    )
    assert result["integrity_valid"] is True
    assert result["status"] == expected
    assert result["smoke_passed"] is (not failed)


def test_rejects_nonfinite_metrics(tmp_path):
    root, protocol, amendment, checkpoint, report, attestation = _fixture(tmp_path)
    value = json.loads(report.read_text())
    value["history"][0]["loss"] = float("nan")
    report.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        validate_report(
            report_path=report,
            checkpoint_path=checkpoint,
            attestation_path=attestation,
            protocol_path=protocol,
            amendment_path=amendment,
            project_root=root,
        )


def test_rejects_source_changed_after_submission(tmp_path):
    root, protocol, amendment, checkpoint, report, attestation = _fixture(tmp_path)
    (root / SOURCE_PATHS["semantic_runtime"]).write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="attested source hash mismatch"):
        validate_report(
            report_path=report,
            checkpoint_path=checkpoint,
            attestation_path=attestation,
            protocol_path=protocol,
            amendment_path=amendment,
            project_root=root,
        )

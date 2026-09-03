import json

import pytest

from scripts.scenario_factory_lib import sha256_file
from scripts.validate_stage2l_v7_route151_smoke import (
    _recompute_checks,
    validate_report,
)


SOURCE_PATHS = {
    "submitter": "scripts/submit_stage2l_v7_route151_smoke.sh",
    "trainer": "scripts/train_stage2l_v7_route151_smoke.py",
    "training_protocol": "configs/scenario_factory/stage2l_training_v7_calibrated_matched_semantics.json",
    "launch_amendment": "configs/scenario_factory/amendments/20260830_stage2l_v7_route151_calibrated_smoke_v1.json",
    "qa_factory_config": "configs/scenario_factory/qa_factory_v3_calibrated_semantics.json",
    "calibrated_objective": "uq_estimator/stage2l_calibrated_objective.py",
    "qa_contract": "uq_estimator/stage2l_qa_contract_v3.py",
    "semantic_bottleneck": "uq_estimator/stage2l_semantic_bottleneck_v2.py",
    "semantic_runtime": "uq_estimator/stage2l_semantic_runtime_v2.py",
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
    protocol_value = {
        "schema": "orion.stage2l_calibrated_training_protocol.v1",
        "launch_locks": {"stage2l_pilot_training_allowed": False},
        "losses": {
            "geometry_normalized_on_off_ranking": {
                "required_oracle_fraction": 0.8
            }
        },
    }
    qa_config_value = {
        "schema": "orion.uq_relevance_qa_factory_config.v3",
        "task_relevance_gates": {
            "minimum_foreground_recall": 0.95,
            "maximum_background_false_positive_rate": 0.05,
        },
        "stance_gates": {
            "minimum_per_variant_accuracy": 1.0,
            "minimum_per_target_class_recall": 1.0,
        },
        "generation_gates": {
            "minimum_hard_stance_target_probability": 0.5,
            "family_tag_accuracy": 1.0,
            "hard_driving_stance_parse_rate": 1.0,
            "hard_driving_structured_stance_agreement": 1.0,
        },
    }
    protocol = _write(root / SOURCE_PATHS["training_protocol"], protocol_value)
    qa_config = _write(root / SOURCE_PATHS["qa_factory_config"], qa_config_value)
    amendment = _write(
        root / SOURCE_PATHS["launch_amendment"],
        {
            "schema": "orion.scenario_factory.amendment.v1",
            "launch_locks": {"stage2l_pilot_training_allowed": False},
        },
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
            "qa_static_audit": "2" * 64,
            "visual_cache": "3" * 64,
            "objective_diagnostic": "4" * 64,
            "orion_config": "5" * 64,
            "base_orion_checkpoint": "6" * 64,
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
        "schema": "orion.stage2l_v7_calibrated_matched_smoke.v1",
        "engineering_smoke_only": True,
        "formal_training_ready": False,
        "stage2l_pilot_training_ready": False,
        "stage2p_ready": False,
        "event_id": "route151_step218",
        "optimizer_steps": 20,
        "record_equivalent_presentations": 400,
        "qa_audit": {
            "passed": True,
            "record_count": 100,
            "matched_group_count": 5,
            "hard_language_record_count": 90,
            "hard_stance_record_count": 15,
            "same_family_preference_anchor_count": 90,
            "distinct_counterfactual_negative_count": 270,
            "checks": {"static": True},
        },
        "before": {
            "first_group_mean_hard_language_nll": 2.0,
            "first_group_same_family_margin_pass_fraction": 0.2,
        },
        "after": {
            "first_group_mean_hard_language_nll": 1.0,
            "first_group_same_family_margin_pass_fraction": 0.9,
            "relevance_support": {
                "foreground_recall": 1.0,
                "background_false_positive_rate": 0.01,
            },
            "ranking": {
                "positive_order_fraction": 1.0,
                "minimum_attained_fraction": 0.9,
            },
            "stance": {
                "per_variant_accuracy": {
                    "zero_uq": 1.0,
                    "off_path_uq": 1.0,
                    "on_path_uq": 0.0 if failed else 1.0,
                },
                "per_target_class_recall": {
                    "maintain": 1.0,
                    "caution": 1.0,
                    "prepare_to_yield": 0.0 if failed else 1.0,
                },
                "minimum_target_probability": 0.8,
            },
            "generation_contract": {
                "family_tag_parse_and_accuracy": 1.0,
                "nonrepeating_text_fraction": 1.0,
                "hard_driving_stance_parse_rate": 1.0,
                "hard_driving_stance_agreement": 1.0,
            },
        },
        "history": history,
        "architecture": {
            "stage1_adapter_frozen_and_task_agnostic": True,
            "task_relevance_owned_by_vlm": True,
            "raw_k_magnitude_side_channel": True,
            "one_optimizer_step_per_20_record_group": True,
            "cross_family_answers_used_as_negatives": False,
            "exact_duplicate_answers_used_as_negatives": False,
            "overall_stance_accuracy_used_as_gate": False,
            "ground_truth_stance_enters_forward": False,
            "legacy_density_uq_used": False,
            "hard_governor_used": False,
            "trajectory_or_control_loss": False,
        },
        "provenance": {
            "records": {"sha256": source_hashes["records"]},
            "visual_cache": {"sha256": source_hashes["visual_cache"]},
            "qa_config": {"sha256": source_hashes["qa_factory_config"]},
            "training_protocol": {"sha256": source_hashes["training_protocol"]},
            "objective_diagnostic": {
                "sha256": source_hashes["objective_diagnostic"]
            },
            "launch_amendment": {"sha256": source_hashes["launch_amendment"]},
            "trainer": {"sha256": source_hashes["trainer"]},
            "base_orion_checkpoint": {
                "sha256": source_hashes["base_orion_checkpoint"]
            },
            "checkpoint": {"sha256": sha256_file(checkpoint)},
        },
    }
    report_value["checks"] = _recompute_checks(
        report_value, protocol_value, qa_config_value
    )
    passed = all(report_value["checks"].values())
    report_value["status"] = (
        "engineering_v7_calibrated_smoke_pass"
        if passed
        else "engineering_v7_calibrated_smoke_failed_gate"
    )
    report = _write(tmp_path / "report.json", report_value)
    attestation = _write(
        tmp_path / "submission_attestation.json",
        {
            "schema": "orion.stage2l_v7_smoke_submission_attestation.v1",
            "training_bounds": {
                "maximum_submissions": 1,
                "maximum_optimizer_steps": 20,
                "formal_training": False,
                "stage2p_training": False,
                "trajectory_or_control_loss": False,
                "automatic_retry_or_extension": False,
            },
            "source_sha256": source_hashes,
        },
    )
    return root, protocol, qa_config, amendment, checkpoint, report, attestation


@pytest.mark.parametrize(
    "failed,expected",
    [(False, "validated_pass"), (True, "validated_failed_gate")],
)
def test_independently_validates_honest_v7_pass_and_failure(
    tmp_path, failed, expected
):
    root, protocol, qa_config, amendment, checkpoint, report, attestation = _fixture(
        tmp_path, failed=failed
    )
    result = validate_report(
        report_path=report,
        checkpoint_path=checkpoint,
        attestation_path=attestation,
        protocol_path=protocol,
        qa_config_path=qa_config,
        amendment_path=amendment,
        project_root=root,
    )
    assert result["integrity_valid"] is True
    assert result["status"] == expected
    assert result["smoke_passed"] is (not failed)


def test_rejects_nonfinite_v7_metrics(tmp_path):
    root, protocol, qa_config, amendment, checkpoint, report, attestation = _fixture(
        tmp_path
    )
    value = json.loads(report.read_text())
    value["history"][0]["loss"] = float("nan")
    report.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        validate_report(
            report_path=report,
            checkpoint_path=checkpoint,
            attestation_path=attestation,
            protocol_path=protocol,
            qa_config_path=qa_config,
            amendment_path=amendment,
            project_root=root,
        )


def test_rejects_v7_source_changed_after_submission(tmp_path):
    root, protocol, qa_config, amendment, checkpoint, report, attestation = _fixture(
        tmp_path
    )
    (root / SOURCE_PATHS["qa_contract"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="attested source hash mismatch"):
        validate_report(
            report_path=report,
            checkpoint_path=checkpoint,
            attestation_path=attestation,
            protocol_path=protocol,
            qa_config_path=qa_config,
            amendment_path=amendment,
            project_root=root,
        )


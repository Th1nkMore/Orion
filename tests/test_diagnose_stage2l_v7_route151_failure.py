import copy

import pytest

from scripts.diagnose_stage2l_v7_route151_failure import diagnose


def _fixtures():
    checks = {
        "all_groups_positive_on_off_order": True,
        "all_groups_attain_oracle_fraction": False,
        "relevance_foreground_recall": True,
    }
    report = {
        "schema": "orion.stage2l_v7_calibrated_matched_smoke.v1",
        "status": "engineering_v7_calibrated_smoke_failed_gate",
        "optimizer_steps": 20,
        "checks": checks,
        "before": {
            "first_group_mean_hard_language_nll": 12.5,
            "first_group_same_family_margin_pass_fraction": 0.2,
            "stance": {"balanced_accuracy": 1 / 3},
        },
        "after": {
            "first_group_mean_hard_language_nll": 5.7,
            "first_group_same_family_margin_pass_fraction": 0.4,
            "ranking": {"attained_fraction": [1.2, 1.1, 0.65, 0.9, 1.0]},
            "relevance_support": {
                "foreground_recall": 1.0,
                "background_false_positive_rate": 0.22,
            },
            "stance": {
                "balanced_accuracy": 2 / 3,
                "per_target_class_recall": {
                    "maintain": 1.0,
                    "caution": 0.0,
                    "prepare_to_yield": 1.0,
                },
                "minimum_target_probability": 0.23,
            },
            "generation_contract": {
                "family_tag_parse_and_accuracy": 0.0,
                "hard_driving_stance_parse_rate": 0.0,
                "nonrepeating_text_fraction": 0.5,
            },
        },
        "history": [
            {
                "group_id": "g0",
                "optimizer_step": 1,
                "language_nll": 8.0,
                "minimum_attained_fraction": 0.5,
                "foreground_balanced_relevance": 0.1,
                "class_balanced_stance": 1.0,
            },
            {
                "group_id": "g0",
                "optimizer_step": 2,
                "language_nll": 7.0,
                "minimum_attained_fraction": 0.4,
                "foreground_balanced_relevance": 0.2,
                "class_balanced_stance": 0.8,
            },
        ],
    }
    validation = {
        "schema": "orion.stage2l_v7_route151_independent_validation.v1",
        "status": "validated_failed_gate",
        "integrity_valid": True,
        "smoke_passed": False,
        "checks": checks,
    }
    tokenizer = {
        "family_markers": {"<x>": {"token_count": 7}}
    }
    parameters = {"total_saved_trainable_parameters": 23_640_360}
    return report, validation, tokenizer, parameters


def test_diagnosis_separates_partial_learning_from_capacity_claim():
    report, validation, tokenizer, parameters = _fixtures()
    result = diagnose(
        report=report,
        validation=validation,
        tokenizer_audit=tokenizer,
        parameter_audit=parameters,
    )
    structural = result["findings"]["partial_structural_learnability"]
    assert structural["ranking_groups_at_or_above_0_8"] == 4
    assert structural["ranking_group_count"] == 5
    capacity = result["findings"]["capacity_interpretation"]
    assert capacity["insufficient_capacity_supported_by_this_smoke"] is False
    assert capacity["capacity_sufficiency_for_formal_training_proven"] is False


def test_diagnosis_records_within_group_interference():
    report, validation, tokenizer, parameters = _fixtures()
    result = diagnose(
        report=report,
        validation=validation,
        tokenizer_audit=tokenizer,
        parameter_audit=parameters,
    )
    regression = result["group_trajectories"]["g0"][
        "within_group_regression_observed"
    ]
    assert regression["attained_fraction"] is True
    assert regression["relevance_loss"] is True
    assert regression["stance_loss"] is False


def test_diagnosis_rejects_validator_disagreement():
    report, validation, tokenizer, parameters = _fixtures()
    validation = copy.deepcopy(validation)
    validation["checks"]["all_groups_attain_oracle_fraction"] = True
    with pytest.raises(ValueError, match="disagree"):
        diagnose(
            report=report,
            validation=validation,
            tokenizer_audit=tokenizer,
            parameter_audit=parameters,
        )

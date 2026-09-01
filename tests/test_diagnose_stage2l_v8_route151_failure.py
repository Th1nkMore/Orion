import copy

import pytest

from scripts.diagnose_stage2l_v8_route151_failure import diagnose


def _fixtures():
    checks = {
        "all_groups_attain_oracle_fraction": False,
        "language_nll_decreases": True,
        "generated_semantics_parse": False,
    }
    report = {
        "schema": "orion.stage2l_v8_gradient_routed_smoke.v1",
        "status": "engineering_v8_gradient_routed_smoke_failed_gate",
        "optimizer_steps": 60,
        "checks": checks,
        "before": {
            "first_group_mean_hard_language_nll": 14.5,
            "first_group_same_family_margin_pass_fraction": 0.13,
            "stance": {"balanced_accuracy": 1 / 3},
        },
        "after": {
            "first_group_mean_hard_language_nll": 1.55,
            "first_group_same_family_margin_pass_fraction": 0.49,
            "ranking": {
                "attained_fraction": [1.1, 0.88, 1.13, 0.38, 1.11],
                "positive_order_fraction": 1.0,
            },
            "relevance_support": {
                "foreground_recall": 1.0,
                "background_false_positive_rate": 0.16,
            },
            "stance": {
                "balanced_accuracy": 5 / 6,
                "per_target_class_recall": {
                    "maintain": 1.0,
                    "caution": 0.5,
                    "prepare_to_yield": 1.0,
                },
                "minimum_target_probability": 0.115,
            },
            "generation_semantics": {
                "semantic_parse_rate": 0.0,
                "semantic_field_accuracy": 0.0,
                "semantic_answer_exact_match": 0.0,
                "nonrepeating_text_fraction": 1 / 3,
            },
        },
        "history": [
            {
                "group_id": "g0",
                "optimizer_step": 1,
                "language_nll": 8.0,
                "minimum_attained_fraction": 0.5,
                "ranking_loss": 0.3,
                "foreground_balanced_relevance": 0.1,
                "dataset_frequency_balanced_stance": 1.0,
            },
            {
                "group_id": "g0",
                "optimizer_step": 2,
                "language_nll": 7.0,
                "minimum_attained_fraction": 0.4,
                "ranking_loss": 0.4,
                "foreground_balanced_relevance": 0.2,
                "dataset_frequency_balanced_stance": 0.8,
            },
        ],
    }
    validation = {
        "schema": "orion.stage2l_v8_route151_independent_validation.v1",
        "status": "validated_failed_gate",
        "integrity_valid": True,
        "smoke_passed": False,
        "checks": checks,
    }
    alignment = {
        "schema": "orion.stage2l_v8_generation_prompt_alignment.v1",
        "status": "alignment_pass",
        "checks": {"generation_prompt_is_training_prefix": True},
    }
    checkpoint = {
        "schema": "orion.stage2l_v8_checkpoint_tensor_audit.v1",
        "status": "finite_complete_checkpoint",
        "total_saved_trainable_parameters": 23_640_360,
        "sections": {"lora": {"all_finite": True}},
    }
    summary = {
        "observation_uncertainty": {
            "peak_score": 0.0,
            "peak_view": "CAM_FRONT",
            "peak_region": "upper_left",
        },
        "task_risk": {
            "level": "low",
            "peak_score": 0.0,
            "peak_view": "CAM_FRONT",
            "peak_region": "upper_left",
        },
    }
    records = [
        {
            "sample_id": "z0",
            "counterfactual": {"variant": "zero_uq", "group_id": "g0"},
            "question_family": "epistemic_limitation",
            "loss_policy": {"hard_language_target": True},
            "target": {"structured_summary": summary},
            "conversation": [
                {"value": "q"},
                {"value": "evidence=unreliable; hidden_content=unknown"},
            ],
        },
        {
            "sample_id": "z1",
            "counterfactual": {"variant": "zero_uq", "group_id": "g0"},
            "question_family": "task_relevance",
            "loss_policy": {"hard_language_target": True},
            "target": {"structured_summary": summary},
            "conversation": [{"value": "q"}, {"value": "a"}],
        },
    ]
    return report, validation, alignment, checkpoint, records


def test_v8_diagnosis_separates_alignment_from_generation_failure():
    result = diagnose(**_as_kwargs(_fixtures()))
    language = result["findings"]["language_result"]
    assert language["prompt_or_label_alignment_failure"] is False
    assert language["free_generation_semantics_learned"] is False
    assert result["decision"]["retry_or_epoch_extension_allowed"] is False


def test_v8_diagnosis_flags_zero_uq_semantic_contract():
    result = diagnose(**_as_kwargs(_fixtures()))
    labels = result["findings"]["label_contract"]
    assert labels["semantic_contract_self_consistent"] is False
    assert labels["zero_uq_epistemic_answers_claiming_unreliable_unknown"] == 1
    assert labels["zero_uq_task_risk_answers_with_arbitrary_non_none_location"] == 1


def test_v8_diagnosis_rejects_validator_disagreement():
    values = list(_fixtures())
    values[1] = copy.deepcopy(values[1])
    values[1]["checks"]["all_groups_attain_oracle_fraction"] = True
    with pytest.raises(ValueError, match="disagree"):
        diagnose(**_as_kwargs(tuple(values)))


def _as_kwargs(values):
    report, validation, alignment, checkpoint, records = values
    return {
        "report": report,
        "validation": validation,
        "prompt_alignment": alignment,
        "checkpoint_audit": checkpoint,
        "records": records,
    }

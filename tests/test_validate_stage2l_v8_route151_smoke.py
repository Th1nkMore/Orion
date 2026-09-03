import json

from scripts.validate_stage2l_v8_route151_smoke import (
    EXPECTED_CHECKS,
    _recompute_checks,
)


def _fixture(failed=False):
    protocol = {
        "losses": {"geometry_normalized_on_off_ranking": {"required_oracle_fraction": 0.8}},
        "gradient_ownership": {
            "qa_language_loss_to_relevance_logits": False,
            "qa_language_loss_to_stance_classifier": False,
        },
    }
    qa = {
        "task_relevance_gates": {"minimum_foreground_recall": 0.95, "maximum_background_false_positive_rate": 0.05},
        "stance_gates": {"minimum_per_variant_accuracy": 1.0, "minimum_per_target_class_recall": 1.0, "minimum_target_probability": 0.5},
        "generation_gates": {"semantic_parse_rate": 1.0, "semantic_field_accuracy": 1.0, "semantic_answer_exact_match": 1.0, "nonrepeating_text_fraction": 1.0},
    }
    report = {
        "optimizer_steps": 60,
        "history": [
            {"records_in_optimizer_unit": 20, "optimizer_steps_inside_group": 0}
            for _ in range(60)
        ],
        "before": {
            "first_group_mean_hard_language_nll": 2.0,
            "first_group_same_family_margin_pass_fraction": 0.2,
        },
        "after": {
            "first_group_mean_hard_language_nll": 1.0,
            "first_group_same_family_margin_pass_fraction": 0.9,
            "relevance_support": {"foreground_recall": 1.0, "background_false_positive_rate": 0.01},
            "ranking": {"positive_order_fraction": 1.0, "minimum_attained_fraction": 0.9},
            "stance": {
                "per_variant_accuracy": {"zero_uq": 1.0, "off_path_uq": 1.0, "on_path_uq": 0.0 if failed else 1.0},
                "per_target_class_recall": {"maintain": 1.0, "caution": 0.0 if failed else 1.0, "prepare_to_yield": 1.0},
                "minimum_target_probability": 0.8,
            },
            "generation_semantics": {
                "semantic_parse_rate": 1.0,
                "semantic_field_accuracy": 1.0,
                "semantic_answer_exact_match": 1.0,
                "nonrepeating_text_fraction": 1.0,
            },
        },
    }
    return report, protocol, qa


def test_v8_recomputation_covers_exact_frozen_gate_set():
    report, protocol, qa = _fixture()
    checks = _recompute_checks(report, protocol, qa)
    assert set(checks) == EXPECTED_CHECKS
    assert all(checks.values())


def test_v8_recomputation_preserves_honest_stance_failure():
    report, protocol, qa = _fixture(failed=True)
    checks = _recompute_checks(report, protocol, qa)
    assert checks["all_stance_variants_correct"] is False
    assert checks["all_stance_classes_recalled"] is False
    assert sum(not value for value in checks.values()) == 2

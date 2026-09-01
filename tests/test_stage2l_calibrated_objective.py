import pytest
import torch

from uq_estimator.stage2l_calibrated_objective import (
    FAMILY_RESPONSE_TAGS,
    canonical_tagged_answer,
    class_balanced_matched_stance_loss,
    counterfactual_answer_preference_loss,
    foreground_balanced_relevance_terms,
    generation_is_nonrepeating,
    geometry_normalized_task_risk_ranking_terms,
    matched_stance_metrics,
    parse_planning_stance,
    relevance_support_metrics,
    same_family_unique_counterfactual_answers,
    tagged_answer_family,
)


def _logit(probability):
    value = torch.as_tensor(probability, dtype=torch.float32)
    return torch.logit(value.clamp(1e-5, 1.0 - 1e-5))


def test_foreground_balanced_relevance_retains_soft_target_optimum():
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 1, 1] = 0.4
    target[0, 0, 1, 2] = 0.2
    exact_logits = _logit(target)
    collapsed_logits = torch.full_like(target, -9.0)

    exact = foreground_balanced_relevance_terms(exact_logits, target)
    collapsed = foreground_balanced_relevance_terms(collapsed_logits, target)
    assert exact.balanced_brier.item() < 1e-8
    assert collapsed.foreground_brier.item() > exact.foreground_brier.item()
    assert collapsed.loss.item() > exact.loss.item()
    assert relevance_support_metrics(exact_logits, target) == pytest.approx(
        {
            "foreground_recall": 1.0,
            "background_false_positive_rate": 0.0,
            "foreground_mean_probability": 0.3,
            "background_mean_probability": 1e-5,
            "foreground_background_probability_gap": 0.29999,
        },
        abs=2e-5,
    )


def test_foreground_balanced_relevance_backpropagates_foreground_and_background():
    target = torch.zeros(1, 1, 3, 3)
    target[0, 0, 1, 1] = 0.8
    logits = torch.zeros_like(target, requires_grad=True)
    terms = foreground_balanced_relevance_terms(logits, target)
    terms.loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0, 1, 1] < 0
    assert logits.grad[0, 0, 0, 0] > 0


def test_geometry_normalized_ranking_is_attainable_for_target_r():
    target = torch.zeros(2, 1, 2, 3)
    target[:, 0, 0, 1] = torch.tensor([0.25, 0.5])
    on_uq = torch.zeros_like(target)
    off_uq = torch.zeros_like(target)
    on_uq[:, 0, 0, 1] = 0.8
    off_uq[:, 0, 1, 2] = 0.9
    exact = geometry_normalized_task_risk_ranking_terms(
        on_uq,
        off_uq,
        _logit(target),
        target,
        required_oracle_fraction=0.8,
    )
    assert torch.allclose(exact.attained_fraction, torch.ones(2), atol=1e-4)
    assert exact.loss.item() == 0.0

    collapsed = geometry_normalized_task_risk_ranking_terms(
        on_uq,
        off_uq,
        torch.zeros_like(target),
        target,
        required_oracle_fraction=0.8,
    )
    assert collapsed.loss.item() > 0.0


def test_geometry_normalized_ranking_rejects_nonseparating_labels():
    target = torch.ones(1, 1, 2, 2) * 0.2
    uq = torch.ones_like(target) * 0.5
    with pytest.raises(ValueError, match="positive on/off-path oracle gap"):
        geometry_normalized_task_risk_ranking_terms(
            uq, uq, torch.zeros_like(target), target
        )


def test_class_balanced_stance_loss_defeats_majority_class_solution():
    targets = {
        "zero_uq": "maintain",
        "off_path_uq": "maintain",
        "on_path_uq": "prepare_to_yield",
    }
    all_maintain = {
        variant: torch.tensor([[5.0, 0.0, -5.0]], requires_grad=True)
        for variant in targets
    }
    correct = {
        "zero_uq": torch.tensor([[5.0, 0.0, -5.0]], requires_grad=True),
        "off_path_uq": torch.tensor([[5.0, 0.0, -5.0]], requires_grad=True),
        "on_path_uq": torch.tensor([[-5.0, 0.0, 5.0]], requires_grad=True),
    }
    majority_loss = class_balanced_matched_stance_loss(all_maintain, targets)
    correct_loss = class_balanced_matched_stance_loss(correct, targets)
    assert majority_loss.item() > correct_loss.item() + 4.0
    majority_loss.backward()
    assert all(value.grad is not None for value in all_maintain.values())


def test_class_balanced_stance_loss_supports_per_sample_targets():
    logits = {
        "zero_uq": torch.randn(2, 3, requires_grad=True),
        "off_path_uq": torch.randn(2, 3, requires_grad=True),
        "on_path_uq": torch.randn(2, 3, requires_grad=True),
    }
    targets = {
        "zero_uq": ["maintain", "maintain"],
        "off_path_uq": ["maintain", "maintain"],
        "on_path_uq": ["caution", "prepare_to_yield"],
    }
    loss = class_balanced_matched_stance_loss(logits, targets)
    assert loss.ndim == 0
    loss.backward()
    assert all(value.grad is not None for value in logits.values())
    with pytest.raises(ValueError, match="count does not match"):
        class_balanced_matched_stance_loss(
            logits, dict(targets, on_path_uq=["caution"])
        )


def test_stance_metrics_report_each_variant_instead_of_only_two_thirds():
    logits = {
        variant: [torch.tensor([[5.0, 0.0, -5.0]])]
        for variant in ("zero_uq", "off_path_uq", "on_path_uq")
    }
    targets = {
        "zero_uq": ["maintain"],
        "off_path_uq": ["maintain"],
        "on_path_uq": ["prepare_to_yield"],
    }
    metrics = matched_stance_metrics(logits, targets)
    assert metrics["per_variant_accuracy"] == {
        "zero_uq": 1.0,
        "off_path_uq": 1.0,
        "on_path_uq": 0.0,
    }
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["minimum_target_probability"] < 1e-3


def _row(variant, family, answer):
    return {
        "counterfactual": {"group_id": "event/frame", "variant": variant},
        "question_family": family,
        "conversation": [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": answer},
        ],
    }


def test_family_tags_are_unique_and_fix_the_v2_driving_prefix():
    assert len(set(FAMILY_RESPONSE_TAGS.values())) == 4
    answer = canonical_tagged_answer(
        "driving_implication",
        "<task_relevance_map> The uncertainty-aware planning stance is maintain.",
    )
    assert answer.startswith("<planning_stance>")
    assert tagged_answer_family(answer) == "driving_implication"
    with pytest.raises(ValueError, match="exactly one"):
        tagged_answer_family("The stance is maintain.")
    assert parse_planning_stance(answer) == "maintain"
    assert generation_is_nonrepeating(answer)
    assert not generation_is_nonrepeating("< go_ be go more go more go more.")


def test_same_family_preference_deduplicates_identical_maintain_answers():
    rows = [
        _row("zero_uq", "driving_implication", "The stance is maintain."),
        _row("off_path_uq", "driving_implication", "The stance is maintain."),
        _row("on_path_uq", "driving_implication", "The stance is prepare_to_yield."),
        _row("observed", "driving_implication", "The stance is maintain."),
    ]
    answers = same_family_unique_counterfactual_answers(rows, rows[0])
    assert len(answers) == 2
    assert answers[0].endswith("maintain.")
    assert answers[1].endswith("prepare_to_yield.")


def test_counterfactual_preference_accepts_variable_distinct_negative_counts():
    target = torch.tensor([0.1, 0.2], requires_grad=True)
    negatives = torch.tensor(
        [[0.6, 0.8], [0.7, 0.9]], requires_grad=True
    )
    assert counterfactual_answer_preference_loss(
        target, negatives, margin=0.2
    ).item() == 0.0
    weak_target = torch.tensor([1.0], requires_grad=True)
    strong_negative = torch.tensor([[0.1]], requires_grad=True)
    loss = counterfactual_answer_preference_loss(
        weak_target, strong_negative, margin=0.2
    )
    assert loss.item() > 0.0
    loss.backward()
    assert weak_target.grad is not None
    assert strong_negative.grad is not None

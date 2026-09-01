import pytest
import torch

from uq_estimator.stage2l_matched_objective import (
    HARD_STANCE_VARIANTS,
    MATCHED_VARIANTS,
    QUESTION_FAMILIES,
    audit_matched_training_records,
    cross_family_answer_preference_loss,
    hard_language_supervision_allowed,
    matched_stance_batch_loss,
    partition_complete_matched_groups,
    same_variant_cross_family_answers,
)


def test_matched_stance_loss_accumulates_all_hard_counterfactuals():
    logits = {
        variant: torch.randn(2, 3, requires_grad=True)
        for variant in HARD_STANCE_VARIANTS
    }
    targets = {
        "zero_uq": "maintain",
        "off_path_uq": "maintain",
        "on_path_uq": "prepare_to_yield",
    }
    loss = matched_stance_batch_loss(logits, targets)
    assert loss.ndim == 0
    loss.backward()
    assert all(
        value.grad is not None and value.grad.abs().sum() > 0
        for value in logits.values()
    )


def test_observed_and_shuffled_are_not_default_hard_stance_labels():
    assert "observed" not in HARD_STANCE_VARIANTS
    assert "view_shuffled_uq" not in HARD_STANCE_VARIANTS
    logits = {
        variant: torch.randn(1, 3)
        for variant in (*HARD_STANCE_VARIANTS, "observed")
    }
    targets = {
        "zero_uq": "maintain",
        "off_path_uq": "maintain",
        "on_path_uq": "prepare_to_yield",
        "observed": "maintain",
    }
    with pytest.raises(ValueError, match="observed"):
        matched_stance_batch_loss(
            logits,
            targets,
            supervised_variants=(*HARD_STANCE_VARIANTS, "observed"),
        )


def test_untrusted_driving_answers_are_diagnostic_only():
    for variant in ("observed", "view_shuffled_uq"):
        assert not hard_language_supervision_allowed(
            variant, "driving_implication"
        )
        assert hard_language_supervision_allowed(variant, "task_relevance")
    for variant in HARD_STANCE_VARIANTS:
        assert hard_language_supervision_allowed(variant, "driving_implication")
    with pytest.raises(ValueError, match="unsupported matched variant"):
        hard_language_supervision_allowed("random_uq", "driving_implication")


def test_incomplete_hard_stance_group_is_rejected():
    logits = {
        "zero_uq": torch.randn(1, 3),
        "off_path_uq": torch.randn(1, 3),
    }
    targets = {
        "zero_uq": "maintain",
        "off_path_uq": "maintain",
    }
    with pytest.raises(ValueError, match="incomplete"):
        matched_stance_batch_loss(logits, targets)


def test_cross_family_preference_compares_all_three_other_answers():
    target = torch.tensor([0.1, 0.2], requires_grad=True)
    negatives = torch.tensor(
        [[0.8, 0.9, 1.0], [0.7, 0.8, 0.9]], requires_grad=True
    )
    zero = cross_family_answer_preference_loss(target, negatives, margin=0.2)
    assert zero.item() == 0.0

    weak_target = torch.tensor([1.0], requires_grad=True)
    strong_negatives = torch.tensor(
        [[0.1, 0.2, 0.3]], requires_grad=True
    )
    loss = cross_family_answer_preference_loss(
        weak_target, strong_negatives, margin=0.2
    )
    assert loss.item() > 0
    loss.backward()
    assert weak_target.grad is not None and weak_target.grad.abs().sum() > 0
    assert strong_negatives.grad is not None
    assert strong_negatives.grad.abs().sum() > 0


def _record(family, answer):
    return {
        "counterfactual": {"group_id": "event/frame", "variant": "on_path_uq"},
        "question_family": family,
        "conversation": [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": answer},
        ],
    }


def _full_group(group_id="event/frame"):
    return [
        {
            "sample_id": "%s/%s/%s" % (group_id, variant, family),
            "event_id": "event",
            "counterfactual": {"group_id": group_id, "variant": variant},
            "question_family": family,
            "conversation": [
                {"from": "human", "value": "question"},
                {"from": "gpt", "value": "%s answer" % family},
            ],
        }
        for variant in MATCHED_VARIANTS
        for family in QUESTION_FAMILIES
    ]


def test_complete_group_partition_is_one_fail_closed_optimizer_unit():
    rows = list(reversed(_full_group()))
    groups = partition_complete_matched_groups(rows)
    assert len(groups) == 1
    assert len(groups[0]) == 20
    audit = audit_matched_training_records(rows)
    assert audit == {
        "record_count": 20,
        "matched_group_count": 1,
        "optimizer_step_count_per_epoch": 1,
        "optimizer_steps_inside_group": 0,
        "hard_language_record_count": 18,
        "hard_stance_record_count": 3,
        "diagnostic_only_driving_record_count": 2,
        "cross_family_pairwise_comparison_count": 54,
    }


def test_matched_partition_rejects_missing_duplicate_and_cross_event_rows():
    rows = _full_group()
    with pytest.raises(ValueError, match="exactly 5x4"):
        partition_complete_matched_groups(rows[:-1])
    duplicate = list(rows)
    duplicate[-1] = dict(duplicate[-2], sample_id="new-sample")
    with pytest.raises(ValueError, match="duplicates a pair|exactly 5x4"):
        partition_complete_matched_groups(duplicate)
    crossed = _full_group()
    crossed[-1] = dict(crossed[-1], event_id="other-event")
    with pytest.raises(ValueError, match="crosses event"):
        partition_complete_matched_groups(crossed)


def test_same_variant_cross_family_answers_are_complete_and_unique():
    rows = [
        _record("observation_semantics", "observation answer"),
        _record("epistemic_limitation", "epistemic answer"),
        _record("task_relevance", "relevance answer"),
        _record("driving_implication", "driving answer"),
    ]
    answers = same_variant_cross_family_answers(rows, rows[0])
    assert set(answers) == {
        "epistemic_limitation",
        "task_relevance",
        "driving_implication",
    }
    with pytest.raises(ValueError, match="duplicates"):
        same_variant_cross_family_answers(rows + [rows[1]], rows[0])
    with pytest.raises(ValueError, match="all cross-family"):
        same_variant_cross_family_answers(rows[:-1], rows[0])

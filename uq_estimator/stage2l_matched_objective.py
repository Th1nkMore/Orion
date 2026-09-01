"""Loss/data contracts for matched Stage2-L semantic training groups."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from uq_estimator.stage2l_semantic_bottleneck import (
    PLANNING_STANCES,
    encode_planning_stances,
)


QUESTION_FAMILIES = (
    "observation_semantics",
    "epistemic_limitation",
    "task_relevance",
    "driving_implication",
)
MATCHED_VARIANTS = (
    "observed",
    "zero_uq",
    "off_path_uq",
    "on_path_uq",
    "view_shuffled_uq",
)
HARD_STANCE_VARIANTS = (
    "zero_uq",
    "off_path_uq",
    "on_path_uq",
)
DIAGNOSTIC_STANCE_VARIANTS = ("view_shuffled_uq",)
UNLABELED_OBSERVED_VARIANT = "observed"


def hard_language_supervision_allowed(variant: str, question_family: str) -> bool:
    """Return whether a QA answer may enter hard causal-language loss.

    Observed and view-shuffled driving implications do not have independent
    action truth.  Their text remains available for diagnostics, but training
    on the generated stance sentence would silently reintroduce the same hard
    stance label that the structured loss excludes.
    """

    variant = str(variant)
    question_family = str(question_family)
    if variant not in MATCHED_VARIANTS:
        raise ValueError("unsupported matched variant: %s" % variant)
    if question_family not in QUESTION_FAMILIES:
        raise ValueError("unsupported QA family: %s" % question_family)
    return not (
        question_family == "driving_implication"
        and variant in (UNLABELED_OBSERVED_VARIANT, *DIAGNOSTIC_STANCE_VARIANTS)
    )


def partition_complete_matched_groups(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[Tuple[Mapping[str, Any], ...], ...]:
    """Return deterministic 20-record optimizer units or fail closed."""

    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    seen_sample_ids = set()
    for row in records:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in seen_sample_ids:
            raise ValueError("matched records require unique non-empty sample ids")
        seen_sample_ids.add(sample_id)
        group_id = str(row.get("counterfactual", {}).get("group_id", ""))
        if not group_id:
            raise ValueError("matched record lacks a counterfactual group id")
        grouped[group_id].append(row)
    if not grouped:
        raise ValueError("matched training records are empty")
    expected = {
        (variant, family)
        for variant in MATCHED_VARIANTS
        for family in QUESTION_FAMILIES
    }
    result = []
    for group_id in sorted(grouped):
        rows = grouped[group_id]
        pairs = [
            (
                str(row.get("counterfactual", {}).get("variant", "")),
                str(row.get("question_family", "")),
            )
            for row in rows
        ]
        if len(rows) != len(expected) or set(pairs) != expected:
            raise ValueError(
                "matched optimizer group is not exactly 5x4: %s" % group_id
            )
        if len(pairs) != len(set(pairs)):
            raise ValueError("matched optimizer group duplicates a pair: %s" % group_id)
        if len({str(row.get("event_id", "")) for row in rows}) != 1:
            raise ValueError("matched optimizer group crosses event boundaries")
        ordered = sorted(
            rows,
            key=lambda row: (
                MATCHED_VARIANTS.index(
                    str(row.get("counterfactual", {}).get("variant", ""))
                ),
                QUESTION_FAMILIES.index(str(row.get("question_family", ""))),
            ),
        )
        result.append(tuple(ordered))
    return tuple(result)


def audit_matched_training_records(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    """Count the exact v6 loss/optimizer contract without model execution."""

    groups = partition_complete_matched_groups(records)
    hard_language = 0
    hard_stance = 0
    diagnostic_driving = 0
    cross_family_comparisons = 0
    for group in groups:
        for row in group:
            variant = str(row["counterfactual"]["variant"])
            family = str(row["question_family"])
            if hard_language_supervision_allowed(variant, family):
                hard_language += 1
                cross_family_comparisons += len(QUESTION_FAMILIES) - 1
            elif family == "driving_implication":
                diagnostic_driving += 1
            if family == "driving_implication" and variant in HARD_STANCE_VARIANTS:
                hard_stance += 1
    return {
        "record_count": len(records),
        "matched_group_count": len(groups),
        "optimizer_step_count_per_epoch": len(groups),
        "optimizer_steps_inside_group": 0,
        "hard_language_record_count": hard_language,
        "hard_stance_record_count": hard_stance,
        "diagnostic_only_driving_record_count": diagnostic_driving,
        "cross_family_pairwise_comparison_count": cross_family_comparisons,
    }


def matched_stance_batch_loss(
    logits_by_variant: Mapping[str, torch.Tensor],
    target_stance_by_variant: Mapping[str, str],
    *,
    supervised_variants: Sequence[str] = HARD_STANCE_VARIANTS,
) -> torch.Tensor:
    """Average stance CE across a complete matched group before optimizer step."""

    variants = tuple(str(value) for value in supervised_variants)
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("supervised stance variants must be unique and non-empty")
    if UNLABELED_OBSERVED_VARIANT in variants:
        raise ValueError("observed adapter output may not receive hard stance CE")
    missing = [
        variant
        for variant in variants
        if variant not in logits_by_variant or variant not in target_stance_by_variant
    ]
    if missing:
        raise ValueError("matched stance group is incomplete: %s" % missing)
    logits = [logits_by_variant[variant] for variant in variants]
    if any(value.ndim != 2 or value.shape[-1] != len(PLANNING_STANCES) for value in logits):
        raise ValueError("each stance logit tensor must have shape [B,3]")
    if len({value.shape[0] for value in logits}) != 1:
        raise ValueError("matched stance variants use different batch sizes")
    stacked = torch.cat(logits, dim=0)
    targets = encode_planning_stances(
        [
            target_stance_by_variant[variant]
            for variant in variants
            for _ in range(logits_by_variant[variant].shape[0])
        ],
        device=stacked.device,
    )
    return F.cross_entropy(stacked, targets)


def cross_family_answer_preference_loss(
    target_nll: torch.Tensor,
    negative_family_nlls: torch.Tensor,
    *,
    margin: float = 0.2,
) -> torch.Tensor:
    """Require the requested QA-family answer to beat every other family."""

    if target_nll.ndim != 1:
        raise ValueError("target NLL must have shape [B]")
    if negative_family_nlls.ndim != 2:
        raise ValueError("negative family NLLs must have shape [B,F-1]")
    if negative_family_nlls.shape[0] != target_nll.shape[0]:
        raise ValueError("target and cross-family NLL batch sizes differ")
    if negative_family_nlls.shape[1] != len(QUESTION_FAMILIES) - 1:
        raise ValueError("every other QA family must be a negative")
    if margin < 0.0:
        raise ValueError("answer preference margin must be non-negative")
    return F.relu(
        float(margin) + target_nll[:, None] - negative_family_nlls
    ).mean()


def same_variant_cross_family_answers(
    records: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Any],
) -> Dict[str, str]:
    """Return the three same-frame/variant answers outside the anchor family."""

    group_id = str(anchor["counterfactual"]["group_id"])
    variant = str(anchor["counterfactual"]["variant"])
    family = str(anchor["question_family"])
    if family not in QUESTION_FAMILIES:
        raise ValueError("anchor has an unsupported QA family")
    answers: Dict[str, str] = {}
    for row in records:
        if (
            str(row["counterfactual"]["group_id"]) == group_id
            and str(row["counterfactual"]["variant"]) == variant
            and str(row["question_family"]) != family
        ):
            other = str(row["question_family"])
            if other in answers:
                raise ValueError("matched group duplicates a QA family")
            answers[other] = str(row["conversation"][1]["value"])
    expected = set(QUESTION_FAMILIES) - {family}
    if set(answers) != expected:
        raise ValueError("matched group does not provide all cross-family negatives")
    return answers


__all__ = [
    "DIAGNOSTIC_STANCE_VARIANTS",
    "HARD_STANCE_VARIANTS",
    "MATCHED_VARIANTS",
    "QUESTION_FAMILIES",
    "UNLABELED_OBSERVED_VARIANT",
    "cross_family_answer_preference_loss",
    "audit_matched_training_records",
    "hard_language_supervision_allowed",
    "matched_stance_batch_loss",
    "partition_complete_matched_groups",
    "same_variant_cross_family_answers",
]

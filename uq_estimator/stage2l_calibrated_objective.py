"""Calibration-preserving objectives for Stage2-L task relevance and stance.

The v6 Route151 smoke exposed three objective defects:

* an unweighted dense BCE could be small on an extremely sparse R target;
* a fixed absolute task-risk margin was unattainable for most target maps; and
* overall stance accuracy admitted the two-of-three majority-class solution.

This module is intentionally separate from the hash-bound v6 implementation.
It supplies the corrected v7 primitives without changing historical artifacts.
Stage-1 observation uncertainty remains frozen and task agnostic throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from uq_estimator.stage2l_matched_objective import HARD_STANCE_VARIANTS
from uq_estimator.stage2l_qa_contract_v3 import (
    FAMILY_RESPONSE_TAGS,
    canonical_tagged_answer,
    generation_is_nonrepeating,
    parse_planning_stance,
    same_family_unique_counterfactual_answers,
    tagged_answer_family,
)
from uq_estimator.stage2l_semantic_bottleneck import (
    PLANNING_STANCES,
    encode_planning_stances,
)


SCHEMA = "orion.stage2l_calibrated_objective.v1"


@dataclass(frozen=True)
class ForegroundBalancedRelevanceTerms:
    """Proper, foreground-balanced dense R objective components."""

    loss: torch.Tensor
    balanced_brier: torch.Tensor
    foreground_brier: torch.Tensor
    background_brier: torch.Tensor
    calibration_bce: torch.Tensor


@dataclass(frozen=True)
class GeometryNormalizedRankingTerms:
    """Learned and target-attainable on/off-path task-risk gaps."""

    loss: torch.Tensor
    learned_gap: torch.Tensor
    oracle_gap: torch.Tensor
    attained_fraction: torch.Tensor


def _validate_dense_relevance(
    relevance_logits: torch.Tensor,
    relevance_target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if relevance_logits.shape != relevance_target.shape:
        raise ValueError("relevance logits and target shapes differ")
    if relevance_logits.ndim < 2:
        raise ValueError("dense relevance tensors require a batch dimension")
    if not relevance_target.is_floating_point():
        raise ValueError("soft relevance target must be floating point")
    if not (
        bool(torch.isfinite(relevance_logits).all())
        and bool(torch.isfinite(relevance_target).all())
    ):
        raise ValueError("dense relevance tensors must be finite")
    if bool((relevance_target < 0.0).any()) or bool(
        (relevance_target > 1.0).any()
    ):
        raise ValueError("soft relevance target must lie in [0,1]")
    if valid_mask is None:
        return torch.ones_like(relevance_target, dtype=torch.bool)
    if valid_mask.shape != relevance_target.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean and match relevance shape")
    if not bool(valid_mask.flatten(1).any(dim=1).all()):
        raise ValueError("every relevance sample requires valid cells")
    return valid_mask


def _relative_foreground_mask(
    relevance_target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    support_fraction_of_peak: float,
) -> torch.Tensor:
    if not 0.0 < support_fraction_of_peak < 1.0:
        raise ValueError("support fraction must lie inside (0,1)")
    masked = relevance_target.masked_fill(~valid_mask, float("-inf"))
    peaks = masked.flatten(1).amax(dim=1)
    if bool((peaks <= 0.0).any()) or not bool(torch.isfinite(peaks).all()):
        raise ValueError("each relevance target requires positive valid support")
    threshold_shape = (relevance_target.shape[0],) + (1,) * (
        relevance_target.ndim - 1
    )
    thresholds = peaks.reshape(threshold_shape) * float(support_fraction_of_peak)
    foreground = valid_mask & relevance_target.ge(thresholds)
    background = valid_mask & ~foreground
    if not bool(foreground.flatten(1).any(dim=1).all()):
        raise ValueError("each relevance target requires foreground cells")
    if not bool(background.flatten(1).any(dim=1).all()):
        raise ValueError("each relevance target requires background cells")
    return foreground


def foreground_balanced_relevance_terms(
    relevance_logits: torch.Tensor,
    relevance_target: torch.Tensor,
    valid_mask: torch.Tensor = None,
    *,
    support_fraction_of_peak: float = 0.1,
    calibration_bce_weight: float = 0.1,
) -> ForegroundBalancedRelevanceTerms:
    """Balance spatial support without changing the calibrated optimum.

    A class-weighted BCE would move the probability optimum away from the soft
    target.  Instead, v7 balances a Brier error over target foreground and
    background, then adds a small unweighted proper BCE term.  Both terms have
    zero logit gradient when ``sigmoid(logits) == relevance_target``.
    """

    if calibration_bce_weight < 0.0:
        raise ValueError("calibration BCE weight must be non-negative")
    valid = _validate_dense_relevance(
        relevance_logits, relevance_target, valid_mask
    )
    foreground = _relative_foreground_mask(
        relevance_target,
        valid,
        support_fraction_of_peak=support_fraction_of_peak,
    )
    background = valid & ~foreground
    squared_error = (relevance_logits.sigmoid() - relevance_target).square()

    def region_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        numerator = (values * mask).flatten(1).sum(dim=1)
        denominator = mask.flatten(1).sum(dim=1).to(values.dtype)
        return numerator / denominator

    foreground_brier = region_mean(squared_error, foreground).mean()
    background_brier = region_mean(squared_error, background).mean()
    balanced_brier = 0.5 * (foreground_brier + background_brier)
    element_bce = F.binary_cross_entropy_with_logits(
        relevance_logits, relevance_target, reduction="none"
    )
    calibration_bce = region_mean(element_bce, valid).mean()
    loss = balanced_brier + float(calibration_bce_weight) * calibration_bce
    return ForegroundBalancedRelevanceTerms(
        loss=loss,
        balanced_brier=balanced_brier,
        foreground_brier=foreground_brier,
        background_brier=background_brier,
        calibration_bce=calibration_bce,
    )


@torch.no_grad()
def relevance_support_metrics(
    relevance_logits: torch.Tensor,
    relevance_target: torch.Tensor,
    valid_mask: torch.Tensor = None,
    *,
    support_fraction_of_peak: float = 0.1,
) -> Dict[str, float]:
    """Report foreground recall and background FPR at target-relative support."""

    valid = _validate_dense_relevance(
        relevance_logits, relevance_target, valid_mask
    )
    foreground = _relative_foreground_mask(
        relevance_target,
        valid,
        support_fraction_of_peak=support_fraction_of_peak,
    )
    background = valid & ~foreground
    target_peaks = relevance_target.masked_fill(~valid, float("-inf")).flatten(1).amax(1)
    threshold_shape = (relevance_target.shape[0],) + (1,) * (
        relevance_target.ndim - 1
    )
    thresholds = target_peaks.reshape(threshold_shape) * float(
        support_fraction_of_peak
    )
    predicted = valid & relevance_logits.sigmoid().ge(thresholds)

    def ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        return numerator.flatten(1).sum(1).float() / denominator.flatten(1).sum(1).float()

    recall = ratio(predicted & foreground, foreground)
    false_positive_rate = ratio(predicted & background, background)
    probabilities = relevance_logits.sigmoid()
    foreground_probability = (
        (probabilities * foreground).flatten(1).sum(1)
        / foreground.flatten(1).sum(1).to(probabilities.dtype)
    )
    background_probability = (
        (probabilities * background).flatten(1).sum(1)
        / background.flatten(1).sum(1).to(probabilities.dtype)
    )
    return {
        "foreground_recall": float(recall.mean().item()),
        "background_false_positive_rate": float(false_positive_rate.mean().item()),
        "foreground_mean_probability": float(foreground_probability.mean().item()),
        "background_mean_probability": float(background_probability.mean().item()),
        "foreground_background_probability_gap": float(
            (foreground_probability - background_probability).mean().item()
        ),
    }


def geometry_normalized_task_risk_ranking_terms(
    on_path_uq: torch.Tensor,
    off_path_uq: torch.Tensor,
    relevance_logits: torch.Tensor,
    relevance_target: torch.Tensor,
    *,
    required_oracle_fraction: float = 0.8,
    epsilon: float = 1e-6,
) -> GeometryNormalizedRankingTerms:
    """Rank on-path above off-path relative to the attainable target-R gap."""

    if not 0.0 < required_oracle_fraction <= 1.0:
        raise ValueError("required oracle fraction must lie inside (0,1]")
    if epsilon <= 0.0:
        raise ValueError("ranking epsilon must be positive")
    if not (
        on_path_uq.shape
        == off_path_uq.shape
        == relevance_logits.shape
        == relevance_target.shape
    ):
        raise ValueError("matched U, learned R and target R shapes differ")
    _validate_dense_relevance(relevance_logits, relevance_target, None)
    for name, value in (("on-path U", on_path_uq), ("off-path U", off_path_uq)):
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError("%s must be a finite floating tensor" % name)
        if bool((value < 0.0).any()) or bool((value > 1.0).any()):
            raise ValueError("%s must lie in [0,1]" % name)

    learned_r = relevance_logits.sigmoid()
    learned_on = (on_path_uq * learned_r).flatten(1).amax(1)
    learned_off = (off_path_uq * learned_r).flatten(1).amax(1)
    learned_gap = learned_on - learned_off
    oracle_on = (on_path_uq * relevance_target).flatten(1).amax(1)
    oracle_off = (off_path_uq * relevance_target).flatten(1).amax(1)
    oracle_gap = (oracle_on - oracle_off).detach()
    if bool((oracle_gap <= float(epsilon)).any()):
        raise ValueError(
            "target geometry must provide a positive on/off-path oracle gap"
        )
    attained_fraction = learned_gap / oracle_gap
    loss = F.relu(float(required_oracle_fraction) - attained_fraction).mean()
    return GeometryNormalizedRankingTerms(
        loss=loss,
        learned_gap=learned_gap,
        oracle_gap=oracle_gap,
        attained_fraction=attained_fraction,
    )


def class_balanced_matched_stance_loss(
    logits_by_variant: Mapping[str, torch.Tensor],
    target_stance_by_variant: Mapping[str, Any],
    *,
    supervised_variants: Sequence[str] = HARD_STANCE_VARIANTS,
) -> torch.Tensor:
    """Average CE within each represented stance class, then across classes."""

    variants = tuple(str(value) for value in supervised_variants)
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("supervised stance variants must be unique and non-empty")
    missing = [
        variant
        for variant in variants
        if variant not in logits_by_variant or variant not in target_stance_by_variant
    ]
    if missing:
        raise ValueError("matched stance group is incomplete: %s" % missing)
    logits = [logits_by_variant[variant] for variant in variants]
    if any(
        value.ndim != 2 or value.shape[-1] != len(PLANNING_STANCES)
        for value in logits
    ):
        raise ValueError("each stance logit tensor must have shape [B,3]")
    if len({value.shape[0] for value in logits}) != 1:
        raise ValueError("matched stance variants use different batch sizes")
    expanded_targets: Dict[str, Tuple[str, ...]] = {}
    for variant, logits_for_variant in zip(variants, logits):
        raw_targets = target_stance_by_variant[variant]
        if isinstance(raw_targets, str):
            values = (raw_targets,) * logits_for_variant.shape[0]
        else:
            values = tuple(str(value) for value in raw_targets)
            if len(values) != logits_for_variant.shape[0]:
                raise ValueError("stance target count does not match batch size")
        if any(value not in PLANNING_STANCES for value in values):
            raise ValueError("unsupported planning stance target")
        expanded_targets[variant] = values
    targets = [
        target
        for variant in variants
        for target in expanded_targets[variant]
    ]
    if len(set(targets)) < 2:
        raise ValueError("class-balanced stance groups require at least two classes")
    stacked = torch.cat(logits, dim=0)
    encoded = encode_planning_stances(
        targets,
        device=stacked.device,
    )
    per_item = F.cross_entropy(stacked, encoded, reduction="none")
    class_losses = [
        per_item[encoded == class_index].mean()
        for class_index in sorted(set(encoded.detach().cpu().tolist()))
    ]
    return torch.stack(class_losses).mean()


@torch.no_grad()
def matched_stance_metrics(
    logits_by_variant: Mapping[str, Sequence[torch.Tensor]],
    target_stance_by_variant: Mapping[str, Sequence[str]],
    *,
    supervised_variants: Sequence[str] = HARD_STANCE_VARIANTS,
) -> Dict[str, Any]:
    """Expose per-variant and macro stance metrics; never use majority accuracy."""

    variants = tuple(str(value) for value in supervised_variants)
    per_variant_accuracy: Dict[str, float] = {}
    target_probabilities = []
    target_class_correct: Dict[int, list] = {}
    for variant in variants:
        if variant not in logits_by_variant or variant not in target_stance_by_variant:
            raise ValueError("matched stance metrics are incomplete")
        batches = tuple(logits_by_variant[variant])
        targets = tuple(str(value) for value in target_stance_by_variant[variant])
        if len(batches) != len(targets) or not batches:
            raise ValueError("stance metric logits and targets differ")
        correct = []
        for logits, target in zip(batches, targets):
            if logits.shape != (1, len(PLANNING_STANCES)):
                raise ValueError("stance metric logits must each have shape [1,3]")
            target_index = PLANNING_STANCES.index(target)
            probabilities = logits.softmax(dim=-1)[0]
            is_correct = int(logits.argmax(dim=-1).item()) == target_index
            correct.append(is_correct)
            target_probabilities.append(float(probabilities[target_index].item()))
            target_class_correct.setdefault(target_index, []).append(is_correct)
        per_variant_accuracy[variant] = sum(correct) / len(correct)
    class_recalls = {
        PLANNING_STANCES[index]: sum(values) / len(values)
        for index, values in target_class_correct.items()
    }
    return {
        "per_variant_accuracy": per_variant_accuracy,
        "per_target_class_recall": class_recalls,
        "balanced_accuracy": sum(class_recalls.values()) / len(class_recalls),
        "minimum_target_probability": min(target_probabilities),
    }


def counterfactual_answer_preference_loss(
    target_nll: torch.Tensor,
    negative_nlls: torch.Tensor,
    *,
    margin: float = 0.2,
) -> torch.Tensor:
    """Prefer the matched answer over distinct same-family counterfactuals."""

    if target_nll.ndim != 1:
        raise ValueError("target NLL must have shape [B]")
    if negative_nlls.ndim != 2 or negative_nlls.shape[1] < 1:
        raise ValueError("counterfactual negative NLLs must have shape [B,N>=1]")
    if negative_nlls.shape[0] != target_nll.shape[0]:
        raise ValueError("target and counterfactual NLL batch sizes differ")
    if margin < 0.0:
        raise ValueError("answer preference margin must be non-negative")
    if not (
        bool(torch.isfinite(target_nll).all())
        and bool(torch.isfinite(negative_nlls).all())
    ):
        raise ValueError("answer preference NLLs must be finite")
    return F.relu(float(margin) + target_nll[:, None] - negative_nlls).mean()


__all__ = [
    "FAMILY_RESPONSE_TAGS",
    "ForegroundBalancedRelevanceTerms",
    "GeometryNormalizedRankingTerms",
    "SCHEMA",
    "canonical_tagged_answer",
    "class_balanced_matched_stance_loss",
    "counterfactual_answer_preference_loss",
    "foreground_balanced_relevance_terms",
    "generation_is_nonrepeating",
    "geometry_normalized_task_risk_ranking_terms",
    "matched_stance_metrics",
    "parse_planning_stance",
    "relevance_support_metrics",
    "same_family_unique_counterfactual_answers",
    "tagged_answer_family",
]

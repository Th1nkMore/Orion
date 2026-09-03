"""Support-aligned dense relevance objective for Stage2-L v9.

The v8 balanced Brier objective reduced average background probability but
still left 16.2% of background cells above the declared support threshold.
This objective keeps the calibrated soft-target loss and adds an explicit
background-support hinge aligned with the reported false-positive metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from uq_estimator.stage2l_calibrated_objective import (
    ForegroundBalancedRelevanceTerms,
    foreground_balanced_relevance_terms,
)


SCHEMA = "orion.stage2l_support_aligned_relevance_objective.v1"


@dataclass(frozen=True)
class SupportAlignedRelevanceTerms:
    loss: torch.Tensor
    calibrated: ForegroundBalancedRelevanceTerms
    background_support_hinge: torch.Tensor
    support_thresholds: torch.Tensor
    schema: str = SCHEMA


def support_aligned_relevance_terms(
    relevance_logits: torch.Tensor,
    relevance_target: torch.Tensor,
    *,
    support_fraction_of_peak: float = 0.1,
    calibration_bce_weight: float = 0.1,
    background_support_weight: float = 1.0,
    background_probability_margin: float = 0.01,
) -> SupportAlignedRelevanceTerms:
    if relevance_logits.shape != relevance_target.shape:
        raise ValueError("relevance logits and targets differ")
    if relevance_logits.ndim < 2:
        raise ValueError("dense relevance tensors require a batch dimension")
    if not 0.0 < support_fraction_of_peak < 1.0:
        raise ValueError("support fraction must lie inside (0,1)")
    if background_support_weight < 0.0:
        raise ValueError("background support weight must be non-negative")
    if not 0.0 <= background_probability_margin < 1.0:
        raise ValueError("background margin must lie inside [0,1)")
    if not (
        bool(torch.isfinite(relevance_logits).all())
        and bool(torch.isfinite(relevance_target).all())
    ):
        raise ValueError("dense relevance tensors must be finite")
    if bool((relevance_target < 0.0).any()) or bool(
        (relevance_target > 1.0).any()
    ):
        raise ValueError("soft relevance target must lie in [0,1]")

    calibrated = foreground_balanced_relevance_terms(
        relevance_logits,
        relevance_target,
        support_fraction_of_peak=support_fraction_of_peak,
        calibration_bce_weight=calibration_bce_weight,
    )
    peaks = relevance_target.flatten(1).amax(dim=1)
    if bool((peaks <= 0.0).any()):
        raise ValueError("each relevance target requires positive support")
    threshold_shape = (relevance_target.shape[0],) + (1,) * (
        relevance_target.ndim - 1
    )
    thresholds = peaks.reshape(threshold_shape) * float(
        support_fraction_of_peak
    )
    foreground = relevance_target.ge(thresholds)
    background = ~foreground
    if not bool(background.flatten(1).any(dim=1).all()):
        raise ValueError("each relevance target requires background cells")
    desired_background_ceiling = (
        thresholds - float(background_probability_margin)
    ).clamp_min(0.0)
    violations = F.relu(
        relevance_logits.sigmoid() - desired_background_ceiling
    ).square()
    per_sample = (
        (violations * background).flatten(1).sum(dim=1)
        / background.flatten(1).sum(dim=1).to(violations.dtype)
    )
    background_hinge = per_sample.mean()
    loss = calibrated.loss + float(background_support_weight) * background_hinge
    return SupportAlignedRelevanceTerms(
        loss=loss,
        calibrated=calibrated,
        background_support_hinge=background_hinge,
        support_thresholds=thresholds,
    )


__all__ = [
    "SCHEMA",
    "SupportAlignedRelevanceTerms",
    "support_aligned_relevance_terms",
]

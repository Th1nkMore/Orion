"""Phased Stage2-L v10 objective focused on VLM-owned task relevance R.

MR2 showed that adding route coverage did not resolve background support false
positives or held-out on/off-path ranking.  V10 therefore removes the redundant
structured-field classification loss and makes the learning sequence explicit:

* ``map_pretrain`` optimizes only dense R support and calibration.
* ``risk_alignment`` retains the map objective and adds K=U*R ranking.

Language QA remains auxiliary in the trainer and may not redefine R.  This
module contains no adapter loss, field-classification loss, trajectory loss,
direct-control loss, Density UQ, or governor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from uq_estimator.stage2l_calibrated_objective import (
    ForegroundBalancedRelevanceTerms,
    GeometryNormalizedRankingTerms,
    foreground_balanced_relevance_terms,
    geometry_normalized_task_risk_ranking_terms,
)


SCHEMA = "orion.stage2l_relevance_objective.v10"
PHASES = ("map_pretrain", "risk_alignment")


@dataclass(frozen=True)
class Stage2LRelevanceObjectiveV10Terms:
    loss: torch.Tensor
    map_loss: torch.Tensor
    calibrated: ForegroundBalancedRelevanceTerms
    support_tversky_loss: torch.Tensor
    foreground_support_hinge: torch.Tensor
    background_support_hinge: torch.Tensor
    ranking: Optional[GeometryNormalizedRankingTerms]
    phase: str
    structured_field_classification_loss_used: bool = False
    trajectory_or_control_loss_used: bool = False
    schema: str = SCHEMA


def _region_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numerator = (values * mask).flatten(1).sum(dim=1)
    denominator = mask.flatten(1).sum(dim=1).to(values.dtype)
    if bool((denominator <= 0.0).any()):
        raise ValueError("each target requires foreground and background support")
    return (numerator / denominator).mean()


def stage2l_relevance_objective_v10(
    relevance_logits: torch.Tensor,
    relevance_target: torch.Tensor,
    *,
    phase: str,
    on_path_uq: torch.Tensor = None,
    off_path_uq: torch.Tensor = None,
    support_fraction_of_peak: float = 0.1,
    calibration_bce_weight: float = 0.1,
    tversky_false_positive_weight: float = 0.7,
    tversky_false_negative_weight: float = 0.3,
    foreground_probability_margin: float = 0.05,
    background_probability_margin: float = 0.01,
    calibrated_weight: float = 1.0,
    tversky_weight: float = 1.0,
    support_hinge_weight: float = 1.0,
    ranking_weight: float = 1.0,
    required_oracle_fraction: float = 0.8,
    epsilon: float = 1e-6,
) -> Stage2LRelevanceObjectiveV10Terms:
    if phase not in PHASES:
        raise ValueError("unsupported Stage2-L v10 phase")
    if relevance_logits.shape != relevance_target.shape or relevance_logits.ndim < 2:
        raise ValueError("R logits and target shapes differ or lack a batch")
    if not (
        relevance_logits.is_floating_point()
        and relevance_target.is_floating_point()
        and bool(torch.isfinite(relevance_logits).all())
        and bool(torch.isfinite(relevance_target).all())
        and not bool((relevance_target < 0.0).any())
        and not bool((relevance_target > 1.0).any())
    ):
        raise ValueError("R logits/targets must be finite and targets lie in [0,1]")
    if not 0.0 < support_fraction_of_peak < 1.0:
        raise ValueError("support fraction must lie inside (0,1)")
    if not (
        0.0 <= tversky_false_positive_weight <= 1.0
        and 0.0 <= tversky_false_negative_weight <= 1.0
        and tversky_false_positive_weight + tversky_false_negative_weight > 0.0
    ):
        raise ValueError("Tversky weights are invalid")
    scalar_weights = (
        calibrated_weight,
        tversky_weight,
        support_hinge_weight,
        ranking_weight,
        foreground_probability_margin,
        background_probability_margin,
    )
    if any(float(value) < 0.0 for value in scalar_weights):
        raise ValueError("objective weights and margins must be non-negative")

    calibrated = foreground_balanced_relevance_terms(
        relevance_logits,
        relevance_target,
        support_fraction_of_peak=support_fraction_of_peak,
        calibration_bce_weight=calibration_bce_weight,
    )
    peaks = relevance_target.flatten(1).amax(dim=1)
    if bool((peaks <= 0.0).any()):
        raise ValueError("each R target requires positive support")
    threshold_shape = (relevance_target.shape[0],) + (1,) * (
        relevance_target.ndim - 1
    )
    thresholds = peaks.reshape(threshold_shape) * float(support_fraction_of_peak)
    foreground = relevance_target.ge(thresholds)
    background = ~foreground
    if not (
        bool(foreground.flatten(1).any(dim=1).all())
        and bool(background.flatten(1).any(dim=1).all())
    ):
        raise ValueError("each R target requires foreground and background")

    probabilities = relevance_logits.sigmoid()
    foreground_float = foreground.to(probabilities.dtype)
    background_float = background.to(probabilities.dtype)
    true_positive = (probabilities * foreground_float).flatten(1).sum(dim=1)
    false_positive = (probabilities * background_float).flatten(1).sum(dim=1)
    false_negative = ((1.0 - probabilities) * foreground_float).flatten(1).sum(dim=1)
    tversky = (true_positive + epsilon) / (
        true_positive
        + float(tversky_false_positive_weight) * false_positive
        + float(tversky_false_negative_weight) * false_negative
        + epsilon
    )
    support_tversky_loss = (1.0 - tversky).mean()

    foreground_floor = (
        thresholds + float(foreground_probability_margin)
    ).clamp_max(1.0)
    background_ceiling = (
        thresholds - float(background_probability_margin)
    ).clamp_min(0.0)
    foreground_hinge = _region_mean(
        F.relu(foreground_floor - probabilities).square(), foreground
    )
    background_hinge = _region_mean(
        F.relu(probabilities - background_ceiling).square(), background
    )
    map_loss = (
        float(calibrated_weight) * calibrated.loss
        + float(tversky_weight) * support_tversky_loss
        + float(support_hinge_weight) * (foreground_hinge + background_hinge)
    )

    ranking = None
    loss = map_loss
    if phase == "risk_alignment":
        if on_path_uq is None or off_path_uq is None:
            raise ValueError("risk-alignment phase requires matched on/off-path U")
        ranking = geometry_normalized_task_risk_ranking_terms(
            on_path_uq,
            off_path_uq,
            relevance_logits,
            relevance_target,
            required_oracle_fraction=required_oracle_fraction,
            epsilon=epsilon,
        )
        loss = loss + float(ranking_weight) * ranking.loss
    elif on_path_uq is not None or off_path_uq is not None:
        raise ValueError("map-pretrain phase must not consume matched U ranking labels")

    return Stage2LRelevanceObjectiveV10Terms(
        loss=loss,
        map_loss=map_loss,
        calibrated=calibrated,
        support_tversky_loss=support_tversky_loss,
        foreground_support_hinge=foreground_hinge,
        background_support_hinge=background_hinge,
        ranking=ranking,
        phase=phase,
    )


__all__ = [
    "PHASES",
    "SCHEMA",
    "Stage2LRelevanceObjectiveV10Terms",
    "stage2l_relevance_objective_v10",
]

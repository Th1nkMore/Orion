"""Calibrated, view-balanced dense relevance objective for Stage2-L v12.

The objective accepts only contextual task-relevance logits and the weak soft
geometry target.  It never consumes observation uncertainty, QA answers,
scenario names, TTC, outcomes, trajectories, or controls.  Foreground and
background cells are balanced first across active camera views and then within
each view, while every loss component has zero gradient at ``sigmoid(logit) =
target``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


SCHEMA = "orion.stage2l_view_balanced_objective.v12"


@dataclass(frozen=True)
class ViewBalancedRegionWeights:
    foreground_mask: torch.Tensor
    background_mask: torch.Tensor
    foreground: torch.Tensor
    background: torch.Tensor
    threshold: torch.Tensor
    active_foreground_views: torch.Tensor
    active_background_views: torch.Tensor


@dataclass(frozen=True)
class ViewBalancedRelevanceTerms:
    loss: torch.Tensor
    balanced_brier: torch.Tensor
    foreground_brier: torch.Tensor
    background_brier: torch.Tensor
    calibration_bce: torch.Tensor
    foreground_support_hinge: torch.Tensor
    background_support_hinge: torch.Tensor
    mean_active_foreground_views: torch.Tensor
    mean_active_background_views: torch.Tensor
    schema: str = SCHEMA
    observation_uq_used: bool = False
    qa_answer_used: bool = False
    trajectory_or_control_used: bool = False


def _validate(
    relevance_logits: torch.Tensor, relevance_target: torch.Tensor
) -> None:
    if relevance_logits.shape != relevance_target.shape:
        raise ValueError("relevance logits and target shapes differ")
    if relevance_logits.ndim != 4:
        raise ValueError("view-balanced relevance requires [B,V,H,W]")
    if not (
        relevance_logits.is_floating_point()
        and relevance_target.is_floating_point()
        and bool(torch.isfinite(relevance_logits).all())
        and bool(torch.isfinite(relevance_target).all())
    ):
        raise ValueError("relevance tensors must be finite floating tensors")
    if bool((relevance_target < 0.0).any()) or bool(
        (relevance_target > 1.0).any()
    ):
        raise ValueError("soft relevance target must lie in [0,1]")


def view_balanced_region_weights(
    relevance_target: torch.Tensor,
    *,
    support_fraction_of_peak: float = 0.1,
) -> ViewBalancedRegionWeights:
    """Construct per-sample region weights with equal active-view mass."""

    if relevance_target.ndim != 4 or not relevance_target.is_floating_point():
        raise ValueError("relevance target must be floating [B,V,H,W]")
    if not bool(torch.isfinite(relevance_target).all()):
        raise ValueError("relevance target must be finite")
    if bool((relevance_target < 0.0).any()) or bool(
        (relevance_target > 1.0).any()
    ):
        raise ValueError("relevance target must lie in [0,1]")
    if not 0.0 < float(support_fraction_of_peak) < 1.0:
        raise ValueError("support fraction must lie inside (0,1)")

    peaks = relevance_target.flatten(1).amax(dim=1)
    if bool((peaks <= 0.0).any()):
        raise ValueError("each relevance target requires positive support")
    threshold = peaks[:, None, None, None] * float(support_fraction_of_peak)
    foreground_mask = relevance_target.ge(threshold)
    background_mask = ~foreground_mask
    foreground_counts = foreground_mask.sum(dim=(-2, -1))
    background_counts = background_mask.sum(dim=(-2, -1))
    active_foreground = foreground_counts.gt(0)
    active_background = background_counts.gt(0)
    foreground_view_counts = active_foreground.sum(dim=1)
    background_view_counts = active_background.sum(dim=1)
    if bool((foreground_view_counts <= 0).any()) or bool(
        (background_view_counts <= 0).any()
    ):
        raise ValueError("each target requires foreground and background regions")

    foreground_denominator = (
        foreground_counts.clamp_min(1).to(relevance_target.dtype)
        * foreground_view_counts[:, None].to(relevance_target.dtype)
    )
    background_denominator = (
        background_counts.clamp_min(1).to(relevance_target.dtype)
        * background_view_counts[:, None].to(relevance_target.dtype)
    )
    foreground = (
        foreground_mask.to(relevance_target.dtype)
        / foreground_denominator[..., None, None]
    )
    background = (
        background_mask.to(relevance_target.dtype)
        / background_denominator[..., None, None]
    )
    if not torch.allclose(
        foreground.flatten(1).sum(dim=1),
        torch.ones_like(peaks),
        atol=1e-6,
        rtol=0.0,
    ) or not torch.allclose(
        background.flatten(1).sum(dim=1),
        torch.ones_like(peaks),
        atol=1e-6,
        rtol=0.0,
    ):
        raise RuntimeError("view-balanced region weights do not sum to one")
    return ViewBalancedRegionWeights(
        foreground_mask=foreground_mask,
        background_mask=background_mask,
        foreground=foreground,
        background=background,
        threshold=threshold,
        active_foreground_views=active_foreground,
        active_background_views=active_background,
    )


def view_balanced_relevance_terms_v12(
    relevance_logits: torch.Tensor,
    relevance_target: torch.Tensor,
    *,
    support_fraction_of_peak: float = 0.1,
    calibration_bce_weight: float = 0.1,
    support_hinge_weight: float = 1.0,
) -> ViewBalancedRelevanceTerms:
    """Return a strictly target-anchored, view-balanced R objective."""

    _validate(relevance_logits, relevance_target)
    if calibration_bce_weight < 0.0 or support_hinge_weight < 0.0:
        raise ValueError("objective weights must be non-negative")
    weights = view_balanced_region_weights(
        relevance_target,
        support_fraction_of_peak=support_fraction_of_peak,
    )
    probability = relevance_logits.sigmoid()
    squared_error = (probability - relevance_target).square()
    foreground_brier = (
        (squared_error * weights.foreground).flatten(1).sum(dim=1).mean()
    )
    background_brier = (
        (squared_error * weights.background).flatten(1).sum(dim=1).mean()
    )
    balanced_brier = 0.5 * (foreground_brier + background_brier)
    calibration_bce = F.binary_cross_entropy_with_logits(
        relevance_logits, relevance_target, reduction="mean"
    )

    # No positive margin is added: p=target lies inside both zero-hinge sets.
    foreground_support_hinge = (
        F.relu(weights.threshold - probability).square()
        * weights.foreground
    ).flatten(1).sum(dim=1).mean()
    background_support_hinge = (
        F.relu(probability - weights.threshold).square()
        * weights.background
    ).flatten(1).sum(dim=1).mean()
    loss = (
        balanced_brier
        + float(calibration_bce_weight) * calibration_bce
        + float(support_hinge_weight)
        * (foreground_support_hinge + background_support_hinge)
    )
    return ViewBalancedRelevanceTerms(
        loss=loss,
        balanced_brier=balanced_brier,
        foreground_brier=foreground_brier,
        background_brier=background_brier,
        calibration_bce=calibration_bce,
        foreground_support_hinge=foreground_support_hinge,
        background_support_hinge=background_support_hinge,
        mean_active_foreground_views=(
            weights.active_foreground_views.sum(dim=1).float().mean()
        ),
        mean_active_background_views=(
            weights.active_background_views.sum(dim=1).float().mean()
        ),
    )


@torch.no_grad()
def view_balanced_weight_summary(
    relevance_target: torch.Tensor,
    *,
    support_fraction_of_peak: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Expose exact old/new foreground mass by view for CPU preflight."""

    weights = view_balanced_region_weights(
        relevance_target,
        support_fraction_of_peak=support_fraction_of_peak,
    )
    foreground_counts = weights.foreground_mask.sum(dim=(-2, -1)).float()
    current = foreground_counts / foreground_counts.sum(dim=1, keepdim=True)
    proposed = weights.foreground.sum(dim=(-2, -1))
    return {
        "current_per_group_view_mass": current,
        "proposed_per_group_view_mass": proposed,
        "active_foreground_views": weights.active_foreground_views,
        "foreground_weight_sum": weights.foreground.flatten(1).sum(dim=1),
        "background_weight_sum": weights.background.flatten(1).sum(dim=1),
    }


__all__ = [
    "SCHEMA",
    "ViewBalancedRegionWeights",
    "ViewBalancedRelevanceTerms",
    "view_balanced_region_weights",
    "view_balanced_relevance_terms_v12",
    "view_balanced_weight_summary",
]

"""Factorized route/actor task-relevance interface for Stage2-L v12.1.

The module accepts only contextual token features.  Observation UQ, QA
answers, outcomes and controls are deliberately absent.  Route-corridor and
future-path-conflict-actor relevance remain separately supervised; their union
is exposed only as a derived diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


SCHEMA = "orion.stage2l_factorized_relevance.v12_1"
COMPONENT_ORDER = ("route", "actor")


@dataclass(frozen=True)
class FactorizedRelevanceOutput:
    route_logits: torch.Tensor
    actor_logits: torch.Tensor
    component_logits: torch.Tensor
    route_probability: torch.Tensor
    actor_probability: torch.Tensor
    derived_union_probability: torch.Tensor
    schema: str = SCHEMA
    observation_uq_used: bool = False
    qa_answer_used: bool = False
    trajectory_or_control_used: bool = False


@dataclass(frozen=True)
class FactorizedRelevanceTerms:
    loss: torch.Tensor
    route_loss: torch.Tensor
    actor_loss: torch.Tensor
    route_active_brier: torch.Tensor
    actor_active_brier: torch.Tensor
    route_inactive_view_anchor: torch.Tensor
    actor_inactive_view_anchor: torch.Tensor
    route_empty_component_anchor: torch.Tensor
    actor_empty_component_anchor: torch.Tensor
    derived_union_brier_diagnostic: torch.Tensor
    active_sample_component_count: torch.Tensor
    empty_sample_component_count: torch.Tensor
    schema: str = SCHEMA
    derived_union_loss_weight: float = 0.0
    observation_uq_used: bool = False
    qa_answer_used: bool = False
    trajectory_or_control_used: bool = False


class FactorizedTaskRelevanceMapHead(nn.Module):
    """Decode route and conflict-actor R from one shared contextual grid."""

    def __init__(self, model_dim: int = 4096, hidden_dim: int = 256) -> None:
        super().__init__()
        if min(model_dim, hidden_dim) <= 0:
            raise ValueError("factorized relevance dimensions must be positive")
        self.shared = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
        )
        self.route_output = nn.Linear(hidden_dim, 1)
        self.actor_output = nn.Linear(hidden_dim, 1)

    def forward(self, contextual_token_grid: torch.Tensor) -> FactorizedRelevanceOutput:
        if contextual_token_grid.ndim != 5:
            raise ValueError("contextual token grid must have shape [B,V,H,W,D]")
        hidden = self.shared(contextual_token_grid)
        route_logits = self.route_output(hidden).squeeze(dim=-1)
        actor_logits = self.actor_output(hidden).squeeze(dim=-1)
        component_logits = torch.stack((route_logits, actor_logits), dim=1)
        route_probability = route_logits.sigmoid()
        actor_probability = actor_logits.sigmoid()
        return FactorizedRelevanceOutput(
            route_logits=route_logits,
            actor_logits=actor_logits,
            component_logits=component_logits,
            route_probability=route_probability,
            actor_probability=actor_probability,
            derived_union_probability=torch.maximum(
                route_probability, actor_probability
            ),
        )


def _validate(
    component_logits: torch.Tensor, component_targets: torch.Tensor
) -> None:
    if component_logits.shape != component_targets.shape:
        raise ValueError("factorized logits and targets differ")
    if component_logits.ndim != 5 or component_logits.shape[1] != 2:
        raise ValueError("factorized relevance requires [B,2,V,H,W]")
    if not (
        component_logits.is_floating_point()
        and component_targets.is_floating_point()
        and bool(torch.isfinite(component_logits).all())
        and bool(torch.isfinite(component_targets).all())
    ):
        raise ValueError("factorized relevance tensors must be finite floating tensors")
    if bool((component_targets < 0.0).any()) or bool(
        (component_targets > 1.0).any()
    ):
        raise ValueError("factorized soft targets must lie in [0,1]")


def _mean_or_zero(values: Tuple[torch.Tensor, ...], reference: torch.Tensor) -> torch.Tensor:
    if not values:
        return reference.sum() * 0.0
    return torch.stack(values).mean()


def _component_terms(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    support_fraction_of_peak: float,
    inactive_view_background_anchor_weight: float,
    empty_component_background_anchor_weight: float,
) -> Dict[str, torch.Tensor]:
    probability = logits.sigmoid()
    squared_error = (probability - target).square()
    active_losses = []
    inactive_view_anchors = []
    empty_component_anchors = []
    active_count = 0
    empty_count = 0
    for sample_index in range(target.shape[0]):
        sample_target = target[sample_index]
        sample_error = squared_error[sample_index]
        peak = sample_target.max()
        if bool(peak <= 0.0):
            empty_count += 1
            empty_component_anchors.append(sample_error.mean())
            continue
        active_count += 1
        threshold = peak * float(support_fraction_of_peak)
        foreground = sample_target.ge(threshold)
        active_views = foreground.flatten(1).any(dim=1)
        foreground_view_losses = []
        background_view_losses = []
        for view_index in torch.nonzero(active_views, as_tuple=False).flatten():
            view_foreground = foreground[view_index]
            view_background = ~view_foreground
            foreground_view_losses.append(
                sample_error[view_index][view_foreground].mean()
            )
            if bool(view_background.any()):
                background_view_losses.append(
                    sample_error[view_index][view_background].mean()
                )
        foreground_loss = torch.stack(foreground_view_losses).mean()
        background_loss = _mean_or_zero(
            tuple(background_view_losses), sample_error
        )
        active_losses.append(0.5 * (foreground_loss + background_loss))
        inactive_views = ~active_views
        if bool(inactive_views.any()):
            inactive_view_anchors.append(sample_error[inactive_views].mean())
    active = _mean_or_zero(tuple(active_losses), squared_error)
    inactive_anchor = _mean_or_zero(tuple(inactive_view_anchors), squared_error)
    empty_anchor = _mean_or_zero(tuple(empty_component_anchors), squared_error)
    total = (
        active
        + float(inactive_view_background_anchor_weight) * inactive_anchor
        + float(empty_component_background_anchor_weight) * empty_anchor
    )
    return {
        "total": total,
        "active": active,
        "inactive_view_anchor": inactive_anchor,
        "empty_component_anchor": empty_anchor,
        "active_count": torch.as_tensor(
            float(active_count), device=target.device, dtype=target.dtype
        ),
        "empty_count": torch.as_tensor(
            float(empty_count), device=target.device, dtype=target.dtype
        ),
    }


def factorized_relevance_terms_v121(
    component_logits: torch.Tensor,
    component_targets: torch.Tensor,
    *,
    support_fraction_of_peak: float = 0.1,
    inactive_view_background_anchor_weight: float = 0.25,
    empty_component_background_anchor_weight: float = 1.0,
    route_component_weight: float = 0.5,
    actor_component_weight: float = 0.5,
) -> FactorizedRelevanceTerms:
    """Return separately calibrated route and conflict-actor R losses."""

    _validate(component_logits, component_targets)
    if not 0.0 < float(support_fraction_of_peak) < 1.0:
        raise ValueError("support fraction must lie inside (0,1)")
    scalar_weights = (
        inactive_view_background_anchor_weight,
        empty_component_background_anchor_weight,
        route_component_weight,
        actor_component_weight,
    )
    if any(float(value) < 0.0 for value in scalar_weights):
        raise ValueError("factorized objective weights must be non-negative")
    if float(route_component_weight) + float(actor_component_weight) <= 0.0:
        raise ValueError("at least one component weight must be positive")

    route = _component_terms(
        component_logits[:, 0],
        component_targets[:, 0],
        support_fraction_of_peak=support_fraction_of_peak,
        inactive_view_background_anchor_weight=inactive_view_background_anchor_weight,
        empty_component_background_anchor_weight=empty_component_background_anchor_weight,
    )
    actor = _component_terms(
        component_logits[:, 1],
        component_targets[:, 1],
        support_fraction_of_peak=support_fraction_of_peak,
        inactive_view_background_anchor_weight=inactive_view_background_anchor_weight,
        empty_component_background_anchor_weight=empty_component_background_anchor_weight,
    )
    total_component_weight = float(route_component_weight) + float(
        actor_component_weight
    )
    loss = (
        float(route_component_weight) * route["total"]
        + float(actor_component_weight) * actor["total"]
    ) / total_component_weight
    probabilities = component_logits.sigmoid()
    derived_union_probability = torch.maximum(
        probabilities[:, 0], probabilities[:, 1]
    )
    derived_union_target = torch.maximum(
        component_targets[:, 0], component_targets[:, 1]
    )
    union_brier = (
        derived_union_probability - derived_union_target
    ).square().mean()
    return FactorizedRelevanceTerms(
        loss=loss,
        route_loss=route["total"],
        actor_loss=actor["total"],
        route_active_brier=route["active"],
        actor_active_brier=actor["active"],
        route_inactive_view_anchor=route["inactive_view_anchor"],
        actor_inactive_view_anchor=actor["inactive_view_anchor"],
        route_empty_component_anchor=route["empty_component_anchor"],
        actor_empty_component_anchor=actor["empty_component_anchor"],
        derived_union_brier_diagnostic=union_brier,
        active_sample_component_count=(
            route["active_count"] + actor["active_count"]
        ),
        empty_sample_component_count=(
            route["empty_count"] + actor["empty_count"]
        ),
    )


__all__ = [
    "COMPONENT_ORDER",
    "SCHEMA",
    "FactorizedRelevanceOutput",
    "FactorizedRelevanceTerms",
    "FactorizedTaskRelevanceMapHead",
    "factorized_relevance_terms_v121",
]

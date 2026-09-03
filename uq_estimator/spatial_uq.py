"""Spatial uncertainty primitives for uncertainty-aware driving.

This module deliberately keeps *perception uncertainty* separate from route
relevance.  :class:`SpatialPatchUQHead` only receives perception features and
therefore cannot learn the shortcut ``route -> uncertainty``.  A route corridor
is introduced later, in :func:`cvar_path_risk`, through a fixed and auditable
aggregation rule.

The implementation follows three standard uncertainty-quantification ideas:

* input-dependent (heteroscedastic) variance trained with a proper Gaussian
  negative log likelihood;
* epistemic variance distilled from an ensemble and decomposed with the law of
  total variance; and
* selective risk summarized by a worst-tail (top-q/CVaR) aggregation rather
  than a global spatial mean.

All functions operate on arbitrary leading dimensions and are independent of
the ORION model implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


SPATIAL_UQ_SCHEMA_VERSION = "spatial-uq/v1"
ENSEMBLE_VARIANCE_SCHEMA_VERSION = "ensemble-variance/v1"
PATH_RISK_SCHEMA_VERSION = "path-risk-cvar/v1"


@dataclass(frozen=True)
class SpatialUQOutput:
    """Versioned output of :class:`SpatialPatchUQHead`.

    Every tensor has the same dynamic leading shape as the input patch tensor
    with its feature dimension removed.  ``epistemic_variance`` is ``None``
    when the optional distilled epistemic head is disabled.
    """

    expected_error: torch.Tensor
    log_variance: torch.Tensor
    aleatoric_variance: torch.Tensor
    failure_probability: torch.Tensor
    epistemic_variance: Optional[torch.Tensor] = None
    schema_version: str = SPATIAL_UQ_SCHEMA_VERSION

    CURRENT_SCHEMA_VERSION: ClassVar[str] = SPATIAL_UQ_SCHEMA_VERSION


class SpatialPatchUQHead(nn.Module):
    """Lightweight per-patch uncertainty head with dynamic spatial shape.

    Args:
        feature_dim: Dimension of each perception feature/token.
        hidden_dim: Width of the shared two-layer patch MLP.
        min_log_variance: Lower bound for predicted log aleatoric variance.
        max_log_variance: Upper bound for predicted log aleatoric variance.
        predict_epistemic: Add a non-negative head intended to be distilled
            from independent-ensemble disagreement.

    ``forward`` intentionally accepts only ``patch_features``.  Route geometry,
    waypoints, and hazard labels belong to the downstream risk aggregator and
    never enter this UQ head.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        min_log_variance: float = -8.0,
        max_log_variance: float = 4.0,
        predict_epistemic: bool = False,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if min_log_variance >= max_log_variance:
            raise ValueError("min_log_variance must be smaller than max_log_variance")

        self.feature_dim = int(feature_dim)
        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)
        self.predict_epistemic = bool(predict_epistemic)

        self.trunk = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # mean error, bounded log variance, and failure logit
        self.output_head = nn.Linear(hidden_dim, 3)
        self.epistemic_head = (
            nn.Linear(hidden_dim, 1) if self.predict_epistemic else None
        )

    def forward(self, patch_features: torch.Tensor) -> SpatialUQOutput:
        """Predict spatial UQ for a tensor shaped ``[..., feature_dim]``."""
        if patch_features.ndim < 2:
            raise ValueError("patch_features must have at least two dimensions")
        if patch_features.shape[-1] != self.feature_dim:
            raise ValueError(
                "last patch_features dimension must equal feature_dim "
                f"({self.feature_dim}), got {patch_features.shape[-1]}"
            )
        if not patch_features.is_floating_point():
            raise TypeError("patch_features must be floating point")

        hidden = self.trunk(patch_features)
        raw_expected_error, raw_log_variance, failure_logit = self.output_head(
            hidden
        ).unbind(dim=-1)

        # Error and variances are non-negative.  A smooth tanh map keeps the
        # log variance inside declared bounds without a hard-clamp dead zone.
        expected_error = F.softplus(raw_expected_error)
        half_range = 0.5 * (self.max_log_variance - self.min_log_variance)
        midpoint = 0.5 * (self.max_log_variance + self.min_log_variance)
        log_variance = midpoint + half_range * torch.tanh(raw_log_variance)
        aleatoric_variance = torch.exp(log_variance)
        failure_probability = torch.sigmoid(failure_logit)

        epistemic_variance = None
        if self.epistemic_head is not None:
            epistemic_variance = F.softplus(
                self.epistemic_head(hidden).squeeze(dim=-1)
            )

        return SpatialUQOutput(
            expected_error=expected_error,
            log_variance=log_variance,
            aleatoric_variance=aleatoric_variance,
            failure_probability=failure_probability,
            epistemic_variance=epistemic_variance,
        )


def paired_cosine_representation_error(
    clean_features: torch.Tensor,
    corrupt_features: torch.Tensor,
    feature_dim: int = -1,
    eps: float = 1e-8,
    detach_target: bool = True,
) -> torch.Tensor:
    """Construct a paired clean/corrupt representation-error target.

    The target is ``1 - cosine_similarity(clean, corrupt)`` and lies in
    ``[0, 2]`` up to numerical precision.  It attributes a change to a paired
    intervention without treating the corruption mask itself as uncertainty.
    By default the result is detached because it is normally a supervision
    target for the UQ head, not an objective for the frozen representation.
    """
    if clean_features.shape != corrupt_features.shape:
        raise ValueError(
            "clean_features and corrupt_features must have identical shapes, "
            f"got {tuple(clean_features.shape)} and {tuple(corrupt_features.shape)}"
        )
    if not clean_features.is_floating_point() or not corrupt_features.is_floating_point():
        raise TypeError("paired representation features must be floating point")

    similarity = F.cosine_similarity(
        clean_features, corrupt_features, dim=feature_dim, eps=eps
    ).clamp(min=-1.0, max=1.0)
    target = 1.0 - similarity
    return target.detach() if detach_target else target


def _reduce_loss(
    loss: torch.Tensor,
    reduction: str,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")

    if weight is not None:
        weight = torch.broadcast_to(weight.to(dtype=loss.dtype, device=loss.device), loss.shape)
        if torch.any(weight < 0):
            raise ValueError("loss weights must be non-negative")
        loss = loss * weight

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if weight is None:
        return loss.mean()
    return loss.sum() / weight.sum().clamp_min(torch.finfo(loss.dtype).eps)


def heteroscedastic_gaussian_nll(
    expected_error: torch.Tensor,
    target_error: torch.Tensor,
    log_variance: torch.Tensor,
    reduction: str = "mean",
    weight: Optional[torch.Tensor] = None,
    include_constant: bool = False,
) -> torch.Tensor:
    """Gaussian NLL for input-dependent perception error.

    The essential ``+ log_variance`` term is retained, preventing the model
    from reducing the residual penalty merely by predicting infinite variance.
    ``include_constant`` adds ``0.5 * log(2*pi)`` when an absolute likelihood
    value is required; it does not affect optimization.
    """
    expected_error, target_error, log_variance = torch.broadcast_tensors(
        expected_error, target_error, log_variance
    )
    squared_residual = (target_error - expected_error).square()
    loss = 0.5 * (torch.exp(-log_variance) * squared_residual + log_variance)
    if include_constant:
        loss = loss + 0.5 * math.log(2.0 * math.pi)
    return _reduce_loss(loss, reduction=reduction, weight=weight)


def brier_loss(
    failure_probability: torch.Tensor,
    failure_target: torch.Tensor,
    reduction: str = "mean",
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Brier proper scoring rule for calibrated failure probabilities."""
    failure_probability, failure_target = torch.broadcast_tensors(
        failure_probability, failure_target
    )
    loss = (failure_probability - failure_target).square()
    return _reduce_loss(loss, reduction=reduction, weight=weight)


def paired_error_ranking_loss(
    clean_expected_error: torch.Tensor,
    corrupt_expected_error: torch.Tensor,
    clean_target_error: torch.Tensor,
    corrupt_target_error: torch.Tensor,
    margin: float = 0.1,
    min_target_increase: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """Rank corrupt above clean *only when its target error truly increases*.

    A pair is active iff
    ``corrupt_target_error > clean_target_error + min_target_increase``.
    Inactive pairs contribute exactly zero, preventing a corruption-type label
    from forcing UQ upward when perception remained correct.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if min_target_increase < 0:
        raise ValueError("min_target_increase must be non-negative")

    clean_expected_error, corrupt_expected_error, clean_target_error, corrupt_target_error = (
        torch.broadcast_tensors(
            clean_expected_error,
            corrupt_expected_error,
            clean_target_error,
            corrupt_target_error,
        )
    )
    active = corrupt_target_error > (clean_target_error + min_target_increase)
    raw = F.relu(margin - (corrupt_expected_error - clean_expected_error))
    loss = torch.where(active, raw, torch.zeros_like(raw))

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction != "mean":
        raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")
    active_count = active.to(dtype=loss.dtype).sum()
    return loss.sum() / active_count.clamp_min(1.0)


@dataclass(frozen=True)
class EnsembleVarianceDecomposition:
    """Law-of-total-variance decomposition for an ensemble."""

    predictive_mean: torch.Tensor
    aleatoric_variance: torch.Tensor
    epistemic_variance: torch.Tensor
    total_variance: torch.Tensor
    schema_version: str = ENSEMBLE_VARIANCE_SCHEMA_VERSION

    CURRENT_SCHEMA_VERSION: ClassVar[str] = ENSEMBLE_VARIANCE_SCHEMA_VERSION


def decompose_ensemble_variance(
    member_means: torch.Tensor,
    member_aleatoric_variances: torch.Tensor,
    member_dim: int = 0,
) -> EnsembleVarianceDecomposition:
    """Decompose predictive variance across independently trained members.

    ``E_k[var(y|model_k)]`` is the aleatoric component and
    ``var_k(E[y|model_k])`` (population variance) is the epistemic component.
    """
    if member_means.shape != member_aleatoric_variances.shape:
        raise ValueError(
            "member means and variances must have identical shapes, got "
            f"{tuple(member_means.shape)} and "
            f"{tuple(member_aleatoric_variances.shape)}"
        )
    if member_means.ndim == 0:
        raise ValueError("ensemble tensors must include a member dimension")
    if torch.any(member_aleatoric_variances < 0):
        raise ValueError("member aleatoric variances must be non-negative")

    predictive_mean = member_means.mean(dim=member_dim)
    aleatoric_variance = member_aleatoric_variances.mean(dim=member_dim)
    epistemic_variance = member_means.var(dim=member_dim, unbiased=False)
    return EnsembleVarianceDecomposition(
        predictive_mean=predictive_mean,
        aleatoric_variance=aleatoric_variance,
        epistemic_variance=epistemic_variance,
        total_variance=aleatoric_variance + epistemic_variance,
    )


@dataclass(frozen=True)
class PathRiskAggregation:
    """Auditable result of top-q/CVaR aggregation over a route corridor."""

    risk: torch.Tensor
    risk_field: torch.Tensor
    selected_mask: torch.Tensor
    valid_cell_count: torch.Tensor
    selected_cell_count: torch.Tensor
    top_q: float
    schema_version: str = PATH_RISK_SCHEMA_VERSION

    CURRENT_SCHEMA_VERSION: ClassVar[str] = PATH_RISK_SCHEMA_VERSION


def cvar_path_risk(
    spatial_failure_probability: torch.Tensor,
    path_corridor_mask: torch.Tensor,
    occupancy_probability: Optional[torch.Tensor] = None,
    ttc_weight: Optional[torch.Tensor] = None,
    top_q: float = 0.2,
    spatial_ndim: int = 2,
) -> PathRiskAggregation:
    """Aggregate worst-tail risk only inside a supplied path corridor.

    Args:
        spatial_failure_probability: Calibrated dense failure probability.
        path_corridor_mask: Boolean mask, or non-negative fixed corridor weight.
            This is the only route-dependent input and is deliberately outside
            :class:`SpatialPatchUQHead`.
        occupancy_probability: Optional dense occupancy/object relevance.
        ttc_weight: Optional non-negative time-to-collision relevance weight.
        top_q: Fraction of valid corridor cells in the worst tail.  ``0.2``
            means the mean of the highest-risk 20 percent.
        spatial_ndim: Number of trailing dimensions to flatten as spatial cells.
            Leading dimensions (for example batch and trajectory mode) are
            independently aggregated.

    Returns:
        :class:`PathRiskAggregation`, including the selected cells and counts
        needed to audit each reported scalar risk.
    """
    if not 0.0 < top_q <= 1.0:
        raise ValueError("top_q must lie in (0, 1]")
    if spatial_ndim <= 0 or spatial_ndim > spatial_failure_probability.ndim:
        raise ValueError("spatial_ndim must select one or more existing dimensions")

    field = spatial_failure_probability
    corridor = path_corridor_mask.to(device=field.device)
    tensors = [field, corridor]
    if occupancy_probability is not None:
        tensors.append(occupancy_probability.to(device=field.device))
    if ttc_weight is not None:
        tensors.append(ttc_weight.to(device=field.device))
    tensors = list(torch.broadcast_tensors(*tensors))

    field = tensors[0]
    corridor = tensors[1]
    offset = 2
    occupancy = tensors[offset] if occupancy_probability is not None else torch.ones_like(field)
    offset += int(occupancy_probability is not None)
    ttc = tensors[offset] if ttc_weight is not None else torch.ones_like(field)

    corridor_weight = corridor.to(dtype=field.dtype)
    if torch.any(corridor_weight < 0):
        raise ValueError("path_corridor_mask weights must be non-negative")
    if torch.any(occupancy < 0):
        raise ValueError("occupancy_probability must be non-negative")
    if torch.any(ttc < 0):
        raise ValueError("ttc_weight must be non-negative")

    valid = corridor_weight > 0
    risk_field = field * corridor_weight * occupancy.to(field.dtype) * ttc.to(field.dtype)

    leading_shape = risk_field.shape[:-spatial_ndim]
    n_cells = math.prod(risk_field.shape[-spatial_ndim:])
    flat_risk = risk_field.reshape(*leading_shape, n_cells)
    flat_valid = valid.reshape(*leading_shape, n_cells)

    neg_inf = torch.full_like(flat_risk, -torch.inf)
    sortable = torch.where(flat_valid, flat_risk, neg_inf)
    sorted_risk, sorted_indices = sortable.sort(dim=-1, descending=True)

    valid_count = flat_valid.sum(dim=-1)
    selected_count = torch.ceil(valid_count.to(field.dtype) * top_q).to(torch.long)
    ranks = torch.arange(n_cells, device=field.device)
    ranks = ranks.reshape((1,) * len(leading_shape) + (n_cells,))
    selected_sorted = (ranks < selected_count.unsqueeze(-1)) & torch.isfinite(sorted_risk)
    selected_values = torch.where(selected_sorted, sorted_risk, torch.zeros_like(sorted_risk))
    denominator = selected_count.clamp_min(1).to(field.dtype)
    risk = selected_values.sum(dim=-1) / denominator
    risk = torch.where(valid_count > 0, risk, torch.zeros_like(risk))

    selected_flat = torch.zeros_like(flat_valid)
    selected_flat.scatter_(dim=-1, index=sorted_indices, src=selected_sorted)
    selected_mask = selected_flat.reshape(risk_field.shape)

    return PathRiskAggregation(
        risk=risk,
        risk_field=risk_field,
        selected_mask=selected_mask,
        valid_cell_count=valid_count,
        selected_cell_count=selected_count,
        top_q=float(top_q),
    )


__all__ = [
    "SPATIAL_UQ_SCHEMA_VERSION",
    "SpatialUQOutput",
    "SpatialPatchUQHead",
    "paired_cosine_representation_error",
    "heteroscedastic_gaussian_nll",
    "brier_loss",
    "paired_error_ranking_loss",
    "EnsembleVarianceDecomposition",
    "decompose_ensemble_variance",
    "PathRiskAggregation",
    "cvar_path_risk",
]

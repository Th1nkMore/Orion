"""Counterfactual evidence-loss targets and a task-agnostic spatial adapter.

The paired reference is privileged supervision only.  At inference the
adapter receives current/previous frozen visual features, view identity, and
patch coordinates; it never receives a route, hazard, corruption label, mask,
severity, or paired reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


COUNTERFACTUAL_EVIDENCE_SCHEMA_VERSION = "orion.counterfactual-evidence/v1"
EVIDENCE_COMPONENTS = (
    "persistent_direction",
    "persistent_magnitude",
    "transient_inconsistency",
)
CLAIM_BOUNDARY = {
    "quantity": "generic spatial observation evidence-loss proxy",
    "counterfactual_target_is_unique_uncertainty_truth": False,
    "task_relevance_is_output": False,
    "path_risk_is_output": False,
    "corruption_metadata_is_model_input": False,
    "paired_reference_is_inference_input": False,
    "actual_orion_failure_is_primary_target": False,
    "model_independence_claimed": False,
}


class CounterfactualEvidenceError(ValueError):
    """Raised when evidence tensors violate the frozen v1 contract."""


def _check_feature_sequence(
    current: torch.Tensor,
    previous: Optional[torch.Tensor],
    previous_valid: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if current.ndim != 5 or not current.is_floating_point():
        raise CounterfactualEvidenceError(
            "current must have floating [B,V,H,W,D] shape"
        )
    if not bool(torch.isfinite(current).all()):
        raise CounterfactualEvidenceError("current features must be finite")
    if previous is None:
        previous = torch.zeros_like(current)
    if previous.shape != current.shape or not previous.is_floating_point():
        raise CounterfactualEvidenceError("previous must match current features")
    if not bool(torch.isfinite(previous).all()):
        raise CounterfactualEvidenceError("previous features must be finite")
    if previous_valid is None:
        previous_valid = torch.zeros(
            current.shape[0], dtype=torch.bool, device=current.device
        )
    if previous_valid.shape != (current.shape[0],):
        raise CounterfactualEvidenceError("previous_valid must have shape [B]")
    return previous, previous_valid.to(device=current.device, dtype=torch.bool)


@dataclass(frozen=True)
class CounterfactualEvidenceTarget:
    values: torch.Tensor
    component_valid: torch.Tensor
    components: tuple[str, ...] = EVIDENCE_COMPONENTS
    schema_version: str = COUNTERFACTUAL_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.values.ndim != 5 or self.values.shape[-1] != len(self.components):
            raise CounterfactualEvidenceError(
                "evidence values must have [B,V,H,W,components] shape"
            )
        if self.component_valid.shape != self.values.shape:
            raise CounterfactualEvidenceError("component validity must match values")
        if self.component_valid.dtype != torch.bool:
            raise CounterfactualEvidenceError("component validity must be boolean")
        if not self.values.is_floating_point() or not bool(torch.isfinite(self.values).all()):
            raise CounterfactualEvidenceError("evidence values must be finite floating point")
        if bool((self.values < 0).any()):
            raise CounterfactualEvidenceError("evidence values must be non-negative")


def counterfactual_evidence_target(
    reference_current: torch.Tensor,
    observed_current: torch.Tensor,
    reference_previous: Optional[torch.Tensor] = None,
    observed_previous: Optional[torch.Tensor] = None,
    previous_valid: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> CounterfactualEvidenceTarget:
    """Measure persistent and transient feature evidence loss at equal pose.

    Direction and magnitude are complementary persistent components.  The
    transient component removes ordinary reference-scene motion by comparing
    observed temporal change with the paired reference temporal change.
    """

    if reference_current.shape != observed_current.shape:
        raise CounterfactualEvidenceError("reference and observed current features differ")
    reference_previous, valid = _check_feature_sequence(
        reference_current, reference_previous, previous_valid
    )
    observed_previous, observed_valid = _check_feature_sequence(
        observed_current, observed_previous, previous_valid
    )
    if not torch.equal(valid, observed_valid):  # defensive; both derive from one input
        raise CounterfactualEvidenceError("reference/observed temporal validity differs")

    direction = 1.0 - F.cosine_similarity(
        observed_current.float(), reference_current.float(), dim=-1, eps=eps
    ).clamp(-1.0, 1.0)
    observed_rms = observed_current.float().square().mean(dim=-1).clamp_min(eps).sqrt()
    reference_rms = reference_current.float().square().mean(dim=-1).clamp_min(eps).sqrt()
    magnitude = (observed_rms.log() - reference_rms.log()).abs()

    observed_change = 1.0 - F.cosine_similarity(
        observed_current.float(), observed_previous.float(), dim=-1, eps=eps
    ).clamp(-1.0, 1.0)
    reference_change = 1.0 - F.cosine_similarity(
        reference_current.float(), reference_previous.float(), dim=-1, eps=eps
    ).clamp(-1.0, 1.0)
    transient = (observed_change - reference_change).abs()
    valid_map = valid[:, None, None, None].expand_as(transient)
    transient = torch.where(valid_map, transient, torch.zeros_like(transient))

    values = torch.stack((direction, magnitude, transient), dim=-1)
    component_valid = torch.ones_like(values, dtype=torch.bool)
    component_valid[..., 2] = valid_map
    return CounterfactualEvidenceTarget(
        values=values.detach(), component_valid=component_valid
    )


def fit_counterfactual_component_scales(
    targets: Sequence[CounterfactualEvidenceTarget], quantile: float = 0.95
) -> torch.Tensor:
    """Fit one robust train-only scale per component."""

    if not targets or not 0.5 <= quantile < 1.0:
        raise CounterfactualEvidenceError("scale fitting needs targets and q in [0.5,1)")
    scales = []
    for component in range(len(EVIDENCE_COMPONENTS)):
        values = torch.cat(
            [
                target.values[..., component][target.component_valid[..., component]]
                .detach()
                .cpu()
                .float()
                .reshape(-1)
                for target in targets
            ]
        )
        if values.numel() == 0:
            raise CounterfactualEvidenceError("component scale has no valid targets")
        scales.append(torch.quantile(values, quantile).clamp_min(1e-4))
    return torch.stack(scales)


def scale_counterfactual_target(
    target: CounterfactualEvidenceTarget, scales: torch.Tensor
) -> CounterfactualEvidenceTarget:
    if scales.shape != (len(EVIDENCE_COMPONENTS),) or bool((scales <= 0).any()):
        raise CounterfactualEvidenceError("component scales have the wrong shape/value")
    values = target.values / scales.to(
        device=target.values.device, dtype=target.values.dtype
    )
    return CounterfactualEvidenceTarget(
        values=values,
        component_valid=target.component_valid,
        components=target.components,
    )


class ObservationEvidenceAdapter(nn.Module):
    """Predict three task-agnostic evidence-loss maps from observation features."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        max_views: int = 6,
        output_bias: Optional[float] = None,
    ):
        super().__init__()
        if min(feature_dim, hidden_dim, max_views) <= 0:
            raise CounterfactualEvidenceError("adapter dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_views = int(max_views)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.current_projection = nn.Conv2d(feature_dim, hidden_dim, 1)
        self.previous_projection = nn.Conv2d(feature_dim, hidden_dim, 1)
        self.scalar_projection = nn.Conv2d(3, hidden_dim, 1)
        self.coordinate_projection = nn.Conv2d(2, hidden_dim, 1)
        self.view_embedding = nn.Embedding(max_views, hidden_dim)
        self.previous_missing = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
        self.context = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4),
            nn.GELU(),
        )
        self.output_projection = nn.Conv2d(hidden_dim, len(EVIDENCE_COMPONENTS), 1)
        if output_bias is not None:
            if not isinstance(output_bias, (float, int)):
                raise CounterfactualEvidenceError("output bias must be numeric")
            nn.init.constant_(self.output_projection.bias, float(output_bias))

    def forward(
        self,
        current: torch.Tensor,
        previous: Optional[torch.Tensor] = None,
        previous_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        previous, valid = _check_feature_sequence(current, previous, previous_valid)
        batch, views, height, width, feature_dim = current.shape
        if feature_dim != self.feature_dim or views > self.max_views:
            raise CounterfactualEvidenceError("adapter input dimensions differ from config")

        current_float = current.float()
        previous_float = previous.float()
        current_norm = self.feature_norm(current_float)
        previous_norm = self.feature_norm(previous_float)
        current_chw = current_norm.permute(0, 1, 4, 2, 3).reshape(
            batch * views, feature_dim, height, width
        )
        previous_chw = previous_norm.permute(0, 1, 4, 2, 3).reshape(
            batch * views, feature_dim, height, width
        )
        hidden = self.current_projection(current_chw)
        temporal = self.previous_projection(previous_chw)
        valid_bv = valid[:, None].expand(batch, views).reshape(batch * views, 1, 1, 1)
        temporal = torch.where(
            valid_bv, temporal, self.previous_missing.to(dtype=temporal.dtype)
        )

        change = 1.0 - F.cosine_similarity(
            current_float, previous_float, dim=-1, eps=1e-6
        ).clamp(-1.0, 1.0)
        change = torch.where(
            valid[:, None, None, None], change, torch.zeros_like(change)
        )
        current_log_rms = current_float.square().mean(dim=-1).clamp_min(1e-6).sqrt().log()
        previous_log_rms = previous_float.square().mean(dim=-1).clamp_min(1e-6).sqrt().log()
        previous_log_rms = torch.where(
            valid[:, None, None, None],
            previous_log_rms,
            torch.zeros_like(previous_log_rms),
        )
        scalars = torch.stack((change, current_log_rms, previous_log_rms), dim=2).reshape(
            batch * views, 3, height, width
        )

        y = torch.linspace(-1.0, 1.0, height, device=current.device)
        x = torch.linspace(-1.0, 1.0, width, device=current.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((yy, xx), dim=0).to(dtype=hidden.dtype)
        coordinates = coordinates[None].expand(batch * views, -1, -1, -1)
        view_ids = torch.arange(views, device=current.device)
        view_ids = view_ids[None].expand(batch, views).reshape(-1)
        view_context = self.view_embedding(view_ids).view(
            batch * views, self.hidden_dim, 1, 1
        )
        fused = (
            hidden
            + temporal
            + self.scalar_projection(scalars)
            + self.coordinate_projection(coordinates)
            + view_context
        )
        output = F.softplus(self.output_projection(self.context(fused)))
        return output.reshape(
            batch, views, len(EVIDENCE_COMPONENTS), height, width
        ).permute(0, 1, 3, 4, 2)


@dataclass(frozen=True)
class EvidenceHurdlePrediction:
    """Factor evidence loss into occurrence probability and conditional size."""

    score: torch.Tensor
    presence_probability: torch.Tensor
    presence_logits: torch.Tensor
    conditional_magnitude: torch.Tensor


class ObservationEvidenceHurdleAdapter(nn.Module):
    """Predict sparse evidence loss with separate presence and magnitude heads.

    The inputs remain task agnostic.  The extra head is an architectural
    response to sparse paired targets: it must first decide whether evidence
    loss is measurable at a cell, then estimate its size conditional on that
    event.  ``forward`` returns their product for drop-in evaluation.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        max_views: int = 6,
        presence_bias: float = -3.0,
        magnitude_bias: float = 0.0,
        use_view_embedding: bool = True,
    ):
        super().__init__()
        if min(feature_dim, hidden_dim, max_views) <= 0:
            raise CounterfactualEvidenceError("adapter dimensions must be positive")
        if not isinstance(use_view_embedding, bool):
            raise CounterfactualEvidenceError("use_view_embedding must be boolean")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_views = int(max_views)
        self.use_view_embedding = use_view_embedding
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.current_projection = nn.Conv2d(feature_dim, hidden_dim, 1)
        self.previous_projection = nn.Conv2d(feature_dim, hidden_dim, 1)
        self.scalar_projection = nn.Conv2d(3, hidden_dim, 1)
        self.coordinate_projection = nn.Conv2d(2, hidden_dim, 1)
        self.view_embedding = (
            nn.Embedding(max_views, hidden_dim) if use_view_embedding else None
        )
        self.previous_missing = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
        self.context = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4),
            nn.GELU(),
        )
        self.presence_projection = nn.Conv2d(
            hidden_dim, len(EVIDENCE_COMPONENTS), 1
        )
        self.magnitude_projection = nn.Conv2d(
            hidden_dim, len(EVIDENCE_COMPONENTS), 1
        )
        nn.init.constant_(self.presence_projection.bias, float(presence_bias))
        nn.init.constant_(self.magnitude_projection.bias, float(magnitude_bias))

    def _context_features(
        self,
        current: torch.Tensor,
        previous: Optional[torch.Tensor],
        previous_valid: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, int, int, int, int]:
        previous, valid = _check_feature_sequence(current, previous, previous_valid)
        batch, views, height, width, feature_dim = current.shape
        if feature_dim != self.feature_dim or views > self.max_views:
            raise CounterfactualEvidenceError("adapter input dimensions differ from config")

        current_float = current.float()
        previous_float = previous.float()
        current_chw = self.feature_norm(current_float).permute(0, 1, 4, 2, 3).reshape(
            batch * views, feature_dim, height, width
        )
        previous_chw = self.feature_norm(previous_float).permute(
            0, 1, 4, 2, 3
        ).reshape(batch * views, feature_dim, height, width)
        hidden = self.current_projection(current_chw)
        temporal = self.previous_projection(previous_chw)
        valid_bv = valid[:, None].expand(batch, views).reshape(batch * views, 1, 1, 1)
        temporal = torch.where(
            valid_bv, temporal, self.previous_missing.to(dtype=temporal.dtype)
        )

        change = 1.0 - F.cosine_similarity(
            current_float, previous_float, dim=-1, eps=1e-6
        ).clamp(-1.0, 1.0)
        change = torch.where(
            valid[:, None, None, None], change, torch.zeros_like(change)
        )
        current_log_rms = current_float.square().mean(dim=-1).clamp_min(1e-6).sqrt().log()
        previous_log_rms = previous_float.square().mean(dim=-1).clamp_min(1e-6).sqrt().log()
        previous_log_rms = torch.where(
            valid[:, None, None, None], previous_log_rms, torch.zeros_like(previous_log_rms)
        )
        scalars = torch.stack((change, current_log_rms, previous_log_rms), dim=2).reshape(
            batch * views, 3, height, width
        )

        y = torch.linspace(-1.0, 1.0, height, device=current.device)
        x = torch.linspace(-1.0, 1.0, width, device=current.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((yy, xx), dim=0).to(dtype=hidden.dtype)
        coordinates = coordinates[None].expand(batch * views, -1, -1, -1)
        if self.view_embedding is None:
            view_context = hidden.new_zeros(batch * views, self.hidden_dim, 1, 1)
        else:
            view_ids = torch.arange(views, device=current.device)
            view_ids = view_ids[None].expand(batch, views).reshape(-1)
            view_context = self.view_embedding(view_ids).view(
                batch * views, self.hidden_dim, 1, 1
            )
        fused = (
            hidden
            + temporal
            + self.scalar_projection(scalars)
            + self.coordinate_projection(coordinates)
            + view_context
        )
        return self.context(fused), batch, views, height, width

    def predict_parts(
        self,
        current: torch.Tensor,
        previous: Optional[torch.Tensor] = None,
        previous_valid: Optional[torch.Tensor] = None,
    ) -> EvidenceHurdlePrediction:
        context, batch, views, height, width = self._context_features(
            current, previous, previous_valid
        )

        def restore(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(
                batch, views, len(EVIDENCE_COMPONENTS), height, width
            ).permute(0, 1, 3, 4, 2)

        presence_logits = restore(self.presence_projection(context))
        presence_probability = torch.sigmoid(presence_logits)
        conditional_magnitude = F.softplus(restore(self.magnitude_projection(context)))
        return EvidenceHurdlePrediction(
            score=presence_probability * conditional_magnitude,
            presence_probability=presence_probability,
            presence_logits=presence_logits,
            conditional_magnitude=conditional_magnitude,
        )

    def forward(
        self,
        current: torch.Tensor,
        previous: Optional[torch.Tensor] = None,
        previous_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.predict_parts(current, previous, previous_valid).score


def counterfactual_evidence_regression_loss(
    prediction: torch.Tensor,
    target: CounterfactualEvidenceTarget,
) -> torch.Tensor:
    if prediction.shape != target.values.shape:
        raise CounterfactualEvidenceError("prediction and evidence target shapes differ")
    per_cell = F.smooth_l1_loss(
        torch.log1p(prediction), torch.log1p(target.values), reduction="none"
    )
    target_weight = 1.0 + 3.0 * target.values / (1.0 + target.values)
    valid_weight = target_weight * target.component_valid.to(target_weight.dtype)
    return (per_cell * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)


def balanced_counterfactual_evidence_regression_loss(
    prediction: torch.Tensor,
    target: CounterfactualEvidenceTarget,
    responsive_weight: float = 0.75,
    response_floor: float = 1e-6,
) -> torch.Tensor:
    """Balance measured-responsive and background cells per component.

    Local single-camera interventions make exact-zero cells the large majority.
    Normalizing the two populations independently removes the trivial all-zero
    optimum without reading a corruption mask.  High-amplitude responsive cells
    retain the same bounded target-derived weighting as the original loss.
    """

    if prediction.shape != target.values.shape:
        raise CounterfactualEvidenceError("prediction and evidence target shapes differ")
    if not 0.0 < responsive_weight < 1.0 or response_floor < 0:
        raise CounterfactualEvidenceError("invalid balanced regression settings")
    per_cell = F.smooth_l1_loss(
        torch.log1p(prediction), torch.log1p(target.values), reduction="none"
    )
    component_losses = []
    for component in range(len(EVIDENCE_COMPONENTS)):
        valid = target.component_valid[..., component]
        responsive = valid & (target.values[..., component] > response_floor)
        background = valid & ~responsive
        terms = []
        weights = []
        if bool(responsive.any()):
            amplitude_weight = 1.0 + 3.0 * (
                target.values[..., component]
                / (1.0 + target.values[..., component])
            )
            selected_weight = amplitude_weight[responsive]
            terms.append(
                (per_cell[..., component][responsive] * selected_weight).sum()
                / selected_weight.sum().clamp_min(1.0)
            )
            weights.append(responsive_weight)
        if bool(background.any()):
            terms.append(per_cell[..., component][background].mean())
            weights.append(1.0 - responsive_weight)
        if terms:
            normalizer = sum(weights)
            component_losses.append(
                sum(weight * term for weight, term in zip(weights, terms))
                / normalizer
            )
    if not component_losses:
        raise CounterfactualEvidenceError("balanced regression has no valid cells")
    return torch.stack(component_losses).mean()


def balanced_evidence_presence_loss(
    presence_logits: torch.Tensor,
    target: CounterfactualEvidenceTarget,
    responsive_weight: float = 0.5,
    response_floor: float = 1e-6,
    support_thresholds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Classify target-derived support without using a corruption mask.

    ``support_thresholds`` enables a train-only high-response definition.  If
    omitted, the legacy numerical-nonzero footprint is retained only for
    reproducibility of the stopped v1 smoke.
    """

    if presence_logits.shape != target.values.shape:
        raise CounterfactualEvidenceError("presence logits and target shapes differ")
    if not 0.0 < responsive_weight < 1.0 or response_floor < 0:
        raise CounterfactualEvidenceError("invalid presence-loss settings")
    if support_thresholds is None:
        threshold = torch.full(
            (len(EVIDENCE_COMPONENTS),),
            float(response_floor),
            dtype=target.values.dtype,
            device=target.values.device,
        )
    else:
        if (
            support_thresholds.shape != (len(EVIDENCE_COMPONENTS),)
            or not bool(torch.isfinite(support_thresholds).all())
            or bool((support_thresholds <= response_floor).any())
        ):
            raise CounterfactualEvidenceError("invalid high-response support thresholds")
        threshold = support_thresholds.to(
            device=target.values.device, dtype=target.values.dtype
        )
    labels = (target.values > threshold).to(presence_logits.dtype)
    per_cell = F.binary_cross_entropy_with_logits(
        presence_logits, labels, reduction="none"
    )
    component_losses = []
    for component in range(len(EVIDENCE_COMPONENTS)):
        valid = target.component_valid[..., component]
        responsive = valid & labels[..., component].bool()
        background = valid & ~responsive
        terms = []
        weights = []
        if bool(responsive.any()):
            terms.append(per_cell[..., component][responsive].mean())
            weights.append(responsive_weight)
        if bool(background.any()):
            terms.append(per_cell[..., component][background].mean())
            weights.append(1.0 - responsive_weight)
        if terms:
            component_losses.append(
                sum(weight * term for weight, term in zip(weights, terms))
                / sum(weights)
            )
    if not component_losses:
        raise CounterfactualEvidenceError("presence loss has no valid cells")
    return torch.stack(component_losses).mean()


def responsive_evidence_magnitude_loss(
    conditional_magnitude: torch.Tensor,
    target: CounterfactualEvidenceTarget,
    response_floor: float = 1e-6,
) -> torch.Tensor:
    """Regress amplitude only where paired features measured a response."""

    if conditional_magnitude.shape != target.values.shape:
        raise CounterfactualEvidenceError("magnitude and target shapes differ")
    if response_floor < 0:
        raise CounterfactualEvidenceError("invalid magnitude-loss response floor")
    responsive = target.component_valid & (target.values > response_floor)
    if not bool(responsive.any()):
        raise CounterfactualEvidenceError("magnitude loss has no responsive cells")
    per_cell = F.smooth_l1_loss(
        torch.log1p(conditional_magnitude),
        torch.log1p(target.values),
        reduction="none",
    )
    amplitude_weight = 1.0 + 3.0 * target.values / (1.0 + target.values)
    selected_weight = amplitude_weight[responsive]
    return (
        (per_cell[responsive] * selected_weight).sum()
        / selected_weight.sum().clamp_min(1.0)
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "COUNTERFACTUAL_EVIDENCE_SCHEMA_VERSION",
    "EVIDENCE_COMPONENTS",
    "EvidenceHurdlePrediction",
    "CounterfactualEvidenceError",
    "CounterfactualEvidenceTarget",
    "ObservationEvidenceAdapter",
    "ObservationEvidenceHurdleAdapter",
    "balanced_evidence_presence_loss",
    "balanced_counterfactual_evidence_regression_loss",
    "counterfactual_evidence_regression_loss",
    "counterfactual_evidence_target",
    "fit_counterfactual_component_scales",
    "responsive_evidence_magnitude_loss",
    "scale_counterfactual_target",
]

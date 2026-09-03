"""Stage2-L tokenizer and explicit task-relevance map primitives.

The tokenizer receives only normalized task-agnostic Stage-1 component maps.
Route and visual semantics enter later through the VLM.  The relevance head is
applied to VLM-fused hidden states at the UQ token positions; it is not a
standalone planning or yield controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA = "orion.uq_relevance_tokenizer.v1"


@dataclass(frozen=True)
class UQRelevanceTokens:
    tokens: torch.Tensor
    token_grid: torch.Tensor
    temporal_summary: torch.Tensor
    latest_scalar_uq: torch.Tensor
    schema: str = SCHEMA


@dataclass(frozen=True)
class TaskRiskBridgeTokens:
    """Compact, inspectable K summaries passed back to the language model."""

    tokens: torch.Tensor
    task_risk: torch.Tensor
    per_view_features: torch.Tensor
    global_features: torch.Tensor


class UQComponentTokenizer(nn.Module):
    """Convert ``[B,T,V,H,W,C]`` maps into view/space-aware VLM tokens."""

    def __init__(
        self,
        component_dim: int = 3,
        model_dim: int = 4096,
        grid_hw: Tuple[int, int] = (10, 10),
        max_views: int = 6,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if min(component_dim, model_dim, max_views, hidden_dim) <= 0:
            raise ValueError("tokenizer dimensions must be positive")
        if len(grid_hw) != 2 or min(map(int, grid_hw)) <= 0:
            raise ValueError("grid_hw must contain two positive values")
        self.component_dim = int(component_dim)
        self.model_dim = int(model_dim)
        self.grid_hw = tuple(map(int, grid_hw))
        self.max_views = int(max_views)
        summary_dim = 3 * self.component_dim
        self.summary_norm = nn.LayerNorm(summary_dim)
        self.summary_projection = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
        )
        self.coordinate_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
        )
        self.view_embedding = nn.Embedding(max_views, model_dim)
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, components: torch.Tensor) -> UQRelevanceTokens:
        if components.ndim != 6 or components.shape[-1] != self.component_dim:
            raise ValueError("components must have shape [B,T,V,H,W,C]")
        if not components.is_floating_point() or not bool(torch.isfinite(components).all()):
            raise ValueError("components must be finite floating point")
        if bool((components < 0).any()) or bool((components > 1).any()):
            raise ValueError("normalized components must lie in [0,1]")
        batch, time, views, height, width, component_dim = components.shape
        if views > self.max_views or time <= 0:
            raise ValueError("view/time dimensions are invalid")
        grid_h, grid_w = self.grid_hw
        pooled = components.permute(0, 1, 2, 5, 3, 4).reshape(
            batch * time * views, component_dim, height, width
        )
        pooled = F.adaptive_avg_pool2d(pooled, self.grid_hw)
        pooled = pooled.reshape(
            batch, time, views, component_dim, grid_h, grid_w
        ).permute(0, 1, 2, 4, 5, 3)
        latest = pooled[:, -1]
        mean = pooled.mean(dim=1)
        delta = latest - pooled[:, 0]
        summary = torch.cat((latest, mean, delta), dim=-1)

        y = torch.linspace(-1.0, 1.0, grid_h, device=components.device)
        x = torch.linspace(-1.0, 1.0, grid_w, device=components.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((yy, xx), dim=-1).to(dtype=components.dtype)
        position = self.coordinate_projection(coordinates)
        view_ids = torch.arange(views, device=components.device)
        view = self.view_embedding(view_ids)[:, None, None, :]
        token_grid = self.summary_projection(self.summary_norm(summary))
        token_grid = self.output_norm(
            token_grid + position[None, None] + view[None]
        )
        tokens = token_grid.reshape(batch, views * grid_h * grid_w, self.model_dim)
        latest_scalar = latest.mean(dim=-1)
        return UQRelevanceTokens(
            tokens=tokens,
            token_grid=token_grid,
            temporal_summary=summary,
            latest_scalar_uq=latest_scalar,
        )


class TaskRelevanceMapHead(nn.Module):
    """Decode explicit R logits from VLM-fused UQ-token hidden states."""

    def __init__(self, model_dim: int = 4096, hidden_dim: int = 256) -> None:
        super().__init__()
        if min(model_dim, hidden_dim) <= 0:
            raise ValueError("relevance head dimensions must be positive")
        self.net = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fused_token_grid: torch.Tensor) -> torch.Tensor:
        if fused_token_grid.ndim != 5:
            raise ValueError("fused_token_grid must have shape [B,V,H,W,D]")
        return self.net(fused_token_grid).squeeze(dim=-1)


class SpatialTaskRelevanceQueryTokenizer(nn.Module):
    """Create U-independent view/grid queries for the first VLM pass.

    These queries acquire task relevance by attending to ORION visual tokens
    and route-language context.  They never receive Stage1 uncertainty, which
    keeps the definitions of U and R structurally separate.
    """

    def __init__(
        self,
        model_dim: int = 4096,
        hidden_dim: int = 256,
        grid_hw: Tuple[int, int] = (10, 10),
        max_views: int = 6,
    ) -> None:
        super().__init__()
        if min(model_dim, hidden_dim, max_views) <= 0:
            raise ValueError("query tokenizer dimensions must be positive")
        if len(grid_hw) != 2 or min(map(int, grid_hw)) <= 0:
            raise ValueError("grid_hw must contain two positive values")
        self.model_dim = int(model_dim)
        self.grid_hw = tuple(map(int, grid_hw))
        self.max_views = int(max_views)
        self.base_query = nn.Parameter(torch.zeros(model_dim))
        self.coordinate_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
        )
        self.view_embedding = nn.Embedding(max_views, model_dim)
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        batch_size: int,
        views: int,
        *,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        if batch_size <= 0 or views <= 0 or views > self.max_views:
            raise ValueError("query batch/view dimensions are invalid")
        reference = self.base_query
        device = reference.device if device is None else device
        dtype = reference.dtype if dtype is None else dtype
        grid_h, grid_w = self.grid_hw
        y = torch.linspace(-1.0, 1.0, grid_h, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, grid_w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((yy, xx), dim=-1)
        position = self.coordinate_projection(coordinates)
        view_ids = torch.arange(views, device=device)
        grid = (
            self.base_query.to(device=device, dtype=dtype)[None, None, None, :]
            + position[None]
            + self.view_embedding(view_ids).to(dtype=dtype)[:, None, None, :]
        )
        grid = self.output_norm(grid)
        return grid.reshape(1, views * grid_h * grid_w, self.model_dim).expand(
            batch_size, -1, -1
        )


class ViewAlignedTaskRelevanceQueryTokenizer(nn.Module):
    """Bind every R query to the matching frozen ORION camera feature cell.

    The input is a six-view, 10x10 feature grid from ORION's frozen image
    backbone.  Each output query sees only the feature at its own canonical
    ``[view, y, x]`` slot before the VLM performs global fusion with the route
    text and native det/map tokens.  Observation uncertainty is never an
    input, so task relevance remains VLM-owned and structurally separate from
    Stage-1 U.
    """

    def __init__(
        self,
        model_dim: int = 4096,
        image_feature_dim: int = 1024,
        hidden_dim: int = 256,
        grid_hw: Tuple[int, int] = (10, 10),
        max_views: int = 6,
    ) -> None:
        super().__init__()
        if min(model_dim, image_feature_dim, hidden_dim, max_views) <= 0:
            raise ValueError("view-aligned query dimensions must be positive")
        if len(grid_hw) != 2 or min(map(int, grid_hw)) <= 0:
            raise ValueError("grid_hw must contain two positive values")
        self.model_dim = int(model_dim)
        self.image_feature_dim = int(image_feature_dim)
        self.grid_hw = tuple(map(int, grid_hw))
        self.max_views = int(max_views)
        self.base_query = nn.Parameter(torch.zeros(model_dim))
        self.coordinate_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
        )
        self.view_embedding = nn.Embedding(max_views, model_dim)
        self.evidence_norm = nn.LayerNorm(image_feature_dim)
        self.evidence_projection = nn.Sequential(
            nn.Linear(image_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        if view_features.ndim != 5:
            raise ValueError("view_features must have shape [B,V,H,W,C]")
        batch, views, grid_h, grid_w, channels = view_features.shape
        if (
            batch <= 0
            or views <= 0
            or views > self.max_views
            or (grid_h, grid_w) != self.grid_hw
            or channels != self.image_feature_dim
        ):
            raise ValueError("view-aligned feature shape differs from the contract")
        if not view_features.is_floating_point() or not bool(
            torch.isfinite(view_features).all()
        ):
            raise ValueError("view-aligned features must be finite floating tensors")
        dtype = self.base_query.dtype
        device = self.base_query.device
        features = view_features.to(device=device, dtype=dtype, non_blocking=True)
        y = torch.linspace(-1.0, 1.0, grid_h, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, grid_w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((yy, xx), dim=-1)
        position = self.coordinate_projection(coordinates)
        view_ids = torch.arange(views, device=device)
        evidence = self.evidence_projection(self.evidence_norm(features))
        grid = self.output_norm(
            self.base_query[None, None, None, None]
            + position[None, None]
            + self.view_embedding(view_ids)[None, :, None, None]
            + evidence
        )
        return grid.reshape(batch, views * grid_h * grid_w, self.model_dim)


class TaskRiskLanguageBridge(nn.Module):
    """Encode ``K = U * R`` as six view tokens and one global token.

    The bridge never receives R or U as separate features.  Consequently a
    zero-U input produces the same bridge representation for every R map and
    cannot by itself request conservative behavior.  The output is a language
    conditioning representation, not a brake/steering command or governor.
    """

    def __init__(
        self,
        model_dim: int = 4096,
        hidden_dim: int = 256,
        max_views: int = 6,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(model_dim, hidden_dim, max_views) <= 0 or epsilon <= 0.0:
            raise ValueError("bridge dimensions and epsilon must be positive")
        self.model_dim = int(model_dim)
        self.max_views = int(max_views)
        self.epsilon = float(epsilon)
        # [max(K), mean(K), rms(K), soft_y(K), soft_x(K), view_coordinate]
        self.feature_norm = nn.LayerNorm(6)
        self.feature_projection = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
        )
        self.view_embedding = nn.Embedding(max_views, model_dim)
        self.token_type_embedding = nn.Embedding(2, model_dim)
        self.output_norm = nn.LayerNorm(model_dim)

    def _summaries(
        self, task_risk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return spatial_signal_summaries(task_risk, epsilon=self.epsilon)

    def forward(
        self,
        latest_scalar_uq: torch.Tensor,
        relevance_logits: torch.Tensor,
    ) -> TaskRiskBridgeTokens:
        if latest_scalar_uq.ndim != 4:
            raise ValueError("bridge U and R must have shape [B,V,H,W]")
        if latest_scalar_uq.shape != relevance_logits.shape:
            raise ValueError("bridge U and R shapes differ")
        if latest_scalar_uq.shape[1] > self.max_views:
            raise ValueError("bridge input exceeds max_views")
        if not (
            bool(torch.isfinite(latest_scalar_uq).all())
            and bool(torch.isfinite(relevance_logits).all())
        ):
            raise ValueError("bridge inputs must be finite")

        task_risk = fixed_task_risk(latest_scalar_uq, relevance_logits)
        per_view, global_features = self._summaries(task_risk)
        batch, views = per_view.shape[:2]
        projected_view = self.feature_projection(self.feature_norm(per_view))
        projected_global = self.feature_projection(
            self.feature_norm(global_features)
        ).unsqueeze(1)
        view_ids = torch.arange(views, device=task_risk.device)
        projected_view = (
            projected_view
            + self.view_embedding(view_ids)[None]
            + self.token_type_embedding.weight[0][None, None]
        )
        projected_global = (
            projected_global
            + self.token_type_embedding.weight[1][None, None]
        )
        tokens = self.output_norm(
            torch.cat((projected_view, projected_global), dim=1)
        )
        if tokens.shape != (batch, views + 1, self.model_dim):
            raise RuntimeError("task-risk bridge token shape is malformed")
        return TaskRiskBridgeTokens(
            tokens=tokens,
            task_risk=task_risk,
            per_view_features=per_view,
            global_features=global_features,
        )


def task_relevance_loss(
    relevance_logits: torch.Tensor,
    relevance_target: torch.Tensor,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    if relevance_logits.shape != relevance_target.shape:
        raise ValueError("relevance logits and target shapes differ")
    if not relevance_target.is_floating_point() or bool((relevance_target < 0).any()) or bool((relevance_target > 1).any()):
        raise ValueError("soft relevance target must lie in [0,1]")
    loss = F.binary_cross_entropy_with_logits(
        relevance_logits, relevance_target, reduction="none"
    )
    if valid_mask is None:
        return loss.mean()
    if valid_mask.shape != loss.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean and match relevance shape")
    return loss[valid_mask].mean() if bool(valid_mask.any()) else loss.sum() * 0.0


def fixed_task_risk(
    latest_scalar_uq: torch.Tensor,
    relevance_logits: torch.Tensor,
) -> torch.Tensor:
    if latest_scalar_uq.shape != relevance_logits.shape:
        raise ValueError("U and R shapes differ")
    if bool((latest_scalar_uq < 0).any()) or bool((latest_scalar_uq > 1).any()):
        raise ValueError("latest scalar UQ must lie in [0,1]")
    return latest_scalar_uq * relevance_logits.sigmoid()


def spatial_signal_summaries(
    signal: torch.Tensor, *, epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return shared six-feature summaries for U or K spatial signals."""

    if signal.ndim != 4:
        raise ValueError("spatial signal must have shape [B,V,H,W]")
    if epsilon <= 0.0 or not bool(torch.isfinite(signal).all()):
        raise ValueError("spatial signal and epsilon must be finite and valid")
    if bool((signal < 0.0).any()):
        raise ValueError("spatial signal must be non-negative")
    batch, views, height, width = signal.shape
    dtype = signal.dtype
    device = signal.device
    y = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
    x = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    mass = signal.sum(dim=(-2, -1))
    denominator = mass + epsilon
    soft_y = (signal * yy).sum(dim=(-2, -1)) / denominator
    soft_x = (signal * xx).sum(dim=(-2, -1)) / denominator
    view_coordinate = torch.linspace(
        -1.0, 1.0, views, dtype=dtype, device=device
    ).expand(batch, -1)
    per_view = torch.stack(
        (
            signal.amax(dim=(-2, -1)),
            signal.mean(dim=(-2, -1)),
            (signal.square().mean(dim=(-2, -1)) + epsilon).sqrt(),
            soft_y,
            soft_x,
            view_coordinate,
        ),
        dim=-1,
    )

    global_mass = mass.sum(dim=-1)
    global_denominator = global_mass + epsilon
    global_soft_y = (soft_y * mass).sum(dim=-1) / global_denominator
    global_soft_x = (soft_x * mass).sum(dim=-1) / global_denominator
    global_soft_view = (
        view_coordinate * mass
    ).sum(dim=-1) / global_denominator
    global_features = torch.stack(
        (
            signal.flatten(1).amax(dim=-1),
            signal.flatten(1).mean(dim=-1),
            (signal.flatten(1).square().mean(dim=-1) + epsilon).sqrt(),
            global_soft_y,
            global_soft_x,
            global_soft_view,
        ),
        dim=-1,
    )
    return per_view, global_features


def matched_task_risk_ranking_loss(
    on_path_risk: torch.Tensor,
    off_path_risk: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    if on_path_risk.shape != off_path_risk.shape:
        raise ValueError("matched risk tensors must have equal shapes")
    if margin < 0.0:
        raise ValueError("ranking margin must be non-negative")
    on_score = on_path_risk.flatten(1).amax(dim=1)
    off_score = off_path_risk.flatten(1).amax(dim=1)
    return F.relu(float(margin) - on_score + off_score).mean()


__all__ = [
    "SCHEMA",
    "TaskRelevanceMapHead",
    "SpatialTaskRelevanceQueryTokenizer",
    "ViewAlignedTaskRelevanceQueryTokenizer",
    "TaskRiskBridgeTokens",
    "TaskRiskLanguageBridge",
    "UQComponentTokenizer",
    "UQRelevanceTokens",
    "fixed_task_risk",
    "matched_task_risk_ranking_loss",
    "spatial_signal_summaries",
    "task_relevance_loss",
]

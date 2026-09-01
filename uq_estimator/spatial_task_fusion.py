"""Task-aware fusion for frozen spatial observation-uncertainty maps.

This module deliberately stops driving gradients at the Stage-1 observation
map.  It turns a multi-view map into compact, auditable tokens and lets a
Stage-2 task adapter fuse them with ORION planning context.  Observation UQ,
task risk, and the resulting trajectory response remain separate tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


STAGE2_TASK_FUSION_CHECKPOINT_SCHEMA = "orion.stage2-spatial-task-fusion/v1"


@dataclass(frozen=True)
class SpatialUQTokens:
    """Selected UQ tokens plus indices needed to render them back to images."""

    tokens: torch.Tensor
    scores: torch.Tensor
    flat_indices: torch.Tensor
    camera_indices: torch.Tensor
    row_indices: torch.Tensor
    column_indices: torch.Tensor
    source_shape: tuple[int, int, int]


@dataclass(frozen=True)
class TaskRiskTrajectoryOutput:
    """Stage-2 outputs; none of these tensors is a Stage-1 UQ target."""

    conditioned_context: torch.Tensor
    yield_logits: torch.Tensor
    conflict_logits: torch.Tensor
    trajectory_residual: torch.Tensor
    token_attention: torch.Tensor


class SpatialUQTokenProjector(nn.Module):
    """Compress every camera map while retaining view and cell provenance.

    A fixed number of cells is selected independently from each camera so a
    numerically large nuisance view cannot suppress all other views.  The UQ
    input is always detached: collision or trajectory losses may train this
    projector, but they cannot redefine the Stage-1 observation signal.
    """

    def __init__(
        self,
        *,
        component_dim: int = 3,
        model_dim: int = 256,
        hidden_dim: int = 256,
        max_views: int = 8,
        tokens_per_view: int = 8,
        score_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if min(component_dim, model_dim, hidden_dim, max_views) <= 0:
            raise ValueError("projector dimensions must be positive")
        if tokens_per_view <= 0:
            raise ValueError("tokens_per_view must be positive")
        if not math.isfinite(score_scale) or score_scale <= 0:
            raise ValueError("score_scale must be finite and positive")
        self.component_dim = int(component_dim)
        self.model_dim = int(model_dim)
        self.max_views = int(max_views)
        self.tokens_per_view = int(tokens_per_view)
        self.score_scale = float(score_scale)

        self.component_projector = nn.Sequential(
            nn.LayerNorm(self.component_dim),
            nn.Linear(self.component_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.model_dim),
        )
        self.position_projector = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.model_dim),
        )
        self.camera_embedding = nn.Embedding(self.max_views, self.model_dim)
        self.output_norm = nn.LayerNorm(self.model_dim)

    def forward(self, observation_uq: torch.Tensor) -> SpatialUQTokens:
        if observation_uq.ndim != 5:
            raise ValueError("observation_uq must have shape [B,V,H,W,K]")
        batch, views, height, width, components = observation_uq.shape
        if components != self.component_dim:
            raise ValueError(
                f"expected component_dim={self.component_dim}, got {components}"
            )
        if views <= 0 or views > self.max_views:
            raise ValueError("view count is outside configured max_views")
        if height <= 0 or width <= 0:
            raise ValueError("spatial dimensions must be positive")
        if height * width < self.tokens_per_view:
            raise ValueError("tokens_per_view exceeds cells available per view")
        if not observation_uq.is_floating_point():
            raise ValueError("observation_uq must be floating point")
        if not bool(torch.isfinite(observation_uq).all()):
            raise ValueError("observation_uq must be finite")
        if bool((observation_uq < 0).any()):
            raise ValueError("observation_uq components must be non-negative")

        # This detach is the architectural boundary between Stage 1 and Stage 2.
        frozen_uq = observation_uq.detach()
        per_view_components = frozen_uq.reshape(
            batch, views, height * width, components
        )
        per_view_scores = per_view_components.mean(dim=-1)
        scores, within_view_indices = torch.topk(
            per_view_scores,
            k=self.tokens_per_view,
            dim=-1,
            largest=True,
            sorted=True,
        )
        gather_index = within_view_indices.unsqueeze(-1).expand(
            -1, -1, -1, components
        )
        selected_components = torch.gather(
            per_view_components, 2, gather_index
        )

        camera_indices = torch.arange(
            views, device=observation_uq.device, dtype=torch.long
        ).view(1, views, 1).expand(batch, -1, self.tokens_per_view)
        row_indices = torch.div(
            within_view_indices, width, rounding_mode="floor"
        )
        column_indices = within_view_indices.remainder(width)
        row_denominator = max(height - 1, 1)
        column_denominator = max(width - 1, 1)
        coordinates = torch.stack(
            (
                2.0 * row_indices.to(observation_uq.dtype) / row_denominator
                - 1.0,
                2.0
                * column_indices.to(observation_uq.dtype)
                / column_denominator
                - 1.0,
            ),
            dim=-1,
        )

        parameter_dtype = self.component_projector[1].weight.dtype
        content = self.component_projector(
            selected_components.to(dtype=parameter_dtype)
        )
        content = content + self.position_projector(
            coordinates.to(dtype=parameter_dtype)
        )
        content = content + self.camera_embedding(camera_indices)
        # Exact zero UQ must produce exact zero tokens despite positional terms.
        gate = scores.to(dtype=parameter_dtype) / (
            scores.to(dtype=parameter_dtype) + self.score_scale
        )
        tokens = gate.unsqueeze(-1) * self.output_norm(content)

        flat_indices = camera_indices * (height * width) + within_view_indices
        flatten = lambda value: value.reshape(batch, views * self.tokens_per_view)
        return SpatialUQTokens(
            tokens=tokens.reshape(
                batch, views * self.tokens_per_view, self.model_dim
            ),
            scores=flatten(scores),
            flat_indices=flatten(flat_indices),
            camera_indices=flatten(camera_indices),
            row_indices=flatten(row_indices),
            column_indices=flatten(column_indices),
            source_shape=(views, height, width),
        )


class TaskRiskTrajectoryAdapter(nn.Module):
    """Fuse spatial UQ with planning context without a scalar speed governor.

    ``conditioned_context`` can be passed to the existing VAE/diffusion
    trajectory decoder.  ``trajectory_residual`` is a lightweight alternative
    for bounded experiments.  The residual path is zero-initialized so adding
    the module to frozen ORION is initially an exact identity.
    """

    YIELD_STATES = ("go", "prepare_yield", "hold", "release")

    def __init__(
        self,
        *,
        model_dim: int = 256,
        num_heads: int = 8,
        trajectory_steps: int = 6,
        task_context_dim: int = 89,
        response_score_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if model_dim <= 0 or trajectory_steps <= 0 or task_context_dim <= 0:
            raise ValueError("model, trajectory, and task-context dimensions must be positive")
        if num_heads <= 0 or model_dim % num_heads:
            raise ValueError("num_heads must divide model_dim")
        if not math.isfinite(response_score_scale) or response_score_scale <= 0:
            raise ValueError("response_score_scale must be finite and positive")
        self.model_dim = int(model_dim)
        self.trajectory_steps = int(trajectory_steps)
        self.task_context_dim = int(task_context_dim)
        self.response_score_scale = float(response_score_scale)
        self.context_norm = nn.LayerNorm(self.model_dim)
        self.uq_norm = nn.LayerNorm(self.model_dim)
        self.task_projector = nn.Sequential(
            nn.LayerNorm(self.task_context_dim),
            nn.Linear(self.task_context_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            self.model_dim,
            num_heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.context_residual = nn.Linear(
            self.model_dim, self.model_dim, bias=False
        )
        self.summary_norm = nn.LayerNorm(self.model_dim)
        hidden_dim = max(32, self.model_dim // 2)
        self.yield_head = nn.Sequential(
            nn.Linear(self.model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.YIELD_STATES)),
        )
        self.conflict_head = nn.Sequential(
            nn.Linear(self.model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.trajectory_steps),
        )
        self.trajectory_head = nn.Sequential(
            nn.Linear(self.model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.trajectory_steps * 2),
        )
        nn.init.zeros_(self.context_residual.weight)
        for head in (self.yield_head, self.conflict_head, self.trajectory_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        # With no observation uncertainty, the only valid behavior state is
        # ``go`` and the trajectory residual must be exactly zero.  This fixed
        # prior cannot be trained into a false intervention.
        self.register_buffer(
            "yield_go_prior_logits",
            torch.tensor((4.0, 0.0, 0.0, 0.0)),
            persistent=True,
        )

    def forward(
        self,
        planning_context: torch.Tensor,
        spatial_uq_tokens: SpatialUQTokens,
        task_context: torch.Tensor | None = None,
    ) -> TaskRiskTrajectoryOutput:
        if planning_context.ndim != 3:
            raise ValueError("planning_context must have shape [B,N,D]")
        tokens = spatial_uq_tokens.tokens
        if tokens.ndim != 3:
            raise ValueError("spatial UQ tokens must have shape [B,M,D]")
        if planning_context.shape[0] != tokens.shape[0]:
            raise ValueError("planning context and UQ batch sizes must match")
        if planning_context.shape[-1] != self.model_dim:
            raise ValueError("planning context feature dimension differs")
        if tokens.shape[-1] != self.model_dim:
            raise ValueError("spatial UQ token feature dimension differs")
        if planning_context.shape[1] <= 0 or tokens.shape[1] <= 0:
            raise ValueError("planning context and UQ tokens must be non-empty")
        if task_context is None:
            task_context = planning_context.new_zeros(
                planning_context.shape[0], self.task_context_dim
            )
        if task_context.shape != (
            planning_context.shape[0], self.task_context_dim
        ):
            raise ValueError("task_context must have shape [B,task_context_dim]")
        if not task_context.is_floating_point() or not bool(torch.isfinite(task_context).all()):
            raise ValueError("task_context must be finite floating point")
        task_parameter_dtype = self.task_projector[1].weight.dtype
        task_embedding = self.task_projector(
            task_context.to(dtype=task_parameter_dtype)
        )

        attended, attention = self.cross_attention(
            self.context_norm(planning_context) + task_embedding.unsqueeze(1),
            self.uq_norm(tokens),
            self.uq_norm(tokens),
            need_weights=True,
            average_attn_weights=True,
        )
        global_score = spatial_uq_tokens.scores.max(dim=1).values.to(
            planning_context
        )
        global_gate = global_score / (
            global_score + self.response_score_scale
        )
        # This gate is intentionally applied after the trainable residual.
        # Otherwise attention biases could change native ORION behavior after
        # training even when Stage 1 reports exact zero uncertainty.
        conditioned = planning_context + global_gate[:, None, None] * self.context_residual(
            attended
        )
        summary = self.summary_norm(
            conditioned.mean(dim=1)
            + global_gate.unsqueeze(-1) * attended.mean(dim=1)
            + task_embedding
        )
        token_scores = spatial_uq_tokens.scores.to(attention)
        token_gate = token_scores / (
            token_scores + self.response_score_scale
        )
        yield_logits = self.yield_go_prior_logits.to(summary).unsqueeze(0)
        yield_logits = yield_logits + global_gate.unsqueeze(-1) * self.yield_head(summary)
        trajectory_residual = self.trajectory_head(summary).reshape(
            planning_context.shape[0], self.trajectory_steps, 2
        )
        trajectory_residual = global_gate[:, None, None] * trajectory_residual
        return TaskRiskTrajectoryOutput(
            conditioned_context=conditioned,
            yield_logits=yield_logits,
            conflict_logits=self.conflict_head(summary),
            trajectory_residual=trajectory_residual,
            token_attention=attention.mean(dim=1) * token_gate,
        )


def build_stage2_task_fusion_modules(
    projector_config: Mapping[str, Any],
    adapter_config: Mapping[str, Any],
) -> tuple[SpatialUQTokenProjector, TaskRiskTrajectoryAdapter]:
    """Build the two Stage-2 modules from an explicit checkpoint contract."""

    projector = SpatialUQTokenProjector(**dict(projector_config))
    adapter = TaskRiskTrajectoryAdapter(**dict(adapter_config))
    if projector.model_dim != adapter.model_dim:
        raise ValueError("Stage-2 projector and adapter model dimensions differ")
    return projector, adapter


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage2_task_fusion_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[SpatialUQTokenProjector, TaskRiskTrajectoryAdapter, dict[str, Any]]:
    """Load Stage-2 weights while rejecting Density/corruption supervision."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Stage-2 spatial task checkpoint is missing: %s" % path)
    observed_sha256 = _sha256(checkpoint_path)
    if expected_sha256 and observed_sha256 != expected_sha256:
        raise RuntimeError("Stage-2 spatial task checkpoint hash differs")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError("Stage-2 spatial task checkpoint must be a mapping")
    if payload.get("schema_version") != STAGE2_TASK_FUSION_CHECKPOINT_SCHEMA:
        raise RuntimeError("Stage-2 spatial task checkpoint schema differs")
    contract = payload.get("supervision_contract") or {}
    if (
        contract.get("stage") != "stage2_task_risk"
        or contract.get("uses_density_uq") is not False
        or contract.get("uses_corruption_label") is not False
        or contract.get("updates_stage1_adapter") is not False
    ):
        raise RuntimeError("Stage-2 checkpoint violates the frozen supervision boundary")
    projector_config = payload.get("projector_config")
    adapter_config = payload.get("adapter_config")
    if not isinstance(projector_config, Mapping) or not isinstance(adapter_config, Mapping):
        raise RuntimeError("Stage-2 checkpoint lacks module configs")
    projector, adapter = build_stage2_task_fusion_modules(
        projector_config, adapter_config
    )
    projector.load_state_dict(payload.get("projector_state") or {}, strict=True)
    adapter.load_state_dict(payload.get("adapter_state") or {}, strict=True)
    projector.to(device)
    adapter.to(device)
    metadata = {
        "path": str(checkpoint_path.resolve()),
        "sha256": observed_sha256,
        "schema_version": payload["schema_version"],
        "projector_config": dict(projector_config),
        "adapter_config": dict(adapter_config),
        "stage1_checkpoint_sha256": payload.get("stage1_checkpoint_sha256"),
        "mechanism_report_sha256": payload.get("mechanism_report_sha256"),
        "training_stage": payload.get("training_stage"),
        "closed_loop_eligible": payload.get("closed_loop_eligible") is True,
    }
    return projector, adapter, metadata


def scatter_selected_token_values(
    values: torch.Tensor,
    tokens: SpatialUQTokens,
) -> torch.Tensor:
    """Scatter selected-token task weights to an auditable sparse view grid."""

    if values.shape != tokens.flat_indices.shape:
        raise ValueError("values must match selected token index shape")
    views, height, width = tokens.source_shape
    output = values.new_zeros((values.shape[0], views * height * width))
    output.scatter_add_(1, tokens.flat_indices, values)
    return output.reshape(values.shape[0], views, height, width)

"""Bounded Stage2-P trajectory response to an already task-relevant K map.

This module receives ``K`` rather than raw observation uncertainty.  Route,
actor, TTC, collision and expert-action labels are therefore never forward
inputs.  They may supervise the trajectory response offline, but task
relevance remains upstream in Stage2-L.  Exact zero K is an architectural
identity, independent of learned parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


CHECKPOINT_SCHEMA = "orion.stage2p_task_risk_trajectory.v1"


@dataclass(frozen=True)
class TaskRiskTokens:
    tokens: torch.Tensor
    scores: torch.Tensor
    flat_indices: torch.Tensor
    camera_indices: torch.Tensor
    row_indices: torch.Tensor
    column_indices: torch.Tensor
    source_shape: tuple[int, int, int]


@dataclass(frozen=True)
class Stage2PTrajectoryOutput:
    conditioned_context: torch.Tensor
    trajectory_residual: torch.Tensor
    token_attention: torch.Tensor
    global_gate: torch.Tensor


class TaskRiskMapTokenProjector(nn.Module):
    """Select spatial K cells while preserving view and grid provenance."""

    def __init__(
        self,
        *,
        model_dim: int = 256,
        hidden_dim: int = 128,
        max_views: int = 8,
        tokens_per_view: int = 8,
        score_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if min(model_dim, hidden_dim, max_views, tokens_per_view) <= 0:
            raise ValueError("Stage2-P projector dimensions must be positive")
        if not math.isfinite(score_scale) or score_scale <= 0.0:
            raise ValueError("Stage2-P score scale must be finite and positive")
        self.model_dim = int(model_dim)
        self.max_views = int(max_views)
        self.tokens_per_view = int(tokens_per_view)
        self.score_scale = float(score_scale)
        self.value_projector = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
        )
        self.position_projector = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
        )
        self.camera_embedding = nn.Embedding(max_views, model_dim)
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, task_risk: torch.Tensor) -> TaskRiskTokens:
        if task_risk.ndim == 5 and task_risk.shape[-1] == 1:
            task_risk = task_risk[..., 0]
        if task_risk.ndim != 4:
            raise ValueError("task_risk must have shape [B,V,H,W]")
        batch, views, height, width = task_risk.shape
        if views <= 0 or views > self.max_views or min(height, width) <= 0:
            raise ValueError("Stage2-P K map shape differs")
        if height * width < self.tokens_per_view:
            raise ValueError("tokens_per_view exceeds the K grid")
        if not task_risk.is_floating_point() or not bool(
            torch.isfinite(task_risk).all()
        ):
            raise ValueError("task_risk must be finite floating point")
        if bool((task_risk < 0.0).any()) or bool((task_risk > 1.0).any()):
            raise ValueError("task_risk must lie in [0,1]")

        frozen = task_risk.detach()
        flattened = frozen.reshape(batch, views, height * width)
        scores, within_view = torch.topk(
            flattened,
            k=self.tokens_per_view,
            dim=-1,
            largest=True,
            sorted=True,
        )
        camera = torch.arange(
            views, device=task_risk.device, dtype=torch.long
        ).view(1, views, 1).expand(batch, -1, self.tokens_per_view)
        rows = torch.div(within_view, width, rounding_mode="floor")
        columns = within_view.remainder(width)
        coordinates = torch.stack(
            (
                2.0 * rows.to(task_risk.dtype) / max(height - 1, 1) - 1.0,
                2.0 * columns.to(task_risk.dtype) / max(width - 1, 1) - 1.0,
            ),
            dim=-1,
        )
        parameter_dtype = self.value_projector[0].weight.dtype
        content = self.value_projector(
            scores.unsqueeze(-1).to(dtype=parameter_dtype)
        )
        content = content + self.position_projector(
            coordinates.to(dtype=parameter_dtype)
        )
        content = content + self.camera_embedding(camera)
        gate = scores.to(dtype=parameter_dtype) / (
            scores.to(dtype=parameter_dtype) + self.score_scale
        )
        tokens = gate.unsqueeze(-1) * self.output_norm(content)
        flat_indices = camera * (height * width) + within_view
        flatten = lambda value: value.reshape(
            batch, views * self.tokens_per_view
        )
        return TaskRiskTokens(
            tokens=tokens.reshape(
                batch, views * self.tokens_per_view, self.model_dim
            ),
            scores=flatten(scores),
            flat_indices=flatten(flat_indices),
            camera_indices=flatten(camera),
            row_indices=flatten(rows),
            column_indices=flatten(columns),
            source_shape=(views, height, width),
        )


class TaskRiskTrajectoryResponse(nn.Module):
    """Predict a bounded residual without changing native ORION context.

    The direct residual is consumed after ORION produces its native trajectory.
    The native VLM memory is returned exactly, so the first interface smoke has
    one and only one response path.
    """

    def __init__(
        self,
        *,
        model_dim: int = 256,
        num_heads: int = 8,
        trajectory_steps: int = 6,
        response_score_scale: float = 0.25,
        lateral_bound_m: float = 2.0,
        longitudinal_bound_m: float = 24.0,
    ) -> None:
        super().__init__()
        if model_dim <= 0 or trajectory_steps <= 0:
            raise ValueError("Stage2-P response dimensions must be positive")
        if num_heads <= 0 or model_dim % num_heads:
            raise ValueError("num_heads must divide model_dim")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (
                response_score_scale,
                lateral_bound_m,
                longitudinal_bound_m,
            )
        ):
            raise ValueError("Stage2-P gates and bounds must be positive")
        self.model_dim = int(model_dim)
        self.trajectory_steps = int(trajectory_steps)
        self.response_score_scale = float(response_score_scale)
        self.context_norm = nn.LayerNorm(model_dim)
        self.risk_norm = nn.LayerNorm(model_dim)
        self.cross_attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        hidden_dim = max(64, model_dim // 2)
        self.summary_norm = nn.LayerNorm(model_dim)
        self.trajectory_head = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, trajectory_steps * 2),
        )
        nn.init.zeros_(self.trajectory_head[-1].weight)
        nn.init.zeros_(self.trajectory_head[-1].bias)
        bounds = torch.tensor(
            (float(lateral_bound_m), float(longitudinal_bound_m))
        ).view(1, 1, 2).expand(1, trajectory_steps, 2).contiguous()
        self.register_buffer("trajectory_bounds_m", bounds, persistent=True)

    def forward(
        self,
        planning_context: torch.Tensor,
        task_risk_tokens: TaskRiskTokens,
    ) -> Stage2PTrajectoryOutput:
        tokens = task_risk_tokens.tokens
        if (
            planning_context.ndim != 3
            or tokens.ndim != 3
            or planning_context.shape[0] != tokens.shape[0]
            or planning_context.shape[-1] != self.model_dim
            or tokens.shape[-1] != self.model_dim
            or min(planning_context.shape[1], tokens.shape[1]) <= 0
        ):
            raise ValueError("Stage2-P planning/token interface differs")
        if not bool(torch.isfinite(planning_context).all()) or not bool(
            torch.isfinite(tokens).all()
        ):
            raise ValueError("Stage2-P inputs must be finite")
        attended, attention = self.cross_attention(
            self.context_norm(planning_context),
            self.risk_norm(tokens),
            self.risk_norm(tokens),
            need_weights=True,
            average_attn_weights=True,
        )
        global_score = task_risk_tokens.scores.max(dim=1).values.to(
            planning_context
        )
        global_gate = global_score / (
            global_score + self.response_score_scale
        )
        summary = self.summary_norm(
            planning_context.mean(dim=1)
            + global_gate.unsqueeze(-1) * attended.mean(dim=1)
        )
        raw = self.trajectory_head(summary).reshape(
            planning_context.shape[0], self.trajectory_steps, 2
        )
        residual = (
            global_gate[:, None, None]
            * torch.tanh(raw)
            * self.trajectory_bounds_m.to(raw)
        )
        token_gate = task_risk_tokens.scores.to(attention) / (
            task_risk_tokens.scores.to(attention) + self.response_score_scale
        )
        return Stage2PTrajectoryOutput(
            conditioned_context=planning_context,
            trajectory_residual=residual,
            token_attention=attention.mean(dim=1) * token_gate,
            global_gate=global_gate,
        )


def build_modules(
    projector_config: Mapping[str, Any],
    response_config: Mapping[str, Any],
) -> tuple[TaskRiskMapTokenProjector, TaskRiskTrajectoryResponse]:
    projector = TaskRiskMapTokenProjector(**dict(projector_config))
    response = TaskRiskTrajectoryResponse(**dict(response_config))
    if projector.model_dim != response.model_dim:
        raise ValueError("Stage2-P projector and response dimensions differ")
    return projector, response


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[
    TaskRiskMapTokenProjector,
    TaskRiskTrajectoryResponse,
    dict[str, Any],
]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Stage2-P checkpoint is missing: %s" % path)
    observed = _sha256(checkpoint_path)
    if expected_sha256 and observed != expected_sha256:
        raise RuntimeError("Stage2-P checkpoint hash differs")
    value = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = value.get("responsibility_contract") or {}
    if (
        value.get("schema") != CHECKPOINT_SCHEMA
        or contract.get("forward_inputs")
        != ["frozen_orion_planning_context", "task_risk_k"]
        or contract.get("raw_observation_u_forward") is not False
        or contract.get("privileged_task_context_forward") is not False
        or contract.get("route_actor_ttc_outcome_forward") is not False
        or value.get("formal_stage2p_ready") is not False
        or value.get("closed_loop_eligible") is not False
    ):
        raise RuntimeError("Stage2-P checkpoint responsibility contract differs")
    projector_config = value.get("projector_config")
    response_config = value.get("response_config")
    if not isinstance(projector_config, Mapping) or not isinstance(
        response_config, Mapping
    ):
        raise RuntimeError("Stage2-P checkpoint configs are missing")
    projector, response = build_modules(projector_config, response_config)
    projector.load_state_dict(value.get("projector_state") or {}, strict=True)
    response.load_state_dict(value.get("response_state") or {}, strict=True)
    projector.to(device)
    response.to(device)
    metadata = {
        "path": str(checkpoint_path.resolve()),
        "sha256": observed,
        "schema": value["schema"],
        "projector_config": dict(projector_config),
        "response_config": dict(response_config),
        "engineering_smoke_only": value.get("engineering_smoke_only") is True,
        "formal_stage2p_ready": False,
        "closed_loop_eligible": False,
    }
    return projector, response, metadata


__all__ = [
    "CHECKPOINT_SCHEMA",
    "Stage2PTrajectoryOutput",
    "TaskRiskMapTokenProjector",
    "TaskRiskTokens",
    "TaskRiskTrajectoryResponse",
    "build_modules",
    "load_checkpoint",
]

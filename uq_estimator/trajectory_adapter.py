"""Privileged path-risk trajectory adapter used as an oracle teacher baseline.

This module is no longer the Stage-2 paper mainline. It deliberately consumes
privileged *path risk* and is retained to generate/check oracle behavior targets
and mechanism upper bounds. The v2 mainline feeds task-agnostic spatial UQ into
fine-tuned ORION/VLM, which learns route relevance itself. A zero-risk input is
an exact identity mapping, preserving its value as an auditable teacher.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrajectoryAdapterOutput:
    """Output of :class:`PathRiskTrajectoryAdapter`.

    Shapes:
        trajectories: ``[B, M, T, 2]`` adapted waypoint displacements.
        residual: ``[B, M, T, 2]`` bounded change from the base trajectory.
        intervention: ``[B, M, T]`` residual magnitude in metres.
        stop_probability: ``[B, M, T]`` auxiliary supervised stop probability.
    """

    trajectories: torch.Tensor
    residual: torch.Tensor
    intervention: torch.Tensor
    stop_probability: torch.Tensor


class PathRiskTrajectoryAdapter(nn.Module):
    """Apply a bounded residual to candidate trajectories using path risk.

    The shared adapter sees the base displacement, cumulative waypoint,
    normalized horizon, and the fixed path-risk signal.  It does not see the
    corruption family or mask identity.  The final residual layer is initialized
    to zero, and multiplication by path risk guarantees exact preservation when
    risk is zero even after training.

    Args:
        context_dim: optional dimension of an ORION planning feature supplied as
            ``context``.  Set to zero to train directly on saved trajectories.
        hidden_dim: width of the lightweight residual MLP.
        max_residual_m: per-coordinate residual bound in metres per waypoint.
    """

    def __init__(
        self,
        context_dim: int = 0,
        hidden_dim: int = 128,
        max_residual_m: float = 2.0,
    ) -> None:
        super().__init__()
        if context_dim < 0:
            raise ValueError("context_dim must be non-negative")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if max_residual_m <= 0:
            raise ValueError("max_residual_m must be positive")

        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_residual_m = float(max_residual_m)

        # base displacement (2), cumulative position (2), risk (1), time (1)
        input_dim = 6 + self.context_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(self.hidden_dim, 2)
        self.stop_head = nn.Linear(self.hidden_dim, 1)

        # Exact identity for trajectory output before training.  The auxiliary
        # stop head starts with a conservative low prior but remains trainable.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.stop_head.weight)
        nn.init.constant_(self.stop_head.bias, -4.0)

    @staticmethod
    def _expand_path_risk(
        path_risk: torch.Tensor,
        batch: int,
        modes: int,
        steps: int,
    ) -> torch.Tensor:
        """Normalize risk to ``[B, M, T]`` and clamp it to ``[0, 1]``."""
        if path_risk.ndim == 2:
            if tuple(path_risk.shape) != (batch, modes):
                raise ValueError("2-D path_risk must have shape [B, M]")
            path_risk = path_risk.unsqueeze(-1).expand(-1, -1, steps)
        elif path_risk.ndim == 3:
            if tuple(path_risk.shape) != (batch, modes, steps):
                raise ValueError("3-D path_risk must have shape [B, M, T]")
        else:
            raise ValueError("path_risk must have shape [B, M] or [B, M, T]")
        if not torch.isfinite(path_risk).all():
            raise ValueError("path_risk must contain only finite values")
        return path_risk.clamp(0.0, 1.0)

    def forward(
        self,
        base_trajectories: torch.Tensor,
        path_risk: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> TrajectoryAdapterOutput:
        """Adapt trajectory displacements.

        Args:
            base_trajectories: ``[B, M, T, 2]`` ORION displacement candidates.
            path_risk: ``[B, M]`` or ``[B, M, T]`` fixed path-overlap risk.
            context: optional ``[B, D_context]`` planning feature.
        """
        if base_trajectories.ndim != 4 or base_trajectories.shape[-1] != 2:
            raise ValueError(
                "base_trajectories must have shape [B, M, T, 2]"
            )
        if not torch.isfinite(base_trajectories).all():
            raise ValueError("base_trajectories must contain only finite values")
        batch, modes, steps, _ = base_trajectories.shape
        risk = self._expand_path_risk(path_risk, batch, modes, steps)

        if self.context_dim == 0:
            if context is not None:
                raise ValueError("context was provided but context_dim is zero")
            context_grid = None
        else:
            if context is None:
                raise ValueError("context is required when context_dim is positive")
            if tuple(context.shape) != (batch, self.context_dim):
                raise ValueError("context must have shape [B, context_dim]")
            context_grid = context[:, None, None, :].expand(
                -1, modes, steps, -1
            )

        cumulative = base_trajectories.cumsum(dim=-2)  # [B, M, T, 2]
        horizon = torch.linspace(
            0.0,
            1.0,
            steps,
            device=base_trajectories.device,
            dtype=base_trajectories.dtype,
        ).view(1, 1, steps, 1).expand(batch, modes, -1, -1)
        features = [
            base_trajectories,
            cumulative,
            risk.unsqueeze(-1),
            horizon,
        ]
        if context_grid is not None:
            features.append(context_grid.to(dtype=base_trajectories.dtype))
        hidden = self.trunk(torch.cat(features, dim=-1))  # [B, M, T, H]

        raw_residual = self.residual_head(hidden)  # [B, M, T, 2]
        residual = (
            risk.unsqueeze(-1)
            * self.max_residual_m
            * torch.tanh(raw_residual)
        )
        adapted = base_trajectories + residual
        stop_probability = torch.sigmoid(
            self.stop_head(hidden).squeeze(-1)
        )
        intervention = torch.linalg.vector_norm(residual, dim=-1)
        return TrajectoryAdapterOutput(
            trajectories=adapted,
            residual=residual,
            intervention=intervention,
            stop_probability=stop_probability,
        )


def trajectory_adapter_loss(
    output: TrajectoryAdapterOutput,
    expert_trajectories: torch.Tensor,
    trajectory_mask: torch.Tensor,
    path_risk: torch.Tensor,
    stop_target: torch.Tensor | None = None,
    clean_risk_threshold: float = 0.05,
    lambda_stop: float = 0.25,
    lambda_clean: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Oracle-imitation loss with explicit clean/off-path preservation.

    All trajectory tensors use displacement coordinates ``[B, M, T, 2]``.
    ``trajectory_mask`` is ``[B, M, T]``.  The function intentionally has no
    ADE-specific claim: the imitation term teaches the oracle response, while
    closed-loop collision, completion and violation metrics remain decisive.
    """
    if output.trajectories.shape != expert_trajectories.shape:
        raise ValueError("expert_trajectories must match adapted trajectories")
    if trajectory_mask.shape != output.trajectories.shape[:-1]:
        raise ValueError("trajectory_mask must have shape [B, M, T]")
    if clean_risk_threshold < 0:
        raise ValueError("clean_risk_threshold must be non-negative")

    mask = trajectory_mask.to(dtype=output.trajectories.dtype)
    point_error = F.smooth_l1_loss(
        output.trajectories,
        expert_trajectories,
        reduction="none",
    ).sum(dim=-1)
    imitation = (point_error * mask).sum() / mask.sum().clamp_min(1.0)

    batch, modes, steps = mask.shape
    expanded_risk = PathRiskTrajectoryAdapter._expand_path_risk(
        path_risk, batch, modes, steps
    )
    preserve_mask = (
        (expanded_risk <= clean_risk_threshold).to(mask.dtype) * mask
    )
    clean_preservation = (
        output.residual.pow(2).sum(dim=-1) * preserve_mask
    ).sum() / preserve_mask.sum().clamp_min(1.0)

    stop = output.trajectories.new_zeros(())
    if stop_target is not None:
        if stop_target.shape != output.stop_probability.shape:
            raise ValueError("stop_target must have shape [B, M, T]")
        stop_point = F.binary_cross_entropy(
            output.stop_probability,
            stop_target.to(dtype=output.stop_probability.dtype),
            reduction="none",
        )
        stop = (stop_point * mask).sum() / mask.sum().clamp_min(1.0)

    total = imitation + lambda_clean * clean_preservation + lambda_stop * stop
    return {
        "total": total,
        "imitation": imitation,
        "clean_preservation": clean_preservation,
        "stop": stop,
    }

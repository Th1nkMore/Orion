"""Magnitude-preserving Stage2-L planning-semantics bottleneck.

The failed v1 head saw only the normalized 4096-D global K bridge token.  This
version retains that contextual path but adds an explicit raw six-feature K
summary.  Its magnitude channels are never LayerNorm'ed, so absolute task-risk
scale cannot disappear before stance classification.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from uq_estimator.stage2l_semantic_bottleneck import PLANNING_STANCES


SCHEMA = "orion.stage2l_magnitude_semantic_bottleneck.v2"
RAW_TASK_RISK_FEATURES = (
    "max_task_risk",
    "mean_task_risk",
    "rms_task_risk",
    "soft_y",
    "soft_x",
    "soft_view",
)


@dataclass(frozen=True)
class MagnitudeSemanticTokens:
    token: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_indices: torch.Tensor
    raw_global_features: torch.Tensor
    magnitude_features: torch.Tensor
    schema: str = SCHEMA


class MagnitudePreservingPlanningStanceBottleneck(nn.Module):
    """Predict stance from normalized context plus unnormalized K magnitude."""

    def __init__(
        self,
        model_dim: int = 4096,
        hidden_dim: int = 256,
        magnitude_scale: float = 10.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if min(model_dim, hidden_dim) <= 0:
            raise ValueError("semantic dimensions must be positive")
        if magnitude_scale <= 0.0 or temperature <= 0.0:
            raise ValueError("semantic scales must be positive")
        self.model_dim = int(model_dim)
        self.magnitude_scale = float(magnitude_scale)
        self.temperature = float(temperature)

        # Deliberately no LayerNorm on max/mean/RMS.  log1p only compresses the
        # dynamic range while remaining strictly monotonic in absolute scale.
        self.magnitude_projection = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.location_projection = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(PLANNING_STANCES)),
        )
        self.stance_embedding = nn.Embedding(len(PLANNING_STANCES), model_dim)
        self.token_type_embedding = nn.Parameter(torch.zeros(model_dim))
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        task_risk_bridge_tokens: torch.Tensor,
        raw_global_features: torch.Tensor,
    ) -> MagnitudeSemanticTokens:
        if task_risk_bridge_tokens.ndim != 3:
            raise ValueError("task-risk bridge tokens must have shape [B,N,D]")
        if task_risk_bridge_tokens.shape[1] < 1:
            raise ValueError("task-risk bridge token span cannot be empty")
        if task_risk_bridge_tokens.shape[-1] != self.model_dim:
            raise ValueError("task-risk bridge token dimension is incompatible")
        if raw_global_features.shape != (
            task_risk_bridge_tokens.shape[0],
            len(RAW_TASK_RISK_FEATURES),
        ):
            raise ValueError("raw K summary must have shape [B,6]")
        if not (
            bool(torch.isfinite(task_risk_bridge_tokens).all())
            and bool(torch.isfinite(raw_global_features).all())
        ):
            raise ValueError("semantic bottleneck inputs must be finite")
        if bool((raw_global_features[:, :3] < 0.0).any()):
            raise ValueError("K magnitude summaries must be non-negative")

        magnitude = torch.log1p(
            self.magnitude_scale * raw_global_features[:, :3]
        )
        magnitude_latent = self.magnitude_projection(magnitude)
        location_latent = self.location_projection(raw_global_features[:, 3:])
        context_latent = self.context_projection(task_risk_bridge_tokens[:, -1])
        fused = torch.cat(
            (magnitude_latent, location_latent, context_latent), dim=-1
        )
        logits = self.classifier(fused)
        probabilities = F.softmax(logits / self.temperature, dim=-1)
        soft_stance = probabilities @ self.stance_embedding.weight
        token = self.output_norm(
            soft_stance + self.token_type_embedding[None]
        ).unsqueeze(1)
        return MagnitudeSemanticTokens(
            token=token,
            logits=logits,
            probabilities=probabilities,
            predicted_indices=logits.argmax(dim=-1),
            raw_global_features=raw_global_features,
            magnitude_features=magnitude,
        )


__all__ = [
    "MagnitudePreservingPlanningStanceBottleneck",
    "MagnitudeSemanticTokens",
    "RAW_TASK_RISK_FEATURES",
    "SCHEMA",
]

"""Explicit Stage2-L planning-semantics bottleneck.

The bottleneck is owned by the VLM side of the architecture.  It receives the
compact task-risk bridge tokens derived from ``K = U * R`` and predicts one of
three planning stances.  The predicted distribution, never a ground-truth
stance, is converted into a differentiable token for language generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA = "orion.stage2l_semantic_bottleneck.v1"
PLANNING_STANCES: Tuple[str, ...] = (
    "maintain",
    "caution",
    "prepare_to_yield",
)


@dataclass(frozen=True)
class PlanningStanceTokens:
    """Inspectable stance prediction and its language-conditioning token."""

    token: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_indices: torch.Tensor
    schema: str = SCHEMA


def planning_stance_index(stance: str) -> int:
    """Return the frozen class index for one planning stance."""

    value = str(stance)
    try:
        return PLANNING_STANCES.index(value)
    except ValueError as exc:
        raise ValueError("unsupported planning stance: %s" % value) from exc


def encode_planning_stances(
    stances: Iterable[str], *, device: torch.device = None
) -> torch.Tensor:
    """Encode stance strings without exposing them to model forward inputs."""

    values = [planning_stance_index(value) for value in stances]
    if not values:
        raise ValueError("at least one planning stance is required")
    return torch.tensor(values, dtype=torch.long, device=device)


def planning_stance_loss(
    logits: torch.Tensor, target_indices: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy supervision for the explicit semantic prediction."""

    if logits.ndim != 2 or logits.shape[-1] != len(PLANNING_STANCES):
        raise ValueError("stance logits must have shape [B,3]")
    if target_indices.ndim != 1 or target_indices.shape[0] != logits.shape[0]:
        raise ValueError("stance targets must have shape [B]")
    if target_indices.dtype != torch.long:
        raise ValueError("stance targets must be torch.long")
    if bool((target_indices < 0).any()) or bool(
        (target_indices >= len(PLANNING_STANCES)).any()
    ):
        raise ValueError("stance target index is out of range")
    return F.cross_entropy(logits, target_indices)


class PlanningStanceSemanticBottleneck(nn.Module):
    """Predict a structured stance and emit one soft semantic token.

    Only the global K bridge token is consumed.  The softmax-weighted stance
    embedding keeps the language path differentiable and prevents teacher
    forcing of the target class into generation.
    """

    def __init__(
        self,
        model_dim: int = 4096,
        hidden_dim: int = 256,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if min(model_dim, hidden_dim) <= 0 or temperature <= 0.0:
            raise ValueError("semantic dimensions and temperature must be positive")
        self.model_dim = int(model_dim)
        self.temperature = float(temperature)
        self.context_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(PLANNING_STANCES)),
        )
        self.stance_embedding = nn.Embedding(len(PLANNING_STANCES), model_dim)
        self.token_type_embedding = nn.Parameter(torch.zeros(model_dim))
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, task_risk_bridge_tokens: torch.Tensor) -> PlanningStanceTokens:
        if task_risk_bridge_tokens.ndim != 3:
            raise ValueError("task-risk bridge tokens must have shape [B,N,D]")
        if task_risk_bridge_tokens.shape[1] < 1:
            raise ValueError("task-risk bridge token span cannot be empty")
        if task_risk_bridge_tokens.shape[-1] != self.model_dim:
            raise ValueError("task-risk bridge token dimension is incompatible")
        if not bool(torch.isfinite(task_risk_bridge_tokens).all()):
            raise ValueError("task-risk bridge tokens must be finite")

        global_context = task_risk_bridge_tokens[:, -1]
        logits = self.classifier(self.context_norm(global_context))
        probabilities = F.softmax(logits / self.temperature, dim=-1)
        soft_stance = probabilities @ self.stance_embedding.weight
        token = self.output_norm(
            soft_stance + self.token_type_embedding[None]
        ).unsqueeze(1)
        return PlanningStanceTokens(
            token=token,
            logits=logits,
            probabilities=probabilities,
            predicted_indices=logits.argmax(dim=-1),
        )


__all__ = [
    "PLANNING_STANCES",
    "SCHEMA",
    "PlanningStanceSemanticBottleneck",
    "PlanningStanceTokens",
    "encode_planning_stances",
    "planning_stance_index",
    "planning_stance_loss",
]

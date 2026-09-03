"""Project density uncertainty into continuous LLM input tokens."""

from __future__ import annotations

import torch
import torch.nn as nn


class UQTokenProjector(nn.Module):
    """Create learned-null, score-gated uncertainty tokens."""

    def __init__(
        self,
        active_dim: int = 16,
        hidden_dim: int = 512,
        llm_dim: int = 4096,
        token_count: int = 1,
    ) -> None:
        super().__init__()
        if active_dim <= 0 or hidden_dim <= 0 or llm_dim <= 0 or token_count <= 0:
            raise ValueError("All UQTokenProjector dimensions must be positive")

        self.active_dim = int(active_dim)
        self.hidden_dim = int(hidden_dim)
        self.llm_dim = int(llm_dim)
        self.token_count = int(token_count)
        self.direction_projector = nn.Sequential(
            nn.Linear(self.active_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.token_count * self.llm_dim),
        )
        self.null_token = nn.Parameter(
            torch.zeros(1, self.token_count, self.llm_dim)
        )
        self.score_basis = nn.Parameter(
            torch.empty(1, self.token_count, self.llm_dim)
        )

        nn.init.normal_(self.score_basis, mean=0.0, std=0.02)
        nn.init.zeros_(self.direction_projector[-1].weight)
        nn.init.zeros_(self.direction_projector[-1].bias)

    def forward(
        self,
        active_embedding: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        if active_embedding.ndim != 2:
            raise ValueError("active_embedding must have shape [B, active_dim]")
        if active_embedding.shape[-1] != self.active_dim:
            raise ValueError(
                f"Expected active_dim={self.active_dim}, "
                f"got {active_embedding.shape[-1]}"
            )
        if score.ndim == 1:
            score = score.unsqueeze(-1)
        if score.ndim != 2 or score.shape[-1] != 1:
            raise ValueError("score must have shape [B, 1]")
        if score.shape[0] != active_embedding.shape[0]:
            raise ValueError("active_embedding and score batch sizes must match")

        parameter_dtype = self.direction_projector[0].weight.dtype
        active_embedding = active_embedding.to(dtype=parameter_dtype)
        score = score.to(dtype=parameter_dtype).clamp(0.0, 1.0)
        direction_delta = self.direction_projector(active_embedding).reshape(
            active_embedding.shape[0], self.token_count, self.llm_dim
        )
        score_content = self.score_basis + direction_delta
        return (
            self.null_token.to(dtype=score_content.dtype)
            + score.unsqueeze(-1) * score_content
        )

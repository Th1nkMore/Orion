"""Ground density uncertainty in the LLM waypoint representation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UQGroundingHead(nn.Module):
    """Predict calibrated density score from a waypoint hidden state."""

    def __init__(self, input_dim: int = 4096) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.score = nn.Linear(input_dim, 1)

    def forward(self, waypoint_feature: torch.Tensor) -> torch.Tensor:
        if waypoint_feature.ndim != 2:
            raise ValueError("waypoint_feature must have shape [B, D]")
        return torch.sigmoid(self.score(self.norm(waypoint_feature.float())))


def grounding_loss(
    predicted_score: torch.Tensor,
    target_score: torch.Tensor,
) -> torch.Tensor:
    if predicted_score.shape != target_score.shape:
        raise ValueError("predicted_score and target_score shapes must match")
    return F.smooth_l1_loss(
        predicted_score.float(),
        target_score.detach().float(),
    )

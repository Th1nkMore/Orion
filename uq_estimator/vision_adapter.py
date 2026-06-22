"""Score-conditioned residual adapter for pre-LLM visual tokens."""

from __future__ import annotations

import torch
import torch.nn as nn


class UQVisionAdapter(nn.Module):
    """Apply an identity-initialized residual to every visual query."""

    def __init__(self, llm_dim: int = 4096, bottleneck_dim: int = 256) -> None:
        super().__init__()
        if llm_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("Adapter dimensions must be positive")
        self.llm_dim = int(llm_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.norm = nn.LayerNorm(self.llm_dim)
        self.down = nn.Linear(self.llm_dim, self.bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(self.bottleneck_dim, self.llm_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        if vision_tokens.ndim != 3:
            raise ValueError("vision_tokens must have shape [B, N, D]")
        if vision_tokens.shape[-1] != self.llm_dim:
            raise ValueError(
                f"Expected llm_dim={self.llm_dim}, "
                f"got {vision_tokens.shape[-1]}"
            )
        if score.ndim == 1:
            score = score.unsqueeze(-1)
        if score.ndim != 2 or score.shape[-1] != 1:
            raise ValueError("score must have shape [B, 1]")
        if score.shape[0] != vision_tokens.shape[0]:
            raise ValueError("score and vision_tokens batch sizes must match")

        parameter_dtype = self.down.weight.dtype
        inputs = vision_tokens.to(dtype=parameter_dtype)
        residual = self.up(self.activation(self.down(self.norm(inputs))))
        gate = score.to(dtype=parameter_dtype).clamp(0.0, 1.0).unsqueeze(-1)
        return (inputs + gate * residual).to(dtype=vision_tokens.dtype)

"""UQ Estimator: uncertainty-aware module for ORION autonomous driving."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

# Number of raw statistical features computed in the dataset
D_STAT_RAW = 5


@dataclass
class UQOutput:
    """Output container for UQEstimator."""

    embedding: torch.Tensor   # [B, d_out]
    score: torch.Tensor       # [B, 1]
    attn_weights: Optional[torch.Tensor] = None  # [B, n_query, N_patches]


class UQEstimator(nn.Module):
    """Uncertainty estimator that consumes Vision Encoder patch tokens.

    Args:
        config: dict with keys matching configs/uq_train.yaml ``model`` section.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        d_patch: int = config["d_patch"]        # 1152
        n_query: int = config["n_query"]         # 16
        d_stat: int = config["d_stat"]           # 64
        d_hidden: int = config["d_hidden"]       # 512
        d_out: int = config["d_out"]             # 256

        # Ablation switches
        self.use_stat_features = config.get("use_stat_features", True)
        self.use_transformer_decoder = config.get("use_transformer_decoder", True)

        # -- 1. Patch projection: D → d_out (reduce dim before decoder) ------
        self.patch_proj = nn.Linear(d_patch, d_out)  # [*, D] → [*, d_out]

        # -- 2. Learnable queries for cross-attention pooling ----------------
        if self.use_transformer_decoder:
            self.queries = nn.Parameter(
                torch.randn(1, n_query, d_out) * (1.0 / math.sqrt(d_out))
            )  # [1, n_query, d_out]

            # -- 3. Two-layer Transformer decoder (query cross-attends patches)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_out,
                nhead=8,
                dim_feedforward=d_out * 2,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # -- 4. Stat-feature projection (5 raw → d_stat) + LayerNorm --------
        fusion_in_dim = d_out
        if self.use_stat_features:
            self.stat_proj = nn.Sequential(
                nn.Linear(D_STAT_RAW, d_stat),
                nn.GELU(),
            )
            self.stat_norm = nn.LayerNorm(d_stat)
            fusion_in_dim += d_stat

        # -- 5. Fusion MLP --------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_out),
        )

        # -- 6. Output heads -------------------------------------------------
        self.score_head = nn.Sequential(
            nn.Linear(d_out, 1),
            nn.Sigmoid(),
        )
        self.embed_head = nn.Sequential(
            nn.Linear(d_out, d_out),
            nn.LayerNorm(d_out),
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        patch_tokens: torch.Tensor,
        stat_features: torch.Tensor,
    ) -> UQOutput:
        """
        Args:
            patch_tokens:  [B, N_views, N_patches, D]
            stat_features: [B, D_STAT_RAW]  (raw 5-dim statistics)

        Returns:
            UQOutput with embedding [B, d_out], score [B, 1].
        """
        B, N_views, N_patches, D = patch_tokens.shape  # [B, N_views, N_patches, D]

        # 1. Reshape views into batch and project to lower dim
        x = patch_tokens.reshape(B * N_views, N_patches, D)  # [B*N_views, N_patches, D]
        x = self.patch_proj(x)  # [B*N_views, N_patches, d_out]
        d = x.shape[-1]  # d_out

        # 2. Patch aggregation: cross-attention decoder or simple mean pool
        if self.use_transformer_decoder:
            queries = self.queries.expand(B * N_views, -1, -1)  # [B*N_views, n_query, d_out]
            attn_out = self.decoder(queries, x)  # [B*N_views, n_query, d_out]
            attn_out = attn_out.mean(dim=1)  # [B*N_views, d_out]
        else:
            attn_out = x.mean(dim=1)  # [B*N_views, d_out]

        # Restore view dimension and average across views
        attn_out = attn_out.reshape(B, N_views, d)  # [B, N_views, d_out]
        attn_out = attn_out.mean(dim=1)  # [B, d_out]

        # 3. Stat-feature branch (optional)
        if self.use_stat_features:
            stat_out = self.stat_proj(stat_features)  # [B, d_stat]
            stat_out = self.stat_norm(stat_out)  # [B, d_stat]
            fused = torch.cat([attn_out, stat_out], dim=-1)  # [B, d_out + d_stat]
        else:
            fused = attn_out  # [B, d_out]

        # 4. Fusion
        embedding = self.fusion(fused)  # [B, d_out]

        # 5. Output heads
        score = self.score_head(embedding)  # [B, 1]
        embedding_out = self.embed_head(embedding)  # [B, d_out]

        return UQOutput(embedding=embedding_out, score=score)


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import yaml

    with open("configs/uq_train.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    model = UQEstimator(model_cfg)

    # Mock data: B=2, N_views=6, N_patches=256, D=1152
    B, N_views, N_patches, D = 2, model_cfg["n_views"], model_cfg["n_patches"], model_cfg["d_patch"]
    patch_tokens = torch.randn(B, N_views, N_patches, D)  # [B, N_views, N_patches, D]
    stat_features = torch.randn(B, D_STAT_RAW)  # [B, 5]

    with torch.no_grad():
        out = model(patch_tokens, stat_features)

    n_params = _count_parameters(model)
    print(f"embedding shape : {out.embedding.shape}")
    print(f"score shape     : {out.score.shape}")
    print(f"score mean      : {out.score.mean().item():.4f}")
    print(f"score range     : [{out.score.min().item():.4f}, {out.score.max().item():.4f}]")
    print(f"total params    : {n_params:,} ({n_params / 1e6:.2f}M)")
    assert n_params < 5_000_000, f"Parameter count {n_params:,} exceeds 5M limit!"
    print("smoke test PASSED")

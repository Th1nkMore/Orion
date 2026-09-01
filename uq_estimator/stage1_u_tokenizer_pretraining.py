"""Task-agnostic pretraining objective for the Stage2-L U tokenizer.

The tokenizer is allowed to see only normalized Stage1 observation-
uncertainty components.  A disposable decoder reconstructs the exact temporal
summary represented by each view/grid token.  Route context, actor geometry,
QA answers, task relevance, TTC, collision outcomes and corruption metadata
are deliberately absent from this API.

After this objective passes held-out reconstruction gates, the decoder is
discarded and the tokenizer is frozen before Stage2-L starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from uq_estimator.uq_relevance_tokenizer import UQComponentTokenizer


SCHEMA = "orion.stage1_u_tokenizer_pretraining.v1"


class UQSummaryReconstructionHead(nn.Module):
    """Disposable decoder for ``latest/mean/delta`` U components."""

    def __init__(
        self,
        *,
        model_dim: int = 4096,
        component_dim: int = 3,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if min(model_dim, component_dim, hidden_dim) <= 0:
            raise ValueError("reconstruction dimensions must be positive")
        self.model_dim = int(model_dim)
        self.summary_dim = 3 * int(component_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.summary_dim),
        )

    def forward(self, token_grid: torch.Tensor) -> torch.Tensor:
        if token_grid.ndim != 5 or token_grid.shape[-1] != self.model_dim:
            raise ValueError("token grid must have shape [B,V,H,W,D]")
        if not (
            token_grid.is_floating_point()
            and bool(torch.isfinite(token_grid).all())
        ):
            raise ValueError("token grid must be finite floating point")
        return self.net(token_grid)


@dataclass(frozen=True)
class Stage1UTokenizerPretrainingTerms:
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    zero_anchor_loss: torch.Tensor
    decoded_summary: torch.Tensor
    target_summary: torch.Tensor
    task_labels_consumed: bool = False
    route_context_consumed: bool = False
    corruption_metadata_consumed: bool = False
    schema: str = SCHEMA


def stage1_u_tokenizer_pretraining_terms(
    *,
    tokenizer: UQComponentTokenizer,
    reconstruction_head: UQSummaryReconstructionHead,
    components: torch.Tensor,
    zero_anchor_weight: float = 0.1,
    smooth_l1_beta: float = 0.05,
) -> Stage1UTokenizerPretrainingTerms:
    """Reconstruct task-agnostic U summaries without any driving labels."""

    if zero_anchor_weight < 0.0 or smooth_l1_beta <= 0.0:
        raise ValueError("pretraining loss weights are invalid")
    tokenized = tokenizer(components)
    decoded = reconstruction_head(tokenized.token_grid)
    target = tokenized.temporal_summary.detach()
    if decoded.shape != target.shape:
        raise ValueError("decoded and target U summaries differ in shape")
    reconstruction_loss = F.smooth_l1_loss(
        decoded, target, beta=float(smooth_l1_beta)
    )

    # A fixed all-zero observation is a definition-level anchor, not a
    # corruption example or task label.  It prevents coordinate/view embeddings
    # from making absence impossible to decode.
    zero_components = torch.zeros_like(components[:1])
    zero_tokenized = tokenizer(zero_components)
    zero_decoded = reconstruction_head(zero_tokenized.token_grid)
    zero_anchor_loss = F.smooth_l1_loss(
        zero_decoded,
        torch.zeros_like(zero_decoded),
        beta=float(smooth_l1_beta),
    )
    loss = reconstruction_loss + float(zero_anchor_weight) * zero_anchor_loss
    return Stage1UTokenizerPretrainingTerms(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        zero_anchor_loss=zero_anchor_loss,
        decoded_summary=decoded,
        target_summary=target,
    )


__all__ = [
    "SCHEMA",
    "Stage1UTokenizerPretrainingTerms",
    "UQSummaryReconstructionHead",
    "stage1_u_tokenizer_pretraining_terms",
]

"""Task-free objectives for aligning the UQFormer uncertainty modality.

These losses operate only on Stage-1 U components, deterministic transforms of
those components, or text embeddings describing the U field itself.  They do
not accept driving-task labels.  The reconstruction decoder is disposable and
must not be passed to ORION after alignment pretraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from uq_estimator.uq_modality_bridge import UQFormerOutput


SCHEMA = "orion.uqformer_alignment_objectives.v1"


class UQFormerReconstructionHead(nn.Module):
    """Disposable cross-attention decoder for the pooled Stage-1 U field."""

    def __init__(
        self,
        *,
        bridge_dim: int = 256,
        component_dim: int = 3,
        max_views: int = 6,
        num_heads: int = 8,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if min(bridge_dim, component_dim, max_views, num_heads, hidden_dim) <= 0:
            raise ValueError("reconstruction dimensions must be positive")
        if bridge_dim % num_heads:
            raise ValueError("bridge_dim must be divisible by num_heads")
        self.bridge_dim = int(bridge_dim)
        self.component_dim = int(component_dim)
        self.summary_dim = 3 * self.component_dim
        self.max_views = int(max_views)
        self.coordinate_projection = nn.Sequential(
            nn.Linear(2, bridge_dim), nn.GELU(), nn.Linear(bridge_dim, bridge_dim)
        )
        self.view_embedding = nn.Embedding(max_views, bridge_dim)
        self.query_type = nn.Parameter(torch.empty(bridge_dim))
        nn.init.normal_(self.query_type, std=0.02)
        self.query_norm = nn.LayerNorm(bridge_dim)
        self.memory_norm = nn.LayerNorm(bridge_dim)
        self.cross_attention = nn.MultiheadAttention(
            bridge_dim, num_heads, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(bridge_dim),
            nn.Linear(bridge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.summary_dim),
        )

    @staticmethod
    def _coordinates(
        count: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.linspace(-1.0, 1.0, count, device=device, dtype=dtype)

    def forward(
        self,
        compact_tokens: torch.Tensor,
        *,
        views: int,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if compact_tokens.ndim != 3 or compact_tokens.shape[-1] != self.bridge_dim:
            raise ValueError("compact_tokens must have shape [B,Q,bridge_dim]")
        if not compact_tokens.is_floating_point() or not bool(
            torch.isfinite(compact_tokens).all()
        ):
            raise ValueError("compact_tokens must be finite floating point")
        if views <= 0 or views > self.max_views:
            raise ValueError("reconstruction view dimension is invalid")
        if len(grid_hw) != 2 or min(map(int, grid_hw)) <= 0:
            raise ValueError("grid_hw must contain two positive values")
        batch = compact_tokens.shape[0]
        device, dtype = compact_tokens.device, compact_tokens.dtype
        grid_h, grid_w = map(int, grid_hw)
        y = self._coordinates(grid_h, device=device, dtype=dtype)
        x = self._coordinates(grid_w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xy = torch.stack((yy, xx), dim=-1)
        view = self.view_embedding(torch.arange(views, device=device)).to(dtype=dtype)
        query = (
            self.coordinate_projection(xy)[None]
            + view[:, None, None]
            + self.query_type.to(dtype=dtype)
        )
        query = query.reshape(1, -1, self.bridge_dim).expand(batch, -1, -1)
        decoded, _ = self.cross_attention(
            self.query_norm(query),
            self.memory_norm(compact_tokens),
            self.memory_norm(compact_tokens),
            need_weights=False,
        )
        decoded = self.output(decoded).reshape(
            batch, views, grid_h, grid_w, self.summary_dim
        )
        return decoded


@dataclass(frozen=True)
class UQFormerReconstructionTerms:
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    zero_anchor_loss: torch.Tensor
    decoded_components: torch.Tensor
    target_components: torch.Tensor
    task_labels_consumed: bool = False
    route_context_consumed: bool = False
    corruption_metadata_consumed: bool = False
    schema: str = SCHEMA


def uqformer_reconstruction_terms(
    *,
    output: UQFormerOutput,
    reconstruction_head: UQFormerReconstructionHead,
    zero_output: Optional[UQFormerOutput] = None,
    zero_anchor_weight: float = 0.1,
    smooth_l1_beta: float = 0.05,
) -> UQFormerReconstructionTerms:
    """Reconstruct U while preserving a definition-level all-zero anchor."""

    if zero_anchor_weight < 0.0 or smooth_l1_beta <= 0.0:
        raise ValueError("reconstruction loss weights are invalid")
    target = output.source_summary.detach()
    decoded = reconstruction_head(
        output.compact_tokens,
        views=target.shape[1],
        grid_hw=target.shape[2:4],
    )
    reconstruction_loss = F.smooth_l1_loss(
        decoded, target, beta=float(smooth_l1_beta)
    )

    if zero_output is not None:
        if not bool(zero_output.zero_input_mask.all()):
            raise ValueError("zero_output must come from all-zero U components")
        zero_target = zero_output.source_summary.detach()
        zero_decoded = reconstruction_head(
            zero_output.compact_tokens,
            views=zero_target.shape[1],
            grid_hw=zero_target.shape[2:4],
        )
        zero_anchor_loss = F.smooth_l1_loss(
            zero_decoded,
            torch.zeros_like(zero_decoded),
            beta=float(smooth_l1_beta),
        )
    elif bool(output.zero_input_mask.any()):
        zero_anchor_loss = F.smooth_l1_loss(
            decoded[output.zero_input_mask],
            torch.zeros_like(decoded[output.zero_input_mask]),
            beta=float(smooth_l1_beta),
        )
    else:
        zero_anchor_loss = decoded.sum() * 0.0
    loss = reconstruction_loss + float(zero_anchor_weight) * zero_anchor_loss
    return UQFormerReconstructionTerms(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        zero_anchor_loss=zero_anchor_loss,
        decoded_components=decoded,
        target_components=target,
    )


@dataclass(frozen=True)
class UQFormerEquivarianceTerms:
    loss: torch.Tensor
    view_spatial_loss: torch.Tensor
    temporal_loss: torch.Tensor
    component_loss: torch.Tensor
    global_invariance_loss: torch.Tensor
    schema: str = SCHEMA


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def uqformer_equivariance_terms(
    *,
    reference: UQFormerOutput,
    transformed: UQFormerOutput,
    view_permutation: Optional[Sequence[int]] = None,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
    temporal_changed: bool = False,
    temporal_change_margin: float = 0.1,
    component_permutation: Optional[Sequence[int]] = None,
    global_invariant: bool = False,
) -> UQFormerEquivarianceTerms:
    """Align deterministic U transforms in the compact structured layout.

    ``view_permutation[new_view] = old_view`` and the component permutation
    follows the same convention.  Latest/mean/delta are semantic statistic
    types, not exchangeable time slots.  A temporal counterfactual therefore
    uses a sensitivity margin instead of the incorrect shortcut of reversing
    the three query tokens.  Exact transformed 9-d values remain supervised by
    reconstruction.  This primitive never needs to know why a U field changed.

    ``global_invariant`` is deliberately opt-in.  It is valid only for a
    nuisance transform that leaves every described U fact unchanged, such as
    equivalent resampling of the same 9-d grid.  View, xy, temporal-statistic,
    or component semantic changes must be allowed to change global U tokens.
    """

    if reference.language_tokens.shape != transformed.language_tokens.shape:
        raise ValueError("equivariance outputs must have matching token shapes")
    target_view = reference.view_spatial_tokens
    if view_permutation is not None:
        permutation = torch.as_tensor(
            view_permutation,
            dtype=torch.long,
            device=target_view.device,
        )
        if sorted(permutation.tolist()) != list(range(target_view.shape[1])):
            raise ValueError("view_permutation must be a complete permutation")
        target_view = target_view.index_select(1, permutation)
    if vertical_flip:
        target_view = target_view.flip(2)
    if horizontal_flip:
        target_view = target_view.flip(3)
    view_loss = F.smooth_l1_loss(transformed.view_spatial_tokens, target_view)

    if temporal_change_margin < 0.0:
        raise ValueError("temporal_change_margin cannot be negative")
    if reference.temporal_tokens.shape[1]:
        if temporal_changed:
            temporal_distance = (
                transformed.temporal_tokens - reference.temporal_tokens
            ).square().mean().add(1e-12).sqrt()
            temporal_loss = F.relu(
                temporal_distance.new_tensor(float(temporal_change_margin))
                - temporal_distance
            )
        else:
            temporal_loss = F.smooth_l1_loss(
                transformed.temporal_tokens, reference.temporal_tokens
            )
    else:
        temporal_loss = _zero_loss(reference.language_tokens)

    if reference.component_tokens.shape[1]:
        target_component = reference.component_tokens
        if component_permutation is not None:
            permutation = torch.as_tensor(
                component_permutation,
                dtype=torch.long,
                device=target_component.device,
            )
            if sorted(permutation.tolist()) != list(
                range(target_component.shape[1])
            ):
                raise ValueError(
                    "component_permutation must be a complete permutation"
                )
            target_component = target_component.index_select(1, permutation)
        component_loss = F.smooth_l1_loss(
            transformed.component_tokens, target_component
        )
    else:
        component_loss = _zero_loss(reference.language_tokens)

    if reference.global_tokens.shape[1] and global_invariant:
        global_loss = F.smooth_l1_loss(
            transformed.global_tokens, reference.global_tokens
        )
    else:
        global_loss = _zero_loss(reference.language_tokens)
    loss = view_loss + temporal_loss + component_loss + global_loss
    return UQFormerEquivarianceTerms(
        loss=loss,
        view_spatial_loss=view_loss,
        temporal_loss=temporal_loss,
        component_loss=component_loss,
        global_invariance_loss=global_loss,
    )


def symmetric_u_text_alignment_loss(
    u_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """CLIP-style alignment for task-free captions describing only U facts."""

    if (
        u_embeddings.ndim != 2
        or text_embeddings.ndim != 2
        or u_embeddings.shape != text_embeddings.shape
    ):
        raise ValueError("U and text embeddings must have matching [B,D] shapes")
    if u_embeddings.shape[0] < 2:
        raise ValueError("contrastive alignment requires at least two pairs")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not bool(torch.isfinite(u_embeddings).all()) or not bool(
        torch.isfinite(text_embeddings).all()
    ):
        raise ValueError("alignment embeddings must be finite")
    u_normalized = F.normalize(u_embeddings, dim=-1)
    text_normalized = F.normalize(text_embeddings, dim=-1)
    logits = u_normalized @ text_normalized.transpose(0, 1)
    logits = logits / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )


__all__ = [
    "SCHEMA",
    "UQFormerEquivarianceTerms",
    "UQFormerReconstructionHead",
    "UQFormerReconstructionTerms",
    "symmetric_u_text_alignment_loss",
    "uqformer_equivariance_terms",
    "uqformer_reconstruction_terms",
]

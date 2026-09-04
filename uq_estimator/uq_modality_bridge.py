"""Task-agnostic observation-uncertainty modality alignment.

``UQFormerBridge`` treats frozen Stage-1 uncertainty components as their own
structured continuous modality.  It does not pretend that a U field is an RGB
image and it does not infer task relevance.  The bridge converts the time axis
to the frozen native ``latest/mean/delta`` contract, preserves explicit
``[view, y, x, temporal-statistic, component]`` structure in a small latent
memory, and uses learned queries to resample it into a compact token span.
The full pooled time sequence is returned only for diagnostics; query
attention is honestly defined over ``[view, y, x]`` source cells.

The public API intentionally has no route, actor, task-risk, TTC, action,
collision, planning, or corruption-family argument.  Those semantics belong
to the downstream VLM.  Query metadata and cross-attention maps are returned
so a later VLM-owned relevance decoder can project its decisions back onto the
original U grid without changing the meaning of this bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA = "orion.uqformer_bridge.v1"
MODALITY = "observation_uncertainty"


@dataclass(frozen=True)
class UQFormerQueryLayout:
    """Auditable metadata for every compact query.

    Negative values mean that an axis is not bound for that query kind.  The
    kind ids are: 0=view/spatial, 1=temporal, 2=component, 3=global.
    """

    kind_ids: torch.Tensor
    view_ids: torch.Tensor
    xy: torch.Tensor
    temporal_statistic_ids: torch.Tensor
    component_ids: torch.Tensor
    view_slice: Tuple[int, int]
    temporal_slice: Tuple[int, int]
    component_slice: Tuple[int, int]
    global_slice: Tuple[int, int]


@dataclass(frozen=True)
class UQFormerOutput:
    """Compact U tokens plus source-aligned, task-free diagnostics.

    ``attention_maps`` has shape ``[B,Q,V,H,W]``.  ``pooled_components``
    retains ``T`` for audits, while ``source_summary`` is the exact flattened
    9-d latest/mean/delta representation consumed by cross-attention.
    """

    language_tokens: torch.Tensor
    compact_tokens: torch.Tensor
    view_spatial_tokens: torch.Tensor
    temporal_tokens: torch.Tensor
    component_tokens: torch.Tensor
    global_tokens: torch.Tensor
    pooled_components: torch.Tensor
    source_summary: torch.Tensor
    source_features: torch.Tensor
    attention_maps: torch.Tensor
    component_mean: torch.Tensor
    component_max: torch.Tensor
    zero_input_mask: torch.Tensor
    query_layout: UQFormerQueryLayout
    modality: str = MODALITY
    schema: str = SCHEMA

    def alignment_embedding(self) -> torch.Tensor:
        """Return a global language-width embedding for U-caption alignment."""

        if self.global_tokens.shape[1] > 0:
            return self.global_tokens.mean(dim=1)
        return self.language_tokens.mean(dim=1)


class _CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        bridge_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(bridge_dim)
        self.memory_norm = nn.LayerNorm(bridge_dim)
        self.cross_attention = nn.MultiheadAttention(
            bridge_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_attention_norm = nn.LayerNorm(bridge_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(bridge_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, bridge_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        attention_bias: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.cross_attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            attn_mask=attention_bias,
            need_weights=True,
            average_attn_weights=True,
        )
        queries = queries + attended
        queries = queries + self.feedforward(self.post_attention_norm(queries))
        return queries, weights


class UQFormerBridge(nn.Module):
    """Resample frozen Stage-1 U fields into compact language-width tokens.

    ``view_query_hw`` controls the spatial query lattice per camera.  The
    default layout emits 24 view/spatial queries for six cameras, three
    explicit latest/mean/delta queries, three component queries, and four
    global queries: 34 U tokens instead of the former 600-token 10x10
    concatenation.

    View/spatial queries are hard-masked to their matching camera and receive
    a soft locality bias.  Other queries can summarize the full U memory.
    Modality, source/query type, view, xy, temporal-statistic, and component
    identities all have explicit representations; matching the LLM width is
    only the final projection, not the alignment mechanism itself.
    """

    def __init__(
        self,
        *,
        component_dim: int = 3,
        model_dim: int = 4096,
        bridge_dim: int = 256,
        grid_hw: Tuple[int, int] = (10, 10),
        max_views: int = 6,
        view_query_hw: Tuple[int, int] = (2, 2),
        temporal_queries: int = 3,
        include_component_queries: bool = True,
        global_queries: int = 4,
        num_heads: int = 8,
        num_layers: int = 2,
        feedforward_dim: int = 1024,
        dropout: float = 0.0,
        spatial_locality_strength: float = 2.0,
    ) -> None:
        super().__init__()
        if min(component_dim, model_dim, bridge_dim, max_views) <= 0:
            raise ValueError("UQFormer dimensions must be positive")
        if bridge_dim % num_heads != 0 or num_heads <= 0:
            raise ValueError("bridge_dim must be divisible by positive num_heads")
        if num_layers <= 0 or feedforward_dim <= 0:
            raise ValueError("UQFormer layer dimensions must be positive")
        if len(grid_hw) != 2 or min(map(int, grid_hw)) <= 0:
            raise ValueError("grid_hw must contain two positive values")
        if len(view_query_hw) != 2 or min(map(int, view_query_hw)) <= 0:
            raise ValueError("view_query_hw must contain two positive values")
        if temporal_queries not in (0, 3):
            raise ValueError(
                "temporal_queries must be 0 or the explicit latest/mean/delta trio"
            )
        if global_queries < 0:
            raise ValueError("query counts cannot be negative")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if spatial_locality_strength < 0.0:
            raise ValueError("attention locality strengths cannot be negative")

        self.component_dim = int(component_dim)
        self.model_dim = int(model_dim)
        self.bridge_dim = int(bridge_dim)
        self.grid_hw = tuple(map(int, grid_hw))
        self.max_views = int(max_views)
        self.view_query_hw = tuple(map(int, view_query_hw))
        self.temporal_query_count = int(temporal_queries)
        self.include_component_queries = bool(include_component_queries)
        self.global_query_count = int(global_queries)
        self.spatial_locality_strength = float(spatial_locality_strength)
        self.summary_dim = 3 * self.component_dim

        # A nonlinear value encoder plus a value-weighted component basis make
        # component identity explicit instead of relying on channel ordering.
        self.value_encoder = nn.Sequential(
            nn.LayerNorm(self.summary_dim),
            nn.Linear(self.summary_dim, bridge_dim),
            nn.GELU(),
            nn.Linear(bridge_dim, bridge_dim),
        )
        self.component_value_basis = nn.Parameter(
            torch.empty(3, component_dim, bridge_dim)
        )
        nn.init.normal_(self.component_value_basis, std=0.02)
        self.coordinate_projection = nn.Sequential(
            nn.Linear(2, bridge_dim), nn.GELU(), nn.Linear(bridge_dim, bridge_dim)
        )
        self.view_embedding = nn.Embedding(max_views, bridge_dim)
        self.component_embedding = nn.Embedding(component_dim, bridge_dim)
        self.temporal_statistic_embedding = nn.Embedding(3, bridge_dim)
        self.source_type_embedding = nn.Parameter(torch.empty(bridge_dim))
        self.modality_embedding = nn.Parameter(torch.empty(bridge_dim))
        self.source_norm = nn.LayerNorm(bridge_dim)

        query_h, query_w = self.view_query_hw
        self.view_query_seed = nn.Parameter(
            torch.empty(query_h, query_w, bridge_dim)
        )
        self.temporal_query_seed = nn.Parameter(
            torch.empty(self.temporal_query_count, bridge_dim)
        )
        component_query_count = component_dim if include_component_queries else 0
        self.component_query_seed = nn.Parameter(
            torch.empty(component_query_count, bridge_dim)
        )
        self.global_query_seed = nn.Parameter(
            torch.empty(self.global_query_count, bridge_dim)
        )
        self.query_type_embedding = nn.Embedding(4, bridge_dim)
        self.query_norm = nn.LayerNorm(bridge_dim)
        for parameter in (
            self.source_type_embedding,
            self.modality_embedding,
            self.view_query_seed,
            self.temporal_query_seed,
            self.component_query_seed,
            self.global_query_seed,
        ):
            nn.init.normal_(parameter, std=0.02)

        self.layers = nn.ModuleList(
            _CrossAttentionBlock(
                bridge_dim=bridge_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.language_projection = nn.Sequential(
            nn.LayerNorm(bridge_dim),
            nn.Linear(bridge_dim, model_dim),
        )
        self.language_modality_embedding = nn.Parameter(torch.empty(model_dim))
        nn.init.normal_(self.language_modality_embedding, std=0.02)
        self.language_norm = nn.LayerNorm(model_dim)

    @property
    def compact_query_count_at_max_views(self) -> int:
        query_h, query_w = self.view_query_hw
        component_count = self.component_dim if self.include_component_queries else 0
        return (
            self.max_views * query_h * query_w
            + self.temporal_query_count
            + component_count
            + self.global_query_count
        )

    def _pool_components(self, components: torch.Tensor) -> torch.Tensor:
        batch, time, views, height, width, channels = components.shape
        pooled = components.permute(0, 1, 2, 5, 3, 4).reshape(
            batch * time * views, channels, height, width
        )
        pooled = F.adaptive_avg_pool2d(pooled, self.grid_hw)
        grid_h, grid_w = self.grid_hw
        return pooled.reshape(
            batch, time, views, channels, grid_h, grid_w
        ).permute(0, 1, 2, 4, 5, 3)

    @staticmethod
    def _coordinates(
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((yy, xx), dim=-1)

    def _temporal_summary(self, pooled: torch.Tensor) -> torch.Tensor:
        latest = pooled[:, -1]
        mean = pooled.mean(dim=1)
        delta = latest - pooled[:, 0]
        return torch.stack((latest, mean, delta), dim=-2)

    def _source_memory(self, summary: torch.Tensor) -> torch.Tensor:
        batch, views, grid_h, grid_w, statistics, components = summary.shape
        if statistics != 3 or components != self.component_dim:
            raise ValueError("source summary must be [latest,mean,delta] x components")
        dtype, device = summary.dtype, summary.device
        xy = self._coordinates(grid_h, grid_w, device=device, dtype=dtype)
        flat_summary = summary.reshape(batch, views, grid_h, grid_w, -1)
        content = self.value_encoder(flat_summary)
        content = content + torch.einsum(
            "bvhwsc,scd->bvhwd", summary, self.component_value_basis
        )
        view = self.view_embedding(torch.arange(views, device=device)).to(dtype=dtype)
        memory = (
            content
            + self.coordinate_projection(xy)[None, None]
            + view[None, :, None, None]
            + self.source_type_embedding.to(dtype=dtype)
            + self.modality_embedding.to(dtype=dtype)
        )
        source_features = self.source_norm(memory)
        return source_features.reshape(batch, -1, self.bridge_dim)

    def _query_layout_and_tokens(
        self,
        *,
        batch: int,
        views: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, UQFormerQueryLayout]:
        query_h, query_w = self.view_query_hw
        query_xy_grid = self._coordinates(
            query_h, query_w, device=device, dtype=dtype
        )
        query_xy = query_xy_grid.reshape(-1, 2)
        spatial_per_view = query_h * query_w
        view_ids = torch.arange(views, device=device).repeat_interleave(
            spatial_per_view
        )
        repeated_xy = query_xy.repeat(views, 1)
        view_tokens = (
            self.view_query_seed.to(dtype=dtype).reshape(-1, self.bridge_dim).repeat(
                views, 1
            )
            + self.view_embedding(view_ids).to(dtype=dtype)
            + self.coordinate_projection(repeated_xy)
            + self.query_type_embedding(
                torch.zeros(len(view_ids), dtype=torch.long, device=device)
            ).to(dtype=dtype)
        )

        if self.temporal_query_count:
            temporal_statistic_ids = torch.arange(3, device=device)
            temporal_tokens = (
                self.temporal_query_seed.to(dtype=dtype)
                + self.temporal_statistic_embedding(temporal_statistic_ids).to(
                    dtype=dtype
                )
                + self.query_type_embedding(
                    torch.ones(
                        self.temporal_query_count, dtype=torch.long, device=device
                    )
                ).to(dtype=dtype)
            )
        else:
            temporal_statistic_ids = torch.empty(
                0, device=device, dtype=torch.long
            )
            temporal_tokens = torch.empty(
                0, self.bridge_dim, device=device, dtype=dtype
            )

        component_count = self.component_dim if self.include_component_queries else 0
        if component_count:
            component_ids = torch.arange(component_count, device=device)
            component_tokens = (
                self.component_query_seed.to(dtype=dtype)
                + self.component_embedding(component_ids).to(dtype=dtype)
                + self.query_type_embedding(
                    torch.full(
                        (component_count,), 2, dtype=torch.long, device=device
                    )
                ).to(dtype=dtype)
            )
        else:
            component_ids = torch.empty(0, device=device, dtype=torch.long)
            component_tokens = torch.empty(
                0, self.bridge_dim, device=device, dtype=dtype
            )

        global_tokens = self.global_query_seed.to(dtype=dtype)
        if self.global_query_count:
            global_tokens = global_tokens + self.query_type_embedding(
                torch.full(
                    (self.global_query_count,), 3, dtype=torch.long, device=device
                )
            ).to(dtype=dtype)

        tokens = torch.cat(
            (view_tokens, temporal_tokens, component_tokens, global_tokens), dim=0
        )
        tokens = tokens + self.modality_embedding.to(dtype=dtype)
        tokens = tokens[None].expand(batch, -1, -1)
        tokens = self.query_norm(tokens)

        view_end = len(view_tokens)
        temporal_end = view_end + len(temporal_tokens)
        component_end = temporal_end + len(component_tokens)
        total = component_end + len(global_tokens)
        kind_ids = torch.cat(
            (
                torch.zeros(view_end, dtype=torch.long, device=device),
                torch.ones(len(temporal_tokens), dtype=torch.long, device=device),
                torch.full(
                    (len(component_tokens),), 2, dtype=torch.long, device=device
                ),
                torch.full((len(global_tokens),), 3, dtype=torch.long, device=device),
            )
        )
        layout_view_ids = torch.full((total,), -1, dtype=torch.long, device=device)
        layout_view_ids[:view_end] = view_ids
        layout_xy = torch.full((total, 2), -2.0, device=device, dtype=dtype)
        layout_xy[:view_end] = repeated_xy
        layout_temporal_statistic = torch.full(
            (total,), -1, device=device, dtype=torch.long
        )
        layout_temporal_statistic[view_end:temporal_end] = (
            temporal_statistic_ids
        )
        layout_component = torch.full(
            (total,), -1, dtype=torch.long, device=device
        )
        layout_component[temporal_end:component_end] = component_ids
        layout = UQFormerQueryLayout(
            kind_ids=kind_ids,
            view_ids=layout_view_ids,
            xy=layout_xy,
            temporal_statistic_ids=layout_temporal_statistic,
            component_ids=layout_component,
            view_slice=(0, view_end),
            temporal_slice=(view_end, temporal_end),
            component_slice=(temporal_end, component_end),
            global_slice=(component_end, total),
        )
        return tokens, layout

    def _attention_bias(
        self,
        *,
        layout: UQFormerQueryLayout,
        views: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        source_xy = self._coordinates(
            grid_h, grid_w, device=device, dtype=dtype
        ).reshape(-1, 2)
        source_view_ids = (
            torch.arange(views, device=device)[:, None]
            .expand(views, grid_h * grid_w)
            .reshape(-1)
        )
        source_xy = source_xy.repeat(views, 1)
        query_count = len(layout.kind_ids)
        source_count = len(source_view_ids)
        bias = torch.zeros(query_count, source_count, device=device, dtype=dtype)

        view_start, view_end = layout.view_slice
        if view_end > view_start:
            query_views = layout.view_ids[view_start:view_end]
            wrong_view = query_views[:, None] != source_view_ids[None]
            local_distance = (
                layout.xy[view_start:view_end, None] - source_xy[None]
            ).square().sum(dim=-1)
            view_bias = -self.spatial_locality_strength * local_distance
            view_bias = view_bias.masked_fill(wrong_view, float("-inf"))
            bias[view_start:view_end] = view_bias

        return bias

    def forward(self, components: torch.Tensor) -> UQFormerOutput:
        if components.ndim != 6 or components.shape[-1] != self.component_dim:
            raise ValueError("components must have shape [B,T,V,H,W,C]")
        if not components.is_floating_point() or not bool(
            torch.isfinite(components).all()
        ):
            raise ValueError("components must be finite floating point")
        if bool((components < 0).any()) or bool((components > 1).any()):
            raise ValueError("normalized components must lie in [0,1]")
        batch, time, views, _, _, _ = components.shape
        if batch <= 0 or time <= 0 or views <= 0 or views > self.max_views:
            raise ValueError("batch/time/view dimensions are invalid")

        pooled = self._pool_components(components)
        summary = self._temporal_summary(pooled)
        grid_h, grid_w = self.grid_hw
        zero_input_mask = summary.abs().amax(dim=(1, 2, 3, 4, 5)) == 0
        memory = self._source_memory(summary)
        queries, layout = self._query_layout_and_tokens(
            batch=batch,
            views=views,
            device=components.device,
            dtype=components.dtype,
        )
        bias = self._attention_bias(
            layout=layout,
            views=views,
            grid_h=grid_h,
            grid_w=grid_w,
            device=components.device,
            dtype=components.dtype,
        )
        attention = None
        for layer in self.layers:
            queries, attention = layer(queries, memory, bias)
        if attention is None:
            raise RuntimeError("UQFormer produced no cross-attention map")
        compact = self.query_norm(queries)
        language = self.language_projection(compact)
        language = self.language_norm(
            language + self.language_modality_embedding.to(dtype=language.dtype)
        )
        attention_maps = attention.reshape(
            batch, attention.shape[1], views, grid_h, grid_w
        )

        view_start, view_end = layout.view_slice
        temporal_start, temporal_end = layout.temporal_slice
        component_start, component_end = layout.component_slice
        global_start, global_end = layout.global_slice
        query_h, query_w = self.view_query_hw
        view_tokens = language[:, view_start:view_end].reshape(
            batch, views, query_h, query_w, self.model_dim
        )
        return UQFormerOutput(
            language_tokens=language,
            compact_tokens=compact,
            view_spatial_tokens=view_tokens,
            temporal_tokens=language[:, temporal_start:temporal_end],
            component_tokens=language[:, component_start:component_end],
            global_tokens=language[:, global_start:global_end],
            pooled_components=pooled,
            source_summary=summary.reshape(
                batch, views, grid_h, grid_w, self.summary_dim
            ),
            source_features=memory.reshape(
                batch, views, grid_h, grid_w, self.bridge_dim
            ),
            attention_maps=attention_maps,
            component_mean=pooled.mean(dim=(1, 2, 3, 4)),
            component_max=pooled.amax(dim=(1, 2, 3, 4)),
            zero_input_mask=zero_input_mask,
            query_layout=layout,
        )


__all__ = [
    "MODALITY",
    "SCHEMA",
    "UQFormerBridge",
    "UQFormerOutput",
    "UQFormerQueryLayout",
]

"""Continuous visibility-token insertion for the Qwen-Drive VLM sidecar.

This module is intentionally Torch-dependent and must only be imported by the
Qwen sidecar/trainer environment. Dense visibility construction and physical
tokenization remain in the NumPy-only CARLA process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


VISIBILITY_VLM_SCHEMA = "orion.qwen-visibility-vlm-insertion/v1"


class VisibilityTokenProjector(nn.Module):
    """Map versioned physical token features into the Qwen hidden space."""

    def __init__(self, feature_dim: int, hidden_dim: int, vlm_hidden_dim: int) -> None:
        super().__init__()
        if min(int(feature_dim), int(hidden_dim), int(vlm_hidden_dim)) <= 0:
            raise ValueError("projector dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.vlm_hidden_dim = int(vlm_hidden_dim)
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.input_projection = nn.Linear(self.feature_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(self.hidden_dim, self.vlm_hidden_dim)
        self.boundary_embeddings = nn.Parameter(torch.zeros(2, self.vlm_hidden_dim))

        # A newly constructed V0 adapter is behavior-neutral in feature value.
        # Sequence insertion itself is not claimed to be baseline-identical.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                "visibility features must have shape [N,%d]" % self.feature_dim
            )
        if not torch.isfinite(features).all():
            raise ValueError("visibility features must be finite")
        hidden = self.input_projection(self.input_norm(features.float()))
        return self.output_projection(self.activation(hidden))


class TypedScalarVisibilityTokenProjector(nn.Module):
    """Encode each physical field in a distinct scalar-basis channel block.

    Unlike the generic V0 projector, this adapter does not normalize the 23
    fields against one another inside each row. Absolute physical thresholds
    therefore remain available, and field identity is deterministic before
    any learned mixing. The adapter still emits one token per physical row and
    does not predict semantic relevance or a driving action.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        vlm_hidden_dim: int,
        scalar_basis_dim: int = 4,
    ) -> None:
        super().__init__()
        dimensions = (
            int(feature_dim),
            int(hidden_dim),
            int(vlm_hidden_dim),
            int(scalar_basis_dim),
        )
        if min(dimensions) <= 0:
            raise ValueError("typed projector dimensions must be positive")
        if int(scalar_basis_dim) != 4:
            raise ValueError("typed projector v1 requires a four-term scalar basis")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.vlm_hidden_dim = int(vlm_hidden_dim)
        self.scalar_basis_dim = int(scalar_basis_dim)
        expanded_dim = self.feature_dim * self.scalar_basis_dim
        self.field_basis_projection = nn.Linear(expanded_dim, self.hidden_dim)
        self.hidden_norm = nn.LayerNorm(self.hidden_dim)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(self.hidden_dim, self.vlm_hidden_dim)
        self.boundary_embeddings = nn.Parameter(torch.zeros(2, self.vlm_hidden_dim))
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def scalar_basis(self, features: torch.Tensor) -> torch.Tensor:
        """Return `[N, feature, basis]` without cross-field normalization."""

        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                "visibility features must have shape [N,%d]" % self.feature_dim
            )
        if not torch.isfinite(features).all():
            raise ValueError("visibility features must be finite")
        values = features.float()
        return torch.stack(
            (
                values,
                values.square(),
                torch.sin(torch.pi * values),
                torch.cos(torch.pi * values),
            ),
            dim=-1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        basis = self.scalar_basis(features)
        expanded = basis.reshape(basis.shape[0], -1)
        hidden = self.field_basis_projection(expanded)
        hidden = self.activation(self.hidden_norm(hidden))
        return self.output_projection(hidden)


class SlotTypedScalarVisibilityTokenProjector(
    TypedScalarVisibilityTokenProjector
):
    """Add a deterministic one-hot G/F sequence-slot identity to each row."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        vlm_hidden_dim: int,
        scalar_basis_dim: int = 4,
        maximum_token_slots: int = 48,
    ) -> None:
        super().__init__(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            vlm_hidden_dim=vlm_hidden_dim,
            scalar_basis_dim=scalar_basis_dim,
        )
        if int(maximum_token_slots) <= 0:
            raise ValueError("maximum_token_slots must be positive")
        self.maximum_token_slots = int(maximum_token_slots)
        expanded_dim = (
            self.feature_dim * self.scalar_basis_dim + self.maximum_token_slots
        )
        self.field_basis_projection = nn.Linear(expanded_dim, self.hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        basis = self.scalar_basis(features)
        token_count = int(basis.shape[0])
        if token_count > self.maximum_token_slots:
            raise ValueError(
                "visibility token count exceeds the configured slot budget"
            )
        slot_identity = torch.eye(
            self.maximum_token_slots,
            device=basis.device,
            dtype=basis.dtype,
        )[:token_count]
        expanded = torch.cat(
            [basis.reshape(token_count, -1), slot_identity], dim=-1
        )
        hidden = self.field_basis_projection(expanded)
        hidden = self.activation(self.hidden_norm(hidden))
        return self.output_projection(hidden)


class VisibilityFieldQueryBlock(nn.Module):
    """One pre-norm cross-attention/MLP block for a physical record query."""

    def __init__(self, hidden_dim: int, attention_heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.field_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, query: torch.Tensor, fields: torch.Tensor) -> torch.Tensor:
        normalized_query = self.query_norm(query)
        attended, _ = self.cross_attention(
            normalized_query,
            self.field_norm(fields),
            self.field_norm(fields),
            need_weights=False,
        )
        query = query + attended
        return query + self.mlp(self.mlp_norm(query))


class FieldQueryVisibilityTokenProjector(nn.Module):
    """Translate typed scalar records into Qwen tokens with inspectable heads.

    Every scalar is encoded separately with field, record-family, slot, and
    continuous-value identity. One query per valid physical record attends only
    to that record's fields. It cannot mix scenes or directly emit risk/action.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        vlm_hidden_dim: int,
        attention_heads: int = 8,
        query_layers: int = 2,
        maximum_token_slots: int = 48,
        scalar_basis_dim: int = 4,
    ) -> None:
        super().__init__()
        dimensions = (
            int(feature_dim),
            int(hidden_dim),
            int(vlm_hidden_dim),
            int(attention_heads),
            int(query_layers),
            int(maximum_token_slots),
        )
        if min(dimensions) <= 0:
            raise ValueError("field-query projector dimensions must be positive")
        if int(scalar_basis_dim) != 4:
            raise ValueError("field-query projector v1 requires four scalar bases")
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if feature_dim < 2:
            raise ValueError("record type flags require at least two fields")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.vlm_hidden_dim = int(vlm_hidden_dim)
        self.maximum_token_slots = int(maximum_token_slots)
        self.scalar_basis_dim = int(scalar_basis_dim)
        self.field_embeddings = nn.Embedding(self.feature_dim, self.hidden_dim)
        self.record_type_embeddings = nn.Embedding(2, self.hidden_dim)
        self.slot_embeddings = nn.Embedding(self.maximum_token_slots, self.hidden_dim)
        self.scalar_projection = nn.Linear(self.scalar_basis_dim, self.hidden_dim)
        self.query_seed = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.blocks = nn.ModuleList(
            VisibilityFieldQueryBlock(self.hidden_dim, int(attention_heads))
            for _ in range(int(query_layers))
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.vlm_hidden_dim)
        self.boundary_embeddings = nn.Parameter(torch.zeros(2, self.vlm_hidden_dim))
        self.field_reconstruction_head = nn.Linear(self.hidden_dim, self.feature_dim)
        self.record_type_head = nn.Linear(self.hidden_dim, 2)
        self.slot_head = nn.Linear(self.hidden_dim, self.maximum_token_slots)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def scalar_basis(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                "visibility features must have shape [N,%d]" % self.feature_dim
            )
        if not torch.isfinite(features).all():
            raise ValueError("visibility features must be finite")
        values = features.float()
        return torch.stack(
            (
                values,
                values.square(),
                torch.sin(torch.pi * values),
                torch.cos(torch.pi * values),
            ),
            dim=-1,
        )

    def identities(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover explicit family and G/F-local slot ids from canonical rows."""

        if features.shape[0] > self.maximum_token_slots:
            raise ValueError("visibility token count exceeds the configured slot budget")
        flags = features[:, :2].float()
        valid_flags = torch.logical_and(
            torch.isclose(flags.sum(dim=1), torch.ones_like(flags[:, 0])),
            torch.logical_or(flags[:, 0] > 0.5, flags[:, 1] > 0.5),
        )
        if not bool(valid_flags.all()):
            raise ValueError("each record requires exactly one global/frontier type flag")
        record_types = torch.argmax(flags, dim=1)
        global_local = torch.cumsum((record_types == 0).long(), dim=0) - 1
        frontier_local = torch.cumsum((record_types == 1).long(), dim=0) - 1
        if int((record_types == 0).sum().item()) > 16:
            raise ValueError("field-query projector supports at most 16 global records")
        if int((record_types == 1).sum().item()) > self.maximum_token_slots - 16:
            raise ValueError("frontier records exceed the configured slot budget")
        slots = torch.where(record_types == 0, global_local, 16 + frontier_local)
        if bool((slots < 0).any()) or bool((slots >= self.maximum_token_slots).any()):
            raise ValueError("derived G/F slot lies outside the configured slot budget")
        return record_types, slots

    def encode_records(self, features: torch.Tensor) -> tuple[torch.Tensor, dict]:
        basis = self.scalar_basis(features)
        record_types, slots = self.identities(features)
        count = int(features.shape[0])
        field_ids = torch.arange(self.feature_dim, device=features.device)
        shared = (
            self.record_type_embeddings(record_types)
            + self.slot_embeddings(slots)
        )
        field_tokens = (
            self.scalar_projection(basis)
            + self.field_embeddings(field_ids).unsqueeze(0)
            + shared.unsqueeze(1)
        )
        query = self.query_seed.expand(count, -1, -1) + shared.unsqueeze(1)
        for block in self.blocks:
            query = block(query, field_tokens)
        hidden = self.output_norm(query[:, 0])
        auxiliary = {
            "field_values": self.field_reconstruction_head(hidden),
            "record_type_logits": self.record_type_head(hidden),
            "slot_logits": self.slot_head(hidden),
            "record_type_targets": record_types,
            "slot_targets": slots,
        }
        return hidden, auxiliary

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.encode_records(features)
        return self.output_projection(hidden)


@dataclass
class VisibilityPrefillResult:
    """Auditable output of one official or visibility-augmented VLM prefill."""

    scene_cache: list
    anchor: torch.Tensor
    base_position_ids: torch.Tensor
    augmented_position_ids: torch.Tensor
    insertion_index: int
    visibility_token_count: int
    enabled: bool
    past_key_values: object = None
    last_hidden_state: Optional[torch.Tensor] = None

    @property
    def base_sequence_length(self) -> int:
        return int(self.base_position_ids.shape[-1])

    @property
    def augmented_sequence_length(self) -> int:
        return int(self.augmented_position_ids.shape[-1])


def _validate_single_scene_tokens(
    token_features: torch.Tensor,
    token_mask: torch.Tensor,
    feature_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.as_tensor(token_features)
    mask = torch.as_tensor(token_mask, dtype=torch.bool, device=features.device)
    if features.ndim == 3:
        if features.shape[0] != 1:
            raise ValueError("V0 visibility insertion supports batch size one")
        features = features[0]
    if mask.ndim == 2:
        if mask.shape[0] != 1:
            raise ValueError("V0 visibility insertion supports batch size one")
        mask = mask[0]
    if features.ndim != 2 or features.shape[1] != int(feature_dim):
        raise ValueError("invalid physical visibility token shape")
    if mask.shape != (features.shape[0],):
        raise ValueError("visibility token mask must have shape [N]")
    if not torch.isfinite(features).all():
        raise ValueError("physical visibility tokens must be finite")
    if not bool(mask.any()):
        raise ValueError("at least one visibility token must be valid")
    return features, mask


def _official_multimodal_embeddings(model, inputs: dict) -> torch.Tensor:
    """Mirror Transformers 5.14.1 image scatter before language decoding."""

    input_ids = inputs["input_ids"]
    embeddings = model.vlm.get_input_embeddings()(input_ids)
    image_outputs = model.vlm.model.get_image_features(
        inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True
    )
    image_features = torch.cat(image_outputs.pooler_output, dim=0).to(
        embeddings.device, embeddings.dtype
    )
    image_mask, _ = model.vlm.model.get_placeholder_mask(
        input_ids, inputs_embeds=embeddings, image_features=image_features
    )
    return embeddings.masked_scatter(image_mask, image_features)


def _last_vision_end_insertion_index(model, input_ids: torch.Tensor) -> int:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("V0 visibility insertion supports one input sequence")
    vision_end_id = int(model.config.vlm_config.vision_end_token_id)
    positions = torch.nonzero(input_ids[0] == vision_end_id, as_tuple=False).flatten()
    if len(positions) == 0:
        raise ValueError("Qwen prompt contains no vision-end token")
    return int(positions[-1].item()) + 1


def prefill_with_visibility_tokens(
    model,
    inputs: dict,
    token_features: Optional[torch.Tensor],
    token_mask: Optional[torch.Tensor],
    projector: Optional[VisibilityTokenProjector],
    enabled: bool,
) -> VisibilityPrefillResult:
    """Run an official or continuous-U VLM prefill and expose its position contract.

    The U block is inserted after the final camera's vision-end marker and
    before the driving-history/navigation text. Its two learned boundary
    vectors and valid physical token vectors are treated as ordinary text
    positions for Qwen3.5 mRoPE. The released Planning Expert receives the
    resulting post-rotary full-attention K/V cache.
    """

    input_ids = inputs["input_ids"]
    base_positions = model._rope_positions(input_ids, inputs["image_grid_thw"])
    insertion_index = _last_vision_end_insertion_index(model, input_ids)
    if not enabled:
        scene_cache, anchor = model._prefill(inputs)
        return VisibilityPrefillResult(
            scene_cache=scene_cache,
            anchor=anchor,
            base_position_ids=base_positions,
            augmented_position_ids=base_positions,
            insertion_index=insertion_index,
            visibility_token_count=0,
            enabled=False,
        )
    if token_features is None or token_mask is None or projector is None:
        raise ValueError("enabled visibility prefill requires tokens, mask, and projector")
    if input_ids.shape[0] != 1:
        raise ValueError("V0 visibility insertion supports batch size one")
    features, mask = _validate_single_scene_tokens(
        token_features, token_mask, projector.feature_dim
    )
    features = features.to(device=input_ids.device)
    mask = mask.to(device=input_ids.device)
    valid_features = features[mask]
    projected = projector(valid_features).to(
        device=input_ids.device,
        dtype=model.vlm.get_input_embeddings().weight.dtype,
    )
    boundaries = projector.boundary_embeddings.to(
        device=input_ids.device, dtype=projected.dtype
    )
    visibility_block = torch.cat(
        [boundaries[0:1], projected, boundaries[1:2]], dim=0
    ).unsqueeze(0)

    base_embeddings = _official_multimodal_embeddings(model, inputs)
    augmented_embeddings = torch.cat(
        [
            base_embeddings[:, :insertion_index],
            visibility_block,
            base_embeddings[:, insertion_index:],
        ],
        dim=1,
    )
    block_length = int(visibility_block.shape[1])
    dummy_token_id = 0
    if dummy_token_id == int(model.config.vlm_config.image_token_id):
        raise ValueError("visibility position placeholder collides with image token")
    dummy_ids = torch.full(
        (1, block_length),
        dummy_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    augmented_ids = torch.cat(
        [
            input_ids[:, :insertion_index],
            dummy_ids,
            input_ids[:, insertion_index:],
        ],
        dim=1,
    )
    augmented_positions = model._rope_positions(
        augmented_ids, inputs["image_grid_thw"]
    )
    outputs = model.vlm.model.language_model(
        input_ids=None,
        position_ids=augmented_positions,
        inputs_embeds=augmented_embeddings,
        use_cache=True,
    )
    anchor = augmented_positions[:, :, -1]
    return VisibilityPrefillResult(
        scene_cache=model._scene_cache(outputs.past_key_values),
        anchor=anchor,
        base_position_ids=base_positions,
        augmented_position_ids=augmented_positions,
        insertion_index=insertion_index,
        visibility_token_count=int(mask.sum().item()),
        enabled=True,
        past_key_values=outputs.past_key_values,
        last_hidden_state=outputs.last_hidden_state,
    )


def visibility_position_contract(result: VisibilityPrefillResult) -> dict:
    """Check that insertion preserves prefix positions and shifts only the suffix."""

    if not result.enabled:
        return {
            "schema": VISIBILITY_VLM_SCHEMA,
            "enabled": False,
            "identity_positions": bool(
                torch.equal(result.base_position_ids, result.augmented_position_ids)
            ),
            "base_sequence_length": result.base_sequence_length,
            "augmented_sequence_length": result.augmented_sequence_length,
        }
    insertion = result.insertion_index
    block_length = result.visibility_token_count + 2
    base = result.base_position_ids
    augmented = result.augmented_position_ids
    prefix_equal = torch.equal(base[..., :insertion], augmented[..., :insertion])
    suffix_shift = augmented[..., insertion + block_length :] - base[..., insertion:]
    suffix_shift_exact = bool(torch.all(suffix_shift == block_length).item())
    u_positions = augmented[..., insertion : insertion + block_length]
    u_contiguous = bool(torch.all(torch.diff(u_positions, dim=-1) == 1).item())
    anchor_exact = torch.equal(result.anchor, augmented[:, :, -1])
    cache_lengths = [int(key.shape[1]) for key, _ in result.scene_cache]
    return {
        "schema": VISIBILITY_VLM_SCHEMA,
        "enabled": True,
        "base_sequence_length": result.base_sequence_length,
        "augmented_sequence_length": result.augmented_sequence_length,
        "insertion_index": insertion,
        "visibility_token_count": result.visibility_token_count,
        "visibility_block_length": block_length,
        "prefix_positions_equal": bool(prefix_equal),
        "suffix_shift_exact": suffix_shift_exact,
        "visibility_positions_contiguous": u_contiguous,
        "anchor_exact": bool(anchor_exact),
        "scene_cache_lengths": cache_lengths,
        "scene_cache_length_exact": bool(
            cache_lengths
            and all(length == result.augmented_sequence_length for length in cache_lengths)
        ),
    }


@dataclass
class VisibilityReasoningResult:
    """Reasoning text and final cache after closing the augmented assistant turn."""

    scene_cache: list
    anchor: torch.Tensor
    reasoning: str
    reasoning_token_ids: list
    prompt_prefill: VisibilityPrefillResult
    final_sequence_length: int
    enabled: bool


def _continue_greedy_reasoning(
    model,
    cache,
    last_hidden_state: torch.Tensor,
    prompt_anchor: torch.Tensor,
    prompt_length: int,
    max_new_tokens: int,
) -> tuple[list, torch.Tensor, str, list, int]:
    """Match released greedy reasoning and turn closure from an arbitrary prefix."""

    if int(max_new_tokens) <= 0:
        raise ValueError("max_new_tokens must be positive")
    processor = model.processor
    terminators = {
        int(processor.im_end_id),
        int(model.config.vlm_config.text_config.eos_token_id),
    }
    content_ids = []
    logits = model.vlm.lm_head(last_hidden_state[:, -1]).float()
    for step in range(int(max_new_tokens)):
        selection_logits = logits
        if step < int(model.config.min_reasoning_tokens):
            selection_logits = logits.clone()
            for token_id in terminators:
                selection_logits[:, token_id] = -torch.inf
        next_token = int(torch.argmax(selection_logits, dim=-1).item())
        if next_token in terminators:
            break
        content_ids.append(next_token)
        token_tensor = torch.tensor(
            [[next_token]], dtype=torch.long, device=prompt_anchor.device
        )
        token_embedding = model.vlm.get_input_embeddings()(token_tensor)
        offset = len(content_ids)
        position_ids = prompt_anchor.unsqueeze(-1) + offset
        outputs = model.vlm.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            past_key_values=cache,
            inputs_embeds=token_embedding,
            cache_position=torch.tensor(
                [prompt_length + offset - 1],
                dtype=torch.long,
                device=prompt_anchor.device,
            ),
            use_cache=True,
        )
        cache = outputs.past_key_values
        logits = model.vlm.lm_head(outputs.last_hidden_state[:, -1]).float()

    close_ids = [int(processor.im_end_id)] + [
        int(token_id) for token_id in processor.newline_ids
    ]
    close_tensor = torch.tensor(
        [close_ids], dtype=torch.long, device=prompt_anchor.device
    )
    close_embeddings = model.vlm.get_input_embeddings()(close_tensor)
    close_offsets = torch.arange(
        len(content_ids) + 1,
        len(content_ids) + len(close_ids) + 1,
        dtype=prompt_anchor.dtype,
        device=prompt_anchor.device,
    )
    close_positions = prompt_anchor.unsqueeze(-1) + close_offsets.view(1, 1, -1)
    close_cache_positions = torch.arange(
        prompt_length + len(content_ids),
        prompt_length + len(content_ids) + len(close_ids),
        dtype=torch.long,
        device=prompt_anchor.device,
    )
    outputs = model.vlm.model.language_model(
        input_ids=None,
        position_ids=close_positions,
        past_key_values=cache,
        inputs_embeds=close_embeddings,
        cache_position=close_cache_positions,
        use_cache=True,
    )
    cache = outputs.past_key_values
    anchor = prompt_anchor + len(content_ids) + len(close_ids)
    decoded = processor.tokenizer.decode(content_ids, skip_special_tokens=True)
    reasoning = decoded.split("</think>")[-1].strip()
    final_length = prompt_length + len(content_ids) + len(close_ids)
    return model._scene_cache(cache), anchor, reasoning, content_ids, final_length


def manual_reasoning_prefill_without_visibility(
    model, inputs: dict, max_new_tokens: int
) -> VisibilityReasoningResult:
    """Reproduce upstream reasoning manually to validate the augmented decoder."""

    input_ids = inputs["input_ids"]
    positions = model._rope_positions(input_ids, inputs["image_grid_thw"])
    embeddings = _official_multimodal_embeddings(model, inputs)
    outputs = model.vlm.model.language_model(
        input_ids=None,
        position_ids=positions,
        inputs_embeds=embeddings,
        use_cache=True,
    )
    prompt = VisibilityPrefillResult(
        scene_cache=model._scene_cache(outputs.past_key_values),
        anchor=positions[:, :, -1],
        base_position_ids=positions,
        augmented_position_ids=positions,
        insertion_index=_last_vision_end_insertion_index(model, input_ids),
        visibility_token_count=0,
        enabled=False,
        past_key_values=outputs.past_key_values,
        last_hidden_state=outputs.last_hidden_state,
    )
    scene_cache, anchor, reasoning, content_ids, final_length = (
        _continue_greedy_reasoning(
            model,
            prompt.past_key_values,
            prompt.last_hidden_state,
            prompt.anchor,
            prompt.base_sequence_length,
            max_new_tokens,
        )
    )
    return VisibilityReasoningResult(
        scene_cache=scene_cache,
        anchor=anchor,
        reasoning=reasoning,
        reasoning_token_ids=content_ids,
        prompt_prefill=prompt,
        final_sequence_length=final_length,
        enabled=False,
    )


def prefill_with_visibility_reasoning(
    model,
    inputs: dict,
    token_features: Optional[torch.Tensor],
    token_mask: Optional[torch.Tensor],
    projector: Optional[VisibilityTokenProjector],
    enabled: bool,
    max_new_tokens: int,
) -> VisibilityReasoningResult:
    """Generate reasoning from a U-augmented prompt and close the turn upstream-style."""

    if not enabled:
        scene_cache, anchor, reasoning = model._prefill_with_reasoning(
            inputs, int(max_new_tokens)
        )
        positions = model._rope_positions(
            inputs["input_ids"], inputs["image_grid_thw"]
        )
        prompt = VisibilityPrefillResult(
            scene_cache=[],
            anchor=positions[:, :, -1],
            base_position_ids=positions,
            augmented_position_ids=positions,
            insertion_index=_last_vision_end_insertion_index(
                model, inputs["input_ids"]
            ),
            visibility_token_count=0,
            enabled=False,
        )
        cache_length = int(scene_cache[0][0].shape[1]) if scene_cache else 0
        return VisibilityReasoningResult(
            scene_cache=scene_cache,
            anchor=anchor,
            reasoning=reasoning,
            reasoning_token_ids=[],
            prompt_prefill=prompt,
            final_sequence_length=cache_length,
            enabled=False,
        )
    prompt = prefill_with_visibility_tokens(
        model,
        inputs,
        token_features=token_features,
        token_mask=token_mask,
        projector=projector,
        enabled=True,
    )
    scene_cache, anchor, reasoning, content_ids, final_length = (
        _continue_greedy_reasoning(
            model,
            prompt.past_key_values,
            prompt.last_hidden_state,
            prompt.anchor,
            prompt.augmented_sequence_length,
            max_new_tokens,
        )
    )
    return VisibilityReasoningResult(
        scene_cache=scene_cache,
        anchor=anchor,
        reasoning=reasoning,
        reasoning_token_ids=content_ids,
        prompt_prefill=prompt,
        final_sequence_length=final_length,
        enabled=True,
    )

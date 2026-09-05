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

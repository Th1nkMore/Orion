"""Pure tensor contracts for the two-pass Stage2-L language bridge."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uq_estimator.uq_relevance_tokenizer import (
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
)


@dataclass(frozen=True)
class TwoPassLanguageConditioning:
    vision_tokens: torch.Tensor
    task_risk: torch.Tensor
    bridge_global_features: torch.Tensor


def extract_relevance_query_grid(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    image_token_index: int,
    visual_token_count: int,
    views: int,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Extract the expanded spatial-query span from VLM hidden states."""
    if hidden_states.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("hidden_states/input_ids ranks are invalid")
    if hidden_states.shape[0] != input_ids.shape[0] or input_ids.shape[0] != 1:
        raise ValueError("current query extraction requires batch size one")
    if min(visual_token_count, views, grid_h, grid_w) <= 0:
        raise ValueError("query span dimensions must be positive")
    locations = torch.where(input_ids[0] == int(image_token_index))[0]
    if locations.numel() != 1:
        raise ValueError("prompt must contain exactly one image token")
    start = int(locations.item()) + int(visual_token_count)
    count = int(views * grid_h * grid_w)
    span = hidden_states[:, start:start + count]
    if span.shape[1] != count:
        raise ValueError("expanded relevance-query span is incomplete")
    return span.reshape(1, views, grid_h, grid_w, hidden_states.shape[-1])


def build_two_pass_language_conditioning(
    *,
    uq_tokenizer: UQComponentTokenizer,
    risk_bridge: TaskRiskLanguageBridge,
    baseline_vision: torch.Tensor,
    components: torch.Tensor,
    relevance_logits: torch.Tensor,
) -> TwoPassLanguageConditioning:
    """Append Stage1-U tokens and compact K tokens to ORION visual tokens."""
    if baseline_vision.ndim != 3 or components.ndim != 6:
        raise ValueError("baseline vision/components ranks are invalid")
    if baseline_vision.shape[0] != components.shape[0]:
        raise ValueError("conditioning batch dimensions differ")
    components = components.to(
        device=baseline_vision.device, dtype=baseline_vision.dtype,
        non_blocking=True,
    )
    relevance_logits = relevance_logits.to(
        device=baseline_vision.device, dtype=baseline_vision.dtype,
        non_blocking=True,
    )
    uq = uq_tokenizer(components)
    bridge = risk_bridge(uq.latest_scalar_uq, relevance_logits)
    if uq.tokens.shape[-1] != baseline_vision.shape[-1]:
        raise ValueError("UQ and ORION token dimensions differ")
    if bridge.tokens.shape[-1] != baseline_vision.shape[-1]:
        raise ValueError("bridge and ORION token dimensions differ")
    vision = torch.cat((baseline_vision, uq.tokens, bridge.tokens), dim=1)
    return TwoPassLanguageConditioning(
        vision_tokens=vision,
        task_risk=bridge.task_risk,
        bridge_global_features=bridge.global_features,
    )


__all__ = [
    "TwoPassLanguageConditioning",
    "build_two_pass_language_conditioning",
    "extract_relevance_query_grid",
]

"""Gradient-safe Stage2-L v10 conditioning without a learned field head.

Stage1 supplies task-agnostic U and a frozen U tokenizer.  ORION/VLM predicts
R in its first pass.  The second pass receives frozen U tokens plus trainable
K-language bridge tokens computed from detached U and R.  Auxiliary QA may
therefore teach the bridge and ORION LoRA to verbalize deterministic semantics
without moving either U or R.  Spatial fields are decoded directly from U/R/K.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import torch

from uq_estimator.stage2l_deterministic_semantics_v10 import (
    DeterministicTaskSemantics,
    deterministic_task_semantics,
)
from uq_estimator.uq_relevance_tokenizer import (
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
    spatial_signal_summaries,
)


SCHEMA = "orion.stage2l_semantic_runtime.v10"


@dataclass(frozen=True)
class VLMTaskConditioningV10:
    vision_tokens: torch.Tensor
    task_risk: torch.Tensor
    deterministic_semantics: DeterministicTaskSemantics
    raw_observation_global_features: torch.Tensor
    raw_task_risk_global_features: torch.Tensor
    uq_tokenizer_frozen: bool
    uq_tokens_detached_for_stage2: bool
    relevance_detached_for_language: bool
    task_risk_bridge_trainable_by_language: bool
    learned_structured_field_head_used: bool
    semantic_classifier_token_count: int
    direct_control_output_enabled: bool = False
    trajectory_loss_enabled: bool = False
    schema: str = SCHEMA


def build_vlm_task_conditioning_v10(
    *,
    uq_tokenizer: UQComponentTokenizer,
    risk_bridge: TaskRiskLanguageBridge,
    baseline_vision: torch.Tensor,
    components: torch.Tensor,
    relevance_logits: torch.Tensor,
    semantic_decoder_kwargs: Mapping[str, Any] = None,
) -> VLMTaskConditioningV10:
    if baseline_vision.ndim != 3 or components.ndim != 6:
        raise ValueError("baseline vision/components ranks are invalid")
    if baseline_vision.shape[0] != components.shape[0]:
        raise ValueError("conditioning batch dimensions differ")
    if any(parameter.requires_grad for parameter in uq_tokenizer.parameters()):
        raise ValueError(
            "Stage2-L v10 requires a frozen task-agnostic U tokenizer"
        )
    components = components.to(
        device=baseline_vision.device,
        dtype=baseline_vision.dtype,
        non_blocking=True,
    )
    relevance_logits = relevance_logits.to(
        device=baseline_vision.device,
        dtype=baseline_vision.dtype,
        non_blocking=True,
    )
    uq = uq_tokenizer(components)
    frozen_scalar_uq = uq.latest_scalar_uq.detach()
    frozen_relevance = relevance_logits.detach()

    # The bridge owns language-facing K representation.  Its parameters may be
    # trained by QA, while detaching numerical inputs prevents the language
    # objective from modifying Stage1 U or the VLM's first-pass R map.
    bridge = risk_bridge(frozen_scalar_uq, frozen_relevance)
    _, observation_global = spatial_signal_summaries(
        frozen_scalar_uq, epsilon=risk_bridge.epsilon
    )
    semantics = deterministic_task_semantics(
        frozen_scalar_uq,
        frozen_relevance,
        **dict(semantic_decoder_kwargs or {}),
    )
    frozen_uq_tokens = uq.tokens.detach()
    if not (
        baseline_vision.shape[-1]
        == frozen_uq_tokens.shape[-1]
        == bridge.tokens.shape[-1]
    ):
        raise ValueError("ORION, U and K bridge token dimensions differ")
    vision_tokens = torch.cat(
        (baseline_vision, frozen_uq_tokens, bridge.tokens), dim=1
    )
    return VLMTaskConditioningV10(
        vision_tokens=vision_tokens,
        task_risk=bridge.task_risk,
        deterministic_semantics=semantics,
        raw_observation_global_features=observation_global,
        raw_task_risk_global_features=bridge.global_features,
        uq_tokenizer_frozen=True,
        uq_tokens_detached_for_stage2=True,
        relevance_detached_for_language=True,
        task_risk_bridge_trainable_by_language=True,
        learned_structured_field_head_used=False,
        semantic_classifier_token_count=0,
    )


__all__ = [
    "SCHEMA",
    "VLMTaskConditioningV10",
    "build_vlm_task_conditioning_v10",
]

"""Tensor contract for the Stage2-L structured semantic bottleneck."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uq_estimator.stage2l_bridge_runtime import (
    build_two_pass_language_conditioning,
)
from uq_estimator.stage2l_semantic_bottleneck import (
    PlanningStanceSemanticBottleneck,
)
from uq_estimator.uq_relevance_tokenizer import (
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
)


@dataclass(frozen=True)
class StructuredSemanticConditioning:
    vision_tokens: torch.Tensor
    task_risk: torch.Tensor
    bridge_global_features: torch.Tensor
    stance_logits: torch.Tensor
    stance_probabilities: torch.Tensor
    predicted_stance_indices: torch.Tensor


def build_structured_semantic_conditioning(
    *,
    uq_tokenizer: UQComponentTokenizer,
    risk_bridge: TaskRiskLanguageBridge,
    stance_bottleneck: PlanningStanceSemanticBottleneck,
    baseline_vision: torch.Tensor,
    components: torch.Tensor,
    relevance_logits: torch.Tensor,
) -> StructuredSemanticConditioning:
    """Append a soft predicted stance token after the existing U/K tokens."""

    base = build_two_pass_language_conditioning(
        uq_tokenizer=uq_tokenizer,
        risk_bridge=risk_bridge,
        baseline_vision=baseline_vision,
        components=components,
        relevance_logits=relevance_logits,
    )
    # The compact bridge occupies the final V+1 tokens of the base span.
    bridge_token_count = int(relevance_logits.shape[1]) + 1
    bridge_tokens = base.vision_tokens[:, -bridge_token_count:]
    stance = stance_bottleneck(bridge_tokens)
    vision = torch.cat((base.vision_tokens, stance.token), dim=1)
    return StructuredSemanticConditioning(
        vision_tokens=vision,
        task_risk=base.task_risk,
        bridge_global_features=base.bridge_global_features,
        stance_logits=stance.logits,
        stance_probabilities=stance.probabilities,
        predicted_stance_indices=stance.predicted_indices,
    )


__all__ = [
    "StructuredSemanticConditioning",
    "build_structured_semantic_conditioning",
]

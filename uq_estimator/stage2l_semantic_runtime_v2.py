"""Two-pass Stage2-L runtime with magnitude-preserving stance semantics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uq_estimator.stage2l_bridge_runtime import build_two_pass_language_conditioning
from uq_estimator.stage2l_semantic_bottleneck_v2 import (
    MagnitudePreservingPlanningStanceBottleneck,
)
from uq_estimator.uq_relevance_tokenizer import (
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
)


@dataclass(frozen=True)
class MagnitudeStructuredConditioning:
    vision_tokens: torch.Tensor
    task_risk: torch.Tensor
    raw_global_features: torch.Tensor
    magnitude_features: torch.Tensor
    stance_logits: torch.Tensor
    stance_probabilities: torch.Tensor
    predicted_stance_indices: torch.Tensor


def build_magnitude_structured_conditioning(
    *,
    uq_tokenizer: UQComponentTokenizer,
    risk_bridge: TaskRiskLanguageBridge,
    stance_bottleneck: MagnitudePreservingPlanningStanceBottleneck,
    baseline_vision: torch.Tensor,
    components: torch.Tensor,
    relevance_logits: torch.Tensor,
) -> MagnitudeStructuredConditioning:
    base = build_two_pass_language_conditioning(
        uq_tokenizer=uq_tokenizer,
        risk_bridge=risk_bridge,
        baseline_vision=baseline_vision,
        components=components,
        relevance_logits=relevance_logits,
    )
    bridge_token_count = int(relevance_logits.shape[1]) + 1
    bridge_tokens = base.vision_tokens[:, -bridge_token_count:]
    stance = stance_bottleneck(bridge_tokens, base.bridge_global_features)
    return MagnitudeStructuredConditioning(
        vision_tokens=torch.cat((base.vision_tokens, stance.token), dim=1),
        task_risk=base.task_risk,
        raw_global_features=stance.raw_global_features,
        magnitude_features=stance.magnitude_features,
        stance_logits=stance.logits,
        stance_probabilities=stance.probabilities,
        predicted_stance_indices=stance.predicted_indices,
    )


__all__ = [
    "MagnitudeStructuredConditioning",
    "build_magnitude_structured_conditioning",
]

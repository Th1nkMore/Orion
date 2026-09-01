"""Stage2-L language conditioning with explicit gradient ownership."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uq_estimator.stage2l_bridge_runtime import build_two_pass_language_conditioning
from uq_estimator.stage2l_semantic_bottleneck_v3 import (
    GradientRoutedPlanningStanceBottleneck,
)
from uq_estimator.uq_relevance_tokenizer import (
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
)


SCHEMA = "orion.stage2l_gradient_routed_semantic_runtime.v3"


@dataclass(frozen=True)
class GradientRoutedConditioning:
    vision_tokens: torch.Tensor
    task_risk: torch.Tensor
    raw_global_features: torch.Tensor
    magnitude_features: torch.Tensor
    stance_logits: torch.Tensor
    stance_probabilities: torch.Tensor
    predicted_stance_indices: torch.Tensor
    relevance_detached_for_language: bool
    stance_probabilities_detached_for_language: bool
    schema: str = SCHEMA


def build_gradient_routed_conditioning(
    *,
    uq_tokenizer: UQComponentTokenizer,
    risk_bridge: TaskRiskLanguageBridge,
    stance_bottleneck: GradientRoutedPlanningStanceBottleneck,
    baseline_vision: torch.Tensor,
    components: torch.Tensor,
    relevance_logits: torch.Tensor,
    detach_relevance_for_language: bool = True,
    detach_stance_probabilities_for_language: bool = True,
) -> GradientRoutedConditioning:
    """Build QA tokens without giving QA loss ownership of R or stance class."""

    language_relevance = (
        relevance_logits.detach()
        if detach_relevance_for_language
        else relevance_logits
    )
    base = build_two_pass_language_conditioning(
        uq_tokenizer=uq_tokenizer,
        risk_bridge=risk_bridge,
        baseline_vision=baseline_vision,
        components=components,
        relevance_logits=language_relevance,
    )
    bridge_token_count = int(relevance_logits.shape[1]) + 1
    bridge_tokens = base.vision_tokens[:, -bridge_token_count:]
    stance = stance_bottleneck(
        bridge_tokens,
        base.bridge_global_features,
        detach_probabilities_for_language=(
            detach_stance_probabilities_for_language
        ),
    )
    return GradientRoutedConditioning(
        vision_tokens=torch.cat((base.vision_tokens, stance.token), dim=1),
        task_risk=base.task_risk,
        raw_global_features=stance.raw_global_features,
        magnitude_features=stance.magnitude_features,
        stance_logits=stance.logits,
        stance_probabilities=stance.probabilities,
        predicted_stance_indices=stance.predicted_indices,
        relevance_detached_for_language=detach_relevance_for_language,
        stance_probabilities_detached_for_language=(
            detach_stance_probabilities_for_language
        ),
    )


__all__ = ["GradientRoutedConditioning", "SCHEMA", "build_gradient_routed_conditioning"]

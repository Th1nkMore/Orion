"""Stage2-L task-field conditioning with strict gradient ownership.

Stage1 provides frozen, task-agnostic observation uncertainty U.  The first
VLM pass predicts dense task relevance R.  Dense-map and matched-ranking losses
own R.  Explicit task-field loss trains the fixed K=U*sigmoid(R) bridge and the
VLM-owned categorical field head from detached predicted R.  Auxiliary QA
language also sees detached K tokens and detached predicted field
probabilities, so neither coarse fields nor free text can distort dense R.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from uq_estimator.stage2l_structured_field_head import (
    StructuredTaskFieldOutput,
    VLMTaskSemanticFieldHead,
)
from uq_estimator.uq_relevance_tokenizer import (
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
    spatial_signal_summaries,
)


SCHEMA = "orion.stage2l_vlm_task_field_runtime.v4"


@dataclass(frozen=True)
class VLMTaskFieldConditioning:
    vision_tokens: torch.Tensor
    task_risk: torch.Tensor
    raw_observation_global_features: torch.Tensor
    raw_task_risk_global_features: torch.Tensor
    field_logits: Mapping[str, torch.Tensor]
    field_probabilities: Mapping[str, torch.Tensor]
    predicted_field_indices: Mapping[str, torch.Tensor]
    relevance_detached_for_task_fields: bool
    relevance_detached_for_language: bool
    task_risk_bridge_detached_for_language: bool
    field_probabilities_detached_for_language: bool
    direct_control_output_enabled: bool = False
    trajectory_loss_enabled: bool = False
    schema: str = SCHEMA


def build_vlm_task_field_conditioning(
    *,
    uq_tokenizer: UQComponentTokenizer,
    risk_bridge: TaskRiskLanguageBridge,
    field_head: VLMTaskSemanticFieldHead,
    baseline_vision: torch.Tensor,
    components: torch.Tensor,
    relevance_logits: torch.Tensor,
) -> VLMTaskFieldConditioning:
    """Build second-pass tokens while preserving task-semantics ownership."""

    if baseline_vision.ndim != 3 or components.ndim != 6:
        raise ValueError("baseline vision/components ranks are invalid")
    if baseline_vision.shape[0] != components.shape[0]:
        raise ValueError("conditioning batch dimensions differ")
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

    # Dense R and matched ranking own the relevance path.  Coarse categorical
    # fields learn to interpret predicted R but may not reshape the map.
    task_field_bridge = risk_bridge(
        uq.latest_scalar_uq, relevance_logits.detach()
    )
    _, observation_global_features = spatial_signal_summaries(
        uq.latest_scalar_uq, epsilon=risk_bridge.epsilon
    )
    fields: StructuredTaskFieldOutput = field_head(
        task_field_bridge.tokens,
        observation_global_features,
        task_field_bridge.global_features,
        detach_probabilities_for_language=True,
    )

    # Auxiliary QA receives the same numerical K realization, but cannot send
    # gradients into R or the bridge encoding.  The VLM can still learn to read
    # these tokens during its second pass.
    language_bridge_tokens = task_field_bridge.tokens.detach()
    if uq.tokens.shape[-1] != baseline_vision.shape[-1]:
        raise ValueError("UQ and ORION token dimensions differ")
    if language_bridge_tokens.shape[-1] != baseline_vision.shape[-1]:
        raise ValueError("bridge and ORION token dimensions differ")
    vision_tokens = torch.cat(
        (
            baseline_vision,
            uq.tokens,
            language_bridge_tokens,
            fields.token,
        ),
        dim=1,
    )
    return VLMTaskFieldConditioning(
        vision_tokens=vision_tokens,
        task_risk=task_field_bridge.task_risk,
        raw_observation_global_features=observation_global_features,
        raw_task_risk_global_features=task_field_bridge.global_features,
        field_logits=fields.logits,
        field_probabilities=fields.probabilities,
        predicted_field_indices=fields.predicted_indices,
        relevance_detached_for_task_fields=True,
        relevance_detached_for_language=True,
        task_risk_bridge_detached_for_language=True,
        field_probabilities_detached_for_language=(
            fields.probabilities_detached_for_language
        ),
    )


__all__ = [
    "SCHEMA",
    "VLMTaskFieldConditioning",
    "build_vlm_task_field_conditioning",
]

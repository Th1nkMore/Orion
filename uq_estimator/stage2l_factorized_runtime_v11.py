"""Identifiable factorized Stage2-L v11 runtime.

The contextual relevance pass is invoked exactly once for a matched group.
Every uncertainty counterfactual then reuses the same relevance tensor and is
combined only through ``K = U * sigmoid(R)``.  This makes it impossible for a
variant-specific U input to alter R through this runtime.

The module does not define R labels, train Stage1, or produce control.  Its
baseline-only output is the explicit no-U ablation used to test whether the
language task can be solved without the U/K interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

import torch

from uq_estimator.stage2l_deterministic_semantics_v10 import (
    deterministic_task_semantics,
)
from uq_estimator.stage2l_identifiability import (
    MatchedTaskRiskAudit,
    REQUIRED_RISK_VARIANTS,
    RelevanceInvarianceAudit,
    audit_matched_task_risk,
    audit_relevance_invariance,
)
from uq_estimator.stage2l_semantic_runtime_v10 import VLMTaskConditioningV10
from uq_estimator.uq_relevance_tokenizer import (
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
    spatial_signal_summaries,
)


SCHEMA = "orion.stage2l_factorized_runtime.v11"


@dataclass(frozen=True)
class ContextualRelevancePassV11:
    """Outputs of the single U-independent first pass for one matched group."""

    baseline_vision: torch.Tensor
    relevance_logits: torch.Tensor


@dataclass(frozen=True)
class MatchedVLMConditioningV11:
    shared_relevance_logits: torch.Tensor
    conditioning_by_variant: Mapping[str, VLMTaskConditioningV10]
    latest_scalar_uq_by_variant: Mapping[str, torch.Tensor]
    no_u_ablation_vision: torch.Tensor
    relevance_invariance: RelevanceInvarianceAudit
    task_risk_audit: MatchedTaskRiskAudit
    relevance_forward_call_count: int
    u_enters_relevance_query: bool
    learned_structured_field_head_used: bool
    direct_control_output_enabled: bool
    trajectory_loss_enabled: bool
    schema: str = SCHEMA


def _validate_components(
    components_by_variant: Mapping[str, torch.Tensor],
) -> Tuple[str, ...]:
    required = tuple(REQUIRED_RISK_VARIANTS)
    if set(components_by_variant) != set(required):
        missing = sorted(set(required) - set(components_by_variant))
        extra = sorted(set(components_by_variant) - set(required))
        raise ValueError(
            "matched components differ from the v11 variants; missing=%s extra=%s"
            % (missing, extra)
        )
    reference = components_by_variant[required[0]]
    if not isinstance(reference, torch.Tensor) or reference.ndim != 6:
        raise ValueError("matched components must have shape [B,T,V,H,W,C]")
    for variant in required:
        value = components_by_variant[variant]
        if not isinstance(value, torch.Tensor) or value.shape != reference.shape:
            raise ValueError("matched component tensor shapes differ")
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError("matched components must be finite floating tensors")
        if bool((value < 0.0).any()) or bool((value > 1.0).any()):
            raise ValueError("matched components must lie in [0,1]")
    return required


def _condition_variant(
    *,
    uq_tokenizer: UQComponentTokenizer,
    risk_bridge: TaskRiskLanguageBridge,
    baseline_vision: torch.Tensor,
    components: torch.Tensor,
    shared_relevance_logits: torch.Tensor,
    semantic_decoder_kwargs: Mapping[str, Any],
) -> Tuple[VLMTaskConditioningV10, torch.Tensor]:
    components = components.to(
        device=baseline_vision.device,
        dtype=baseline_vision.dtype,
        non_blocking=True,
    )
    tokenized = uq_tokenizer(components)
    frozen_scalar_uq = tokenized.latest_scalar_uq.detach()
    frozen_relevance = shared_relevance_logits.detach()
    bridge = risk_bridge(frozen_scalar_uq, frozen_relevance)
    _, observation_global = spatial_signal_summaries(
        frozen_scalar_uq, epsilon=risk_bridge.epsilon
    )
    semantics = deterministic_task_semantics(
        frozen_scalar_uq,
        frozen_relevance,
        **dict(semantic_decoder_kwargs),
    )
    frozen_uq_tokens = tokenized.tokens.detach()
    if not (
        baseline_vision.shape[-1]
        == frozen_uq_tokens.shape[-1]
        == bridge.tokens.shape[-1]
    ):
        raise ValueError("ORION, U and K bridge token dimensions differ")
    vision_tokens = torch.cat(
        (baseline_vision, frozen_uq_tokens, bridge.tokens), dim=1
    )
    return (
        VLMTaskConditioningV10(
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
        ),
        frozen_scalar_uq,
    )


def build_matched_vlm_conditioning_v11(
    *,
    relevance_forward: Callable[[], ContextualRelevancePassV11],
    uq_tokenizer: UQComponentTokenizer,
    risk_bridge: TaskRiskLanguageBridge,
    components_by_variant: Mapping[str, torch.Tensor],
    semantic_decoder_kwargs: Mapping[str, Any] = None,
    risk_audit_kwargs: Mapping[str, Any] = None,
) -> MatchedVLMConditioningV11:
    """Run R once and build every matched U/K/language condition.

    ``relevance_forward`` receives no uncertainty argument.  It is called once
    by construction; all variants receive the same tensor object.  The
    resulting invariance audit is therefore an implementation invariant, while
    the K and answer-preference audits remain empirical release gates.
    """

    if not callable(relevance_forward):
        raise TypeError("relevance_forward must be callable")
    required = _validate_components(components_by_variant)
    if any(parameter.requires_grad for parameter in uq_tokenizer.parameters()):
        raise ValueError("Stage2-L v11 requires a frozen U tokenizer")

    contextual = relevance_forward()
    if not isinstance(contextual, ContextualRelevancePassV11):
        raise TypeError("relevance_forward returned an invalid v11 context")
    baseline = contextual.baseline_vision
    relevance = contextual.relevance_logits
    reference_components = components_by_variant[required[0]]
    if baseline.ndim != 3 or relevance.ndim != 4:
        raise ValueError("baseline vision or relevance rank differs from v11")
    if baseline.shape[0] != reference_components.shape[0]:
        raise ValueError("baseline and matched component batch sizes differ")
    expected_relevance_shape = (
        reference_components.shape[0],
        reference_components.shape[2],
        uq_tokenizer.grid_hw[0],
        uq_tokenizer.grid_hw[1],
    )
    if tuple(relevance.shape) != expected_relevance_shape:
        raise ValueError("shared relevance shape differs from U-token grid")
    if not baseline.is_floating_point() or not bool(torch.isfinite(baseline).all()):
        raise ValueError("baseline vision must be a finite floating tensor")
    if not relevance.is_floating_point() or not bool(torch.isfinite(relevance).all()):
        raise ValueError("shared relevance must be a finite floating tensor")

    conditioning: Dict[str, VLMTaskConditioningV10] = {}
    scalar_uq: Dict[str, torch.Tensor] = {}
    semantic_kwargs = dict(semantic_decoder_kwargs or {})
    for variant in required:
        conditioning[variant], scalar_uq[variant] = _condition_variant(
            uq_tokenizer=uq_tokenizer,
            risk_bridge=risk_bridge,
            baseline_vision=baseline,
            components=components_by_variant[variant],
            shared_relevance_logits=relevance,
            semantic_decoder_kwargs=semantic_kwargs,
        )

    relevance_by_variant = {variant: relevance for variant in required}
    relevance_audit = audit_relevance_invariance(relevance_by_variant)
    risk_audit = audit_matched_task_risk(
        relevance.detach(), scalar_uq, **dict(risk_audit_kwargs or {})
    )
    return MatchedVLMConditioningV11(
        shared_relevance_logits=relevance,
        conditioning_by_variant=conditioning,
        latest_scalar_uq_by_variant=scalar_uq,
        no_u_ablation_vision=baseline,
        relevance_invariance=relevance_audit,
        task_risk_audit=risk_audit,
        relevance_forward_call_count=1,
        u_enters_relevance_query=False,
        learned_structured_field_head_used=False,
        direct_control_output_enabled=False,
        trajectory_loss_enabled=False,
    )


__all__ = [
    "ContextualRelevancePassV11",
    "MatchedVLMConditioningV11",
    "SCHEMA",
    "build_matched_vlm_conditioning_v11",
]

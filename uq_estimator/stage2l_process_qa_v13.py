"""Structured process supervision for Stage2-L v13.

The language model is supervised on an auditable sequence rather than a
free-form rationale:

``observation evidence -> epistemic limit -> task binding -> decision``.

The dense task-relevance map and its hidden tokens are produced by the
U-independent first VLM pass.  Stage1 U tokens and those R hidden tokens enter
the second VLM pass directly; no learned K bridge is part of this contract.
This module assembles process-language targets, audits matched counterfactual
semantics, and selects the explicitly authorized trainable parameter scope.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple

import torch.nn as nn

from uq_estimator.stage2l_identifiability import REQUIRED_RISK_VARIANTS
from uq_estimator.stage2l_qa_contract_v5 import parse_semantic_fields


SCHEMA = "orion.stage2l_process_qa.v13"
PROCESS_FAMILIES = (
    "observation_semantics",
    "epistemic_limitation",
    "task_relevance",
    "driving_implication",
)
PROCESS_TAGS = (
    "OBSERVATION",
    "EPISTEMIC_LIMIT",
    "TASK_BINDING",
    "DECISION",
)
PROCESS_QUESTION = (
    "Assess the observation uncertainty through the required auditable "
    "process. First localize and characterize unreliable evidence. Then "
    "state what the signal cannot reveal. Next bind that evidence to the "
    "route and task-relevance map. Finally choose a high-level driving "
    "stance. Do not invent content hidden by unreliable evidence."
)
TRAINING_ARMS = ("lora", "partial_unfreeze")
_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class StructuredProcessChain:
    group_id: str
    variant: str
    question: str
    answer: str
    step_answers: Mapping[str, str]
    step_fields: Mapping[str, Mapping[str, str]]
    schema: str = SCHEMA


@dataclass(frozen=True)
class MatchedProcessAudit:
    group_id: str
    checks: Mapping[str, bool]
    variants: Tuple[str, ...]
    schema: str = SCHEMA

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


@dataclass(frozen=True)
class TrainableScope:
    arm: str
    parameter_groups: Mapping[str, Tuple[nn.Parameter, ...]]
    parameter_counts: Mapping[str, int]
    trainable_parameter_names: Tuple[str, ...]
    partial_layer_indices: Tuple[int, ...]
    schema: str = SCHEMA

    @property
    def total_trainable_parameters(self) -> int:
        return sum(self.parameter_counts.values())


def _row_identity(row: Mapping[str, object]) -> Tuple[str, str, str]:
    counterfactual = row.get("counterfactual")
    if not isinstance(counterfactual, Mapping):
        raise ValueError("process row lacks counterfactual identity")
    group_id = str(counterfactual.get("group_id", ""))
    variant = str(counterfactual.get("variant", ""))
    family = str(row.get("question_family", ""))
    if not group_id or not variant or not family:
        raise ValueError("process row identity is incomplete")
    return group_id, variant, family


def build_structured_process_chain(
    rows_by_family: Mapping[str, Mapping[str, object]],
) -> StructuredProcessChain:
    """Build one deterministic, step-supervised answer from V5 QA rows."""

    if tuple(rows_by_family) != PROCESS_FAMILIES:
        raise ValueError("process rows must follow the frozen family order")
    identities = [_row_identity(rows_by_family[key]) for key in PROCESS_FAMILIES]
    group_ids = {value[0] for value in identities}
    variants = {value[1] for value in identities}
    families = tuple(value[2] for value in identities)
    if len(group_ids) != 1 or len(variants) != 1:
        raise ValueError("process chain crosses a matched group or variant")
    if families != PROCESS_FAMILIES:
        raise ValueError("process row family identity differs from its key")

    step_answers: Dict[str, str] = {}
    step_fields: Dict[str, Mapping[str, str]] = {}
    rendered = []
    for tag, family in zip(PROCESS_TAGS, PROCESS_FAMILIES):
        conversation = rows_by_family[family].get("conversation")
        if not isinstance(conversation, Sequence) or len(conversation) != 2:
            raise ValueError("process row conversation is invalid")
        answer = str(conversation[1].get("value", ""))
        fields = parse_semantic_fields(answer, family)
        if not answer or not fields:
            raise ValueError("process step answer is empty")
        step_answers[family] = answer
        step_fields[family] = fields
        rendered.append("<%s> %s </%s>" % (tag, answer, tag))
    return StructuredProcessChain(
        group_id=next(iter(group_ids)),
        variant=next(iter(variants)),
        question=PROCESS_QUESTION,
        answer="\n".join(rendered),
        step_answers=step_answers,
        step_fields=step_fields,
    )


def build_structured_process_row(
    rows_by_family: Mapping[str, Mapping[str, object]],
) -> MutableMapping[str, object]:
    """Return a trainer-compatible row whose answer is the full process."""

    chain = build_structured_process_chain(rows_by_family)
    row = deepcopy(rows_by_family["task_relevance"])
    row["question_family"] = "structured_process_v13"
    row["conversation"] = [
        {"from": "human", "value": chain.question},
        {"from": "gpt", "value": chain.answer},
    ]
    target = dict(row.get("target", {}))
    target["process_schema"] = SCHEMA
    target["process_step_fields"] = {
        key: dict(value) for key, value in chain.step_fields.items()
    }
    row["target"] = target
    return row


def build_process_step_row(
    rows_by_family: Mapping[str, Mapping[str, object]], family: str
) -> MutableMapping[str, object]:
    """Return one explicitly supervised process step with the shared input."""

    if family not in PROCESS_FAMILIES:
        raise ValueError("unknown process family")
    chain = build_structured_process_chain(rows_by_family)
    row = deepcopy(rows_by_family[family])
    row["question_family"] = "structured_process_v13/%s" % family
    row["conversation"] = [
        {
            "from": "human",
            "value": (
                "%s Return only the %s process step."
                % (chain.question, family)
            ),
        },
        {"from": "gpt", "value": chain.step_answers[family]},
    ]
    return row


def audit_matched_process_chains(
    chains_by_variant: Mapping[str, StructuredProcessChain],
) -> MatchedProcessAudit:
    """Audit causal semantics that must hold before GPU optimization."""

    required = tuple(REQUIRED_RISK_VARIANTS)
    if set(chains_by_variant) != set(required):
        raise ValueError("matched process audit variants differ")
    if any(chains_by_variant[key].variant != key for key in required):
        raise ValueError("process chain variant key differs")
    group_ids = {chains_by_variant[key].group_id for key in required}
    if len(group_ids) != 1:
        raise ValueError("matched process audit crosses groups")

    fields = {
        variant: chains_by_variant[variant].step_fields
        for variant in required
    }
    zero_observation = fields["zero_uq"]["observation_semantics"]
    zero_limit = fields["zero_uq"]["epistemic_limitation"]
    zero_task = fields["zero_uq"]["task_relevance"]
    zero_decision = fields["zero_uq"]["driving_implication"]
    on_observation = fields["on_path_uq"]["observation_semantics"]
    off_observation = fields["off_path_uq"]["observation_semantics"]
    shuffled_observation = fields["view_shuffled_uq"]["observation_semantics"]
    on_task = fields["on_path_uq"]["task_relevance"]
    off_task = fields["off_path_uq"]["task_relevance"]
    on_decision = fields["on_path_uq"]["driving_implication"]
    off_decision = fields["off_path_uq"]["driving_implication"]

    nonzero_variants = ("on_path_uq", "off_path_uq", "view_shuffled_uq")
    checks = {
        "zero_u_explicit_absence": (
            zero_observation["uq_view"] == "none"
            and zero_observation["uq_region"] == "none"
            and zero_limit["evidence"] == "not_flagged"
            and zero_task["risk_level"] == "none"
        ),
        "zero_u_does_not_default_caution": zero_decision["stance"] == "maintain",
        "nonzero_u_respects_epistemic_limit": all(
            fields[variant]["epistemic_limitation"]["hidden_content"] == "unknown"
            and fields[variant]["epistemic_limitation"]["task_relevance"]
            == "separate"
            for variant in nonzero_variants
        ),
        "on_off_u_locations_are_distinct": (
            (on_observation["uq_view"], on_observation["uq_region"])
            != (off_observation["uq_view"], off_observation["uq_region"])
        ),
        "view_shuffle_changes_observation_binding": (
            (on_observation["uq_view"], on_observation["uq_region"])
            != (
                shuffled_observation["uq_view"],
                shuffled_observation["uq_region"],
            )
        ),
        "on_path_not_less_relevant_than_off_path": (
            {"not_applicable": 0, "low": 1, "high": 2}[on_task["relevance_level"]]
            >= {"not_applicable": 0, "low": 1, "high": 2}[
                off_task["relevance_level"]
            ]
        ),
        "off_path_does_not_force_conservatism": off_decision["stance"]
        == "maintain",
        "on_path_decision_is_not_less_conservative": (
            {"maintain": 0, "caution": 1, "prepare_to_yield": 2}[
                on_decision["stance"]
            ]
            >= {"maintain": 0, "caution": 1, "prepare_to_yield": 2}[
                off_decision["stance"]
            ]
        ),
    }
    return MatchedProcessAudit(
        group_id=next(iter(group_ids)), checks=checks, variants=required
    )


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = bool(enabled)


def detached_conditioning_gradient_anchor(tokens: torch.Tensor) -> torch.Tensor:
    """Anchor checkpointed ORION gradients without training token producers.

    ORION uses re-entrant gradient checkpointing.  When every conditioning
    input is frozen, PyTorch otherwise treats a checkpointed decoder segment
    as gradient-free even if that segment contains trainable LoRA weights.
    A detached leaf restores the decoder graph while guaranteeing that the
    gradient cannot reach Stage1, the U tokenizer, or the R-token producer.
    """

    if not tokens.is_floating_point():
        raise TypeError("ORION conditioning tokens must be floating point")
    return tokens.detach().requires_grad_(True)


def configure_trainable_scope(
    *,
    lm: nn.Module,
    relevance_queries: nn.Module,
    relevance_head: nn.Module,
    arm: str,
    partial_unfreeze_layers: int = 4,
) -> TrainableScope:
    """Select LoRA-only or LoRA-plus-last-layers capacity arms."""

    if arm not in TRAINING_ARMS:
        raise ValueError("unsupported Stage2-L v13 training arm")
    if partial_unfreeze_layers < 1:
        raise ValueError("partial layer count must be positive")
    for module in (lm, relevance_queries, relevance_head):
        _set_requires_grad(module, False)

    named_lm = tuple(lm.named_parameters())
    lora = tuple(parameter for name, parameter in named_lm if "lora_" in name)
    if not lora:
        raise ValueError("ORION language model exposes no LoRA parameters")
    for parameter in lora:
        parameter.requires_grad = True

    layer_indices = sorted(
        {
            int(match.group(1))
            for name, _ in named_lm
            for match in [_LAYER_PATTERN.search(name)]
            if match is not None
        }
    )
    selected_layers: Tuple[int, ...] = ()
    partial = ()
    if arm == "partial_unfreeze":
        if len(layer_indices) < partial_unfreeze_layers:
            raise ValueError("language model has too few decoder layers")
        selected_layers = tuple(layer_indices[-partial_unfreeze_layers:])
        selected = set(selected_layers)
        partial_values = []
        for name, parameter in named_lm:
            match = _LAYER_PATTERN.search(name)
            if match is not None and int(match.group(1)) in selected:
                parameter.requires_grad = True
                if "lora_" not in name:
                    partial_values.append(parameter)
        partial = tuple(partial_values)
        if not partial:
            raise ValueError("partial arm selected no base decoder parameters")

    for module in (relevance_queries, relevance_head):
        _set_requires_grad(module, True)
    groups = {
        "orion_lora": tuple(
            parameter for parameter in lora if parameter.requires_grad
        ),
        "partial_decoder": partial,
        "relevance_queries": tuple(relevance_queries.parameters()),
        "relevance_head": tuple(relevance_head.parameters()),
    }
    if arm == "lora" and groups["partial_decoder"]:
        raise RuntimeError("LoRA arm unexpectedly selected base decoder weights")
    trainable_names = tuple(
        name for name, parameter in named_lm if parameter.requires_grad
    )
    if any(
        "lora_" not in name
        and not (
            (match := _LAYER_PATTERN.search(name)) is not None
            and int(match.group(1)) in selected_layers
        )
        for name in trainable_names
    ):
        raise RuntimeError("unexpected ORION language parameter escaped scope")
    counts = {
        key: sum(parameter.numel() for parameter in values)
        for key, values in groups.items()
    }
    if any(counts[key] <= 0 for key in (
        "orion_lora",
        "relevance_queries",
        "relevance_head",
    )):
        raise RuntimeError("required v13 trainable group is empty")
    return TrainableScope(
        arm=arm,
        parameter_groups=groups,
        parameter_counts=counts,
        trainable_parameter_names=trainable_names,
        partial_layer_indices=selected_layers,
    )


__all__ = [
    "PROCESS_FAMILIES",
    "PROCESS_QUESTION",
    "PROCESS_TAGS",
    "SCHEMA",
    "TRAINING_ARMS",
    "MatchedProcessAudit",
    "StructuredProcessChain",
    "TrainableScope",
    "audit_matched_process_chains",
    "build_process_step_row",
    "build_structured_process_chain",
    "build_structured_process_row",
    "configure_trainable_scope",
]

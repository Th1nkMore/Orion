from __future__ import annotations

import torch
from torch import nn

from uq_estimator.stage2l_process_qa_v13 import (
    PROCESS_FAMILIES,
    PROCESS_TAGS,
    audit_matched_process_chains,
    build_process_step_row,
    build_structured_process_chain,
    build_structured_process_row,
    configure_trainable_scope,
    detached_conditioning_gradient_anchor,
)


def _answer(family: str, variant: str) -> str:
    locations = {
        "zero_uq": ("none", "none"),
        "on_path_uq": ("CAM_FRONT", "lower_center"),
        "off_path_uq": ("CAM_FRONT", "upper_left"),
        "view_shuffled_uq": ("CAM_BACK", "middle_right"),
    }
    view, region = locations[variant]
    if family == "observation_semantics":
        return (
            "Observation uncertainty: uq_level=%s; uq_view=%s; "
            "uq_region=%s; uq_trend=%s."
            % (
                "low" if variant == "zero_uq" else "high",
                view,
                region,
                "stable" if variant == "zero_uq" else "rising",
            )
        )
    if family == "epistemic_limitation":
        return (
            "Epistemic limitation: evidence=%s; evidence_view=%s; "
            "evidence_region=%s; hidden_content=%s; task_relevance=separate."
            % (
                "not_flagged" if variant == "zero_uq" else "unreliable",
                view,
                region,
                "not_applicable" if variant == "zero_uq" else "unknown",
            )
        )
    if family == "task_relevance":
        if variant == "zero_uq":
            return (
                "Task relevance map: relevance_level=not_applicable; "
                "risk_level=none; risk_view=none; risk_region=none."
            )
        if variant == "off_path_uq":
            return (
                "Task relevance map: relevance_level=low; risk_level=none; "
                "risk_view=none; risk_region=none."
            )
        return (
            "Task relevance map: relevance_level=high; risk_level=medium; "
            "risk_view=%s; risk_region=%s." % (view, region)
        )
    if family == "driving_implication":
        stance = "caution" if variant == "on_path_uq" else "maintain"
        return (
            "Uncertainty response: stance=%s; direct_control=no; "
            "response_basis=observation_uncertainty." % stance
        )
    raise AssertionError(family)


def test_detached_conditioning_gradient_anchor_stops_upstream_gradient() -> None:
    source = torch.randn(2, 3, requires_grad=True)
    produced = source.square()

    anchor = detached_conditioning_gradient_anchor(produced)
    anchor.sum().backward()

    assert anchor.is_leaf
    assert anchor.requires_grad
    assert anchor.grad is not None
    assert source.grad is None


def test_detached_conditioning_gradient_anchor_rejects_non_float() -> None:
    with torch.no_grad():
        tokens = torch.ones(2, 3, dtype=torch.long)
    try:
        detached_conditioning_gradient_anchor(tokens)
    except TypeError as error:
        assert "floating point" in str(error)
    else:
        raise AssertionError("integer conditioning tokens must be rejected")


def _rows(group: str, variant: str):
    return {
        family: {
            "event_id": "route147_step223",
            "question_family": family,
            "counterfactual": {"group_id": group, "variant": variant},
            "conversation": [
                {"from": "human", "value": "question"},
                {"from": "gpt", "value": _answer(family, variant)},
            ],
            "target": {},
        }
        for family in PROCESS_FAMILIES
    }


def test_process_chain_renders_every_auditable_step_in_order() -> None:
    chain = build_structured_process_chain(_rows("group", "on_path_uq"))
    offsets = [chain.answer.index("<%s>" % tag) for tag in PROCESS_TAGS]
    assert offsets == sorted(offsets)
    assert chain.step_fields["epistemic_limitation"]["hidden_content"] == "unknown"

    row = build_structured_process_row(_rows("group", "on_path_uq"))
    assert row["question_family"] == "structured_process_v13"
    assert row["target"]["process_schema"] == "orion.stage2l_process_qa.v13"
    step = build_process_step_row(
        _rows("group", "on_path_uq"), "task_relevance"
    )
    assert step["question_family"].endswith("task_relevance")


def test_matched_process_audit_protects_epistemic_and_zero_u_semantics() -> None:
    chains = {
        variant: build_structured_process_chain(_rows("group", variant))
        for variant in (
            "zero_uq",
            "on_path_uq",
            "off_path_uq",
            "view_shuffled_uq",
        )
    }
    audit = audit_matched_process_chains(chains)
    assert audit.passed
    assert all(audit.checks.values())


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(6)])
        self.lora_adapter = nn.Parameter(torch.zeros(4, 4))
        self.embed = nn.Embedding(8, 4)


def _small_module():
    return nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4))


def test_trainable_scope_distinguishes_lora_and_partial_capacity() -> None:
    lm = _TinyLM()
    modules = [_small_module() for _ in range(3)]
    lora = configure_trainable_scope(
        lm=lm,
        relevance_queries=modules[0],
        relevance_head=modules[1],
        arm="lora",
        partial_unfreeze_layers=2,
    )
    assert lora.partial_layer_indices == ()
    assert lora.parameter_counts["partial_decoder"] == 0
    assert "task_risk_bridge" not in lora.parameter_counts
    assert lm.lora_adapter.requires_grad
    assert not lm.layers[-1].weight.requires_grad

    partial = configure_trainable_scope(
        lm=lm,
        relevance_queries=modules[0],
        relevance_head=modules[1],
        arm="partial_unfreeze",
        partial_unfreeze_layers=2,
    )
    assert partial.partial_layer_indices == (4, 5)
    assert partial.parameter_counts["partial_decoder"] > 0
    assert "task_risk_bridge" not in partial.parameter_counts
    assert lm.layers[4].weight.requires_grad
    assert lm.layers[5].weight.requires_grad
    assert not lm.layers[3].weight.requires_grad
    assert not lm.embed.weight.requires_grad

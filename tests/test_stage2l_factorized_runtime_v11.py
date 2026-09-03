import pytest
import torch

from uq_estimator.stage2l_factorized_runtime_v11 import (
    ContextualRelevancePassV11,
    build_matched_vlm_conditioning_v11,
)
from uq_estimator.uq_relevance_tokenizer import (
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
)


def _modules():
    tokenizer = UQComponentTokenizer(
        component_dim=3,
        model_dim=16,
        grid_hw=(2, 2),
        max_views=3,
        hidden_dim=8,
    )
    tokenizer.requires_grad_(False)
    bridge = TaskRiskLanguageBridge(
        model_dim=16, hidden_dim=8, max_views=3
    )
    return tokenizer, bridge


def _components(batch=2):
    shape = (batch, 2, 3, 2, 2, 3)
    values = {
        "zero_uq": torch.zeros(shape),
        "on_path_uq": torch.zeros(shape),
        "off_path_uq": torch.zeros(shape),
        "view_shuffled_uq": torch.zeros(shape),
    }
    values["on_path_uq"][:, -1, 0, 1, 1] = 0.9
    values["off_path_uq"][:, -1, 0, 0, 0] = 0.9
    values["view_shuffled_uq"][:, -1, 2, 1, 1] = 0.9
    return values


def test_v11_runs_relevance_once_and_reuses_it_for_matched_u():
    tokenizer, bridge = _modules()
    components = _components()
    baseline = torch.randn(2, 5, 16)
    relevance = torch.full((2, 3, 2, 2), -9.0)
    relevance[:, 0, 1, 1] = 9.0
    calls = []

    def relevance_forward():
        calls.append(True)
        return ContextualRelevancePassV11(baseline, relevance)

    result = build_matched_vlm_conditioning_v11(
        relevance_forward=relevance_forward,
        uq_tokenizer=tokenizer,
        risk_bridge=bridge,
        components_by_variant=components,
        risk_audit_kwargs={
            "required_on_over_off_margin": 0.5,
            "maximum_off_path_risk": 0.01,
            "minimum_fraction": 1.0,
        },
        semantic_decoder_kwargs={"camera_order": ("A", "B", "C")},
    )
    assert len(calls) == 1
    assert result.relevance_forward_call_count == 1
    assert result.shared_relevance_logits is relevance
    assert result.no_u_ablation_vision is baseline
    assert result.u_enters_relevance_query is False
    assert result.relevance_invariance.all_invariant is True
    assert all(result.task_risk_audit.aggregate_gates.values())
    assert torch.equal(
        result.conditioning_by_variant["zero_uq"].task_risk,
        torch.zeros_like(relevance),
    )
    for value in result.conditioning_by_variant.values():
        assert value.vision_tokens.shape == (2, 5 + 12 + 4, 16)


def test_v11_language_gradient_stops_at_u_and_r():
    tokenizer, bridge = _modules()
    components = {
        key: value.requires_grad_(True) for key, value in _components(batch=1).items()
    }
    baseline = torch.randn(1, 5, 16)
    relevance = torch.randn(1, 3, 2, 2, requires_grad=True)
    result = build_matched_vlm_conditioning_v11(
        relevance_forward=lambda: ContextualRelevancePassV11(
            baseline, relevance
        ),
        uq_tokenizer=tokenizer,
        risk_bridge=bridge,
        components_by_variant=components,
        semantic_decoder_kwargs={"camera_order": ("A", "B", "C")},
    )
    loss = sum(
        value.vision_tokens[:, -4:].square().mean()
        for value in result.conditioning_by_variant.values()
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in bridge.parameters())
    assert all(parameter.grad is None for parameter in tokenizer.parameters())
    assert relevance.grad is None
    assert all(value.grad is None for value in components.values())


def test_v11_rejects_variant_drift_and_trainable_u_tokenizer():
    tokenizer, bridge = _modules()
    components = _components(batch=1)
    components.pop("view_shuffled_uq")
    context = lambda: ContextualRelevancePassV11(
        torch.randn(1, 5, 16), torch.zeros(1, 3, 2, 2)
    )
    with pytest.raises(ValueError, match="matched components"):
        build_matched_vlm_conditioning_v11(
            relevance_forward=context,
            uq_tokenizer=tokenizer,
            risk_bridge=bridge,
            components_by_variant=components,
        )

    components = _components(batch=1)
    tokenizer.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen U tokenizer"):
        build_matched_vlm_conditioning_v11(
            relevance_forward=context,
            uq_tokenizer=tokenizer,
            risk_bridge=bridge,
            components_by_variant=components,
        )

import inspect

import pytest


torch = pytest.importorskip("torch")

from uq_estimator.stage2l_semantic_runtime_v4 import (
    build_vlm_task_field_conditioning,
)
from uq_estimator.stage2l_structured_field_head import (
    VLMTaskSemanticFieldHead,
)
from uq_estimator.uq_relevance_tokenizer import (
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
)


def _modules():
    return (
        UQComponentTokenizer(
            component_dim=3,
            model_dim=12,
            grid_hw=(2, 2),
            max_views=2,
            hidden_dim=8,
        ),
        TaskRiskLanguageBridge(
            model_dim=12, hidden_dim=8, max_views=2
        ),
        VLMTaskSemanticFieldHead(model_dim=12, hidden_dim=8),
    )


def _forward(components=None, relevance=None):
    uq, bridge, fields = _modules()
    components = (
        torch.rand(1, 2, 2, 4, 4, 3)
        if components is None
        else components
    )
    relevance = (
        torch.randn(1, 2, 2, 2, requires_grad=True)
        if relevance is None
        else relevance
    )
    output = build_vlm_task_field_conditioning(
        uq_tokenizer=uq,
        risk_bridge=bridge,
        field_head=fields,
        baseline_vision=torch.randn(1, 3, 12),
        components=components,
        relevance_logits=relevance,
    )
    return uq, bridge, fields, relevance, output


def test_auxiliary_language_cannot_redefine_r_bridge_or_classifiers():
    uq, bridge, fields, relevance, output = _forward()
    output.vision_tokens.square().mean().backward()
    assert relevance.grad is None
    assert all(parameter.grad is None for parameter in bridge.parameters())
    assert all(
        parameter.grad is None
        for classifier in fields.classifiers.values()
        for parameter in classifier.parameters()
    )
    assert all(
        parameter.grad is None
        for module in (
            fields.observation_magnitude_projection,
            fields.observation_location_projection,
            fields.task_risk_magnitude_projection,
            fields.task_risk_location_projection,
            fields.context_projection,
        )
        for parameter in module.parameters()
    )
    assert any(parameter.grad is not None for parameter in uq.parameters())
    assert any(
        parameter.grad is not None
        for embedding in fields.field_embeddings.values()
        for parameter in embedding.parameters()
    )


def test_explicit_task_field_loss_trains_bridge_and_head_but_not_dense_r():
    _, bridge, fields, relevance, output = _forward()
    loss = sum(value.square().mean() for value in output.field_logits.values())
    loss.backward()
    assert relevance.grad is None
    assert any(parameter.grad is not None for parameter in bridge.parameters())
    assert any(
        parameter.grad is not None
        for classifier in fields.classifiers.values()
        for parameter in classifier.parameters()
    )
    assert output.relevance_detached_for_task_fields is True


def test_zero_u_has_identical_task_fields_for_different_relevance_maps():
    torch.manual_seed(7)
    uq, bridge, fields = _modules()
    components = torch.zeros(1, 2, 2, 4, 4, 3)
    baseline = torch.randn(1, 3, 12)
    low = build_vlm_task_field_conditioning(
        uq_tokenizer=uq,
        risk_bridge=bridge,
        field_head=fields,
        baseline_vision=baseline,
        components=components,
        relevance_logits=torch.full((1, 2, 2, 2), -20.0),
    )
    high = build_vlm_task_field_conditioning(
        uq_tokenizer=uq,
        risk_bridge=bridge,
        field_head=fields,
        baseline_vision=baseline,
        components=components,
        relevance_logits=torch.full((1, 2, 2, 2), 20.0),
    )
    assert torch.equal(low.task_risk, torch.zeros_like(low.task_risk))
    assert torch.equal(high.task_risk, torch.zeros_like(high.task_risk))
    for field in low.field_logits:
        assert torch.allclose(low.field_logits[field], high.field_logits[field])


def test_runtime_has_no_route_input_or_direct_control_output():
    parameters = inspect.signature(
        build_vlm_task_field_conditioning
    ).parameters
    assert "route" not in parameters
    _, _, _, _, output = _forward()
    assert output.direct_control_output_enabled is False
    assert output.trajectory_loss_enabled is False

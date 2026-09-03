import pytest


torch = pytest.importorskip("torch")

from uq_estimator.stage2l_semantic_bottleneck_v3 import (
    GradientRoutedPlanningStanceBottleneck,
)
from uq_estimator.stage2l_semantic_runtime_v3 import (
    build_gradient_routed_conditioning,
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
        GradientRoutedPlanningStanceBottleneck(
            model_dim=12, hidden_dim=8
        ),
    )


def test_language_path_does_not_own_relevance_or_stance_classifier():
    uq, bridge, stance = _modules()
    relevance = torch.randn(1, 2, 2, 2, requires_grad=True)
    output = build_gradient_routed_conditioning(
        uq_tokenizer=uq,
        risk_bridge=bridge,
        stance_bottleneck=stance,
        baseline_vision=torch.randn(1, 3, 12),
        components=torch.rand(1, 2, 2, 4, 4, 3),
        relevance_logits=relevance,
    )
    output.vision_tokens.square().mean().backward()
    assert relevance.grad is None
    assert all(parameter.grad is None for parameter in stance.classifier.parameters())
    assert stance.stance_embedding.weight.grad is not None
    assert any(parameter.grad is not None for parameter in bridge.parameters())
    assert any(parameter.grad is not None for parameter in uq.parameters())


def test_explicit_stance_loss_still_trains_classifier():
    uq, bridge, stance = _modules()
    output = build_gradient_routed_conditioning(
        uq_tokenizer=uq,
        risk_bridge=bridge,
        stance_bottleneck=stance,
        baseline_vision=torch.randn(1, 3, 12),
        components=torch.rand(1, 2, 2, 4, 4, 3),
        relevance_logits=torch.randn(1, 2, 2, 2, requires_grad=True),
    )
    output.stance_logits.square().mean().backward()
    assert any(parameter.grad is not None for parameter in stance.classifier.parameters())


def test_routing_can_be_disabled_only_explicitly():
    uq, bridge, stance = _modules()
    relevance = torch.randn(1, 2, 2, 2, requires_grad=True)
    output = build_gradient_routed_conditioning(
        uq_tokenizer=uq,
        risk_bridge=bridge,
        stance_bottleneck=stance,
        baseline_vision=torch.randn(1, 3, 12),
        components=torch.rand(1, 2, 2, 4, 4, 3),
        relevance_logits=relevance,
        detach_relevance_for_language=False,
        detach_stance_probabilities_for_language=False,
    )
    output.vision_tokens.square().mean().backward()
    assert relevance.grad is not None
    assert any(parameter.grad is not None for parameter in stance.classifier.parameters())

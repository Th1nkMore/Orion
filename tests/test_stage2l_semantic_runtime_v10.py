import pytest
import torch

from uq_estimator.stage2l_semantic_runtime_v10 import (
    build_vlm_task_conditioning_v10,
)
from uq_estimator.uq_relevance_tokenizer import (
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
)


def _modules():
    tokenizer = UQComponentTokenizer(
        model_dim=16, hidden_dim=8, grid_hw=(2, 2), max_views=6
    )
    bridge = TaskRiskLanguageBridge(
        model_dim=16, hidden_dim=8, max_views=6
    )
    return tokenizer, bridge


def test_runtime_rejects_task_trainable_u_tokenizer():
    tokenizer, bridge = _modules()
    with pytest.raises(ValueError, match="frozen task-agnostic U tokenizer"):
        build_vlm_task_conditioning_v10(
            uq_tokenizer=tokenizer,
            risk_bridge=bridge,
            baseline_vision=torch.zeros(1, 3, 16),
            components=torch.zeros(1, 2, 6, 4, 4, 3),
            relevance_logits=torch.zeros(1, 6, 2, 2),
        )


def test_qa_gradient_reaches_bridge_but_not_u_tokenizer_or_r():
    tokenizer, bridge = _modules()
    tokenizer.requires_grad_(False)
    baseline = torch.randn(1, 3, 16, requires_grad=True)
    components = torch.rand(1, 2, 6, 4, 4, 3)
    relevance_logits = torch.randn(1, 6, 2, 2, requires_grad=True)
    output = build_vlm_task_conditioning_v10(
        uq_tokenizer=tokenizer,
        risk_bridge=bridge,
        baseline_vision=baseline,
        components=components,
        relevance_logits=relevance_logits,
    )
    output.vision_tokens.sum().backward()
    assert all(parameter.grad is None for parameter in tokenizer.parameters())
    assert relevance_logits.grad is None
    assert any(parameter.grad is not None for parameter in bridge.parameters())
    assert baseline.grad is not None
    assert output.uq_tokenizer_frozen is True
    assert output.relevance_detached_for_language is True
    assert output.learned_structured_field_head_used is False
    assert output.semantic_classifier_token_count == 0


def test_runtime_token_span_contains_only_orion_u_and_k_bridge_tokens():
    tokenizer, bridge = _modules()
    tokenizer.requires_grad_(False)
    output = build_vlm_task_conditioning_v10(
        uq_tokenizer=tokenizer,
        risk_bridge=bridge,
        baseline_vision=torch.zeros(2, 3, 16),
        components=torch.zeros(2, 2, 6, 4, 4, 3),
        relevance_logits=torch.zeros(2, 6, 2, 2),
    )
    # 3 ORION tokens + (6*2*2) frozen U tokens + (6+1) K bridge tokens.
    assert output.vision_tokens.shape == (2, 34, 16)
    assert output.task_risk.shape == (2, 6, 2, 2)
    assert output.deterministic_semantics.structured_fields[0][
        "relevance_level"
    ] == "not_applicable"

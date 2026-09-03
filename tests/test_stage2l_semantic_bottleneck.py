import pytest
import torch

from uq_estimator.stage2l_semantic_bottleneck import (
    PLANNING_STANCES,
    PlanningStanceSemanticBottleneck,
    encode_planning_stances,
    planning_stance_index,
    planning_stance_loss,
)
from uq_estimator.stage2l_semantic_runtime import (
    build_structured_semantic_conditioning,
)
from uq_estimator.uq_relevance_tokenizer import (
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
)


def test_semantic_bottleneck_emits_one_soft_predicted_token():
    module = PlanningStanceSemanticBottleneck(model_dim=16, hidden_dim=8)
    bridge_tokens = torch.randn(2, 7, 16, requires_grad=True)
    output = module(bridge_tokens)
    assert output.token.shape == (2, 1, 16)
    assert output.logits.shape == (2, 3)
    assert output.probabilities.shape == (2, 3)
    assert output.predicted_indices.shape == (2,)
    assert torch.allclose(
        output.probabilities.sum(dim=-1), torch.ones(2), atol=1e-6
    )
    weights = torch.arange(16, dtype=output.token.dtype)[None, None]
    (output.token * weights).sum().backward()
    assert bridge_tokens.grad is not None
    assert torch.isfinite(bridge_tokens.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_stance_labels_are_loss_only_and_never_forward_inputs():
    module = PlanningStanceSemanticBottleneck(model_dim=12, hidden_dim=6)
    bridge_tokens = torch.randn(1, 7, 12)
    before = module(bridge_tokens)
    maintain = encode_planning_stances(["maintain"])
    prepare = encode_planning_stances(["prepare_to_yield"])
    maintain_loss = planning_stance_loss(before.logits, maintain)
    prepare_loss = planning_stance_loss(before.logits, prepare)
    after = module(bridge_tokens)
    assert torch.equal(before.token, after.token)
    assert torch.equal(before.logits, after.logits)
    assert maintain_loss.item() != prepare_loss.item()


def test_structured_runtime_appends_one_token_and_backpropagates_to_r():
    uq = UQComponentTokenizer(model_dim=16, hidden_dim=8, grid_hw=(2, 2))
    bridge = TaskRiskLanguageBridge(model_dim=16, hidden_dim=8)
    stance = PlanningStanceSemanticBottleneck(model_dim=16, hidden_dim=8)
    baseline = torch.randn(1, 7, 16)
    components = torch.rand(1, 3, 6, 4, 4, 3)
    relevance = torch.randn(1, 6, 2, 2, requires_grad=True)
    result = build_structured_semantic_conditioning(
        uq_tokenizer=uq,
        risk_bridge=bridge,
        stance_bottleneck=stance,
        baseline_vision=baseline,
        components=components,
        relevance_logits=relevance,
    )
    assert result.vision_tokens.shape == (1, 7 + 24 + 7 + 1, 16)
    assert result.stance_logits.shape == (1, len(PLANNING_STANCES))
    assert result.stance_probabilities.shape == (1, len(PLANNING_STANCES))
    target = encode_planning_stances(["caution"])
    loss = planning_stance_loss(result.stance_logits, target)
    loss.backward()
    assert relevance.grad is not None
    assert torch.isfinite(relevance.grad).all()
    assert relevance.grad.abs().sum() > 0


def test_zero_u_semantics_remain_relevance_invariant():
    uq = UQComponentTokenizer(model_dim=16, hidden_dim=8, grid_hw=(2, 2))
    bridge = TaskRiskLanguageBridge(model_dim=16, hidden_dim=8)
    stance = PlanningStanceSemanticBottleneck(model_dim=16, hidden_dim=8)
    baseline = torch.randn(1, 7, 16)
    components = torch.zeros(1, 3, 6, 4, 4, 3)
    low = build_structured_semantic_conditioning(
        uq_tokenizer=uq,
        risk_bridge=bridge,
        stance_bottleneck=stance,
        baseline_vision=baseline,
        components=components,
        relevance_logits=torch.full((1, 6, 2, 2), -9.0),
    )
    high = build_structured_semantic_conditioning(
        uq_tokenizer=uq,
        risk_bridge=bridge,
        stance_bottleneck=stance,
        baseline_vision=baseline,
        components=components,
        relevance_logits=torch.full((1, 6, 2, 2), 9.0),
    )
    assert torch.equal(low.task_risk, high.task_risk)
    assert torch.equal(low.stance_logits, high.stance_logits)
    assert torch.equal(low.stance_probabilities, high.stance_probabilities)
    assert torch.equal(low.vision_tokens, high.vision_tokens)


def test_stance_contract_rejects_unknown_or_malformed_targets():
    assert planning_stance_index("maintain") == 0
    assert planning_stance_index("prepare_to_yield") == 2
    with pytest.raises(ValueError, match="unsupported"):
        planning_stance_index("prepare_right")
    with pytest.raises(ValueError, match=r"shape \[B\]"):
        planning_stance_loss(torch.randn(1, 3), torch.zeros(1, 1, dtype=torch.long))

import pytest
import torch

from uq_estimator.uq_relevance_tokenizer import (
    SpatialTaskRelevanceQueryTokenizer,
    TaskRelevanceMapHead,
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
    fixed_task_risk,
    matched_task_risk_ranking_loss,
    task_relevance_loss,
)


def test_tokenizer_preserves_view_grid_and_temporal_delta():
    model = UQComponentTokenizer(
        component_dim=3, model_dim=32, hidden_dim=16, grid_hw=(4, 5)
    )
    components = torch.zeros(2, 3, 6, 8, 10, 3)
    components[:, -1, 0, 2:4, 3:5] = 0.9
    output = model(components)
    assert output.tokens.shape == (2, 6 * 4 * 5, 32)
    assert output.token_grid.shape == (2, 6, 4, 5, 32)
    assert output.temporal_summary.shape == (2, 6, 4, 5, 9)
    assert output.latest_scalar_uq.shape == (2, 6, 4, 5)
    assert output.temporal_summary[..., 6:].max() > 0


def test_relevance_head_and_fixed_product_are_differentiable():
    tokenizer = UQComponentTokenizer(
        component_dim=3, model_dim=16, hidden_dim=8, grid_hw=(2, 2)
    )
    head = TaskRelevanceMapHead(model_dim=16, hidden_dim=8)
    components = torch.rand(2, 2, 6, 4, 4, 3)
    tokenized = tokenizer(components)
    logits = head(tokenized.token_grid)
    target = torch.rand_like(logits)
    risk = fixed_task_risk(tokenized.latest_scalar_uq, logits)
    loss = task_relevance_loss(logits, target) + risk.mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in tokenizer.parameters())
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_matched_ranking_rewards_onpath_over_offpath():
    on = torch.tensor([[[[0.8]]], [[[0.3]]]])
    off = torch.tensor([[[[0.1]]], [[[0.4]]]])
    loss = matched_task_risk_ranking_loss(on, off, margin=0.2)
    assert loss.item() == pytest.approx(0.15)


def test_tokenizer_rejects_out_of_range_components():
    model = UQComponentTokenizer(model_dim=8, hidden_dim=4, grid_hw=(2, 2))
    value = torch.zeros(1, 1, 6, 2, 2, 3)
    value[..., 0] = 1.1
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        model(value)


def test_task_risk_bridge_is_compact_and_differentiable():
    bridge = TaskRiskLanguageBridge(model_dim=24, hidden_dim=12)
    uncertainty = torch.rand(2, 6, 4, 5)
    relevance_logits = torch.randn(2, 6, 4, 5, requires_grad=True)
    output = bridge(uncertainty, relevance_logits)
    assert output.tokens.shape == (2, 7, 24)
    assert output.task_risk.shape == uncertainty.shape
    assert output.per_view_features.shape == (2, 6, 6)
    assert output.global_features.shape == (2, 6)
    loss = output.tokens.square().mean() + output.task_risk.mean()
    loss.backward()
    assert torch.isfinite(relevance_logits.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in bridge.parameters()
    )


def test_zero_uq_bridge_is_invariant_to_relevance():
    bridge = TaskRiskLanguageBridge(model_dim=16, hidden_dim=8)
    uncertainty = torch.zeros(1, 6, 3, 3)
    low = bridge(uncertainty, torch.full_like(uncertainty, -8.0))
    high = bridge(uncertainty, torch.full_like(uncertainty, 8.0))
    assert torch.equal(low.task_risk, high.task_risk)
    assert torch.equal(low.per_view_features, high.per_view_features)
    assert torch.equal(low.global_features, high.global_features)
    assert torch.equal(low.tokens, high.tokens)
    low.tokens.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in bridge.parameters()
    )


def test_bridge_separates_matched_on_and_off_path_risk():
    bridge = TaskRiskLanguageBridge(model_dim=16, hidden_dim=8)
    relevance_logits = torch.full((1, 6, 5, 5), -8.0)
    relevance_logits[:, 0, 2, 2] = 8.0
    on_path = torch.zeros_like(relevance_logits)
    off_path = torch.zeros_like(relevance_logits)
    on_path[:, 0, 2, 2] = 0.9
    off_path[:, 0, 0, 0] = 0.9
    on = bridge(on_path, relevance_logits)
    off = bridge(off_path, relevance_logits)
    assert on.global_features[0, 0] > 0.89
    assert off.global_features[0, 0] < 0.001
    assert on.global_features[0, 0] > off.global_features[0, 0]


def test_task_relevance_queries_are_spatial_and_u_independent():
    queries = SpatialTaskRelevanceQueryTokenizer(
        model_dim=20, hidden_dim=10, grid_hw=(3, 4)
    )
    output = queries(batch_size=2, views=6)
    assert output.shape == (2, 6 * 3 * 4, 20)
    assert torch.equal(output[0], output[1])
    assert not torch.equal(output[0, 0], output[0, 1])
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in queries.parameters()
    )

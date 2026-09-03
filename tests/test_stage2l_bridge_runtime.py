import pytest
import torch

from uq_estimator.stage2l_bridge_runtime import (
    build_two_pass_language_conditioning,
    extract_relevance_query_grid,
)
from uq_estimator.uq_relevance_tokenizer import (
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
)


def test_extracts_expanded_relevance_query_grid():
    input_ids = torch.tensor([[10, -200, 11, 12]])
    # One text token, 7 visual tokens, 24 query tokens, then two text tokens.
    hidden = torch.arange(34 * 8, dtype=torch.float32).reshape(1, 34, 8)
    grid = extract_relevance_query_grid(
        hidden,
        input_ids,
        image_token_index=-200,
        visual_token_count=7,
        views=6,
        grid_h=2,
        grid_w=2,
    )
    assert grid.shape == (1, 6, 2, 2, 8)
    assert torch.equal(grid.reshape(1, 24, 8), hidden[:, 8:32])
    with pytest.raises(ValueError, match="exactly one"):
        extract_relevance_query_grid(
            hidden, torch.tensor([[-200, -200]]), image_token_index=-200,
            visual_token_count=7, views=6, grid_h=2, grid_w=2,
        )


def test_builds_orion_uq_and_seven_bridge_token_span():
    uq = UQComponentTokenizer(
        model_dim=16, hidden_dim=8, grid_hw=(2, 2)
    )
    bridge = TaskRiskLanguageBridge(model_dim=16, hidden_dim=8)
    baseline = torch.randn(1, 7, 16)
    components = torch.rand(1, 3, 6, 4, 4, 3)
    relevance = torch.randn(1, 6, 2, 2)
    result = build_two_pass_language_conditioning(
        uq_tokenizer=uq,
        risk_bridge=bridge,
        baseline_vision=baseline,
        components=components,
        relevance_logits=relevance,
    )
    assert result.vision_tokens.shape == (1, 7 + 24 + 7, 16)
    assert result.task_risk.shape == (1, 6, 2, 2)
    assert result.bridge_global_features.shape == (1, 6)


def test_zero_u_conditioning_is_relevance_invariant():
    uq = UQComponentTokenizer(
        model_dim=16, hidden_dim=8, grid_hw=(2, 2)
    )
    bridge = TaskRiskLanguageBridge(model_dim=16, hidden_dim=8)
    baseline = torch.randn(1, 7, 16)
    components = torch.zeros(1, 3, 6, 4, 4, 3)
    low = build_two_pass_language_conditioning(
        uq_tokenizer=uq, risk_bridge=bridge, baseline_vision=baseline,
        components=components, relevance_logits=torch.full((1, 6, 2, 2), -9.0),
    )
    high = build_two_pass_language_conditioning(
        uq_tokenizer=uq, risk_bridge=bridge, baseline_vision=baseline,
        components=components, relevance_logits=torch.full((1, 6, 2, 2), 9.0),
    )
    assert torch.equal(low.task_risk, high.task_risk)
    assert torch.equal(low.vision_tokens, high.vision_tokens)

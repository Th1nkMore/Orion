import torch

from uq_estimator.stage2l_semantic_bottleneck import (
    PLANNING_STANCES,
    encode_planning_stances,
    planning_stance_loss,
)
from uq_estimator.stage2l_semantic_bottleneck_v2 import (
    MagnitudePreservingPlanningStanceBottleneck,
)
from uq_estimator.stage2l_semantic_runtime_v2 import (
    build_magnitude_structured_conditioning,
)
from uq_estimator.uq_relevance_tokenizer import (
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
)


def test_v2_bottleneck_emits_soft_token_and_preserves_raw_magnitude_gradient():
    torch.manual_seed(7)
    module = MagnitudePreservingPlanningStanceBottleneck(
        model_dim=16, hidden_dim=8
    )
    bridge = torch.randn(2, 7, 16, requires_grad=True)
    raw = torch.tensor(
        [
            [0.05, 0.01, 0.02, -0.2, 0.1, 0.0],
            [0.80, 0.30, 0.45, 0.3, -0.1, 0.2],
        ],
        requires_grad=True,
    )
    output = module(bridge, raw)
    assert output.token.shape == (2, 1, 16)
    assert output.logits.shape == (2, len(PLANNING_STANCES))
    assert output.probabilities.shape == (2, len(PLANNING_STANCES))
    assert torch.allclose(
        output.probabilities.sum(dim=-1), torch.ones(2), atol=1e-6
    )
    assert output.magnitude_features[1, 0] > output.magnitude_features[0, 0]
    planning_stance_loss(
        output.logits,
        encode_planning_stances(["maintain", "prepare_to_yield"]),
    ).backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert raw.grad[:, :3].abs().sum() > 0
    assert bridge.grad is not None and bridge.grad.abs().sum() > 0


def test_v2_forward_has_no_ground_truth_stance_input():
    module = MagnitudePreservingPlanningStanceBottleneck(
        model_dim=12, hidden_dim=6
    )
    bridge = torch.randn(1, 7, 12)
    raw = torch.tensor([[0.4, 0.1, 0.2, 0.0, 0.0, 0.0]])
    before = module(bridge, raw)
    maintain_loss = planning_stance_loss(
        before.logits, encode_planning_stances(["maintain"])
    )
    prepare_loss = planning_stance_loss(
        before.logits, encode_planning_stances(["prepare_to_yield"])
    )
    after = module(bridge, raw)
    assert torch.equal(before.token, after.token)
    assert torch.equal(before.logits, after.logits)
    assert maintain_loss.item() != prepare_loss.item()


def test_v2_runtime_appends_token_and_backpropagates_through_raw_k_path():
    uq = UQComponentTokenizer(model_dim=16, hidden_dim=8, grid_hw=(2, 2))
    bridge = TaskRiskLanguageBridge(model_dim=16, hidden_dim=8)
    stance = MagnitudePreservingPlanningStanceBottleneck(
        model_dim=16, hidden_dim=8
    )
    baseline = torch.randn(1, 7, 16)
    components = torch.rand(1, 3, 6, 4, 4, 3)
    relevance = torch.randn(1, 6, 2, 2, requires_grad=True)
    result = build_magnitude_structured_conditioning(
        uq_tokenizer=uq,
        risk_bridge=bridge,
        stance_bottleneck=stance,
        baseline_vision=baseline,
        components=components,
        relevance_logits=relevance,
    )
    assert result.vision_tokens.shape == (1, 7 + 24 + 7 + 1, 16)
    assert result.raw_global_features.shape == (1, 6)
    assert result.magnitude_features.shape == (1, 3)
    planning_stance_loss(
        result.stance_logits,
        encode_planning_stances(["prepare_to_yield"]),
    ).backward()
    assert relevance.grad is not None
    assert torch.isfinite(relevance.grad).all()
    assert relevance.grad.abs().sum() > 0


def test_v2_zero_u_semantics_are_relevance_invariant():
    uq = UQComponentTokenizer(model_dim=16, hidden_dim=8, grid_hw=(2, 2))
    bridge = TaskRiskLanguageBridge(model_dim=16, hidden_dim=8)
    stance = MagnitudePreservingPlanningStanceBottleneck(
        model_dim=16, hidden_dim=8
    )
    baseline = torch.randn(1, 7, 16)
    components = torch.zeros(1, 3, 6, 4, 4, 3)

    def run(value):
        return build_magnitude_structured_conditioning(
            uq_tokenizer=uq,
            risk_bridge=bridge,
            stance_bottleneck=stance,
            baseline_vision=baseline,
            components=components,
            relevance_logits=torch.full((1, 6, 2, 2), value),
        )

    low, high = run(-9.0), run(9.0)
    assert torch.equal(low.task_risk, high.task_risk)
    assert torch.equal(low.raw_global_features, high.raw_global_features)
    assert torch.equal(low.stance_logits, high.stance_logits)
    assert torch.equal(low.vision_tokens, high.vision_tokens)

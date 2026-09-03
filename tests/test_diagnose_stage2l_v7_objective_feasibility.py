import torch

from scripts.diagnose_stage2l_v7_objective_feasibility import (
    RawKStanceHead,
    _raw_k_features,
)


def test_raw_k_features_preserve_zero_and_magnitude_order():
    zero = torch.zeros(1, 2, 3, 3)
    low = zero.clone()
    high = zero.clone()
    low[0, 0, 1, 1] = 0.2
    high[0, 0, 1, 1] = 0.8
    features = _raw_k_features(torch.cat((zero, low, high), dim=0))
    assert features.shape == (3, 6)
    assert features[0, 0] == 0.0
    assert features[0, 1] == 0.0
    assert features[2, 0] > features[1, 0] > features[0, 0]
    assert features[2, 1] > features[1, 1] > features[0, 1]


def test_raw_k_stance_head_has_expected_shape_and_gradient():
    head = RawKStanceHead(hidden_dim=8)
    features = torch.randn(5, 6, requires_grad=True)
    logits = head(features)
    assert logits.shape == (5, 3)
    logits.square().mean().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0


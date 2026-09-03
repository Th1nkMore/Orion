import pytest
import torch

from uq_estimator.vision_adapter import UQVisionAdapter


def test_vision_adapter_is_identity_at_initialization():
    adapter = UQVisionAdapter(llm_dim=8, bottleneck_dim=4)
    tokens = torch.randn(2, 5, 8)
    score = torch.tensor([[0.2], [0.9]])
    torch.testing.assert_close(adapter(tokens, score), tokens)


def test_vision_adapter_zero_score_is_identity_after_update():
    adapter = UQVisionAdapter(llm_dim=8, bottleneck_dim=4)
    with torch.no_grad():
        adapter.up.weight.fill_(0.1)
    tokens = torch.randn(2, 5, 8)
    score = torch.zeros(2, 1)
    torch.testing.assert_close(adapter(tokens, score), tokens)


def test_vision_adapter_validates_shape():
    adapter = UQVisionAdapter(llm_dim=8, bottleneck_dim=4)
    with pytest.raises(ValueError):
        adapter(torch.randn(2, 8), torch.ones(2, 1))

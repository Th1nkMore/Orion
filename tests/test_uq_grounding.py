import pytest
import torch

from uq_estimator.grounding import UQGroundingHead, grounding_loss


def test_grounding_head_output_contract():
    head = UQGroundingHead(input_dim=32)
    predicted = head(torch.randn(4, 32))
    assert predicted.shape == (4, 1)
    assert torch.all((predicted >= 0) & (predicted <= 1))


def test_grounding_loss_detaches_target():
    predicted = torch.rand(3, 1, requires_grad=True)
    target = torch.rand(3, 1, requires_grad=True)
    loss = grounding_loss(predicted, target)
    loss.backward()
    assert predicted.grad is not None
    assert target.grad is None


def test_grounding_rejects_bad_shape():
    head = UQGroundingHead(input_dim=8)
    with pytest.raises(ValueError):
        head(torch.randn(2, 1, 8))

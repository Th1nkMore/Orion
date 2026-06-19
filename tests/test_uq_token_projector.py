import pytest
import torch

from uq_estimator.token_projector import UQTokenProjector


def test_projector_output_shape_and_initial_null_behavior():
    model = UQTokenProjector(
        active_dim=16,
        hidden_dim=32,
        llm_dim=64,
        token_count=1,
    )
    active = torch.randn(3, 16)
    score = torch.tensor([[0.0], [0.5], [1.0]])
    output = model(active, score)
    assert output.shape == (3, 1, 64)
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_score_gates_uncertainty_delta():
    model = UQTokenProjector(
        active_dim=2,
        hidden_dim=4,
        llm_dim=3,
        token_count=1,
    )
    with torch.no_grad():
        model.projector[-1].weight.fill_(0.1)
        model.projector[-1].bias.fill_(0.2)
        model.null_token.fill_(0.3)

    active = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    output = model(active, torch.tensor([[0.0], [1.0]]))
    torch.testing.assert_close(output[0], model.null_token[0])
    assert not torch.allclose(output[1], model.null_token[0])


def test_multiple_tokens_and_half_precision():
    model = UQTokenProjector(
        active_dim=4,
        hidden_dim=8,
        llm_dim=16,
        token_count=4,
    ).half()
    output = model(
        torch.randn(2, 4, dtype=torch.float32),
        torch.rand(2, 1, dtype=torch.float32),
    )
    assert output.shape == (2, 4, 16)
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()


def test_invalid_shapes_raise():
    model = UQTokenProjector(active_dim=4, hidden_dim=8, llm_dim=16)
    with pytest.raises(ValueError):
        model(torch.randn(2, 5), torch.rand(2, 1))
    with pytest.raises(ValueError):
        model(torch.randn(2, 4), torch.rand(2, 2))

import pytest
import torch

from uq_estimator.trajectory_adapter import (
    PathRiskTrajectoryAdapter,
    trajectory_adapter_loss,
)


def test_adapter_is_exact_identity_at_initialization():
    model = PathRiskTrajectoryAdapter(hidden_dim=16, max_residual_m=1.0)
    base = torch.randn(2, 3, 6, 2)
    risk = torch.rand(2, 3)
    output = model(base, risk)
    assert torch.equal(output.trajectories, base)
    assert torch.count_nonzero(output.residual) == 0


def test_zero_risk_remains_identity_after_training_parameters_change():
    model = PathRiskTrajectoryAdapter(hidden_dim=16, max_residual_m=1.0)
    with torch.no_grad():
        model.residual_head.weight.normal_()
        model.residual_head.bias.normal_()
    base = torch.randn(1, 2, 6, 2)
    output = model(base, torch.zeros(1, 2))
    assert torch.equal(output.trajectories, base)


def test_residual_is_bounded_by_path_risk():
    model = PathRiskTrajectoryAdapter(hidden_dim=8, max_residual_m=0.75)
    with torch.no_grad():
        model.residual_head.weight.fill_(4.0)
        model.residual_head.bias.fill_(4.0)
    base = torch.ones(1, 2, 4, 2)
    risk = torch.tensor([[0.25, 1.0]])
    output = model(base, risk)
    bounds = risk[..., None, None] * 0.75 + 1e-6
    assert torch.all(output.residual.abs() <= bounds)


def test_stepwise_risk_and_context_shapes():
    model = PathRiskTrajectoryAdapter(
        context_dim=5, hidden_dim=8, max_residual_m=1.0
    )
    base = torch.randn(2, 3, 4, 2)
    risk = torch.rand(2, 3, 4)
    output = model(base, risk, context=torch.randn(2, 5))
    assert output.trajectories.shape == (2, 3, 4, 2)
    assert output.intervention.shape == (2, 3, 4)
    assert output.stop_probability.shape == (2, 3, 4)


def test_adapter_loss_backpropagates_and_preserves_low_risk():
    model = PathRiskTrajectoryAdapter(hidden_dim=16, max_residual_m=2.0)
    base = torch.randn(2, 2, 6, 2)
    risk = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    expert = base.clone()
    expert[1] = 0.0
    mask = torch.ones(2, 2, 6)
    stop_target = torch.zeros(2, 2, 6)
    stop_target[1] = 1.0

    output = model(base, risk)
    losses = trajectory_adapter_loss(
        output,
        expert,
        mask,
        risk,
        stop_target=stop_target,
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert model.residual_head.weight.grad is not None
    assert model.residual_head.weight.grad.abs().sum() > 0
    assert model.stop_head.weight.grad is not None


@pytest.mark.parametrize(
    "bad_risk",
    (torch.zeros(2), torch.zeros(1, 3), torch.zeros(1, 2, 5)),
)
def test_adapter_rejects_invalid_risk_shapes(bad_risk):
    model = PathRiskTrajectoryAdapter(hidden_dim=8)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 2, 4, 2), bad_risk)


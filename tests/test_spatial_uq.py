"""CPU-only tests for the paper-grounded spatial UQ primitives."""

import inspect
import math

import pytest
import torch

from uq_estimator.spatial_uq import (
    SPATIAL_UQ_SCHEMA_VERSION,
    SpatialPatchUQHead,
    brier_loss,
    cvar_path_risk,
    decompose_ensemble_variance,
    heteroscedastic_gaussian_nll,
    paired_cosine_representation_error,
    paired_error_ranking_loss,
)


def test_patch_head_preserves_dynamic_leading_shape_and_bounds():
    head = SpatialPatchUQHead(
        feature_dim=7,
        hidden_dim=13,
        min_log_variance=-3.0,
        max_log_variance=2.0,
        predict_epistemic=True,
    )

    for shape in ((2, 3, 5, 7), (4, 11, 7)):
        result = head(torch.randn(*shape))
        expected_shape = shape[:-1]
        assert result.schema_version == SPATIAL_UQ_SCHEMA_VERSION
        assert result.expected_error.shape == expected_shape
        assert result.log_variance.shape == expected_shape
        assert result.aleatoric_variance.shape == expected_shape
        assert result.failure_probability.shape == expected_shape
        assert result.epistemic_variance is not None
        assert result.epistemic_variance.shape == expected_shape
        assert torch.all(result.expected_error >= 0)
        assert torch.all(result.log_variance >= -3.0)
        assert torch.all(result.log_variance <= 2.0)
        assert torch.all(result.aleatoric_variance > 0)
        assert torch.allclose(
            result.aleatoric_variance, result.log_variance.exp()
        )
        assert torch.all((0 <= result.failure_probability) & (result.failure_probability <= 1))
        assert torch.all(result.epistemic_variance >= 0)


def test_patch_head_epistemic_is_optional_and_route_is_not_an_input():
    head = SpatialPatchUQHead(feature_dim=5, hidden_dim=8, predict_epistemic=False)
    result = head(torch.randn(2, 9, 5))
    assert result.epistemic_variance is None

    # Architectural guardrail: route relevance only enters cvar_path_risk.
    assert list(inspect.signature(SpatialPatchUQHead.forward).parameters) == [
        "self",
        "patch_features",
    ]


def test_paired_cosine_representation_error_known_cases():
    clean = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    corrupt = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
    target = paired_cosine_representation_error(clean, corrupt)
    torch.testing.assert_close(target, torch.tensor([[0.0, 1.0, 2.0]]))
    assert not target.requires_grad


def test_paired_cosine_target_can_propagate_when_requested():
    clean = torch.randn(2, 3, 4, requires_grad=True)
    corrupt = torch.randn(2, 3, 4, requires_grad=True)
    target = paired_cosine_representation_error(
        clean, corrupt, detach_target=False
    )
    target.sum().backward()
    assert clean.grad is not None
    assert corrupt.grad is not None


def test_heteroscedastic_nll_contains_log_variance_term():
    # Zero residual leaves exactly 0.5 * log_variance (without the constant).
    nll = heteroscedastic_gaussian_nll(
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        reduction="none",
    )
    torch.testing.assert_close(nll, torch.tensor([1.0]))

    # The complete Gaussian form is also available for absolute NLL reporting.
    full = heteroscedastic_gaussian_nll(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        include_constant=True,
    )
    assert full.item() == pytest.approx(0.5 * math.log(2.0 * math.pi))


def test_brier_loss_is_a_proper_squared_probability_score():
    probability = torch.tensor([0.2, 0.9])
    target = torch.tensor([0.0, 1.0])
    result = brier_loss(probability, target)
    assert result.item() == pytest.approx((0.2**2 + 0.1**2) / 2)


def test_ranking_activates_only_for_a_real_target_increase():
    clean_prediction = torch.tensor([0.6, 0.6, 0.6], requires_grad=True)
    corrupt_prediction = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
    clean_target = torch.tensor([0.2, 0.4, 0.7])
    corrupt_target = torch.tensor([0.8, 0.4, 0.3])

    per_pair = paired_error_ranking_loss(
        clean_prediction,
        corrupt_prediction,
        clean_target,
        corrupt_target,
        margin=0.2,
        reduction="none",
    )
    # Only pair 0 genuinely becomes harder. Equal/decreasing targets stay off.
    torch.testing.assert_close(per_pair, torch.tensor([0.3, 0.0, 0.0]))

    mean_loss = paired_error_ranking_loss(
        clean_prediction,
        corrupt_prediction,
        clean_target,
        corrupt_target,
        margin=0.2,
    )
    assert mean_loss.item() == pytest.approx(0.3)
    mean_loss.backward()
    assert clean_prediction.grad is not None
    assert corrupt_prediction.grad is not None


def test_ranking_all_inactive_returns_differentiable_zero():
    clean_prediction = torch.tensor([0.9], requires_grad=True)
    corrupt_prediction = torch.tensor([0.1], requires_grad=True)
    loss = paired_error_ranking_loss(
        clean_prediction,
        corrupt_prediction,
        clean_target_error=torch.tensor([0.5]),
        corrupt_target_error=torch.tensor([0.4]),
    )
    assert loss.item() == 0.0
    loss.backward()
    assert clean_prediction.grad is not None
    assert corrupt_prediction.grad is not None


def test_ensemble_variance_uses_law_of_total_variance():
    means = torch.tensor([[1.0, 4.0], [3.0, 8.0]])
    aleatoric = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
    result = decompose_ensemble_variance(means, aleatoric, member_dim=0)

    torch.testing.assert_close(result.predictive_mean, torch.tensor([2.0, 6.0]))
    torch.testing.assert_close(result.aleatoric_variance, torch.tensor([3.0, 2.0]))
    torch.testing.assert_close(result.epistemic_variance, torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(result.total_variance, torch.tensor([4.0, 6.0]))


def test_cvar_path_risk_selects_worst_tail_and_reports_audit_data():
    failure = torch.tensor([[[[0.1, 0.9, 0.4, 0.8], [1.0, 1.0, 1.0, 1.0]]]])
    corridor = torch.tensor([[[[1, 1, 1, 1], [0, 0, 0, 0]]]], dtype=torch.bool)

    result = cvar_path_risk(failure, corridor, top_q=0.5)
    assert result.risk.shape == (1, 1)
    assert result.risk.item() == pytest.approx((0.9 + 0.8) / 2)
    assert result.valid_cell_count.item() == 4
    assert result.selected_cell_count.item() == 2
    assert result.selected_mask.sum().item() == 2
    # Extremely high off-route values are excluded rather than globally averaged.
    assert not result.selected_mask[..., 1, :].any()


def test_cvar_path_risk_applies_occupancy_ttc_and_dynamic_leading_dims():
    failure = torch.ones(2, 3, 2, 2) * 0.5
    corridor = torch.ones(1, 3, 2, 2, dtype=torch.bool)
    occupancy = torch.tensor([[[[1.0, 0.0], [0.5, 1.0]]]])
    ttc = torch.tensor([[[[1.0, 1.0], [2.0, 0.5]]]])

    result = cvar_path_risk(
        failure,
        corridor,
        occupancy_probability=occupancy,
        ttc_weight=ttc,
        top_q=0.25,
    )
    # The maximum cell risk is 0.5 for both occupancy/TTC combinations 1*1
    # and 0.5*2. top 25% of four cells selects one of them.
    assert result.risk.shape == (2, 3)
    torch.testing.assert_close(result.risk, torch.full((2, 3), 0.5))
    assert torch.all(result.selected_cell_count == 1)


def test_cvar_path_risk_empty_corridor_is_zero_and_invalid_q_fails():
    failure = torch.rand(2, 4, 5)
    empty = torch.zeros_like(failure, dtype=torch.bool)
    result = cvar_path_risk(failure, empty, top_q=0.2)
    torch.testing.assert_close(result.risk, torch.zeros(2))
    assert not result.selected_mask.any()
    assert torch.all(result.valid_cell_count == 0)
    assert torch.all(result.selected_cell_count == 0)

    with pytest.raises(ValueError, match="top_q"):
        cvar_path_risk(failure, empty, top_q=0.0)

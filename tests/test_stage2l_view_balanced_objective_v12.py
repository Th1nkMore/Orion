import torch

from uq_estimator.stage2l_view_balanced_objective_v12 import (
    view_balanced_region_weights,
    view_balanced_relevance_terms_v12,
    view_balanced_weight_summary,
)


def test_foreground_weight_is_equal_across_active_views_not_cells():
    target = torch.zeros(1, 6, 10, 10)
    target[0, 0, 4, 5] = 1.0
    target[0, 1, 2:4, 2:4] = 1.0
    weights = view_balanced_region_weights(target)
    per_view = weights.foreground.sum(dim=(-2, -1))[0]
    assert torch.allclose(per_view[:2], torch.tensor([0.5, 0.5]))
    assert torch.count_nonzero(per_view[2:]) == 0
    assert torch.allclose(weights.foreground.flatten(1).sum(1), torch.ones(1))
    assert torch.allclose(weights.background.flatten(1).sum(1), torch.ones(1))


def test_exact_soft_target_is_stationary_and_support_hinges_are_zero():
    target = torch.full((1, 6, 4, 4), 0.05)
    target[0, 0, 1, 1] = 0.8
    target[0, 1, 2, 2] = 0.6
    logits = torch.logit(target).detach().requires_grad_(True)
    terms = view_balanced_relevance_terms_v12(logits, target)
    terms.loss.backward()
    assert terms.balanced_brier.item() == 0.0
    assert terms.foreground_support_hinge.item() == 0.0
    assert terms.background_support_hinge.item() == 0.0
    assert torch.max(torch.abs(logits.grad)).item() < 1e-7


def test_weight_summary_exposes_old_cell_bias_and_new_view_balance():
    target = torch.zeros(1, 6, 10, 10)
    target[0, 0, 4, 5] = 1.0
    target[0, 1, 2:4, 2:4] = 1.0
    summary = view_balanced_weight_summary(target)
    current = summary["current_per_group_view_mass"][0]
    proposed = summary["proposed_per_group_view_mass"][0]
    assert torch.allclose(current[:2], torch.tensor([0.2, 0.8]))
    assert torch.allclose(proposed[:2], torch.tensor([0.5, 0.5]))


def test_requires_positive_relevance_support():
    target = torch.zeros(1, 6, 10, 10)
    try:
        view_balanced_region_weights(target)
    except ValueError as error:
        assert "positive support" in str(error)
    else:
        raise AssertionError("zero-support target should fail")

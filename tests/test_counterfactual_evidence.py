import inspect

import torch

from uq_estimator.counterfactual_evidence import (
    CounterfactualEvidenceTarget,
    EVIDENCE_COMPONENTS,
    ObservationEvidenceAdapter,
    ObservationEvidenceHurdleAdapter,
    balanced_evidence_presence_loss,
    balanced_counterfactual_evidence_regression_loss,
    counterfactual_evidence_regression_loss,
    counterfactual_evidence_target,
    fit_counterfactual_component_scales,
    responsive_evidence_magnitude_loss,
    scale_counterfactual_target,
)


def test_balanced_regression_prevents_zero_background_from_drowning_response():
    values = torch.zeros(1, 1, 2, 2, 3)
    values[0, 0, 0, 0] = torch.tensor([1.0, 0.5, 0.8])
    target = CounterfactualEvidenceTarget(
        values=values,
        component_valid=torch.ones_like(values, dtype=torch.bool),
    )
    prediction = torch.full_like(values, 0.05, requires_grad=True)
    loss = balanced_counterfactual_evidence_regression_loss(
        prediction, target, responsive_weight=0.75
    )
    loss.backward()
    assert torch.isfinite(loss)
    # Responsive cells must be pushed upward while zero background is pushed down.
    assert bool((prediction.grad[0, 0, 0, 0] < 0).all())
    assert bool((prediction.grad[0, 0, 1, 1] > 0).all())


def test_reference_equals_observed_has_zero_targets_and_invalid_first_transient():
    reference = torch.randn(2, 2, 3, 4, 8)
    valid = torch.tensor([False, True])
    target = counterfactual_evidence_target(
        reference,
        reference.clone(),
        reference_previous=reference * 0.9,
        observed_previous=reference * 0.9,
        previous_valid=valid,
    )
    assert target.values.shape == (2, 2, 3, 4, 3)
    assert torch.allclose(target.values, torch.zeros_like(target.values), atol=1e-6)
    assert not target.component_valid[0, ..., 2].any()
    assert target.component_valid[1, ..., 2].all()


def test_targets_separate_direction_magnitude_and_transient_change():
    reference_current = torch.tensor([[[[[1.0, 0.0]]]]])
    reference_previous = torch.tensor([[[[[1.0, 0.0]]]]])
    observed_current = torch.tensor([[[[[0.0, 2.0]]]]])
    observed_previous = torch.tensor([[[[[1.0, 0.0]]]]])
    target = counterfactual_evidence_target(
        reference_current,
        observed_current,
        reference_previous,
        observed_previous,
        torch.tensor([True]),
    )
    direction, magnitude, transient = target.values.flatten()
    assert torch.isclose(direction, torch.tensor(1.0))
    assert magnitude > 0
    assert torch.isclose(transient, torch.tensor(1.0))


def test_train_only_scaling_and_regression_loss_are_finite():
    reference = torch.randn(2, 1, 2, 3, 6)
    observed = reference + 0.3 * torch.randn_like(reference)
    target = counterfactual_evidence_target(
        reference,
        observed,
        reference,
        observed,
        torch.tensor([True, True]),
    )
    scales = fit_counterfactual_component_scales([target])
    scaled = scale_counterfactual_target(target, scales)
    prediction = torch.full_like(scaled.values, 0.5, requires_grad=True)
    loss = counterfactual_evidence_regression_loss(prediction, scaled)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None


def test_adapter_is_task_agnostic_and_preserves_spatial_view_shape():
    signature = inspect.signature(ObservationEvidenceAdapter.forward)
    assert tuple(signature.parameters) == (
        "self",
        "current",
        "previous",
        "previous_valid",
    )
    model = ObservationEvidenceAdapter(feature_dim=8, hidden_dim=12, max_views=3)
    current = torch.randn(2, 3, 5, 6, 8)
    previous = torch.randn_like(current)
    output = model(current, previous, torch.tensor([True, False]))
    assert output.shape == (2, 3, 5, 6, len(EVIDENCE_COMPONENTS))
    assert (output >= 0).all()


def test_hurdle_adapter_is_task_agnostic_and_factors_sparse_score():
    signature = inspect.signature(ObservationEvidenceHurdleAdapter.forward)
    assert tuple(signature.parameters) == (
        "self",
        "current",
        "previous",
        "previous_valid",
    )
    model = ObservationEvidenceHurdleAdapter(
        feature_dim=8, hidden_dim=12, max_views=3
    )
    current = torch.randn(2, 3, 5, 6, 8)
    parts = model.predict_parts(current, None, torch.tensor([False, False]))
    assert parts.score.shape == (2, 3, 5, 6, len(EVIDENCE_COMPONENTS))
    assert torch.allclose(
        parts.score,
        parts.presence_probability * parts.conditional_magnitude,
    )
    assert bool(((parts.presence_probability >= 0) & (parts.presence_probability <= 1)).all())
    assert bool((parts.conditional_magnitude >= 0).all())


def test_hurdle_adapter_without_view_embedding_is_view_equivariant():
    model = ObservationEvidenceHurdleAdapter(
        feature_dim=8,
        hidden_dim=12,
        max_views=3,
        use_view_embedding=False,
    ).eval()
    assert model.view_embedding is None
    current = torch.randn(2, 3, 5, 6, 8)
    previous = torch.randn_like(current)
    valid = torch.tensor([True, False])
    permutation = torch.tensor([2, 0, 1])
    with torch.no_grad():
        expected = model(current, previous, valid)[:, permutation]
        actual = model(
            current[:, permutation], previous[:, permutation], valid
        )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_hurdle_losses_separate_presence_from_conditional_magnitude():
    values = torch.zeros(1, 1, 1, 2, 3)
    values[..., 0, :] = torch.tensor([1.0, 0.5, 0.8])
    target = CounterfactualEvidenceTarget(
        values=values,
        component_valid=torch.ones_like(values, dtype=torch.bool),
    )
    logits = torch.zeros_like(values, requires_grad=True)
    magnitude = torch.full_like(values, 0.1, requires_grad=True)
    loss = balanced_evidence_presence_loss(logits, target)
    loss = loss + responsive_evidence_magnitude_loss(magnitude, target)
    loss.backward()
    assert bool((logits.grad[..., 0, :] < 0).all())
    assert bool((logits.grad[..., 1, :] > 0).all())
    assert bool((magnitude.grad[..., 0, :] < 0).all())
    assert bool((magnitude.grad[..., 1, :] == 0).all())


def test_high_response_support_excludes_small_nonzero_vit_footprint():
    values = torch.tensor([[[[[0.05, 0.20, 0.30], [0.01, 0.02, 0.03]]]]])
    target = CounterfactualEvidenceTarget(
        values=values,
        component_valid=torch.ones_like(values, dtype=torch.bool),
    )
    logits = torch.zeros_like(values, requires_grad=True)
    loss = balanced_evidence_presence_loss(
        logits,
        target,
        support_thresholds=torch.tensor([0.10, 0.10, 0.10]),
    )
    loss.backward()
    # Only the two high-amplitude component cells are positives; every small
    # nonzero response is explicitly background for the support head.
    expected_positive = values > torch.tensor([0.10, 0.10, 0.10])
    assert torch.equal(logits.grad < 0, expected_positive)

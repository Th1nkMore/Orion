import pytest
import torch

from uq_estimator.stage2l_identifiability import (
    audit_answer_preferences,
    audit_matched_task_risk,
    audit_relevance_invariance,
)


def _matched_uncertainty(batch_size=2):
    shape = (batch_size, 3, 3, 3)
    zero = torch.zeros(shape)
    on = torch.zeros(shape)
    off = torch.zeros(shape)
    shuffled = torch.zeros(shape)
    on[:, 0, 1, 1] = 0.9
    off[:, 0, 0, 0] = 0.9
    shuffled[:, 2, 1, 1] = 0.9
    return {
        "zero_uq": zero,
        "on_path_uq": on,
        "off_path_uq": off,
        "view_shuffled_uq": shuffled,
    }


def test_factorized_relevance_requires_variant_invariance():
    logits = torch.randn(2, 3, 3, 3)
    values = {variant: logits.clone() for variant in _matched_uncertainty()}
    audit = audit_relevance_invariance(values)
    assert audit.all_invariant is True
    assert all(value == 0.0 for value in audit.maximum_absolute_drift.values())

    values["off_path_uq"][0, 0, 0, 0] += 0.01
    failed = audit_relevance_invariance(values)
    assert failed.all_invariant is False
    assert failed.invariant_by_variant["off_path_uq"] is False


def test_matched_risk_is_identified_by_u_location_under_shared_r():
    relevance = torch.full((2, 3, 3, 3), -9.0)
    relevance[:, 0, 1, 1] = 9.0
    audit = audit_matched_task_risk(
        relevance,
        _matched_uncertainty(),
        required_on_over_off_margin=0.5,
        required_on_over_shuffled_margin=0.5,
        maximum_off_path_risk=0.01,
        minimum_fraction=1.0,
        minimum_shuffled_fraction=1.0,
    )
    assert all(audit.aggregate_gates.values())
    assert audit.on_over_off_fraction == 1.0
    assert audit.on_over_shuffled_fraction == 1.0
    assert torch.equal(
        audit.risk_peak_by_variant["zero_uq"], torch.zeros(2)
    )


def test_matched_risk_fails_closed_on_unmatched_magnitude_or_nonzero_zero_u():
    relevance = torch.zeros(2, 3, 3, 3)
    unmatched = _matched_uncertainty()
    unmatched["off_path_uq"][:, 0, 0, 0] = 0.4
    nonzero = audit_matched_task_risk(relevance, unmatched)
    assert nonzero.aggregate_gates["on_off_u_mass_matched"] is False
    assert nonzero.aggregate_gates["on_off_u_peak_matched"] is False

    bad_zero = _matched_uncertainty()
    bad_zero["zero_uq"][:, 0, 0, 0] = 0.01
    failed = audit_matched_task_risk(relevance, bad_zero)
    assert failed.aggregate_gates["zero_u_exact"] is False
    assert failed.aggregate_gates["zero_task_risk_exact"] is False


def test_answer_preference_audit_requires_each_matched_variant():
    variants = tuple(_matched_uncertainty())
    targets = {variant: torch.tensor([0.1, 0.2]) for variant in variants}
    alternatives = {variant: torch.tensor([0.8, 0.9]) for variant in variants}
    passed = audit_answer_preferences(
        targets, alternatives, minimum_fraction=1.0
    )
    assert passed.all_passed is True

    alternatives["view_shuffled_uq"] = torch.tensor([0.05, 0.05])
    failed = audit_answer_preferences(
        targets, alternatives, minimum_fraction=0.5
    )
    assert failed.gates["view_shuffled_uq"] is False
    assert failed.all_passed is False


def test_identifiability_audits_reject_missing_or_invalid_inputs():
    values = _matched_uncertainty()
    values.pop("view_shuffled_uq")
    with pytest.raises(ValueError, match="missing variants"):
        audit_matched_task_risk(torch.zeros(2, 3, 3, 3), values)

    values = _matched_uncertainty()
    values["on_path_uq"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        audit_matched_task_risk(torch.zeros(2, 3, 3, 3), values)

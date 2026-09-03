import pytest


torch = pytest.importorskip("torch")

from uq_estimator.stage2l_support_aligned_objective_v9 import (
    support_aligned_relevance_terms,
)


def _target():
    value = torch.zeros(1, 1, 2, 3)
    value[0, 0, 0, 1] = 1.0
    value[0, 0, 1, 1] = 0.5
    return value


def test_background_support_hinge_penalizes_gate_false_positives():
    target = _target()
    clean = torch.logit(target.clamp(0.001, 0.999))
    noisy = clean.clone()
    noisy[0, 0, 0, 0] = torch.logit(torch.tensor(0.4))
    clean_terms = support_aligned_relevance_terms(clean, target)
    noisy_terms = support_aligned_relevance_terms(noisy, target)
    assert noisy_terms.background_support_hinge > clean_terms.background_support_hinge
    assert noisy_terms.loss > clean_terms.loss


def test_background_support_gradient_pushes_false_positive_down():
    target = _target()
    logits = torch.zeros_like(target, requires_grad=True)
    terms = support_aligned_relevance_terms(logits, target)
    terms.loss.backward()
    assert logits.grad[0, 0, 0, 0] > 0.0
    assert torch.isfinite(logits.grad).all()


def test_support_objective_rejects_missing_positive_target():
    with pytest.raises(ValueError, match="positive.*support"):
        support_aligned_relevance_terms(
            torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)
        )

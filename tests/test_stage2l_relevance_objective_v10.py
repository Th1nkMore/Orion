import pytest
import torch

from uq_estimator.stage2l_relevance_objective_v10 import (
    stage2l_relevance_objective_v10,
)


def _target():
    target = torch.zeros(1, 1, 4, 4)
    target[:, :, 2:, 1:3] = 1.0
    return target


def test_map_pretrain_has_no_ranking_or_field_control_losses():
    target = _target()
    logits = torch.where(target.bool(), torch.tensor(4.0), torch.tensor(-4.0))
    terms = stage2l_relevance_objective_v10(
        logits, target, phase="map_pretrain"
    )
    assert terms.ranking is None
    assert terms.structured_field_classification_loss_used is False
    assert terms.trajectory_or_control_loss_used is False
    assert torch.isfinite(terms.loss)


def test_background_false_positive_is_penalized_by_support_terms():
    target = _target()
    good = torch.where(target.bool(), torch.tensor(4.0), torch.tensor(-4.0))
    false_positive = good.clone()
    false_positive[:, :, :2] = 4.0
    good_terms = stage2l_relevance_objective_v10(
        good, target, phase="map_pretrain"
    )
    bad_terms = stage2l_relevance_objective_v10(
        false_positive, target, phase="map_pretrain"
    )
    assert bad_terms.background_support_hinge > good_terms.background_support_hinge
    assert bad_terms.support_tversky_loss > good_terms.support_tversky_loss
    assert bad_terms.loss > good_terms.loss


def test_risk_alignment_requires_matched_u_and_adds_ranking():
    target = _target()
    logits = torch.where(target.bool(), torch.tensor(2.0), torch.tensor(-2.0))
    on_u = target.clone() * 0.9
    off_u = torch.zeros_like(target)
    off_u[:, :, 0, 0] = 0.9
    terms = stage2l_relevance_objective_v10(
        logits,
        target,
        phase="risk_alignment",
        on_path_uq=on_u,
        off_path_uq=off_u,
    )
    assert terms.ranking is not None
    assert terms.loss >= terms.map_loss
    with pytest.raises(ValueError):
        stage2l_relevance_objective_v10(
            logits, target, phase="risk_alignment"
        )
    with pytest.raises(ValueError):
        stage2l_relevance_objective_v10(
            logits,
            target,
            phase="map_pretrain",
            on_path_uq=on_u,
            off_path_uq=off_u,
        )

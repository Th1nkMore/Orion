import io

import torch

from uq_estimator.stage2l_factorized_relevance_v121 import (
    FactorizedTaskRelevanceMapHead,
    factorized_relevance_terms_v121,
)


def _soft_targets():
    target = torch.full((1, 2, 3, 4, 4), 0.05)
    target[0, 0, 0, 1, 1] = 0.75
    target[0, 1, 1, 2, 2] = 0.85
    return target


def test_factorized_head_shapes_union_and_serialization():
    torch.manual_seed(0)
    head = FactorizedTaskRelevanceMapHead(model_dim=8, hidden_dim=4)
    context = torch.randn(2, 6, 3, 3, 8)
    output = head(context)
    assert output.route_logits.shape == (2, 6, 3, 3)
    assert output.actor_logits.shape == (2, 6, 3, 3)
    assert output.component_logits.shape == (2, 2, 6, 3, 3)
    assert torch.equal(
        output.derived_union_probability,
        torch.maximum(output.route_probability, output.actor_probability),
    )
    payload = io.BytesIO()
    torch.save(head.state_dict(), payload)
    payload.seek(0)
    restored = FactorizedTaskRelevanceMapHead(model_dim=8, hidden_dim=4)
    restored.load_state_dict(
        torch.load(payload, map_location="cpu", weights_only=True)
    )
    assert torch.equal(
        restored(context).component_logits, output.component_logits
    )


def test_exact_soft_component_targets_are_stationary():
    target = _soft_targets()
    logits = torch.logit(target).detach().requires_grad_(True)
    terms = factorized_relevance_terms_v121(logits, target)
    terms.loss.backward()
    assert terms.loss.item() < 1e-12
    assert terms.derived_union_brier_diagnostic.item() < 1e-12
    assert torch.max(torch.abs(logits.grad)).item() < 1e-8


def test_route_and_actor_losses_are_independent():
    target = _soft_targets()
    logits = torch.zeros_like(target)
    base = factorized_relevance_terms_v121(logits, target)
    changed_actor = target.clone()
    changed_actor[:, 1] = 0.25
    actor_changed = factorized_relevance_terms_v121(logits, changed_actor)
    assert torch.equal(base.route_loss, actor_changed.route_loss)
    changed_route = target.clone()
    changed_route[:, 0] = 0.25
    route_changed = factorized_relevance_terms_v121(logits, changed_route)
    assert torch.equal(base.actor_loss, route_changed.actor_loss)


def test_empty_actor_component_keeps_negative_supervision_and_finite_gradients():
    target = torch.zeros(1, 2, 3, 4, 4)
    target[0, 0, 0, 1, 1] = 1.0
    logits = torch.zeros_like(target, requires_grad=True)
    terms = factorized_relevance_terms_v121(logits, target)
    terms.loss.backward()
    assert terms.empty_sample_component_count.item() == 1.0
    assert terms.actor_empty_component_anchor.item() > 0.0
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:, 1]) > 0


def test_equal_active_view_averaging_is_not_cell_count_weighted():
    target = torch.zeros(1, 2, 2, 4, 4)
    target[0, 0, 0, 1, 1] = 1.0
    target[0, 0, 1, 1:3, 1:3] = 1.0
    logits = torch.zeros_like(target)
    first = factorized_relevance_terms_v121(logits, target)
    target[0, 0, 1, 0:4, 0:4] = 1.0
    second = factorized_relevance_terms_v121(logits, target)
    assert torch.equal(first.route_active_brier, second.route_active_brier)


def test_rejects_wrong_component_axis():
    logits = torch.zeros(1, 1, 2, 3, 3)
    try:
        factorized_relevance_terms_v121(logits, logits.clone())
    except ValueError as error:
        assert "[B,2,V,H,W]" in str(error)
    else:
        raise AssertionError("one-component input should fail")

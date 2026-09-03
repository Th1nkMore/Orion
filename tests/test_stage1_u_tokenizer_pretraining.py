import pytest
import torch

from uq_estimator.stage1_u_tokenizer_pretraining import (
    UQSummaryReconstructionHead,
    stage1_u_tokenizer_pretraining_terms,
)
from uq_estimator.uq_relevance_tokenizer import UQComponentTokenizer


def _modules():
    tokenizer = UQComponentTokenizer(
        model_dim=16, hidden_dim=8, grid_hw=(2, 3), max_views=6
    )
    decoder = UQSummaryReconstructionHead(
        model_dim=16, hidden_dim=8, component_dim=3
    )
    return tokenizer, decoder


def test_pretraining_uses_only_components_and_reconstructs_exact_summary_shape():
    tokenizer, decoder = _modules()
    components = torch.rand(2, 4, 6, 8, 12, 3)
    terms = stage1_u_tokenizer_pretraining_terms(
        tokenizer=tokenizer,
        reconstruction_head=decoder,
        components=components,
    )
    assert terms.decoded_summary.shape == (2, 6, 2, 3, 9)
    assert terms.decoded_summary.shape == terms.target_summary.shape
    assert torch.isfinite(terms.loss)
    assert terms.task_labels_consumed is False
    assert terms.route_context_consumed is False
    assert terms.corruption_metadata_consumed is False


def test_pretraining_gradients_update_tokenizer_and_disposable_decoder():
    tokenizer, decoder = _modules()
    terms = stage1_u_tokenizer_pretraining_terms(
        tokenizer=tokenizer,
        reconstruction_head=decoder,
        components=torch.rand(2, 3, 6, 6, 9, 3),
    )
    terms.loss.backward()
    assert any(parameter.grad is not None for parameter in tokenizer.parameters())
    assert any(parameter.grad is not None for parameter in decoder.parameters())
    assert terms.target_summary.requires_grad is False


def test_zero_anchor_is_definition_level_and_finite():
    tokenizer, decoder = _modules()
    terms = stage1_u_tokenizer_pretraining_terms(
        tokenizer=tokenizer,
        reconstruction_head=decoder,
        components=torch.zeros(1, 2, 6, 4, 6, 3),
        zero_anchor_weight=1.0,
    )
    assert torch.isfinite(terms.zero_anchor_loss)
    assert torch.equal(terms.target_summary, torch.zeros_like(terms.target_summary))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"zero_anchor_weight": -0.1}, "weights"),
        ({"smooth_l1_beta": 0.0}, "weights"),
    ],
)
def test_pretraining_rejects_invalid_loss_configuration(kwargs, message):
    tokenizer, decoder = _modules()
    with pytest.raises(ValueError, match=message):
        stage1_u_tokenizer_pretraining_terms(
            tokenizer=tokenizer,
            reconstruction_head=decoder,
            components=torch.rand(1, 2, 6, 4, 6, 3),
            **kwargs,
        )

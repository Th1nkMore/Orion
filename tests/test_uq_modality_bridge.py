import inspect

import pytest
import torch
import torch.nn.functional as F

from uq_estimator.uq_modality_bridge import UQFormerBridge
from uq_estimator.uqformer_alignment import (
    UQFormerReconstructionHead,
    symmetric_u_text_alignment_loss,
    uqformer_equivariance_terms,
    uqformer_reconstruction_terms,
)


def _bridge(*, model_dim=32, dropout=0.0):
    return UQFormerBridge(
        component_dim=3,
        model_dim=model_dim,
        bridge_dim=16,
        grid_hw=(3, 4),
        max_views=6,
        view_query_hw=(2, 2),
        temporal_queries=3,
        include_component_queries=True,
        global_queries=4,
        num_heads=4,
        num_layers=2,
        feedforward_dim=32,
        dropout=dropout,
    )


def test_bridge_accepts_only_stage1_components_and_emits_compact_structure():
    assert list(inspect.signature(UQFormerBridge.forward).parameters) == [
        "self",
        "components",
    ]
    model = _bridge()
    components = torch.rand(2, 4, 6, 8, 12, 3)
    output = model(components)
    # 6 views x 2 x 2, latest/mean/delta, 3 components, 4 global.
    assert model.compact_query_count_at_max_views == 34
    assert output.compact_tokens.shape == (2, 34, 16)
    assert output.language_tokens.shape == (2, 34, 32)
    assert output.view_spatial_tokens.shape == (2, 6, 2, 2, 32)
    assert output.temporal_tokens.shape == (2, 3, 32)
    assert output.component_tokens.shape == (2, 3, 32)
    assert output.global_tokens.shape == (2, 4, 32)
    assert output.pooled_components.shape == (2, 4, 6, 3, 4, 3)
    assert output.source_summary.shape == (2, 6, 3, 4, 9)
    assert output.source_features.shape == (2, 6, 3, 4, 16)
    assert output.attention_maps.shape == (2, 34, 6, 3, 4)
    assert output.modality == "observation_uncertainty"
    assert output.query_layout.temporal_statistic_ids[
        slice(*output.query_layout.temporal_slice)
    ].tolist() == [0, 1, 2]


def test_source_is_exact_native_nine_dimensional_temporal_summary():
    model = _bridge()
    components = torch.rand(1, 3, 6, 3, 4, 3)
    output = model(components)
    latest = components[:, -1]
    mean = components.mean(dim=1)
    delta = latest - components[:, 0]
    expected = torch.cat((latest, mean, delta), dim=-1)
    assert torch.allclose(output.source_summary, expected)


def test_language_width_changes_only_boundary_not_source_or_latent_topology():
    components = torch.rand(2, 3, 6, 7, 9, 3)
    torch.manual_seed(123)
    narrow = _bridge(model_dim=24)(components)
    torch.manual_seed(123)
    wide = _bridge(model_dim=48)(components)
    assert narrow.source_summary.shape == wide.source_summary.shape
    assert narrow.source_features.shape == wide.source_features.shape
    assert narrow.compact_tokens.shape == wide.compact_tokens.shape
    assert narrow.attention_maps.shape == wide.attention_maps.shape
    assert narrow.language_tokens.shape[:-1] == wide.language_tokens.shape[:-1]
    assert narrow.language_tokens.shape[-1] == 24
    assert wide.language_tokens.shape[-1] == 48
    # With identical latent initialization, the pre-boundary values are exact.
    assert torch.equal(narrow.source_summary, wide.source_summary)
    assert torch.equal(narrow.source_features, wide.source_features)
    assert torch.equal(narrow.compact_tokens, wide.compact_tokens)


def test_view_queries_attend_only_their_camera_and_attention_is_auditable():
    output = _bridge()(torch.rand(1, 3, 6, 8, 12, 3))
    attention = output.attention_maps
    assert torch.allclose(
        attention.flatten(2).sum(dim=-1),
        torch.ones(attention.shape[:2]),
        atol=1e-6,
    )
    view_start, view_end = output.query_layout.view_slice
    for query_index in range(view_start, view_end):
        expected_view = int(output.query_layout.view_ids[query_index])
        other_views = [value for value in range(6) if value != expected_view]
        assert torch.equal(
            attention[:, query_index, other_views],
            torch.zeros_like(attention[:, query_index, other_views]),
        )


def test_zero_u_is_continuous_finite_and_has_no_binary_embedding_shortcut():
    model = _bridge()
    assert not hasattr(model, "input_state_embedding")
    zero = torch.zeros(2, 3, 6, 8, 12, 3)
    epsilon = torch.full_like(zero, 1e-7)
    zero_output = model(zero)
    epsilon_output = model(epsilon)
    assert bool(zero_output.zero_input_mask.all())
    assert not bool(epsilon_output.zero_input_mask.any())
    assert torch.isfinite(zero_output.language_tokens).all()
    assert torch.equal(
        zero_output.source_summary, torch.zeros_like(zero_output.source_summary)
    )
    # No hard zero/nonzero branch: an infinitesimal input produces a small,
    # continuous latent change rather than switching a learned state token.
    assert (
        epsilon_output.compact_tokens - zero_output.compact_tokens
    ).abs().max() < 1e-3


def test_reconstruction_zero_anchor_and_bridge_gradients_are_finite():
    model = _bridge()
    decoder = UQFormerReconstructionHead(
        bridge_dim=16,
        component_dim=3,
        max_views=6,
        num_heads=4,
        hidden_dim=32,
    )
    output = model(torch.rand(2, 3, 6, 8, 12, 3))
    zero_output = model(torch.zeros(1, 3, 6, 8, 12, 3))
    terms = uqformer_reconstruction_terms(
        output=output,
        reconstruction_head=decoder,
        zero_output=zero_output,
    )
    assert terms.decoded_components.shape == output.source_summary.shape
    assert terms.target_components.shape[-1] == 9
    assert torch.isfinite(terms.loss)
    assert terms.task_labels_consumed is False
    assert terms.route_context_consumed is False
    assert terms.corruption_metadata_consumed is False
    terms.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert any(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
        if gradient is not None
    )


def test_equivariance_primitive_is_zero_for_identity_and_validates_maps():
    output = _bridge()(torch.rand(2, 3, 6, 8, 12, 3))
    identity = uqformer_equivariance_terms(
        reference=output,
        transformed=output,
        global_invariant=True,
    )
    assert identity.loss.item() == pytest.approx(0.0)
    transformed = _bridge()(torch.rand(2, 3, 6, 8, 12, 3))
    terms = uqformer_equivariance_terms(
        reference=output,
        transformed=transformed,
        view_permutation=[5, 4, 3, 2, 1, 0],
        horizontal_flip=True,
        vertical_flip=True,
        temporal_changed=True,
        component_permutation=[2, 1, 0],
    )
    assert torch.isfinite(terms.loss)
    assert terms.global_invariance_loss.item() == pytest.approx(0.0)
    assert (
        inspect.signature(uqformer_equivariance_terms)
        .parameters["global_invariant"]
        .default
        is False
    )
    with pytest.raises(ValueError, match="complete permutation"):
        uqformer_equivariance_terms(
            reference=output,
            transformed=output,
            view_permutation=[0, 0, 1, 2, 3, 4],
        )


def test_task_free_u_text_alignment_prefers_matching_pairs():
    embeddings = F.normalize(torch.eye(4), dim=-1)
    matching = symmetric_u_text_alignment_loss(embeddings, embeddings)
    mismatching = symmetric_u_text_alignment_loss(
        embeddings, embeddings.roll(1, dims=0)
    )
    assert torch.isfinite(matching)
    assert matching < mismatching


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"temporal_queries": 4}, "latest/mean/delta"),
        ({"bridge_dim": 15, "num_heads": 4}, "divisible"),
        ({"spatial_locality_strength": -1.0}, "locality"),
    ],
)
def test_bridge_rejects_ambiguous_or_invalid_configuration(kwargs, message):
    defaults = {
        "model_dim": 16,
        "bridge_dim": 16,
        "grid_hw": (2, 2),
        "num_heads": 4,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        UQFormerBridge(**defaults)


def test_bridge_rejects_nonfinite_or_out_of_range_components():
    model = _bridge()
    invalid = torch.zeros(1, 2, 6, 4, 4, 3)
    invalid[..., 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(invalid)
    invalid.zero_()
    invalid[..., 0] = 1.1
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        model(invalid)

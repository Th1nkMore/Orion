import torch

from uq_estimator.stage2l_structured_field_head import (
    TASK_FIELD_VOCABULARIES,
    VLMTaskSemanticFieldHead,
    dataset_frequency_balanced_field_loss,
    dataset_frequency_balanced_partial_field_loss,
    decode_task_field_predictions,
    encode_task_field_targets,
)


def _targets():
    return [
        {
            "relevance_level": "not_applicable",
            "risk_level": "none",
            "risk_view": "none",
            "risk_region": "none",
            "stance": "maintain",
        },
        {
            "relevance_level": "high",
            "risk_level": "medium",
            "risk_view": "CAM_FRONT",
            "risk_region": "lower_center",
            "stance": "caution",
        },
    ]


def test_structured_field_head_shapes_and_decoding():
    torch.manual_seed(4)
    head = VLMTaskSemanticFieldHead(model_dim=32, hidden_dim=16)
    tokens = torch.randn(2, 7, 32)
    raw = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8, 0.2, 0.3, 0.5, -0.2, 0.0],
    ])
    result = head(tokens, raw, raw)
    assert result.token.shape == (2, 1, 32)
    for field, vocabulary in TASK_FIELD_VOCABULARIES.items():
        assert result.logits[field].shape == (2, len(vocabulary))
        assert result.predicted_indices[field].shape == (2,)
    decoded = decode_task_field_predictions(result.predicted_indices)
    assert len(decoded) == 2
    assert all(set(row) == set(TASK_FIELD_VOCABULARIES) for row in decoded)


def test_language_token_does_not_own_field_classifiers_when_detached():
    torch.manual_seed(5)
    head = VLMTaskSemanticFieldHead(model_dim=16, hidden_dim=8)
    tokens = torch.randn(2, 7, 16, requires_grad=True)
    raw = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8, 0.2, 0.3, 0.5, -0.2, 0.0],
    ], requires_grad=True)
    output = head(
        tokens, raw, raw, detach_probabilities_for_language=True
    )
    output.token.square().mean().backward()
    assert tokens.grad is None
    assert raw.grad is None
    assert all(
        parameter.grad is None
        for classifier in head.classifiers.values()
        for parameter in classifier.parameters()
    )
    assert any(
        parameter.grad is not None
        for embedding in head.field_embeddings.values()
        for parameter in embedding.parameters()
    )


def test_explicit_field_loss_owns_classifiers_and_shared_encoder():
    torch.manual_seed(6)
    head = VLMTaskSemanticFieldHead(model_dim=16, hidden_dim=8)
    tokens = torch.randn(2, 7, 16, requires_grad=True)
    raw = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8, 0.2, 0.3, 0.5, -0.2, 0.0],
    ], requires_grad=True)
    output = head(tokens, raw, raw)
    targets = encode_task_field_targets(_targets())
    counts = {
        field: {
            value: sum(row[field] == value for row in _targets())
            for value in vocabulary
        }
        for field, vocabulary in TASK_FIELD_VOCABULARIES.items()
    }
    loss, per_field = dataset_frequency_balanced_field_loss(
        output.logits, targets, counts
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert set(per_field) == set(TASK_FIELD_VOCABULARIES)
    assert tokens.grad is not None
    assert raw.grad is not None
    assert any(
        parameter.grad is not None
        for classifier in head.classifiers.values()
        for parameter in classifier.parameters()
    )


def test_encoded_targets_round_trip():
    encoded = encode_task_field_targets(_targets())
    assert decode_task_field_predictions(encoded) == tuple(_targets())


def test_partial_field_loss_supports_family_specific_supervision():
    torch.manual_seed(8)
    head = VLMTaskSemanticFieldHead(model_dim=16, hidden_dim=8)
    output = head(
        torch.randn(2, 7, 16),
        torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.3, 0.5, -0.2, 0.0],
        ]),
        torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.3, 0.5, -0.2, 0.0],
        ]),
    )
    encoded = encode_task_field_targets(_targets())
    target = {"stance": encoded["stance"]}
    counts = {
        "stance": {
            value: sum(row["stance"] == value for row in _targets())
            for value in TASK_FIELD_VOCABULARIES["stance"]
        }
    }
    loss, per_field = dataset_frequency_balanced_partial_field_loss(
        output.logits, target, counts
    )
    assert torch.isfinite(loss)
    assert set(per_field) == {"stance"}


def test_observation_summary_distinguishes_zero_u_from_zero_k_off_path():
    torch.manual_seed(9)
    head = VLMTaskSemanticFieldHead(model_dim=16, hidden_dim=8)
    tokens = torch.zeros(1, 3, 16)
    zero = torch.zeros(1, 6)
    zero[:, 2] = 1e-3
    off_path_u = zero.clone()
    off_path_u[:, 0] = 0.9
    off_path_u[:, 1] = 0.2
    zero_output = head(tokens, zero, zero)
    off_path_output = head(tokens, off_path_u, zero)
    assert any(
        not torch.allclose(
            zero_output.logits[field], off_path_output.logits[field]
        )
        for field in TASK_FIELD_VOCABULARIES
    )


def test_small_field_head_can_overfit_zero_off_path_and_on_path_semantics():
    torch.manual_seed(10)
    head = VLMTaskSemanticFieldHead(model_dim=16, hidden_dim=16)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.02, weight_decay=0.0)
    tokens = torch.zeros(4, 3, 16)
    tokens[2, -1, :4] = 0.3
    tokens[3, -1, :4] = 0.8
    observation = torch.tensor([
        [0.0, 0.0, 0.001, 0.0, 0.0, 0.0],
        [0.9, 0.2, 0.3, 0.5, -0.5, 0.0],
        [0.9, 0.2, 0.3, 0.5, 0.0, 0.0],
        [0.9, 0.2, 0.3, 0.5, 0.0, 0.0],
    ])
    task_risk = torch.tensor([
        [0.0, 0.0, 0.001, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.001, 0.0, 0.0, 0.0],
        [0.3, 0.04, 0.1, 0.5, 0.0, 0.0],
        [0.7, 0.1, 0.2, 0.5, 0.0, 0.0],
    ])
    targets = [
        {
            "relevance_level": "not_applicable",
            "risk_level": "none",
            "risk_view": "none",
            "risk_region": "none",
            "stance": "maintain",
        },
        {
            "relevance_level": "low",
            "risk_level": "none",
            "risk_view": "none",
            "risk_region": "none",
            "stance": "maintain",
        },
        {
            "relevance_level": "high",
            "risk_level": "low",
            "risk_view": "CAM_FRONT",
            "risk_region": "lower_center",
            "stance": "caution",
        },
        {
            "relevance_level": "high",
            "risk_level": "medium",
            "risk_view": "CAM_FRONT",
            "risk_region": "lower_center",
            "stance": "prepare_to_yield",
        },
    ]
    encoded = encode_task_field_targets(targets)
    counts = {
        field: {
            value: sum(row[field] == value for row in targets)
            for value in vocabulary
        }
        for field, vocabulary in TASK_FIELD_VOCABULARIES.items()
    }
    for _ in range(200):
        output = head(tokens, observation, task_risk)
        loss, _ = dataset_frequency_balanced_field_loss(
            output.logits, encoded, counts
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    output = head(tokens, observation, task_risk)
    assert decode_task_field_predictions(output.predicted_indices) == tuple(targets)

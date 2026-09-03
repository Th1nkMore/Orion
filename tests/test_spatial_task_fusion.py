import hashlib

import pytest
import torch

from uq_estimator.spatial_task_fusion import (
    SpatialUQTokenProjector,
    STAGE2_TASK_FUSION_CHECKPOINT_SCHEMA,
    TaskRiskTrajectoryAdapter,
    load_stage2_task_fusion_checkpoint,
    scatter_selected_token_values,
)


def _modules():
    projector = SpatialUQTokenProjector(
        component_dim=3,
        model_dim=16,
        hidden_dim=12,
        max_views=6,
        tokens_per_view=2,
    )
    adapter = TaskRiskTrajectoryAdapter(
        model_dim=16,
        num_heads=4,
        trajectory_steps=6,
    )
    return projector, adapter


def test_zero_uq_is_exact_identity_and_zero_response():
    projector, adapter = _modules()
    uq = torch.zeros(2, 6, 4, 5, 3)
    context = torch.randn(2, 7, 16)
    tokens = projector(uq)
    output = adapter(context, tokens)
    assert torch.count_nonzero(tokens.tokens) == 0
    assert torch.equal(output.conditioned_context, context)
    assert output.yield_logits.argmax(dim=-1).tolist() == [0, 0]
    assert torch.count_nonzero(output.conflict_logits) == 0
    assert torch.count_nonzero(output.trajectory_residual) == 0
    assert torch.count_nonzero(output.token_attention) == 0


def test_stage2_gradient_stops_at_observation_uq_boundary():
    projector, adapter = _modules()
    uq = torch.rand(1, 6, 4, 5, 3, requires_grad=True)
    context = torch.randn(1, 7, 16, requires_grad=True)
    output = adapter(context, projector(uq))
    loss = output.conditioned_context.square().mean()
    loss = loss + output.yield_logits.square().mean()
    loss.backward()
    assert uq.grad is None
    assert context.grad is not None
    assert projector.component_projector[1].weight.grad is not None
    assert adapter.context_residual.weight.grad is not None


def test_task_context_is_trainable_but_cannot_break_zero_uq_identity():
    projector, adapter = _modules()
    context = torch.randn(2, 7, 16, requires_grad=True)
    task = torch.randn(2, 89, requires_grad=True)
    zero_output = adapter(context, projector(torch.zeros(2, 6, 4, 5, 3)), task)
    assert torch.equal(zero_output.conditioned_context, context)
    assert torch.count_nonzero(zero_output.trajectory_residual) == 0
    assert zero_output.yield_logits.argmax(dim=-1).tolist() == [0, 0]

    output = adapter(context, projector(torch.rand(2, 6, 4, 5, 3)), task)
    loss = output.conflict_logits.square().mean() + output.yield_logits.square().mean()
    loss.backward()
    assert task.grad is not None
    assert adapter.task_projector[1].weight.grad is not None


def test_selection_preserves_each_view_and_cell_provenance():
    projector, _ = _modules()
    uq = torch.zeros(1, 6, 4, 5, 3)
    for view in range(6):
        uq[0, view, view % 4, (view + 1) % 5] = 10.0 + view
        uq[0, view, (view + 1) % 4, (view + 2) % 5] = 5.0 + view
    selected = projector(uq)
    assert selected.tokens.shape == (1, 12, 16)
    assert selected.source_shape == (6, 4, 5)
    assert selected.camera_indices.reshape(1, 6, 2)[0, :, 0].tolist() == list(
        range(6)
    )
    expected_rows = [view % 4 for view in range(6)]
    expected_columns = [(view + 1) % 5 for view in range(6)]
    assert selected.row_indices.reshape(1, 6, 2)[0, :, 0].tolist() == expected_rows
    assert (
        selected.column_indices.reshape(1, 6, 2)[0, :, 0].tolist()
        == expected_columns
    )


def test_task_attention_can_be_scattered_to_multiview_grid():
    projector, adapter = _modules()
    uq = torch.rand(2, 6, 4, 5, 3)
    selected = projector(uq)
    output = adapter(torch.randn(2, 7, 16), selected)
    grid = scatter_selected_token_values(output.token_attention, selected)
    assert grid.shape == (2, 6, 4, 5)
    assert torch.allclose(
        grid.flatten(1).sum(dim=1), output.token_attention.sum(dim=1)
    )


def test_state_dict_round_trip_is_exact():
    projector, adapter = _modules()
    uq = torch.rand(1, 6, 4, 5, 3)
    context = torch.randn(1, 7, 16)
    reference_tokens = projector(uq)
    reference = adapter(context, reference_tokens)

    projector_copy, adapter_copy = _modules()
    projector_copy.load_state_dict(projector.state_dict(), strict=True)
    adapter_copy.load_state_dict(adapter.state_dict(), strict=True)
    copied_tokens = projector_copy(uq)
    copied = adapter_copy(context, copied_tokens)
    assert torch.equal(copied_tokens.flat_indices, reference_tokens.flat_indices)
    assert torch.equal(copied_tokens.tokens, reference_tokens.tokens)
    assert torch.equal(copied.conditioned_context, reference.conditioned_context)
    assert torch.equal(copied.yield_logits, reference.yield_logits)
    assert torch.equal(copied.conflict_logits, reference.conflict_logits)
    assert torch.equal(copied.trajectory_residual, reference.trajectory_residual)


def test_invalid_shapes_fail_closed():
    projector, adapter = _modules()
    try:
        projector(torch.zeros(1, 6, 4, 3))
    except ValueError as exc:
        assert "[B,V,H,W,K]" in str(exc)
    else:
        raise AssertionError("projector accepted a rank-4 UQ tensor")

    selected = projector(torch.zeros(1, 6, 4, 5, 3))
    try:
        adapter(torch.zeros(1, 7, 15), selected)
    except ValueError as exc:
        assert "feature dimension" in str(exc)
    else:
        raise AssertionError("adapter accepted a mismatched context dimension")


def test_stage2_checkpoint_round_trip_and_supervision_boundary(tmp_path):
    projector, adapter = _modules()
    path = tmp_path / "stage2.pt"
    payload = {
        "schema_version": STAGE2_TASK_FUSION_CHECKPOINT_SCHEMA,
        "projector_config": {
            "component_dim": 3,
            "model_dim": 16,
            "hidden_dim": 12,
            "max_views": 6,
            "tokens_per_view": 2,
        },
        "adapter_config": {
            "model_dim": 16,
            "num_heads": 4,
            "trajectory_steps": 6,
        },
        "projector_state": projector.state_dict(),
        "adapter_state": adapter.state_dict(),
        "supervision_contract": {
            "stage": "stage2_task_risk",
            "uses_density_uq": False,
            "uses_corruption_label": False,
            "updates_stage1_adapter": False,
        },
        "stage1_checkpoint_sha256": "1" * 64,
        "mechanism_report_sha256": "2" * 64,
    }
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded_projector, loaded_adapter, metadata = (
        load_stage2_task_fusion_checkpoint(path, expected_sha256=digest)
    )
    assert metadata["sha256"] == digest
    assert metadata["stage1_checkpoint_sha256"] == "1" * 64
    assert metadata["closed_loop_eligible"] is False
    assert all(
        torch.equal(value, loaded_projector.state_dict()[name])
        for name, value in projector.state_dict().items()
    )
    assert all(
        torch.equal(value, loaded_adapter.state_dict()[name])
        for name, value in adapter.state_dict().items()
    )

    payload["supervision_contract"]["uses_density_uq"] = True
    rejected = tmp_path / "density.pt"
    torch.save(payload, rejected)
    with pytest.raises(RuntimeError, match="supervision boundary"):
        load_stage2_task_fusion_checkpoint(rejected)

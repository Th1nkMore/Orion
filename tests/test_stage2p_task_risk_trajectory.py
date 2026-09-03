import hashlib

import pytest
import torch

from uq_estimator.stage2p_task_risk_trajectory import (
    CHECKPOINT_SCHEMA,
    TaskRiskMapTokenProjector,
    TaskRiskTrajectoryResponse,
    load_checkpoint,
)


def _modules():
    projector = TaskRiskMapTokenProjector(
        model_dim=32,
        hidden_dim=16,
        max_views=6,
        tokens_per_view=2,
    )
    response = TaskRiskTrajectoryResponse(
        model_dim=32,
        num_heads=4,
        trajectory_steps=6,
        lateral_bound_m=2.0,
        longitudinal_bound_m=24.0,
    )
    return projector, response


def test_zero_k_is_exact_identity_after_arbitrary_training_state():
    projector, response = _modules()
    for parameter in list(projector.parameters()) + list(response.parameters()):
        if parameter.requires_grad:
            torch.nn.init.uniform_(parameter, -0.5, 0.5)
    context = torch.randn(2, 7, 32)
    zero_k = torch.zeros(2, 6, 4, 4)
    output = response(context, projector(zero_k))
    assert torch.equal(output.conditioned_context, context)
    assert torch.count_nonzero(output.trajectory_residual) == 0
    assert torch.count_nonzero(output.token_attention) == 0
    assert torch.count_nonzero(output.global_gate) == 0


def test_nonzero_k_response_is_bounded_and_k_is_detached():
    projector, response = _modules()
    torch.nn.init.normal_(response.trajectory_head[-1].weight, std=2.0)
    task_risk = torch.rand(2, 6, 4, 4, requires_grad=True)
    context = torch.randn(2, 7, 32, requires_grad=True)
    output = response(context, projector(task_risk))
    output.trajectory_residual.sum().backward()
    assert task_risk.grad is None
    assert context.grad is not None
    assert output.trajectory_residual[..., 0].abs().max() <= 2.0
    assert output.trajectory_residual[..., 1].abs().max() <= 24.0
    assert torch.equal(output.conditioned_context, context)
    assert output.conditioned_context is context


def test_projector_rejects_invalid_task_risk():
    projector, _ = _modules()
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        projector(torch.full((1, 6, 4, 4), 1.1))
    with pytest.raises(ValueError, match="shape"):
        projector(torch.zeros(1, 6, 4))


def test_checkpoint_keeps_engineering_smoke_fail_closed(tmp_path):
    projector, response = _modules()
    path = tmp_path / "stage2p.pt"
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "engineering_smoke_only": True,
        "formal_stage2p_ready": False,
        "closed_loop_eligible": False,
        "projector_config": {
            "model_dim": 32,
            "hidden_dim": 16,
            "max_views": 6,
            "tokens_per_view": 2,
        },
        "response_config": {
            "model_dim": 32,
            "num_heads": 4,
            "trajectory_steps": 6,
            "lateral_bound_m": 2.0,
            "longitudinal_bound_m": 24.0,
        },
        "projector_state": projector.state_dict(),
        "response_state": response.state_dict(),
        "responsibility_contract": {
            "forward_inputs": [
                "frozen_orion_planning_context",
                "task_risk_k",
            ],
            "raw_observation_u_forward": False,
            "privileged_task_context_forward": False,
            "route_actor_ttc_outcome_forward": False,
        },
    }
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded_projector, loaded_response, metadata = load_checkpoint(
        path, expected_sha256=digest
    )
    assert loaded_projector.model_dim == 32
    assert loaded_response.model_dim == 32
    assert metadata["engineering_smoke_only"] is True
    assert metadata["closed_loop_eligible"] is False

    payload["responsibility_contract"]["privileged_task_context_forward"] = True
    rejected = tmp_path / "rejected.pt"
    torch.save(payload, rejected)
    with pytest.raises(RuntimeError, match="responsibility"):
        load_checkpoint(rejected)

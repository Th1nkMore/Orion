import json

import pytest
import torch

from uq_estimator.privileged_yield_labels import SCHEMA_VERSION as LABEL_SCHEMA
from uq_estimator.spatial_task_fusion import TaskRiskTrajectoryOutput
from uq_estimator.stage2_task_training import (
    SCHEMA_VERSION,
    Stage2DataIntegrityError,
    Stage2TaskResponseDataset,
    build_stage2_manifest,
    stage2_task_response_loss,
    validate_stage2_record,
)
from uq_estimator.stage2_artifact_capture import Stage2ArtifactWriter


def write_artifacts(tmp_path, step=10, route_group="Town05/Route197/seed0"):
    writer = Stage2ArtifactWriter(
        tmp_path / "capture",
        route_group=route_group,
        uq_source="oracle_spatial_uq",
        camera_order=[f"camera_{index}" for index in range(6)],
    )
    writer.write(
        step=step,
        planning_context=torch.randn(4, 256),
        task_context=torch.randn(89),
        observation_uq=torch.rand(6, 10, 10, 3),
    )
    return writer.finalize()


def write_label(tmp_path, step=10):
    path = tmp_path / "labels.jsonl"
    record = {
        "schema_version": LABEL_SCHEMA,
        "source": {"step": step, "sim_time_seconds": 0.5},
        "supervision_contract": {
            "stage": "stage2_task_risk",
            "uses_observation_uq_target": False,
            "uses_density_uq": False,
            "uses_corruption_label": False,
        },
        "base_plan_cumulative_m": [[0.0, float(i)] for i in range(1, 7)],
        "yield_label": {"state": "hold", "state_index": 2},
        "conflict": {"per_horizon_conflict": [False, True, True, False, False, False]},
        "trajectory_residual_m": [[0.0, -float(i)] for i in range(6)],
    }
    path.write_text(json.dumps(record) + "\n")
    return path


def write_mechanism_report(tmp_path, success=True):
    path = tmp_path / ("mechanism_pass.json" if success else "mechanism_fail.json")
    path.write_text(json.dumps({
        "primary_success": success,
        "decision": (
            "planning_mechanism_upper_bound_supported"
            if success else "do_not_train_or_control_with_learned_uq_yet"
        ),
    }))
    return path


def test_build_and_load_stage2_manifest(tmp_path):
    manifest = tmp_path / "stage2.jsonl"
    summary = build_stage2_manifest(
        write_label(tmp_path),
        write_artifacts(tmp_path),
        manifest,
        route_group="Town05/Route197/seed0",
        mechanism_report_path=write_mechanism_report(tmp_path),
    )
    assert summary["sample_count"] == 1
    dataset = Stage2TaskResponseDataset(manifest, model_dim=256)
    sample = dataset[0]
    assert sample["planning_context"].shape == (4, 256)
    assert sample["task_context"].shape == (89,)
    assert sample["observation_uq"].shape == (6, 10, 10, 3)
    assert sample["yield_target"].item() == 2
    assert sample["trajectory_target"].shape == (6, 2)


def test_stage2_record_rejects_density_or_corruption_supervision(tmp_path):
    record = {
        "schema_version": SCHEMA_VERSION,
        "supervision_contract": {
            "stage": "stage2_task_risk",
            "uses_observation_uq_target": False,
            "uses_density_uq": True,
            "uses_corruption_label": False,
        },
    }
    with pytest.raises(Stage2DataIntegrityError, match="exclude"):
        validate_stage2_record(record)


def test_builder_rejects_failed_planning_oracle(tmp_path):
    with pytest.raises(Stage2DataIntegrityError, match="failed planning"):
        build_stage2_manifest(
            write_label(tmp_path),
            write_artifacts(tmp_path),
            tmp_path / "rejected.jsonl",
            route_group="Town05/Route197/seed0",
            mechanism_report_path=write_mechanism_report(tmp_path, success=False),
        )


def test_multitask_loss_is_finite_and_backpropagates():
    batch_size, tokens, dim = 2, 4, 8
    planning_context = torch.randn(batch_size, tokens, dim)
    yield_logits = torch.randn(batch_size, 4, requires_grad=True)
    conflict_logits = torch.randn(batch_size, 6, requires_grad=True)
    residual = torch.randn(batch_size, 6, 2, requires_grad=True)
    conditioned = planning_context.clone().requires_grad_(True)
    output = TaskRiskTrajectoryOutput(
        conditioned_context=conditioned,
        yield_logits=yield_logits,
        conflict_logits=conflict_logits,
        trajectory_residual=residual,
        token_attention=torch.zeros(batch_size, 6),
    )
    losses = stage2_task_response_loss(output, {
        "planning_context": planning_context,
        "yield_target": torch.tensor([0, 2]),
        "conflict_target": torch.tensor([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
        ], dtype=torch.float32),
        "trajectory_target": torch.zeros(batch_size, 6, 2),
    })
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert yield_logits.grad is not None
    assert conflict_logits.grad is not None
    assert residual.grad is not None

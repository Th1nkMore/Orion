import json

import pytest
import torch

from uq_estimator.stage2_artifact_capture import (
    ARTIFACT_INDEX_SCHEMA,
    Stage2ArtifactWriter,
    sha256_file,
)


def test_writer_preserves_full_spatial_map_and_context_provenance(tmp_path):
    root = tmp_path / "capture"
    writer = Stage2ArtifactWriter(
        root,
        route_group="Town04/Route147/seed0",
        uq_source="learned_stage1_spatial_uq",
        camera_order=["front", "left"],
        stage1_checkpoint_sha256="1" * 64,
    )
    record = writer.write(
        step=7,
        planning_context=torch.randn(1, 12, 256, requires_grad=True),
        task_context=torch.randn(1, 89, requires_grad=True),
        observation_uq=torch.rand(1, 2, 4, 5, 3, requires_grad=True),
        raw_observation_uq=torch.rand(1, 2, 4, 5, 3),
        metadata={"sim_time_seconds": 0.35},
    )
    index_path = writer.finalize()
    payload = json.loads(index_path.read_text())
    assert payload["schema_version"] == ARTIFACT_INDEX_SCHEMA
    assert payload["record_count"] == 1
    assert record["planning_context_shape"] == [12, 256]
    assert record["task_context_shape"] == [89]
    assert record["observation_uq_shape"] == [2, 4, 5, 3]
    assert record["planning_context_sha256"] == sha256_file(
        record["planning_context_path"]
    )
    assert not torch.load(
        record["planning_context_path"], weights_only=False
    )["planning_context"].requires_grad


def test_writer_refuses_overwrite_duplicate_and_bad_provenance(tmp_path):
    root = tmp_path / "capture"
    writer = Stage2ArtifactWriter(
        root,
        route_group="route",
        uq_source="oracle_spatial_uq",
        camera_order=["front"],
    )
    context = torch.zeros(2, 256)
    task = torch.zeros(89)
    uq = torch.zeros(1, 2, 2, 3)
    writer.write(
        step=0,
        planning_context=context,
        task_context=task,
        observation_uq=uq,
    )
    with pytest.raises(ValueError, match="duplicated"):
        writer.write(
            step=0,
            planning_context=context,
            task_context=task,
            observation_uq=uq,
        )
    writer.finalize()
    with pytest.raises(FileExistsError, match="overwrite"):
        Stage2ArtifactWriter(
            root,
            route_group="route",
            uq_source="oracle_spatial_uq",
            camera_order=["front"],
        )
    with pytest.raises(ValueError, match="must not claim"):
        Stage2ArtifactWriter(
            tmp_path / "bad",
            route_group="route",
            uq_source="oracle_spatial_uq",
            camera_order=["front"],
            stage1_checkpoint_sha256="1" * 64,
        )

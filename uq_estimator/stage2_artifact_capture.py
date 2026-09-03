"""Auditable artifact writer for real ORION Stage-2 training inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ARTIFACT_INDEX_SCHEMA = "orion.stage2_feature_artifact_index.v2"
ALLOWED_UQ_SOURCES = (
    "learned_stage1_spatial_uq",
    "oracle_spatial_uq",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_sample(value: torch.Tensor, name: str, rank: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError("%s must be a floating tensor" % name)
    detached = value.detach().float()
    if detached.ndim == rank + 1:
        if detached.shape[0] != 1:
            raise ValueError("%s capture requires batch size one" % name)
        detached = detached[0]
    if detached.ndim != rank or min(detached.shape) <= 0:
        raise ValueError("%s has the wrong rank or an empty axis" % name)
    if not bool(torch.isfinite(detached).all()):
        raise ValueError("%s must be finite" % name)
    return detached.cpu().half().contiguous()


class Stage2ArtifactWriter:
    """Write immutable per-frame planning context and spatial-UQ tensors."""

    def __init__(
        self,
        root: str | Path,
        *,
        route_group: str,
        uq_source: str,
        camera_order: Sequence[str],
        stage1_checkpoint_sha256: str | None = None,
    ) -> None:
        if not str(route_group).strip():
            raise ValueError("route_group must be non-empty")
        if uq_source not in ALLOWED_UQ_SOURCES:
            raise ValueError("unsupported Stage-2 UQ artifact source")
        if uq_source == "learned_stage1_spatial_uq":
            if not isinstance(stage1_checkpoint_sha256, str) or len(stage1_checkpoint_sha256) != 64:
                raise ValueError("learned Stage-1 artifacts require checkpoint SHA256")
        elif stage1_checkpoint_sha256 is not None:
            raise ValueError("oracle artifacts must not claim a Stage-1 checkpoint")
        if not camera_order or len(set(camera_order)) != len(camera_order):
            raise ValueError("camera_order must be non-empty and unique")
        self.root = Path(root)
        if self.root.exists():
            raise FileExistsError("refusing to overwrite Stage-2 artifact root: %s" % self.root)
        self.root.mkdir(parents=True)
        self.context_root = self.root / "planning_context"
        self.task_context_root = self.root / "task_context"
        self.uq_root = self.root / "observation_uq"
        self.raw_uq_root = self.root / "raw_observation_uq"
        self.context_root.mkdir()
        self.task_context_root.mkdir()
        self.uq_root.mkdir()
        self.raw_uq_root.mkdir()
        self.route_group = str(route_group)
        self.uq_source = str(uq_source)
        self.camera_order = tuple(str(value) for value in camera_order)
        self.stage1_checkpoint_sha256 = stage1_checkpoint_sha256
        self.records: list[dict[str, Any]] = []
        self._steps: set[int] = set()
        self._finalized = False

    def write(
        self,
        *,
        step: int,
        planning_context: torch.Tensor,
        task_context: torch.Tensor,
        observation_uq: torch.Tensor,
        raw_observation_uq: torch.Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("cannot append to a finalized Stage-2 artifact index")
        step = int(step)
        if step < 0 or step in self._steps:
            raise ValueError("Stage-2 artifact step is negative or duplicated")
        context = _single_sample(planning_context, "planning_context", 2)
        task = _single_sample(task_context, "task_context", 1)
        observation = _single_sample(observation_uq, "observation_uq", 4)
        if context.shape[-1] != 256:
            raise ValueError("planning_context must use ORION's 256-D pre-LLM space")
        if task.shape != (89,):
            raise ValueError("task_context must contain ORION's 89-D command/ego state")
        if observation.shape[0] != len(self.camera_order) or observation.shape[-1] != 3:
            raise ValueError("observation_uq must match camera order and three components")
        if bool((observation < 0).any()):
            raise ValueError("observation_uq must be non-negative")
        raw = None
        if raw_observation_uq is not None:
            raw = _single_sample(raw_observation_uq, "raw_observation_uq", 4)
            if raw.shape != observation.shape or bool((raw < 0).any()):
                raise ValueError("raw observation UQ shape/value differs")

        stem = "%06d.pt" % step
        context_path = self.context_root / stem
        task_context_path = self.task_context_root / stem
        uq_path = self.uq_root / stem
        torch.save({"planning_context": context}, context_path)
        torch.save({"task_context": task}, task_context_path)
        torch.save({"observation_uq": observation}, uq_path)
        raw_path = None
        if raw is not None:
            raw_path = self.raw_uq_root / stem
            torch.save({"raw_observation_uq": raw}, raw_path)
        record = {
            "step": step,
            "route_group": self.route_group,
            "uq_source": self.uq_source,
            "planning_context_path": str(context_path.resolve()),
            "planning_context_sha256": sha256_file(context_path),
            "planning_context_shape": list(context.shape),
            "task_context_path": str(task_context_path.resolve()),
            "task_context_sha256": sha256_file(task_context_path),
            "task_context_shape": list(task.shape),
            "observation_uq_path": str(uq_path.resolve()),
            "observation_uq_sha256": sha256_file(uq_path),
            "observation_uq_shape": list(observation.shape),
            "raw_observation_uq_path": (
                str(raw_path.resolve()) if raw_path is not None else None
            ),
            "raw_observation_uq_sha256": (
                sha256_file(raw_path) if raw_path is not None else None
            ),
            "metadata": dict(metadata or {}),
        }
        self.records.append(record)
        self._steps.add(step)
        return record

    def finalize(self) -> Path:
        if self._finalized:
            return self.root / "artifact_index.json"
        if not self.records:
            raise RuntimeError("refusing to finalize an empty Stage-2 artifact index")
        payload = {
            "schema_version": ARTIFACT_INDEX_SCHEMA,
            "route_group": self.route_group,
            "uq_source": self.uq_source,
            "stage1_checkpoint_sha256": self.stage1_checkpoint_sha256,
            "camera_order": list(self.camera_order),
            "record_count": len(self.records),
            "records": sorted(self.records, key=lambda row: int(row["step"])),
        }
        index_path = self.root / "artifact_index.json"
        index_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self._finalized = True
        return index_path


__all__ = [
    "ALLOWED_UQ_SOURCES",
    "ARTIFACT_INDEX_SCHEMA",
    "Stage2ArtifactWriter",
    "sha256_file",
]

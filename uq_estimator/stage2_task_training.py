"""Dataset and losses for Stage-2 task-risk/trajectory learning.

Stage 2 consumes a frozen Stage-1 spatial observation-evidence tensor plus an
ORION planning context.  Its supervision comes only from privileged dynamic
conflict labels and safe trajectory targets.  Corruption labels and legacy
Density UQ are explicitly forbidden from this manifest contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from uq_estimator.privileged_yield_labels import YIELD_STATES
from uq_estimator.spatial_task_fusion import TaskRiskTrajectoryOutput
from uq_estimator.stage2_artifact_capture import (
    ALLOWED_UQ_SOURCES,
    ARTIFACT_INDEX_SCHEMA,
    sha256_file,
)


SCHEMA_VERSION = "orion.stage2_task_response_sample.v2"


class Stage2DataIntegrityError(RuntimeError):
    pass


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_tensor(value, shape, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tuple(tensor.shape) != tuple(shape):
        raise Stage2DataIntegrityError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise Stage2DataIntegrityError(f"{name} must be finite")
    return tensor


def validate_stage2_record(record: Mapping, *, check_files: bool = False) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise Stage2DataIntegrityError("unexpected Stage-2 sample schema")
    contract = record.get("supervision_contract") or {}
    required_false = (
        "uses_observation_uq_target",
        "uses_density_uq",
        "uses_corruption_label",
    )
    if contract.get("stage") != "stage2_task_risk":
        raise Stage2DataIntegrityError("sample is not Stage-2 task supervision")
    if any(contract.get(key) is not False for key in required_false):
        raise Stage2DataIntegrityError(
            "Stage-2 manifest must exclude UQ, Density, and corruption targets"
        )
    gate = record.get("mechanism_gate") or {}
    if gate.get("oracle_primary_success") is not True:
        raise Stage2DataIntegrityError(
            "Stage-2 training requires a successful planning-mechanism oracle"
        )
    if not isinstance(gate.get("report_sha256"), str) or len(gate["report_sha256"]) != 64:
        raise Stage2DataIntegrityError("mechanism gate report hash is missing")
    artifacts = record.get("artifacts") or {}
    for key in (
        "planning_context_path",
        "task_context_path",
        "observation_uq_path",
    ):
        value = artifacts.get(key)
        if not isinstance(value, str) or not value:
            raise Stage2DataIntegrityError(f"missing artifact path: {key}")
        if check_files and not Path(value).is_file():
            raise Stage2DataIntegrityError(f"artifact does not exist: {value}")
    for path_key, hash_key in (
        ("planning_context_path", "planning_context_sha256"),
        ("task_context_path", "task_context_sha256"),
        ("observation_uq_path", "observation_uq_sha256"),
    ):
        digest = artifacts.get(hash_key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise Stage2DataIntegrityError(f"missing artifact hash: {hash_key}")
        if check_files and sha256_file(artifacts[path_key]) != digest:
            raise Stage2DataIntegrityError(f"artifact hash differs: {path_key}")
    uq_input = record.get("uq_input_contract") or {}
    if uq_input.get("source") not in ALLOWED_UQ_SOURCES:
        raise Stage2DataIntegrityError("Stage-2 UQ input source is invalid")
    stage1_sha = uq_input.get("stage1_checkpoint_sha256")
    if uq_input["source"] == "learned_stage1_spatial_uq":
        if not isinstance(stage1_sha, str) or len(stage1_sha) != 64:
            raise Stage2DataIntegrityError(
                "learned Stage-2 input lacks Stage-1 checkpoint provenance"
            )
    elif stage1_sha is not None:
        raise Stage2DataIntegrityError(
            "oracle Stage-2 input must not claim a learned Stage-1 checkpoint"
        )
    labels = record.get("labels") or {}
    state = labels.get("yield_state")
    state_index = labels.get("yield_state_index")
    if state not in YIELD_STATES or state_index != YIELD_STATES.index(state):
        raise Stage2DataIntegrityError("yield state/index mismatch")
    conflict = labels.get("per_horizon_conflict")
    if not isinstance(conflict, list) or len(conflict) != 6:
        raise Stage2DataIntegrityError("per_horizon_conflict must contain six labels")
    if any(value not in (False, True, 0, 1) for value in conflict):
        raise Stage2DataIntegrityError("conflict labels must be binary")
    _finite_tensor(labels.get("trajectory_residual_m"), (6, 2), "trajectory residual")
    _finite_tensor(record.get("base_plan_cumulative_m"), (6, 2), "base plan")


def build_stage2_manifest(
    privileged_labels_path: str | Path,
    artifact_index_path: str | Path,
    output_path: str | Path,
    *,
    route_group: str,
    mechanism_report_path: str | Path,
) -> dict[str, Any]:
    """Join privileged labels with independently extracted frozen features."""

    label_records = [
        json.loads(line)
        for line in Path(privileged_labels_path).read_text().splitlines()
        if line.strip()
    ]
    artifact_payload = json.loads(Path(artifact_index_path).read_text())
    if artifact_payload.get("schema_version") != ARTIFACT_INDEX_SCHEMA:
        raise Stage2DataIntegrityError("unexpected artifact-index schema")
    if artifact_payload.get("route_group") != route_group:
        raise Stage2DataIntegrityError("artifact-index route group differs")
    uq_source = artifact_payload.get("uq_source")
    if uq_source not in ALLOWED_UQ_SOURCES:
        raise Stage2DataIntegrityError("artifact-index UQ source is invalid")
    stage1_checkpoint_sha256 = artifact_payload.get("stage1_checkpoint_sha256")
    if uq_source == "learned_stage1_spatial_uq":
        if (
            not isinstance(stage1_checkpoint_sha256, str)
            or len(stage1_checkpoint_sha256) != 64
        ):
            raise Stage2DataIntegrityError(
                "learned artifact index lacks Stage-1 checkpoint provenance"
            )
    elif stage1_checkpoint_sha256 is not None:
        raise Stage2DataIntegrityError(
            "oracle artifact index must not claim a Stage-1 checkpoint"
        )
    if artifact_payload.get("record_count") != len(
        artifact_payload.get("records", [])
    ):
        raise Stage2DataIntegrityError("artifact-index record count differs")
    artifact_by_step = {
        int(record["step"]): record for record in artifact_payload.get("records", [])
    }
    if len(artifact_by_step) != len(artifact_payload.get("records", [])):
        raise Stage2DataIntegrityError("duplicate step in artifact index")
    mechanism_report_path = Path(mechanism_report_path)
    mechanism_report = json.loads(mechanism_report_path.read_text())
    if mechanism_report.get("primary_success") is not True:
        raise Stage2DataIntegrityError(
            "refusing Stage-2 data from a failed planning-mechanism oracle"
        )
    mechanism_gate = {
        "report_path": str(mechanism_report_path),
        "report_sha256": _sha256(mechanism_report_path),
        "oracle_primary_success": True,
        "decision": mechanism_report.get("decision"),
    }
    output_records = []
    missing_steps = []
    for label_record in label_records:
        step = int(label_record["source"]["step"])
        artifact = artifact_by_step.get(step)
        if artifact is None:
            missing_steps.append(step)
            continue
        yield_label = label_record["yield_label"]
        conflict = label_record["conflict"]
        if (
            artifact.get("route_group") != route_group
            or artifact.get("uq_source") != uq_source
        ):
            raise Stage2DataIntegrityError(
                "artifact record provenance differs from its index"
            )
        record = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": f"{route_group}#step={step}",
            "route_group": route_group,
            "source": label_record["source"],
            "supervision_contract": label_record["supervision_contract"],
            "mechanism_gate": mechanism_gate,
            "uq_input_contract": {
                "source": uq_source,
                "stage1_checkpoint_sha256": stage1_checkpoint_sha256,
                "map_is_task_risk_target": False,
                "map_updates_stage1_adapter": False,
            },
            "artifacts": {
                "planning_context_path": str(artifact["planning_context_path"]),
                "planning_context_sha256": str(
                    artifact["planning_context_sha256"]
                ),
                "task_context_path": str(artifact["task_context_path"]),
                "task_context_sha256": str(
                    artifact["task_context_sha256"]
                ),
                "observation_uq_path": str(artifact["observation_uq_path"]),
                "observation_uq_sha256": str(
                    artifact["observation_uq_sha256"]
                ),
            },
            "base_plan_cumulative_m": label_record["base_plan_cumulative_m"],
            "labels": {
                "yield_state": yield_label["state"],
                "yield_state_index": int(yield_label["state_index"]),
                "per_horizon_conflict": conflict["per_horizon_conflict"],
                "trajectory_residual_m": label_record["trajectory_residual_m"],
            },
        }
        validate_stage2_record(record, check_files=True)
        output_records.append(record)
    if missing_steps:
        raise Stage2DataIntegrityError(
            f"artifact index is missing {len(missing_steps)} label steps"
        )
    if not output_records:
        raise Stage2DataIntegrityError("Stage-2 manifest would be empty")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for record in output_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_path": str(output_path),
        "sample_count": len(output_records),
        "route_group": route_group,
        "state_counts": {
            state: sum(
                record["labels"]["yield_state"] == state
                for record in output_records
            )
            for state in YIELD_STATES
        },
        "mechanism_gate": mechanism_gate,
    }


def _load_tensor(path: str, key: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping):
        if key not in payload:
            raise Stage2DataIntegrityError(f"{path} is missing tensor key {key}")
        payload = payload[key]
    if not isinstance(payload, torch.Tensor):
        raise Stage2DataIntegrityError(f"{path} does not contain a tensor")
    return payload.detach().float()


class Stage2TaskResponseDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        model_dim: int = 256,
        observation_components: int = 3,
    ) -> None:
        self.records = [
            json.loads(line)
            for line in Path(manifest_path).read_text().splitlines()
            if line.strip()
        ]
        if not self.records:
            raise Stage2DataIntegrityError("Stage-2 manifest is empty")
        self.model_dim = int(model_dim)
        self.observation_components = int(observation_components)
        for record in self.records:
            validate_stage2_record(record, check_files=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        artifacts = record["artifacts"]
        context = _load_tensor(
            artifacts["planning_context_path"], "planning_context"
        )
        task_context = _load_tensor(
            artifacts["task_context_path"], "task_context"
        )
        observation = _load_tensor(
            artifacts["observation_uq_path"], "observation_uq"
        )
        if context.ndim != 2 or context.shape[0] <= 0:
            raise Stage2DataIntegrityError("planning context must have shape [N,D]")
        if context.shape[1] != self.model_dim:
            raise Stage2DataIntegrityError("planning context dimension mismatch")
        if task_context.shape != (89,) or not bool(torch.isfinite(task_context).all()):
            raise Stage2DataIntegrityError("task context must have finite [89] shape")
        if observation.ndim != 4 or observation.shape[-1] != self.observation_components:
            raise Stage2DataIntegrityError(
                "observation UQ must have shape [V,H,W,K]"
            )
        if min(observation.shape[:3]) <= 0:
            raise Stage2DataIntegrityError("observation UQ spatial shape is empty")
        if not bool(torch.isfinite(context).all()):
            raise Stage2DataIntegrityError("planning context must be finite")
        if not bool(torch.isfinite(observation).all()) or bool((observation < 0).any()):
            raise Stage2DataIntegrityError(
                "observation UQ must be finite and non-negative"
            )
        labels = record["labels"]
        return {
            "sample_id": record["sample_id"],
            "route_group": record["route_group"],
            "planning_context": context,
            "task_context": task_context,
            "observation_uq": observation,
            "base_plan": _finite_tensor(
                record["base_plan_cumulative_m"], (6, 2), "base plan"
            ),
            "yield_target": torch.tensor(
                labels["yield_state_index"], dtype=torch.long
            ),
            "conflict_target": torch.tensor(
                labels["per_horizon_conflict"], dtype=torch.float32
            ),
            "trajectory_target": _finite_tensor(
                labels["trajectory_residual_m"], (6, 2), "trajectory residual"
            ),
        }


@dataclass(frozen=True)
class Stage2LossWeights:
    yield_state: float = 1.0
    future_conflict: float = 1.0
    trajectory: float = 2.0
    go_identity: float = 1.0
    context_identity: float = 0.1

    def __post_init__(self) -> None:
        if any(float(value) < 0 for value in self.__dict__.values()):
            raise ValueError("Stage-2 loss weights must be non-negative")


def stage2_task_response_loss(
    output: TaskRiskTrajectoryOutput,
    batch: Mapping[str, torch.Tensor],
    *,
    weights: Stage2LossWeights | None = None,
) -> dict[str, torch.Tensor]:
    """Compute task losses without backpropagating into Stage-1 UQ maps."""

    weights = weights or Stage2LossWeights()
    yield_target = batch["yield_target"].long()
    conflict_target = batch["conflict_target"].to(output.conflict_logits)
    trajectory_target = batch["trajectory_target"].to(output.trajectory_residual)
    planning_context = batch["planning_context"].to(output.conditioned_context)
    batch_size = yield_target.shape[0]
    if output.yield_logits.shape != (batch_size, len(YIELD_STATES)):
        raise ValueError("yield logits shape mismatch")
    if output.conflict_logits.shape != conflict_target.shape:
        raise ValueError("conflict logits shape mismatch")
    if output.trajectory_residual.shape != trajectory_target.shape:
        raise ValueError("trajectory residual shape mismatch")
    if output.conditioned_context.shape != planning_context.shape:
        raise ValueError("conditioned context shape mismatch")

    yield_loss = F.cross_entropy(output.yield_logits, yield_target)
    conflict_loss = F.binary_cross_entropy_with_logits(
        output.conflict_logits, conflict_target
    )
    trajectory_loss = F.smooth_l1_loss(
        output.trajectory_residual, trajectory_target
    )
    go_mask = yield_target == YIELD_STATES.index("go")
    if bool(go_mask.any()):
        go_identity = output.trajectory_residual[go_mask].abs().mean()
        context_identity = (
            output.conditioned_context[go_mask] - planning_context[go_mask]
        ).abs().mean()
    else:
        go_identity = output.trajectory_residual.sum() * 0.0
        context_identity = output.conditioned_context.sum() * 0.0
    total = (
        weights.yield_state * yield_loss
        + weights.future_conflict * conflict_loss
        + weights.trajectory * trajectory_loss
        + weights.go_identity * go_identity
        + weights.context_identity * context_identity
    )
    return {
        "loss": total,
        "loss_yield_state": yield_loss,
        "loss_future_conflict": conflict_loss,
        "loss_trajectory": trajectory_loss,
        "loss_go_identity": go_identity,
        "loss_context_identity": context_identity,
    }

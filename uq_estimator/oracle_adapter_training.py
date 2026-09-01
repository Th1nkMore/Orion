"""Training utilities for the same-rollout oracle trajectory adapter.

This module is intentionally independent of ORION's existing training entry
points.  It trains only :class:`PathRiskTrajectoryAdapter` on exported
closed-loop trajectories.  Successful controlled-stop oracle rollouts are the
only source of trajectory-imitation targets.  Off-policy failures and failed
oracle rollouts remain diagnostic and contribute no trajectory gradients.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from uq_estimator.oracle_adapter_dataset import validate_sample
from uq_estimator.trajectory_adapter import PathRiskTrajectoryAdapter


ROLE_ORACLE_IMITATION = "oracle_imitation"
ROLE_PRESERVATION = "preservation"
ROLE_FAILED_ORACLE = "failed_oracle_diagnostic"
ROLE_OFF_FAILURE = "off_failure_diagnostic"
ROLE_DIAGNOSTIC = "diagnostic_only"
DIAGNOSTIC_ROLES = {ROLE_FAILED_ORACLE, ROLE_OFF_FAILURE, ROLE_DIAGNOSTIC}


class OracleAdapterTrainingError(ValueError):
    """Raised when the dataset cannot support the requested training claim."""


@dataclass(frozen=True)
class TrainingExample:
    sample_id: str
    rollout_id: str
    route_key: str
    role: str
    base_trajectory: list[list[float]]
    target_trajectory: list[list[float]]
    trajectory_mask: list[int]
    path_risk: float
    stop_target: float
    imitation_weight: float
    preservation_weight: float
    stop_weight: float
    recover: bool


@dataclass(frozen=True)
class SplitResult:
    train_indices: list[int]
    validation_indices: list[int]
    train_routes: list[str]
    validation_routes: list[str]
    mode: str


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleAdapterTrainingError(f"Cannot read {path}: {exc}") from exc


def load_exported_samples(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Load and structurally validate an exported JSONL dataset."""
    root = Path(dataset_dir)
    manifest = _load_json(root / "dataset_manifest.json")
    samples_path = root / manifest.get("samples_file", "samples.jsonl")
    if not samples_path.is_file():
        raise OracleAdapterTrainingError(f"Missing samples file: {samples_path}")
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(samples_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OracleAdapterTrainingError(
                f"Invalid JSONL at {samples_path}:{line_number}"
            ) from exc
        validate_sample(sample, check_files=True)
        sample_id = str(sample["sample_id"])
        if sample_id in seen:
            raise OracleAdapterTrainingError(f"Duplicate sample_id {sample_id}")
        seen.add(sample_id)
        samples.append(sample)
    if len(samples) != int(manifest.get("sample_count", -1)):
        raise OracleAdapterTrainingError(
            f"Manifest declares {manifest.get('sample_count')} samples, "
            f"loaded {len(samples)}"
        )
    if not samples:
        raise OracleAdapterTrainingError("Dataset contains no samples")
    return samples


def route_key(sample: dict[str, Any]) -> str:
    """Return a repetition-invariant Town/route split key."""
    terminal = sample.get("terminal_result", {})
    town = str(terminal.get("town") or "unknown-town")
    route_id = str(terminal.get("route_id") or "unknown-route")
    route_id = re.sub(r"_rep\d+$", "", route_id)
    return f"{town}/{route_id}"


def _terminal_success(sample: dict[str, Any]) -> bool:
    terminal = sample.get("terminal_result", {})
    collisions = terminal.get("collisions", {})
    infractions = terminal.get("infractions", {})
    collision_count = sum(int(collisions.get(key, 0)) for key in collisions)
    liveness_count = sum(
        int(infractions.get(key, 0))
        for key in ("vehicle_blocked", "route_timeout", "scenario_timeouts")
    )
    return bool(
        terminal.get("eligible")
        and terminal.get("entry_status") == "Finished"
        and terminal.get("route_status") == "Completed"
        and float(terminal.get("route_completion", 0.0)) >= 99.999
        and collision_count == 0
        and liveness_count == 0
    )


def assign_rollout_roles(
    samples: Sequence[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign one auditable training role to each complete rollout."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["source"]["rollout_id"])].append(sample)

    roles: dict[str, str] = {}
    for rollout_id, rows in grouped.items():
        variants = {row["condition"]["variant"] for row in rows}
        relevance = {row["condition"]["relevance"] for row in rows}
        risk_modes = {row["controls"]["risk_mode"] for row in rows}
        names = {row["condition"]["name"] for row in rows}
        route_keys = {route_key(row) for row in rows}
        if any(len(values) != 1 for values in (variants, relevance, risk_modes, names, route_keys)):
            raise OracleAdapterTrainingError(
                f"Rollout {rollout_id} mixes incompatible condition metadata"
            )
        if any(row["terminal_result"] != rows[0]["terminal_result"] for row in rows[1:]):
            raise OracleAdapterTrainingError(
                f"Rollout {rollout_id} has inconsistent terminal outcomes"
            )

        variant = next(iter(variants))
        relevance_value = next(iter(relevance))
        risk_mode = next(iter(risk_modes))
        condition_name = next(iter(names))
        success = _terminal_success(rows[0])
        has_oracle_stop = any(
            row["labels"]["stop_required"]
            and float(row["oracle"]["path_risk"]) > 0.0
            for row in rows
        )
        is_oracle = risk_mode == "oracle" and "oracle" in condition_name

        if (
            is_oracle
            and success
            and variant == "hazard"
            and relevance_value == "on_path"
            and has_oracle_stop
        ):
            role = ROLE_ORACLE_IMITATION
        elif success and (
            variant == "nohazard"
            or relevance_value == "off_path"
            or all(float(row["oracle"]["path_risk"]) <= 0.0 for row in rows)
        ):
            role = ROLE_PRESERVATION
        elif is_oracle:
            role = ROLE_FAILED_ORACLE
        elif risk_mode == "off":
            role = ROLE_OFF_FAILURE
        else:
            role = ROLE_DIAGNOSTIC
        roles[rollout_id] = role

    role_counts = Counter(roles.values())
    sample_role_counts = Counter(
        roles[str(sample["source"]["rollout_id"])] for sample in samples
    )
    return roles, {
        "rollout_count": len(grouped),
        "rollout_role_counts": dict(sorted(role_counts.items())),
        "sample_role_counts": dict(sorted(sample_role_counts.items())),
    }


def _trajectory(value: Any, field: str) -> list[list[float]]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (6, 2) or not np.isfinite(array).all():
        raise OracleAdapterTrainingError(f"{field} must be finite [6, 2]")
    return array.tolist()


def prepare_examples(
    samples: Sequence[dict[str, Any]],
    roles: dict[str, str],
) -> tuple[list[TrainingExample], dict[str, Any]]:
    """Turn JSON samples into role-gated optimization examples."""
    examples: list[TrainingExample] = []
    for sample in samples:
        rollout_id = str(sample["source"]["rollout_id"])
        role = roles[rollout_id]
        base = _trajectory(
            sample["expert"]["base_plan_displacements_m"],
            "base_plan_displacements_m",
        )
        executed_expert = _trajectory(
            sample["expert"]["trajectory_displacements_m"],
            "trajectory_displacements_m",
        )
        mask = [int(value) for value in sample["expert"]["trajectory_mask"]]
        if len(mask) != 6 or any(value not in (0, 1) for value in mask):
            raise OracleAdapterTrainingError("trajectory_mask must be six binary values")
        path_risk = float(sample["oracle"]["path_risk"])
        positive_active = role == ROLE_ORACLE_IMITATION and path_risk > 0.0
        preservation = (
            role == ROLE_PRESERVATION
            or (role == ROLE_ORACLE_IMITATION and not positive_active)
        )
        diagnostic = role in DIAGNOSTIC_ROLES
        if positive_active:
            target = executed_expert
            imitation_weight = 1.0
            preservation_weight = 0.0
            stop_target = float(bool(sample["labels"]["stop_required"]))
            stop_weight = 1.0
        elif preservation:
            # Crucially, do not imitate the executed trajectory in preservation
            # frames: a future oracle event can already have changed it.  The
            # target is the frozen ORION base plan itself.
            target = base
            imitation_weight = 0.0
            preservation_weight = 1.0
            stop_target = 0.0
            stop_weight = 1.0
        else:
            # Failed/off rollouts are retained for audit counts only.  They do
            # not contribute either failed future poses or a misleading
            # high-risk identity target.
            assert diagnostic
            target = base
            imitation_weight = 0.0
            preservation_weight = 0.0
            stop_target = 0.0
            stop_weight = 0.0
        examples.append(
            TrainingExample(
                sample_id=str(sample["sample_id"]),
                rollout_id=rollout_id,
                route_key=route_key(sample),
                role=role,
                base_trajectory=base,
                target_trajectory=target,
                trajectory_mask=mask,
                path_risk=path_risk,
                stop_target=stop_target,
                imitation_weight=imitation_weight,
                preservation_weight=preservation_weight,
                stop_weight=stop_weight,
                recover=bool(sample["labels"]["recover"]),
            )
        )

    audit = {
        "example_count": len(examples),
        "imitation_examples": sum(example.imitation_weight > 0 for example in examples),
        "preservation_examples": sum(
            example.preservation_weight > 0 for example in examples
        ),
        "diagnostic_only_examples": sum(
            example.role in DIAGNOSTIC_ROLES for example in examples
        ),
        "recovery_examples": sum(example.recover for example in examples),
        "failed_or_off_trajectory_imitation_examples": sum(
            example.imitation_weight > 0 and example.role in DIAGNOSTIC_ROLES
            for example in examples
        ),
    }
    if audit["failed_or_off_trajectory_imitation_examples"] != 0:
        raise OracleAdapterTrainingError(
            "Failed/off rollouts must never enter trajectory imitation"
        )
    return examples, audit


def split_examples_by_route(
    examples: Sequence[TrainingExample],
    *,
    validation_fraction: float,
    seed: int,
    allow_single_route_smoke: bool = False,
) -> SplitResult:
    """Create a deterministic route-disjoint split.

    A one-route dataset is rejected unless explicitly running a smoke test.  In
    that case the returned mode names the leakage and no held-out claim is
    allowed.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise OracleAdapterTrainingError("validation_fraction must lie in (0, 1)")
    by_route: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_route[example.route_key].append(index)
    routes = sorted(by_route)
    generator = random.Random(seed)

    if len(routes) == 1:
        if not allow_single_route_smoke:
            raise OracleAdapterTrainingError(
                "Route-disjoint validation requires at least two routes; use "
                "--smoke only for a non-claiming pipeline check"
            )
        imitation = [
            index for index, example in enumerate(examples)
            if example.imitation_weight > 0
        ]
        preservation = [
            index for index, example in enumerate(examples)
            if example.preservation_weight > 0
        ]
        diagnostic = [
            index for index, example in enumerate(examples)
            if example.imitation_weight == 0 and example.preservation_weight == 0
        ]
        train: list[int] = []
        validation: list[int] = []
        for group in (imitation, preservation):
            generator.shuffle(group)
            if len(group) <= 1:
                train.extend(group)
                continue
            validation_size = max(1, int(round(len(group) * validation_fraction)))
            validation_size = min(validation_size, len(group) - 1)
            validation.extend(group[:validation_size])
            train.extend(group[validation_size:])
        train.extend(diagnostic)
        if not train or not validation:
            raise OracleAdapterTrainingError("Single-route smoke split is empty")
        return SplitResult(
            train_indices=sorted(train),
            validation_indices=sorted(validation),
            train_routes=routes,
            validation_routes=routes,
            mode="single_route_smoke_not_route_disjoint",
        )

    positive_routes = {
        example.route_key
        for example in examples
        if example.imitation_weight > 0
    }
    if len(positive_routes) < 2:
        raise OracleAdapterTrainingError(
            "Route-disjoint imitation validation needs successful oracle "
            "examples on at least two routes"
        )
    shuffled = routes[:]
    generator.shuffle(shuffled)
    validation_count = max(1, int(round(len(routes) * validation_fraction)))
    validation_count = min(validation_count, len(routes) - 1)
    validation_routes = set(shuffled[:validation_count])
    if not (validation_routes & positive_routes):
        validation_routes.remove(next(iter(validation_routes)))
        validation_routes.add(sorted(positive_routes)[0])
    if not (positive_routes - validation_routes):
        move = sorted(validation_routes & positive_routes)[0]
        replacement = next(
            (route for route in shuffled if route not in validation_routes and route != move),
            None,
        )
        if replacement is None:
            raise OracleAdapterTrainingError("Cannot keep oracle examples in both splits")
        validation_routes.remove(move)
        validation_routes.add(replacement)
    train_routes = set(routes) - validation_routes
    train = [index for index, example in enumerate(examples) if example.route_key in train_routes]
    validation = [
        index for index, example in enumerate(examples) if example.route_key in validation_routes
    ]
    if set(train_routes) & set(validation_routes):
        raise OracleAdapterTrainingError("Route leakage detected")
    return SplitResult(
        train_indices=train,
        validation_indices=validation,
        train_routes=sorted(train_routes),
        validation_routes=sorted(validation_routes),
        mode="route_disjoint",
    )


class OracleAdapterTensorDataset(Dataset):
    def __init__(self, examples: Sequence[TrainingExample], indices: Sequence[int]):
        self.examples = list(examples)
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        example = self.examples[self.indices[item]]
        return {
            "base": torch.tensor(example.base_trajectory, dtype=torch.float32),
            "target": torch.tensor(example.target_trajectory, dtype=torch.float32),
            "mask": torch.tensor(example.trajectory_mask, dtype=torch.float32),
            "path_risk": torch.tensor(example.path_risk, dtype=torch.float32),
            "stop_target": torch.tensor(example.stop_target, dtype=torch.float32),
            "imitation_weight": torch.tensor(
                example.imitation_weight, dtype=torch.float32
            ),
            "preservation_weight": torch.tensor(
                example.preservation_weight, dtype=torch.float32
            ),
            "stop_weight": torch.tensor(example.stop_weight, dtype=torch.float32),
            "recover": torch.tensor(float(example.recover), dtype=torch.float32),
            "role": example.role,
            "sample_id": example.sample_id,
        }


def oracle_adapter_loss(
    output,
    batch: dict[str, Any],
    *,
    lambda_imitation: float = 1.0,
    lambda_preservation: float = 0.5,
    lambda_stop: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Role-masked loss that cannot imitate failed/off expert trajectories."""
    target = batch["target"][:, None]
    mask = batch["mask"][:, None]
    imitation_mask = mask * batch["imitation_weight"][:, None, None]
    preservation_mask = mask * batch["preservation_weight"][:, None, None]
    stop_mask = mask * batch["stop_weight"][:, None, None]

    point_error = F.smooth_l1_loss(
        output.trajectories, target, reduction="none"
    ).sum(dim=-1)
    imitation = (point_error * imitation_mask).sum() / imitation_mask.sum().clamp_min(1.0)
    residual_sq = output.residual.pow(2).sum(dim=-1)
    preservation = (residual_sq * preservation_mask).sum() / preservation_mask.sum().clamp_min(1.0)
    stop_target = batch["stop_target"][:, None, None].expand_as(
        output.stop_probability
    )
    stop_point = F.binary_cross_entropy(
        output.stop_probability, stop_target, reduction="none"
    )
    stop = (stop_point * stop_mask).sum() / stop_mask.sum().clamp_min(1.0)
    total = (
        lambda_imitation * imitation
        + lambda_preservation * preservation
        + lambda_stop * stop
    )

    with torch.no_grad():
        stop_predictions = output.stop_probability >= 0.5
        stop_correct = (
            (stop_predictions == (stop_target >= 0.5)).to(stop_mask.dtype)
            * stop_mask
        ).sum()
        preservation_intervention = (
            output.intervention * preservation_mask
        ).sum() / preservation_mask.sum().clamp_min(1.0)
        recovery_mask = mask * batch["recover"][:, None, None]
        recovery_stop_probability = (
            output.stop_probability * recovery_mask
        ).sum() / recovery_mask.sum().clamp_min(1.0)
    return {
        "total": total,
        "imitation": imitation,
        "preservation": preservation,
        "stop": stop,
        "imitation_points": imitation_mask.sum(),
        "preservation_points": preservation_mask.sum(),
        "stop_points": stop_mask.sum(),
        "stop_correct": stop_correct,
        "preservation_intervention_m": preservation_intervention,
        "recovery_stop_probability": recovery_stop_probability,
        "recovery_points": recovery_mask.sum(),
    }


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_epoch(
    model: PathRiskTrajectoryAdapter,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    lambda_imitation: float,
    lambda_preservation: float,
    lambda_stop: float,
    max_steps: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = Counter()
    batches = 0
    for step, raw_batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = _move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(
                batch["base"][:, None],
                batch["path_risk"][:, None],
            )
            losses = oracle_adapter_loss(
                output,
                batch,
                lambda_imitation=lambda_imitation,
                lambda_preservation=lambda_preservation,
                lambda_stop=lambda_stop,
            )
            if training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batches += 1
        for key in ("total", "imitation", "preservation", "stop"):
            totals[key] += float(losses[key].detach())
        for key in (
            "imitation_points",
            "preservation_points",
            "stop_points",
            "stop_correct",
            "recovery_points",
        ):
            totals[key] += float(losses[key].detach())
        totals["preservation_intervention_sum"] += float(
            losses["preservation_intervention_m"].detach()
        ) * float(losses["preservation_points"].detach())
        totals["recovery_stop_sum"] += float(
            losses["recovery_stop_probability"].detach()
        ) * float(losses["recovery_points"].detach())
    if batches == 0:
        raise OracleAdapterTrainingError("DataLoader produced no batches")
    result = {
        key: totals[key] / batches
        for key in ("total", "imitation", "preservation", "stop")
    }
    result.update(
        {
            "batches": batches,
            "imitation_points": totals["imitation_points"],
            "preservation_points": totals["preservation_points"],
            "stop_points": totals["stop_points"],
            "stop_accuracy": (
                totals["stop_correct"] / totals["stop_points"]
                if totals["stop_points"] > 0
                else None
            ),
            "preservation_intervention_m": (
                totals["preservation_intervention_sum"]
                / totals["preservation_points"]
                if totals["preservation_points"] > 0
                else None
            ),
            "recovery_stop_probability": (
                totals["recovery_stop_sum"] / totals["recovery_points"]
                if totals["recovery_points"] > 0
                else None
            ),
            "recovery_points": totals["recovery_points"],
        }
    )
    return result


def train_adapter(
    examples: Sequence[TrainingExample],
    split: SplitResult,
    *,
    hidden_dim: int = 128,
    max_residual_m: float = 2.0,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    lambda_imitation: float = 1.0,
    lambda_preservation: float = 0.5,
    lambda_stop: float = 0.25,
    device: str = "cpu",
    seed: int = 42,
    max_steps_per_epoch: int | None = None,
) -> tuple[PathRiskTrajectoryAdapter, list[dict[str, Any]], float]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise OracleAdapterTrainingError("epochs, batch_size, and learning_rate must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_device = torch.device(device)
    model = PathRiskTrajectoryAdapter(
        context_dim=0,
        hidden_dim=hidden_dim,
        max_residual_m=max_residual_m,
    ).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        OracleAdapterTensorDataset(examples, split.train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )
    validation_loader = DataLoader(
        OracleAdapterTensorDataset(examples, split.validation_indices),
        batch_size=batch_size,
        shuffle=False,
    )
    history: list[dict[str, Any]] = []
    best_validation = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=torch_device,
            optimizer=optimizer,
            lambda_imitation=lambda_imitation,
            lambda_preservation=lambda_preservation,
            lambda_stop=lambda_stop,
            max_steps=max_steps_per_epoch,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device=torch_device,
            optimizer=None,
            lambda_imitation=lambda_imitation,
            lambda_preservation=lambda_preservation,
            lambda_stop=lambda_stop,
            max_steps=max_steps_per_epoch,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise OracleAdapterTrainingError("Training produced no checkpoint state")
    model.load_state_dict(best_state)
    return model, history, float(best_validation)


def make_mock_samples(seed: int = 42) -> list[dict[str, Any]]:
    """Create two-route in-memory records for a dependency-light CLI smoke."""
    generator = random.Random(seed)
    samples: list[dict[str, Any]] = []
    for route_number in (1, 2):
        terminal_success = {
            "eligible": True,
            "entry_status": "Finished",
            "route_status": "Completed",
            "route_id": f"RouteScenario_mock_{route_number}_rep0",
            "town": f"Town0{route_number}",
            "route_completion": 100.0,
            "collisions": {"pedestrian": 0, "vehicle": 0, "layout": 0},
            "infractions": {
                "vehicle_blocked": 0,
                "route_timeout": 0,
                "scenario_timeouts": 0,
            },
        }
        for frame in range(12):
            active = 3 <= frame < 7
            base = [[0.0, 1.0 + 0.05 * generator.random()] for _ in range(6)]
            expert = [[0.0, 0.05] for _ in range(6)] if active else base
            rollout_id = f"mock/oracle_route_{route_number}"
            samples.append(
                {
                    "sample_id": f"{rollout_id}#{frame}",
                    "source": {"rollout_id": rollout_id},
                    "condition": {
                        "name": "front_corrupt_transient_oracle_stop",
                        "variant": "hazard",
                        "relevance": "on_path",
                    },
                    "oracle": {"path_risk": 1.0 if active else 0.0},
                    "controls": {"risk_mode": "oracle"},
                    "expert": {
                        "base_plan_displacements_m": base,
                        "trajectory_displacements_m": expert,
                        "trajectory_mask": [1] * 6,
                    },
                    "labels": {
                        "stop_required": active,
                        "recover": frame == 7,
                    },
                    "terminal_result": terminal_success,
                }
            )
        # A collision-producing off rollout is present only to audit that its
        # future trajectory receives zero imitation weight.
        failed = dict(terminal_success)
        failed["collisions"] = {"pedestrian": 1, "vehicle": 0, "layout": 0}
        for frame in range(3):
            rollout_id = f"mock/off_route_{route_number}"
            base = [[0.0, 1.0] for _ in range(6)]
            samples.append(
                {
                    "sample_id": f"{rollout_id}#{frame}",
                    "source": {"rollout_id": rollout_id},
                    "condition": {
                        "name": "front_corrupt_transient_off",
                        "variant": "hazard",
                        "relevance": "on_path",
                    },
                    "oracle": {"path_risk": 1.0},
                    "controls": {"risk_mode": "off"},
                    "expert": {
                        "base_plan_displacements_m": base,
                        "trajectory_displacements_m": [[0.0, 2.0] for _ in range(6)],
                        "trajectory_mask": [1] * 6,
                    },
                    "labels": {"stop_required": False, "recover": False},
                    "terminal_result": failed,
                }
            )
    return samples


__all__ = [
    "ROLE_ORACLE_IMITATION",
    "ROLE_PRESERVATION",
    "ROLE_FAILED_ORACLE",
    "ROLE_OFF_FAILURE",
    "ROLE_DIAGNOSTIC",
    "OracleAdapterTrainingError",
    "TrainingExample",
    "SplitResult",
    "load_exported_samples",
    "route_key",
    "assign_rollout_roles",
    "prepare_examples",
    "split_examples_by_route",
    "OracleAdapterTensorDataset",
    "oracle_adapter_loss",
    "run_epoch",
    "train_adapter",
    "make_mock_samples",
]

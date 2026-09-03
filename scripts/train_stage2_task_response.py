#!/usr/bin/env python3
"""Train the new task-aware ORION Stage-2 module on frozen spatial UQ.

This Stage-2A entry point cannot update Stage 1 and refuses manifests that use
legacy Density UQ or corruption labels.  It trains only the spatial projector
and auxiliary relevance/response heads; it does *not* train ORION's VLM or
VAE/diffusion trajectory decoder.  ``optimization_smoke`` is deliberately an
in-sample engineering check; only ``route_heldout`` produces held-out metrics,
and neither mode authorizes loading the checkpoint into closed-loop ORION.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uq_estimator.privileged_yield_labels import YIELD_STATES
from uq_estimator.spatial_task_fusion import (
    STAGE2_TASK_FUSION_CHECKPOINT_SCHEMA,
    SpatialUQTokenProjector,
    TaskRiskTrajectoryAdapter,
)
from uq_estimator.stage2_task_training import (
    Stage2TaskResponseDataset,
    stage2_task_response_loss,
)


REPORT_SCHEMA = "orion.stage2-task-response-training-report/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=("optimization_smoke", "route_heldout"),
        default="optimization_smoke",
    )
    parser.add_argument("--heldout-route-group")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tokens-per-view", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _split_indices(dataset: Stage2TaskResponseDataset, args: argparse.Namespace):
    route_groups = [record["route_group"] for record in dataset.records]
    unique_routes = sorted(set(route_groups))
    if args.mode == "optimization_smoke":
        indices = list(range(len(dataset)))
        return indices, indices, unique_routes, unique_routes
    heldout = str(args.heldout_route_group or "").strip()
    if not heldout:
        raise ValueError("route_heldout mode requires --heldout-route-group")
    if len(unique_routes) < 2:
        raise ValueError("route-heldout training requires at least two route groups")
    validation = [index for index, route in enumerate(route_groups) if route == heldout]
    training = [index for index, route in enumerate(route_groups) if route != heldout]
    if not training or not validation:
        raise ValueError("held-out route group is absent or consumes all samples")
    return training, validation, sorted(set(route_groups[i] for i in training)), [heldout]


def _balanced_loader(
    dataset: Stage2TaskResponseDataset,
    indices: list[int],
    *,
    batch_size: int,
    seed: int,
) -> DataLoader:
    state_indices = [
        int(dataset.records[index]["labels"]["yield_state_index"])
        for index in indices
    ]
    counts = {state: state_indices.count(state) for state in set(state_indices)}
    weights = [1.0 / counts[state] for state in state_indices]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=max(len(indices), batch_size),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )


def _evaluation_loader(
    dataset: Stage2TaskResponseDataset, indices: list[int], batch_size: int
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _forward(
    projector: SpatialUQTokenProjector,
    adapter: TaskRiskTrajectoryAdapter,
    batch: dict[str, Any],
):
    selected = projector(batch["observation_uq"].detach())
    return adapter(
        batch["planning_context"], selected, batch["task_context"]
    )


@torch.no_grad()
def evaluate(
    projector: SpatialUQTokenProjector,
    adapter: TaskRiskTrajectoryAdapter,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    projector.eval()
    adapter.eval()
    totals: dict[str, float] = {}
    samples = 0
    yield_correct = 0
    conflict_correct = 0
    conflict_elements = 0
    residual_abs = 0.0
    residual_elements = 0
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        output = _forward(projector, adapter, batch)
        losses = stage2_task_response_loss(output, batch)
        batch_size = int(batch["yield_target"].shape[0])
        samples += batch_size
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch_size
        yield_correct += int(
            (output.yield_logits.argmax(dim=-1) == batch["yield_target"]).sum()
        )
        predicted_conflict = output.conflict_logits.sigmoid() >= 0.5
        target_conflict = batch["conflict_target"] >= 0.5
        conflict_correct += int((predicted_conflict == target_conflict).sum())
        conflict_elements += int(target_conflict.numel())
        residual_abs += float(
            (output.trajectory_residual - batch["trajectory_target"])
            .abs()
            .sum()
        )
        residual_elements += int(batch["trajectory_target"].numel())
    if samples <= 0:
        raise RuntimeError("evaluation loader is empty")
    metrics = {name: value / samples for name, value in totals.items()}
    metrics.update({
        "yield_accuracy": yield_correct / samples,
        "conflict_binary_accuracy": conflict_correct / conflict_elements,
        "trajectory_residual_mae_m": residual_abs / residual_elements,
        "sample_count": float(samples),
    })
    return metrics


@torch.no_grad()
def zero_uq_identity_audit(
    dataset: Stage2TaskResponseDataset,
    projector: SpatialUQTokenProjector,
    adapter: TaskRiskTrajectoryAdapter,
    device: torch.device,
) -> dict[str, Any]:
    sample = dataset[0]
    context = sample["planning_context"].unsqueeze(0).to(device)
    task = sample["task_context"].unsqueeze(0).to(device)
    zero_uq = torch.zeros_like(sample["observation_uq"]).unsqueeze(0).to(device)
    output = adapter(context, projector(zero_uq), task)
    checks = {
        "conditioned_context_exact_identity": torch.equal(
            output.conditioned_context, context
        ),
        "trajectory_residual_exact_zero": int(
            torch.count_nonzero(output.trajectory_residual)
        ) == 0,
        "token_attention_exact_zero": int(
            torch.count_nonzero(output.token_attention)
        ) == 0,
        "yield_state_go": int(output.yield_logits.argmax(dim=-1)[0]) == 0,
    }
    return {"checks": checks, "passed": all(checks.values())}


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.batch_size, args.tokens_per_view, args.hidden_dim) <= 0:
        raise ValueError("positive training sizes are required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Stage-2 output directory")
    output_dir.mkdir(parents=True)
    dataset = Stage2TaskResponseDataset(manifest_path)
    train_indices, validation_indices, train_routes, validation_routes = (
        _split_indices(dataset, args)
    )
    first = dataset[0]
    views, height, width, components = first["observation_uq"].shape
    if height * width < args.tokens_per_view:
        raise ValueError("tokens-per-view exceeds the captured spatial grid")
    device = _device(args.device)
    projector_config = {
        "component_dim": int(components),
        "model_dim": 256,
        "hidden_dim": int(args.hidden_dim),
        "max_views": int(views),
        "tokens_per_view": int(args.tokens_per_view),
    }
    adapter_config = {
        "model_dim": 256,
        "num_heads": int(args.num_heads),
        "trajectory_steps": 6,
        "task_context_dim": 89,
        "response_score_scale": 0.25,
    }
    projector = SpatialUQTokenProjector(**projector_config).to(device)
    adapter = TaskRiskTrajectoryAdapter(**adapter_config).to(device)
    optimizer = torch.optim.AdamW(
        list(projector.parameters()) + list(adapter.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = _balanced_loader(
        dataset,
        train_indices,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    validation_loader = _evaluation_loader(
        dataset, validation_indices, args.batch_size
    )
    best_loss = float("inf")
    best_epoch = None
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        projector.train()
        adapter.train()
        train_loss = 0.0
        train_samples = 0
        for raw_batch in train_loader:
            batch = _move(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = _forward(projector, adapter, batch)
            losses = stage2_task_response_loss(output, batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(projector.parameters()) + list(adapter.parameters()), 5.0
            )
            optimizer.step()
            batch_size = int(batch["yield_target"].shape[0])
            train_loss += float(losses["loss"].detach()) * batch_size
            train_samples += batch_size
        metrics = evaluate(projector, adapter, validation_loader, device)
        row = {
            "epoch": epoch,
            "balanced_train_loss": train_loss / train_samples,
            "evaluation": metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_epoch = epoch
            best_state = {
                "projector": copy.deepcopy(projector.state_dict()),
                "adapter": copy.deepcopy(adapter.state_dict()),
            }
    if best_state is None:
        raise RuntimeError("Stage-2 training produced no checkpoint candidate")
    projector.load_state_dict(best_state["projector"])
    adapter.load_state_dict(best_state["adapter"])
    identity_audit = zero_uq_identity_audit(dataset, projector, adapter, device)
    if not identity_audit["passed"]:
        raise RuntimeError("trained Stage-2 module violates zero-UQ identity")

    stage1_hashes = {
        record["uq_input_contract"].get("stage1_checkpoint_sha256")
        for record in dataset.records
        if record["uq_input_contract"].get("stage1_checkpoint_sha256")
    }
    mechanism_hashes = {
        record["mechanism_gate"]["report_sha256"] for record in dataset.records
    }
    if len(stage1_hashes) > 1 or len(mechanism_hashes) != 1:
        raise RuntimeError("mixed Stage-1 or mechanism provenance is not allowed")
    checkpoint_path = output_dir / "stage2_spatial_task_fusion.pt"
    payload = {
        "schema_version": STAGE2_TASK_FUSION_CHECKPOINT_SCHEMA,
        "projector_config": projector_config,
        "adapter_config": adapter_config,
        "projector_state": {
            name: value.detach().cpu() for name, value in projector.state_dict().items()
        },
        "adapter_state": {
            name: value.detach().cpu() for name, value in adapter.state_dict().items()
        },
        "supervision_contract": {
            "stage": "stage2_task_risk",
            "uses_density_uq": False,
            "uses_corruption_label": False,
            "updates_stage1_adapter": False,
        },
        "stage1_checkpoint_sha256": next(iter(stage1_hashes), None),
        "mechanism_report_sha256": next(iter(mechanism_hashes)),
        "training_mode": args.mode,
        "training_stage": "stage2a_auxiliary_relevance_pretraining",
        "trains_orion_vlm": False,
        "trains_trajectory_decoder": False,
        "closed_loop_eligible": False,
        "best_epoch": best_epoch,
    }
    torch.save(payload, checkpoint_path)
    report = {
        "schema_version": REPORT_SCHEMA,
        "mode": args.mode,
        "training_stage": "stage2a_auxiliary_relevance_pretraining",
        "claim_boundary": (
            "In-sample auxiliary-head optimization smoke; ORION VLM and "
            "trajectory decoder are not trained, so the checkpoint must not "
            "be loaded for closed-loop control."
            if args.mode == "optimization_smoke"
            else "Route-held-out auxiliary-head evaluation only; ORION VLM/decoder joint training and closed-loop validation remain required."
        ),
        "closed_loop_eligible": False,
        "trains_orion_vlm": False,
        "trains_trajectory_decoder": False,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "sample_count": len(dataset),
        "training_route_groups": train_routes,
        "evaluation_route_groups": validation_routes,
        "uq_sources": sorted({
            record["uq_input_contract"]["source"] for record in dataset.records
        }),
        "supervision_contract": payload["supervision_contract"],
        "best_epoch": best_epoch,
        "best_evaluation": history[best_epoch - 1]["evaluation"],
        "history": history,
        "zero_uq_identity_audit": identity_audit,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "device": str(device),
        "seed": args.seed,
    }
    report_path = output_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

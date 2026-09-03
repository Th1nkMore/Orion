#!/usr/bin/env python3
"""Audit the deliberately in-sample Route147 Stage-2A optimization smoke."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uq_estimator.privileged_yield_labels import YIELD_STATES
from uq_estimator.spatial_task_fusion import load_stage2_task_fusion_checkpoint
from uq_estimator.stage2_task_training import Stage2TaskResponseDataset


SCHEMA_VERSION = "orion.stage2a-optimization-smoke-audit/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _variant(route_group: str) -> str:
    value = route_group.rsplit("/", 1)[-1]
    if value not in {"onpath_oracle", "offpath_oracle", "zero_uq"}:
        raise ValueError(f"unexpected optimization-smoke variant: {value}")
    return value


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    device = _device(args.device)
    dataset = Stage2TaskResponseDataset(args.manifest.resolve())
    projector, adapter, metadata = load_stage2_task_fusion_checkpoint(
        args.checkpoint.resolve(),
        expected_sha256=args.expected_checkpoint_sha256,
        device=device,
    )
    if metadata.get("training_stage") != "stage2a_auxiliary_relevance_pretraining":
        raise ValueError("checkpoint is not a Stage-2A auxiliary artifact")
    if metadata.get("closed_loop_eligible"):
        raise ValueError("Stage-2A checkpoint must never be closed-loop eligible")
    projector.eval()
    adapter.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    records = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            size = int(batch["yield_target"].shape[0])
            observation = batch["observation_uq"].to(device)
            context = batch["planning_context"].to(device)
            task = batch["task_context"].to(device)
            output = adapter(context, projector(observation), task)
            predictions = output.yield_logits.argmax(dim=-1).cpu()
            target = batch["yield_target"].cpu()
            residual_target = batch["trajectory_target"].to(device)
            for index in range(size):
                source_record = dataset.records[offset + index]
                variant = _variant(source_record["route_group"])
                event_active = bool(source_record["source"].get("event_active"))
                zero_input = int(torch.count_nonzero(observation[index])) == 0
                context_delta = output.conditioned_context[index] - context[index]
                residual = output.trajectory_residual[index]
                records.append({
                    "sample_id": source_record["sample_id"],
                    "variant": variant,
                    "event_active": event_active,
                    "target_state": YIELD_STATES[int(target[index])],
                    "predicted_state": YIELD_STATES[int(predictions[index])],
                    "target_non_go": int(target[index]) != 0,
                    "predicted_non_go": int(predictions[index]) != 0,
                    "zero_input": zero_input,
                    "context_delta_mean_abs": float(context_delta.abs().mean()),
                    "context_delta_max_abs": float(context_delta.abs().max()),
                    "trajectory_residual_mean_abs": float(residual.abs().mean()),
                    "trajectory_residual_max_abs": float(residual.abs().max()),
                    "trajectory_target_mae": float(
                        (residual - residual_target[index]).abs().mean()
                    ),
                })
            offset += size
    if offset != len(dataset):
        raise RuntimeError("audit did not consume the whole dataset")

    groups = defaultdict(list)
    for record in records:
        groups[record["variant"]].append(record)
    summaries = {}
    for variant, rows in sorted(groups.items()):
        active = [row for row in rows if row["event_active"]]
        positives = [row for row in rows if row["target_non_go"]]
        summaries[variant] = {
            "sample_count": len(rows),
            "event_active_count": len(active),
            "target_state_counts": dict(Counter(row["target_state"] for row in rows)),
            "predicted_state_counts": dict(Counter(row["predicted_state"] for row in rows)),
            "state_accuracy": sum(
                row["target_state"] == row["predicted_state"] for row in rows
            ) / len(rows),
            "target_positive_count": len(positives),
            "target_positive_recall": (
                sum(row["target_state"] == row["predicted_state"] for row in positives)
                / len(positives)
                if positives else None
            ),
            "false_non_go_rate": sum(
                row["predicted_non_go"] and not row["target_non_go"] for row in rows
            ) / len(rows),
            "active_context_delta_mean_abs": _mean([
                row["context_delta_mean_abs"] for row in active
            ]),
            "active_trajectory_residual_mean_abs": _mean([
                row["trajectory_residual_mean_abs"] for row in active
            ]),
            "active_trajectory_target_mae": _mean([
                row["trajectory_target_mae"] for row in active
            ]),
            "all_zero_inputs": all(row["zero_input"] for row in rows),
            "all_context_exact_identity": all(
                row["context_delta_max_abs"] == 0.0 for row in rows
            ),
            "all_trajectory_exact_zero": all(
                row["trajectory_residual_max_abs"] == 0.0 for row in rows
            ),
        }
    required = {"onpath_oracle", "offpath_oracle", "zero_uq"}
    if set(summaries) != required:
        raise ValueError("optimization smoke does not contain all three variants")
    onpath = summaries["onpath_oracle"]
    offpath = summaries["offpath_oracle"]
    zero = summaries["zero_uq"]
    checks = {
        "checkpoint_closed_loop_ineligible": not metadata["closed_loop_eligible"],
        "onpath_has_positive_targets": onpath["target_positive_count"] > 0,
        "onpath_positive_state_recall_at_least_75pct": (
            onpath["target_positive_recall"] is not None
            and onpath["target_positive_recall"] >= 0.75
        ),
        "onpath_active_residual_mae_below_5cm": (
            onpath["active_trajectory_target_mae"] is not None
            and onpath["active_trajectory_target_mae"] <= 0.05
        ),
        "offpath_never_predicts_non_go": offpath["false_non_go_rate"] == 0.0,
        "offpath_active_residual_below_5cm": (
            offpath["active_trajectory_residual_mean_abs"] is not None
            and offpath["active_trajectory_residual_mean_abs"] <= 0.05
        ),
        "zero_uq_inputs_are_exact_zero": zero["all_zero_inputs"],
        "zero_uq_context_exact_identity": zero["all_context_exact_identity"],
        "zero_uq_trajectory_exact_zero": zero["all_trajectory_exact_zero"],
        "zero_uq_never_predicts_non_go": zero["false_non_go_rate"] == 0.0,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "optimization_smoke_nonclaim",
        "claim_boundary": (
            "In-sample auxiliary-head audit only. Passing proves implementation "
            "can fit the controlled location task; it does not validate learned UQ, "
            "ORION trajectory behavior, generalization, or closed-loop safety."
        ),
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": metadata["sha256"],
        "checkpoint_training_stage": metadata["training_stage"],
        "closed_loop_eligible": metadata["closed_loop_eligible"],
        "device": str(device),
        "checks": checks,
        "passed": all(checks.values()),
        "variants": summaries,
        "records": records,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite Stage-2A audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

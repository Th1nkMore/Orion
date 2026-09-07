#!/usr/bin/env python3
"""Train the A1a field-query bridge on physical reconstruction only."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List

import numpy as np
import torch
from torch.nn import functional as F

from uq_estimator.qwen_visibility_vlm import FieldQueryVisibilityTokenProjector


SCHEMA = "orion.qwen-visibility-field-query-auxiliary/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scene(record: Dict[str, Any]) -> np.ndarray:
    with np.load(record["token_artifact"], allow_pickle=False) as artifact:
        global_rows = np.asarray(artifact["visibility_tokens_global"], dtype=np.float32)
        frontier_rows = np.asarray(artifact["visibility_tokens_frontier"], dtype=np.float32)
        global_mask = np.asarray(artifact["visibility_tokens_global_mask"], dtype=bool)
        frontier_mask = np.asarray(artifact["visibility_tokens_frontier_mask"], dtype=bool)
    return np.concatenate([global_rows[global_mask], frontier_rows[frontier_mask]])


def auxiliary_loss(model, features: torch.Tensor) -> Dict[str, torch.Tensor]:
    _, auxiliary = model.encode_records(features)
    field = F.smooth_l1_loss(auxiliary["field_values"], features)
    record_type = F.cross_entropy(
        auxiliary["record_type_logits"], auxiliary["record_type_targets"]
    )
    slot = F.cross_entropy(auxiliary["slot_logits"], auxiliary["slot_targets"])
    return {
        "total": 10.0 * field + record_type + slot,
        "field": field,
        "record_type": record_type,
        "slot": slot,
    }


def evaluate(model, records: List[Dict[str, Any]], device: str, names: List[str]) -> Dict[str, Any]:
    absolute = []
    type_correct = 0
    slot_correct = 0
    count = 0
    losses = defaultdict(float)
    model.eval()
    with torch.inference_mode():
        for record in records:
            features = torch.from_numpy(load_scene(record)).to(device)
            _, auxiliary = model.encode_records(features)
            absolute.append((auxiliary["field_values"] - features).abs().cpu().numpy())
            type_correct += int(
                (auxiliary["record_type_logits"].argmax(-1) == auxiliary["record_type_targets"])
                .sum()
                .item()
            )
            slot_correct += int(
                (auxiliary["slot_logits"].argmax(-1) == auxiliary["slot_targets"])
                .sum()
                .item()
            )
            count += int(features.shape[0])
            for key, value in auxiliary_loss(model, features).items():
                losses[key] += float(value.item()) * int(features.shape[0])
    errors = np.concatenate(absolute, axis=0)
    per_field = {name: float(errors[:, index].mean()) for index, name in enumerate(names)}
    return {
        "record_count": count,
        "mean_absolute_error": float(errors.mean()),
        "worst_field_mean_absolute_error": float(max(per_field.values())),
        "per_field_mean_absolute_error": per_field,
        "record_type_accuracy": type_correct / count,
        "slot_accuracy": slot_correct / count,
        "mean_losses": {key: value / count for key, value in losses.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite A1a output directory")
    if args.epochs != 30 or args.seed != 20260907 or args.learning_rate != 3e-4:
        raise ValueError("A1a v1 hyperparameters are preregistered and immutable")
    if sha256(args.manifest) != args.expected_manifest_sha256:
        raise ValueError("manifest hash mismatch")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = list(manifest["records"])
    by_split = {
        split: [row for row in records if row["split"] == split]
        for split in ("train", "validation", "held_out")
    }
    if [len(by_split[x]) for x in ("train", "validation", "held_out")] != [70, 10, 10]:
        raise ValueError("A1a requires the audited 70/10/10 route split")
    with np.load(records[0]["token_artifact"], allow_pickle=False) as artifact:
        names = [str(x) for x in artifact["visibility_tokens_feature_names"].tolist()]
    config = {
        "feature_dim": 23,
        "hidden_dim": 256,
        "vlm_hidden_dim": 2560,
        "attention_heads": 8,
        "query_layers": 2,
        "maximum_token_slots": 48,
        "scalar_basis_dim": 4,
    }
    model = FieldQueryVisibilityTokenProjector(**config).to(args.device)
    excluded = ("output_projection.", "boundary_embeddings")
    trainable = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(excluded)
    ]
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith(excluded))
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    history = []
    start = time.monotonic()
    model.train()
    for epoch in range(args.epochs):
        order = list(by_split["train"])
        random.Random(args.seed + epoch).shuffle(order)
        totals = defaultdict(float)
        seen = 0
        for record in order:
            features = torch.from_numpy(load_scene(record)).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            losses = auxiliary_loss(model, features)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            n = int(features.shape[0])
            seen += n
            for key, value in losses.items():
                totals[key] += float(value.detach().item()) * n
        history.append({"epoch": epoch + 1, **{key: value / seen for key, value in totals.items()}})
        print(json.dumps(history[-1], sort_keys=True), flush=True)

    metrics = {
        split: evaluate(model, rows, args.device, names)
        for split, rows in by_split.items()
    }
    thresholds = {
        "maximum_macro_mae": 0.03,
        "maximum_worst_field_mae": 0.12,
        "minimum_record_type_accuracy": 0.99,
        "minimum_slot_accuracy": 0.99,
    }
    passed = {}
    for split in ("validation", "held_out"):
        row = metrics[split]
        passed[split] = bool(
            row["mean_absolute_error"] <= thresholds["maximum_macro_mae"]
            and row["worst_field_mean_absolute_error"] <= thresholds["maximum_worst_field_mae"]
            and row["record_type_accuracy"] >= thresholds["minimum_record_type_accuracy"]
            and row["slot_accuracy"] >= thresholds["minimum_slot_accuracy"]
        )
    args.output_dir.mkdir(parents=True)
    checkpoint = args.output_dir / "field_query_auxiliary.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "config": config,
            "state_dict": model.state_dict(),
            "feature_names": names,
            "manifest_sha256": args.expected_manifest_sha256,
        },
        checkpoint,
    )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "stage": "A1a_physical_record_pretraining",
        "qwen_loaded": False,
        "images_loaded": False,
        "lora_installed": False,
        "planning_expert_loaded": False,
        "risk_action_or_trajectory_heads": False,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": args.expected_manifest_sha256,
        "config": config,
        "training": {
            "epochs": args.epochs,
            "optimizer_steps": args.epochs * len(by_split["train"]),
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "elapsed_seconds": time.monotonic() - start,
            "trainable_parameter_count": sum(x.numel() for x in trainable),
            "qwen_output_projection_trained": False,
        },
        "history": history,
        "metrics": metrics,
        "thresholds": thresholds,
        "passed": passed,
        "checkpoint": str(checkpoint.resolve()),
        "claim_boundary": {
            "physical_bridge_integrity_only": True,
            "qwen_language_alignment_claim_allowed": False,
            "visual_alignment_claim_allowed": False,
            "planning_or_safety_claim_allowed": False,
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"metrics": metrics, "passed": passed}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

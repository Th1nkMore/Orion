#!/usr/bin/env python3
"""Bounded task-agnostic pretraining for the Stage2-L U tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

from uq_estimator.stage1_u_tokenizer_pretraining import (
    UQSummaryReconstructionHead,
    stage1_u_tokenizer_pretraining_terms,
)
from uq_estimator.uq_relevance_tokenizer import UQComponentTokenizer


SCHEMA = "orion.stage1_u_tokenizer_pretraining_run.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_components(rows: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    arrays = []
    for row in rows:
        path = Path(str(row["path"]))
        if sha256_file(path) != str(row["sha256"]):
            raise ValueError("Stage1 U map hash mismatch: %s" % path)
        with np.load(path, allow_pickle=False) as payload:
            array = np.asarray(payload["uncertainty_components"], dtype=np.float32)
        if array.shape != (4, 6, 40, 40, 3):
            raise ValueError("Stage1 U component tensor shape differs")
        arrays.append(torch.from_numpy(array))
    return torch.stack(arrays).to(device=device, non_blocking=True)


@torch.no_grad()
def evaluate(
    tokenizer: UQComponentTokenizer,
    decoder: UQSummaryReconstructionHead,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    tokenizer.eval()
    decoder.eval()
    totals = torch.zeros(3, dtype=torch.float64)
    counts = torch.zeros(3, dtype=torch.float64)
    for start in range(0, len(rows), batch_size):
        components = _load_components(rows[start : start + batch_size], device)
        tokenized = tokenizer(components)
        decoded = decoder(tokenized.token_grid)
        absolute = (decoded - tokenized.temporal_summary).abs()
        for index, span in enumerate(((0, 3), (3, 6), (6, 9))):
            values = absolute[..., span[0] : span[1]]
            totals[index] += values.double().sum().cpu()
            counts[index] += values.numel()
    zero = torch.zeros(1, 4, 6, 40, 40, 3, device=device)
    zero_tokens = tokenizer(zero)
    zero_decoded = decoder(zero_tokens.token_grid)
    metrics = totals / counts.clamp_min(1.0)
    return {
        "summary_mae": float((totals.sum() / counts.sum()).item()),
        "latest_component_mae": float(metrics[0].item()),
        "temporal_mean_component_mae": float(metrics[1].item()),
        "temporal_delta_component_mae": float(metrics[2].item()),
        "zero_anchor_mae": float(zero_decoded.abs().mean().item()),
    }


def _validate_protocol(protocol: Mapping[str, Any], audit_path: Path) -> None:
    if protocol.get("schema") != "orion.stage1_u_tokenizer_pretraining_protocol.v1":
        raise ValueError("unsupported tokenizer pretraining protocol")
    if protocol.get("training_inputs") != ["normalized_frozen_stage1_uncertainty_components"]:
        raise ValueError("protocol permits non-Stage1-U training inputs")
    forbidden = protocol.get("forbidden_inputs", {})
    if not forbidden or any(value is not True for value in forbidden.values()):
        raise ValueError("protocol does not prohibit every task input")
    expected = protocol.get("source_audit", {}).get("sha256")
    if expected != sha256_file(audit_path):
        raise ValueError("source audit hash differs from protocol")
    if protocol.get("launch_locks", {}).get("stage2l_v10_training_allowed") is not False:
        raise ValueError("tokenizer pretraining must not unlock Stage2-L")


def train(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite %s" % args.output_dir)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    _validate_protocol(protocol, args.source_audit)
    if audit.get("passed") is not True:
        raise ValueError("Stage1 U source audit did not pass")
    if int(protocol["bounded_pretraining"]["optimizer_steps"]) != args.steps:
        raise ValueError("optimizer steps differ from frozen protocol")
    if int(protocol["bounded_pretraining"]["batch_size"]) != args.batch_size:
        raise ValueError("batch size differs from frozen protocol")
    if not torch.cuda.is_available():
        raise RuntimeError("bounded tokenizer pretraining requires CUDA")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    rows = audit["maps"]
    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    if not train_rows or not dev_rows:
        raise ValueError("source audit lacks train/dev maps")

    tokenizer = UQComponentTokenizer(
        model_dim=4096, hidden_dim=256, grid_hw=(10, 10), max_views=6
    ).to(device)
    decoder = UQSummaryReconstructionHead(
        model_dim=4096, hidden_dim=256, component_dim=3
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(tokenizer.parameters()) + list(decoder.parameters()),
        lr=float(protocol["bounded_pretraining"]["learning_rate"]),
        weight_decay=float(protocol["bounded_pretraining"]["weight_decay"]),
    )
    before = {
        "train": evaluate(tokenizer, decoder, train_rows, device=device, batch_size=args.batch_size),
        "dev": evaluate(tokenizer, decoder, dev_rows, device=device, batch_size=args.batch_size),
    }
    rng = random.Random(args.seed)
    history: List[Dict[str, float]] = []
    event_presentations: Counter = Counter()
    tokenizer.train()
    decoder.train()
    for step in range(args.steps):
        selected = rng.sample(train_rows, k=min(args.batch_size, len(train_rows)))
        components = _load_components(selected, device)
        optimizer.zero_grad(set_to_none=True)
        terms = stage1_u_tokenizer_pretraining_terms(
            tokenizer=tokenizer,
            reconstruction_head=decoder,
            components=components,
            zero_anchor_weight=float(protocol["objective"]["zero_anchor_weight"]),
            smooth_l1_beta=float(protocol["objective"]["smooth_l1_beta"]),
        )
        if not bool(torch.isfinite(terms.loss)):
            raise FloatingPointError("non-finite tokenizer pretraining loss")
        terms.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(tokenizer.parameters()) + list(decoder.parameters()), 1.0
        )
        optimizer.step()
        for row in selected:
            event_presentations[str(row["event_id"])] += 1
        history.append({
            "step": step + 1,
            "loss": float(terms.loss.detach().cpu()),
            "reconstruction_loss": float(terms.reconstruction_loss.detach().cpu()),
            "zero_anchor_loss": float(terms.zero_anchor_loss.detach().cpu()),
        })

    after = {
        "train": evaluate(tokenizer, decoder, train_rows, device=device, batch_size=args.batch_size),
        "dev": evaluate(tokenizer, decoder, dev_rows, device=device, batch_size=args.batch_size),
    }
    thresholds = protocol["release_gates"]
    checks = {
        "all_losses_finite": all(np.isfinite(row["loss"]) for row in history),
        "every_train_event_presented": len(event_presentations) == int(audit["event_count"]) - sum(1 for row in audit["events"] if row["split"] == "dev"),
        "train_reconstruction_improved": after["train"]["summary_mae"] < before["train"]["summary_mae"],
        "dev_reconstruction_improved": after["dev"]["summary_mae"] < before["dev"]["summary_mae"],
        "dev_summary_mae": after["dev"]["summary_mae"] <= float(thresholds["maximum_dev_summary_mae"]),
        "dev_latest_component_mae": after["dev"]["latest_component_mae"] <= float(thresholds["maximum_dev_latest_component_mae"]),
        "zero_anchor_mae": after["dev"]["zero_anchor_mae"] <= float(thresholds["maximum_zero_anchor_mae"]),
        "task_or_route_labels_consumed": False,
    }
    # Negative statement above is an attestation, not a pass gate.
    passed = all(value for key, value in checks.items() if key != "task_or_route_labels_consumed")
    status = "bounded_task_agnostic_tokenizer_pretraining_pass" if passed else "bounded_task_agnostic_tokenizer_pretraining_failed_gate"
    args.output_dir.mkdir(parents=True)
    checkpoint_path = args.output_dir / "stage1_u_tokenizer_task_agnostic_v1.pt"
    torch.save({
        "schema": SCHEMA,
        "status": status,
        "task_agnostic": True,
        "stage1_checkpoint_sha256": audit["stage1_checkpoint_sha256"],
        "source_audit_sha256": sha256_file(args.source_audit),
        "optimizer_steps": args.steps,
        "uq_tokenizer": {key: value.detach().cpu() for key, value in tokenizer.state_dict().items()},
        "reconstruction_decoder_included": False,
        "stage2l_ready": False,
    }, checkpoint_path)
    report = {
        "schema": SCHEMA,
        "status": status,
        "passed": passed,
        "engineering_preexperiment_only": True,
        "formal_stage2l_ready": False,
        "stage2p_ready": False,
        "source_audit": {"path": str(args.source_audit), "sha256": sha256_file(args.source_audit)},
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "train_map_count": len(train_rows),
        "dev_map_count": len(dev_rows),
        "optimizer_steps": args.steps,
        "before": before,
        "after": after,
        "checks": checks,
        "history": history,
        "event_presentations": dict(sorted(event_presentations.items())),
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        "decoder_discarded": True,
        "forbidden_inputs_consumed": {
            "route_context": False,
            "task_relevance": False,
            "qa_text_or_fields": False,
            "ttc_collision_or_control": False,
            "corruption_metadata": False,
        },
        "claim_boundary": "Tokenizer representation pretraining only; no VLM understanding, task relevance, trajectory, closed-loop or safety claim.",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps({
        "status": report["status"],
        "passed": report["passed"],
        "before_dev_mae": report["before"]["dev"]["summary_mae"],
        "after_dev_mae": report["after"]["dev"]["summary_mae"],
        "checkpoint_sha256": report["checkpoint"]["sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

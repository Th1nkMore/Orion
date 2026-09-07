#!/usr/bin/env python3
"""Reload and independently recompute an A1a physical bridge checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from uq_estimator.qwen_visibility_vlm import FieldQueryVisibilityTokenProjector


SCHEMA = "orion.qwen-visibility-field-query-auxiliary-audit/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene(record: Dict[str, Any]) -> np.ndarray:
    with np.load(record["token_artifact"], allow_pickle=False) as artifact:
        g = np.asarray(artifact["visibility_tokens_global"], dtype=np.float32)
        f = np.asarray(artifact["visibility_tokens_frontier"], dtype=np.float32)
        gm = np.asarray(artifact["visibility_tokens_global_mask"], dtype=bool)
        fm = np.asarray(artifact["visibility_tokens_frontier_mask"], dtype=bool)
    return np.concatenate([g[gm], f[fm]])


def recompute(model, records, device, names):
    errors = []
    type_correct = slot_correct = count = 0
    model.eval()
    with torch.inference_mode():
        for record in records:
            features = torch.from_numpy(scene(record)).to(device)
            _, auxiliary = model.encode_records(features)
            errors.append((auxiliary["field_values"] - features).abs().cpu().numpy())
            type_correct += int((auxiliary["record_type_logits"].argmax(-1) == auxiliary["record_type_targets"]).sum())
            slot_correct += int((auxiliary["slot_logits"].argmax(-1) == auxiliary["slot_targets"]).sum())
            count += len(features)
    errors = np.concatenate(errors)
    per_field = {name: float(errors[:, index].mean()) for index, name in enumerate(names)}
    return {
        "record_count": count,
        "mean_absolute_error": float(errors.mean()),
        "worst_field_mean_absolute_error": max(per_field.values()),
        "per_field_mean_absolute_error": per_field,
        "record_type_accuracy": type_correct / count,
        "slot_accuracy": slot_correct / count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite A1a audit")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    manifest_path = Path(report["manifest"])
    failures = []
    if sha256(manifest_path) != report["manifest_sha256"]:
        failures.append("manifest hash mismatch")
    if checkpoint.get("manifest_sha256") != report["manifest_sha256"]:
        failures.append("checkpoint lineage mismatch")
    model = FieldQueryVisibilityTokenProjector(**checkpoint["config"]).to(args.device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    projection_zero = bool((model.output_projection.weight == 0).all()) and bool((model.output_projection.bias == 0).all())
    boundary_zero = bool((model.boundary_embeddings == 0).all())
    if not projection_zero:
        failures.append("Qwen output projection changed during physical-only training")
    if not boundary_zero:
        failures.append("Qwen boundary embeddings changed during physical-only training")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    by_split = {split: [row for row in records if row["split"] == split] for split in ("train", "validation", "held_out")}
    metrics = {
        split: recompute(model, rows, args.device, checkpoint["feature_names"])
        for split, rows in by_split.items()
    }
    for split in metrics:
        reference = report["metrics"][split]
        for key in ("record_count", "mean_absolute_error", "worst_field_mean_absolute_error", "record_type_accuracy", "slot_accuracy"):
            if not math.isclose(float(metrics[split][key]), float(reference[key]), rel_tol=1e-6, abs_tol=1e-7):
                failures.append("metric mismatch: %s/%s" % (split, key))
        for name, value in metrics[split]["per_field_mean_absolute_error"].items():
            if not math.isclose(value, reference["per_field_mean_absolute_error"][name], rel_tol=1e-6, abs_tol=1e-7):
                failures.append("field metric mismatch: %s/%s" % (split, name))
    thresholds = report["thresholds"]
    passed = {
        split: bool(
            metrics[split]["mean_absolute_error"] <= thresholds["maximum_macro_mae"]
            and metrics[split]["worst_field_mean_absolute_error"] <= thresholds["maximum_worst_field_mae"]
            and metrics[split]["record_type_accuracy"] >= thresholds["minimum_record_type_accuracy"]
            and metrics[split]["slot_accuracy"] >= thresholds["minimum_slot_accuracy"]
        )
        for split in ("validation", "held_out")
    }
    if passed != report["passed"]:
        failures.append("pass decision mismatch")
    result = {
        "schema": SCHEMA,
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "report_sha256": sha256(args.report),
        "checkpoint_sha256": sha256(args.checkpoint),
        "recomputed_metrics": metrics,
        "recomputed_passed": passed,
        "qwen_output_projection_remains_zero": projection_zero,
        "boundary_embeddings_remain_zero": boundary_zero,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "failures": failures, "recomputed_passed": passed}, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

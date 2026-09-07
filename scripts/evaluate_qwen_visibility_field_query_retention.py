#!/usr/bin/env python3
"""Measure whether A1b retained the A1a physical field representation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from uq_estimator.qwen_visibility_vlm import FieldQueryVisibilityTokenProjector


SCHEMA = "orion.qwen-visibility-field-query-retention/v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scene(record):
    with np.load(record["token_artifact"], allow_pickle=False) as artifact:
        names = [str(x) for x in artifact["visibility_tokens_feature_names"].tolist()]
        global_rows = np.asarray(artifact["visibility_tokens_global"], dtype=np.float32)
        frontier_rows = np.asarray(artifact["visibility_tokens_frontier"], dtype=np.float32)
        global_mask = np.asarray(artifact["visibility_tokens_global_mask"], dtype=bool)
        frontier_mask = np.asarray(artifact["visibility_tokens_frontier_mask"], dtype=bool)
    return np.concatenate([global_rows[global_mask], frontier_rows[frontier_mask]]), names


def evaluate(model, records, device):
    errors = []
    type_correct = slot_correct = count = 0
    names = None
    model.eval()
    with torch.inference_mode():
        for record in records:
            array, current_names = load_scene(record)
            names = names or current_names
            if names != current_names:
                raise ValueError("field order changed")
            features = torch.from_numpy(array).to(device)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1b-report", type=Path, required=True)
    parser.add_argument("--a1b-checkpoint", type=Path, required=True)
    parser.add_argument("--a1a-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite retention report")
    a1b = json.loads(args.a1b_report.read_text(encoding="utf-8"))
    a1a = json.loads(args.a1a_report.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.a1b_checkpoint, map_location=args.device, weights_only=False)
    config = dict(a1b["protocol"]["projector"])
    if config.pop("type") != "field_query":
        raise ValueError("A1b did not use the field-query bridge")
    model = FieldQueryVisibilityTokenProjector(**config).to(args.device)
    model.load_state_dict(checkpoint["adaptation"]["projector"], strict=True)
    manifest_path = Path(a1b["manifest_path"])
    if sha256(manifest_path) != a1b["manifest_sha256"]:
        raise ValueError("manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_split = {
        split: [row for row in manifest["records"] if row["split"] == split]
        for split in ("train", "validation", "held_out")
    }
    metrics = {split: evaluate(model, rows, args.device) for split, rows in by_split.items()}
    ratios = {
        split: {
            "macro_mae_over_a1a": metrics[split]["mean_absolute_error"] / a1a["metrics"][split]["mean_absolute_error"],
            "worst_field_mae_over_a1a": metrics[split]["worst_field_mean_absolute_error"] / a1a["metrics"][split]["worst_field_mean_absolute_error"],
        }
        for split in metrics
    }
    retained = {
        split: bool(
            metrics[split]["mean_absolute_error"] <= a1a["thresholds"]["maximum_macro_mae"]
            and metrics[split]["worst_field_mean_absolute_error"] <= a1a["thresholds"]["maximum_worst_field_mae"]
            and metrics[split]["record_type_accuracy"] >= a1a["thresholds"]["minimum_record_type_accuracy"]
            and metrics[split]["slot_accuracy"] >= a1a["thresholds"]["minimum_slot_accuracy"]
        )
        for split in ("validation", "held_out")
    }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "a1b_report_sha256": sha256(args.a1b_report),
        "a1b_checkpoint_sha256": sha256(args.a1b_checkpoint),
        "a1a_report_sha256": sha256(args.a1a_report),
        "metrics": metrics,
        "change_relative_to_a1a": ratios,
        "passes_original_a1a_gates": retained,
        "claim_boundary": {"representation_retention_diagnostic_only": True, "language_or_safety_claim_allowed": False},
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "change_relative_to_a1a": ratios, "passes_original_a1a_gates": retained}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

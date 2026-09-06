#!/usr/bin/env python3
"""Independently audit the V1k route-diverse Qwen grounding baseline."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCHEMA = "orion.qwen-visibility-route-diverse-baseline-audit/v1"
REPORT_SCHEMA = "orion.qwen-visibility-grounding-smoke-report/v1"
CONFIG_SCHEMA = "orion.qwen-visibility-route-diverse-route-readout-config/v1"
STAGE = "V1k_route_diverse_random_row_route_readout_baseline"
CONTROLS = ("true_u", "zero_u", "spatial_shuffle")
SPLITS = ("validation", "held_out")
LABELS = ("ON_ROUTE", "OFF_ROUTE")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _finite_positive(values: Iterable[Any]) -> bool:
    values = [float(value) for value in values]
    return bool(values) and all(math.isfinite(value) and value > 0.0 for value in values)


def summarize_evaluations(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Recompute original-target and control-semantic accuracy from raw rows."""

    summary: Dict[str, Any] = {}
    for control in CONTROLS:
        selected = [row for row in rows if row.get("control") == control]
        correct = sum(row.get("parsed") == row.get("expected_answer") for row in selected)
        semantic = sum(
            row.get("parsed") == row.get("control_expected_answer") for row in selected
        )
        summary[control] = {
            "count": len(selected),
            "original_target_correct": correct,
            "original_target_accuracy": correct / len(selected) if selected else None,
            "control_semantic_correct": semantic,
            "control_semantic_accuracy": semantic / len(selected) if selected else None,
            "output_counts": dict(sorted(Counter(row.get("parsed") for row in selected).items())),
            "by_split": {},
        }
        for split in SPLITS:
            split_rows = [row for row in selected if row.get("split") == split]
            split_correct = sum(
                row.get("parsed") == row.get("expected_answer") for row in split_rows
            )
            label_counts = {}
            for label in LABELS:
                label_rows = [
                    row for row in split_rows if row.get("expected_answer") == label
                ]
                label_counts[label] = {
                    "count": len(label_rows),
                    "correct": sum(row.get("parsed") == label for row in label_rows),
                }
            summary[control]["by_split"][split] = {
                "count": len(split_rows),
                "correct": split_correct,
                "accuracy": split_correct / len(split_rows) if split_rows else None,
                "by_original_label": label_counts,
            }
    true_accuracy = summary["true_u"]["original_target_accuracy"]
    stronger_control = max(
        summary["zero_u"]["original_target_accuracy"],
        summary["spatial_shuffle"]["original_target_accuracy"],
    )
    summary["causal_gap_vs_stronger_control"] = true_accuracy - stronger_control
    return summary


def audit(report_path: Path, output_path: Path) -> Dict[str, Any]:
    report_path = Path(report_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite baseline audit: %s" % output_path)
    report = _read_json(report_path)
    failures = []

    if report.get("schema") != REPORT_SCHEMA:
        failures.append("report_schema")
    if report.get("stage") != STAGE:
        failures.append("report_stage")
    if report.get("status") != "complete":
        failures.append("report_status")
    if report.get("objective") != {"type": "random_row_route_readout", "fields": ["route"]}:
        failures.append("objective")
    if report.get("claim_boundary") != {
        "bounded_route_disjoint_grounding_baseline_only": True,
        "reportable_generalization": True,
        "safety_claim_allowed": False,
    }:
        failures.append("claim_boundary")
    if report.get("optimizer_controls") != []:
        failures.append("optimizer_controls")
    if report.get("hidden_actor_labels_used") is not False:
        failures.append("hidden_actor_labels")
    if report.get("planning_expert_in_optimizer") is not False:
        failures.append("planning_expert_optimizer")

    recorded_files = {
        "protocol": (Path(str(report.get("protocol_path", ""))), report.get("protocol_sha256")),
        "manifest": (Path(str(report.get("manifest_path", ""))), report.get("manifest_sha256")),
        "curriculum": (Path(str(report.get("curriculum_path", ""))), report.get("curriculum_sha256")),
    }
    loaded = {}
    for name, (path, expected_hash) in recorded_files.items():
        if not path.is_file():
            failures.append("missing_%s" % name)
            continue
        if _sha256(path) != expected_hash:
            failures.append("%s_hash" % name)
        loaded[name] = _read_json(path)

    protocol = loaded.get("protocol", {})
    manifest = loaded.get("manifest", {})
    curriculum = loaded.get("curriculum", {})
    if protocol.get("schema") != CONFIG_SCHEMA or protocol.get("stage") != STAGE:
        failures.append("protocol_identity")
    data_audit_ref = protocol.get("data_audit", {})
    data_audit_path = Path(str(data_audit_ref.get("path", "")))
    if not data_audit_path.is_file():
        failures.append("missing_data_audit")
    else:
        if _sha256(data_audit_path) != data_audit_ref.get("sha256"):
            failures.append("data_audit_hash")
        data_audit = _read_json(data_audit_path)
        if data_audit.get("passed") is not True or data_audit.get("failures") != []:
            failures.append("data_audit_status")

    records = manifest.get("records", [])
    examples = curriculum.get("examples", [])
    sample_ids = {row.get("sample_id") for row in records}
    example_by_id = {row.get("example_id"): row for row in examples}
    if len(records) != 90 or len(sample_ids) != 90 or None in sample_ids:
        failures.append("manifest_samples")
    if set(report.get("sample_ids", [])) != sample_ids:
        failures.append("report_sample_ids")
    if len(examples) != 90 or len(example_by_id) != 90 or None in example_by_id:
        failures.append("curriculum_examples")
    if report.get("curriculum_example_count") != 90:
        failures.append("curriculum_example_count")

    history = report.get("history", [])
    schedule = curriculum.get("training_schedule", [])
    if len(history) != 240 or [row.get("optimizer_step") for row in history] != list(range(1, 241)):
        failures.append("optimizer_steps")
    if [row.get("example_id") for row in history] != schedule:
        failures.append("training_schedule")
    if any(example_by_id.get(row.get("example_id"), {}).get("split") != "train" for row in history):
        failures.append("non_train_optimizer_example")
    numeric_history_fields = (
        "loss",
        "projector_gradient_norm_before_clip",
        "lora_gradient_norm_before_clip",
        "projector_update_norm",
        "lora_update_norm",
    )
    for field in numeric_history_fields:
        if not _finite_positive(row.get(field, float("nan")) for row in history):
            failures.append("history_%s" % field)
    if any(int(row.get("projector_nonzero_gradient_tensors", 0)) <= 0 for row in history):
        failures.append("projector_nonzero_gradients")
    if any(int(row.get("lora_nonzero_gradient_tensors", 0)) <= 0 for row in history):
        failures.append("lora_nonzero_gradients")

    scope = report.get("scope", {})
    expected_scope = {
        "projector_trainable_parameter_count": 1_391_616,
        "model_trainable_parameter_count": 1_572_864,
        "planning_expert_trainable_parameter_count": 0,
        "vision_trainable_parameter_count": 0,
        "embedding_trainable": False,
        "lm_head_trainable": False,
    }
    for key, expected in expected_scope.items():
        if scope.get(key) != expected:
            failures.append("scope_%s" % key)
    if len(scope.get("projector_trainable_names", [])) != 7:
        failures.append("projector_trainable_names")
    if len(scope.get("model_trainable_names", [])) != 64:
        failures.append("model_trainable_names")
    if len(report.get("installed_lora_modules", [])) != 32:
        failures.append("installed_lora_modules")

    checkpoint = report.get("checkpoint", {})
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if not checkpoint_path.is_file():
        failures.append("missing_checkpoint")
    else:
        if _sha256(checkpoint_path) != checkpoint.get("sha256"):
            failures.append("checkpoint_hash")
        if checkpoint_path.stat().st_size != checkpoint.get("bytes"):
            failures.append("checkpoint_bytes")
    if checkpoint.get("contains_base_model_weights") is not False:
        failures.append("checkpoint_base_weights")
    if checkpoint.get("contains_optimizer_state") is not False:
        failures.append("checkpoint_optimizer_state")
    if checkpoint.get("projector_tensor_count") != 7 or checkpoint.get("lora_tensor_count") != 64:
        failures.append("checkpoint_tensor_counts")

    rows = report.get("evaluations", [])
    expected_eval_ids = set(curriculum.get("evaluation_example_ids", []))
    if len(rows) != 60:
        failures.append("evaluation_count")
    for control in CONTROLS:
        control_rows = [row for row in rows if row.get("control") == control]
        if {row.get("example_id") for row in control_rows} != expected_eval_ids:
            failures.append("evaluation_ids:%s" % control)
    for row in rows:
        example = example_by_id.get(row.get("example_id"))
        if example is None:
            failures.append("unknown_evaluation_example")
            continue
        expected = example.get("expected_answer")
        control_expected = example.get("control_expected_answers", {}).get(row.get("control"))
        if row.get("split") != example.get("split") or row.get("expected_answer") != expected:
            failures.append("evaluation_provenance:%s" % row.get("example_id"))
        if row.get("control_expected_answer") != control_expected:
            failures.append("evaluation_control_target:%s" % row.get("example_id"))
        if row.get("canonical_exact") != (row.get("parsed") == expected):
            failures.append("canonical_exact:%s" % row.get("example_id"))
        if row.get("control_semantic_exact") != (row.get("parsed") == control_expected):
            failures.append("control_semantic_exact:%s" % row.get("example_id"))

    metrics = summarize_evaluations(rows)
    passed = not failures
    audit_result = {
        "schema": SCHEMA,
        "passed": passed,
        "status": (
            "valid_route_diverse_baseline_without_causal_route_readout"
            if passed and metrics["causal_gap_vs_stronger_control"] <= 0.0
            else "valid_route_diverse_baseline_with_positive_causal_gap"
            if passed
            else "invalid_run"
        ),
        "failures": failures,
        "report": {"path": str(report_path), "sha256": _sha256(report_path)},
        "checkpoint": checkpoint,
        "metrics": metrics,
        "loss": {
            "first": float(history[0]["loss"]) if history else None,
            "last": float(history[-1]["loss"]) if history else None,
            "first_30_mean": sum(float(row["loss"]) for row in history[:30]) / 30 if len(history) >= 30 else None,
            "last_30_mean": sum(float(row["loss"]) for row in history[-30:]) / 30 if len(history) >= 30 else None,
        },
        "interpretation_boundary": {
            "shuffle_is_diagnostic_not_a_hard_gate": True,
            "planning_or_closed_loop_claim_allowed": False,
            "architecture_sweep_authorized": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.report, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

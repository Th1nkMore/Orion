"""Fail-closed evaluation of the bounded Qwen visibility grounding overfit."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping


GROUNDING_OVERFIT_AUDIT_SCHEMA = "orion.qwen-visibility-grounding-overfit-audit/v1"
EXPECTED_SAMPLE_IDS = (
    "route151-step-000000",
    "route151-step-000200",
    "route151-step-000260",
    "route151-step-000280",
    "route151-step-000300",
)
EXPECTED_CONTROLS = ("true_u", "zero_u", "spatial_shuffle")
TARGET_FIELDS = ("frontier", "route", "margin", "action")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rates(rows: Iterable[Mapping[str, object]]) -> dict:
    rows = list(rows)
    count = len(rows)
    if not count:
        return {
            "records": 0,
            "canonical_exact": 0.0,
            "valid_json": 0.0,
            "field_accuracy": {field: 0.0 for field in TARGET_FIELDS},
        }
    return {
        "records": count,
        "canonical_exact": sum(bool(row["canonical_exact"]) for row in rows)
        / count,
        "valid_json": sum(row.get("parsed") is not None for row in rows) / count,
        "field_accuracy": {
            field: sum(bool(row["field_correct"][field]) for row in rows) / count
            for field in TARGET_FIELDS
        },
    }


def _metrics_by_control(rows: Iterable[Mapping[str, object]]) -> dict:
    rows = list(rows)
    return {
        control: _rates(row for row in rows if row.get("control") == control)
        for control in EXPECTED_CONTROLS
    }


def audit_grounding_overfit_report(report_path: Path, output_path: Path) -> dict:
    """Audit provenance/process first, then report a separate causal-capacity gate."""

    report_path = Path(report_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite grounding audit: %s" % output_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures = []
    if report.get("schema") != "orion.qwen-visibility-grounding-smoke-report/v1":
        failures.append("report_schema")
    if report.get("status") != "complete":
        failures.append("report_status")
    if report.get("stage") != "V1c_route151_plumbing_overfit":
        failures.append("report_stage")
    if report.get("claim_boundary") != {
        "plumbing_overfit_only": True,
        "reportable_generalization": False,
        "safety_claim_allowed": False,
    }:
        failures.append("claim_boundary")
    if tuple(report.get("sample_ids", ())) != EXPECTED_SAMPLE_IDS:
        failures.append("sample_ids")
    if report.get("optimizer_controls") != []:
        failures.append("optimizer_controls")
    if report.get("hidden_actor_labels_used") is not False:
        failures.append("hidden_actor_labels")
    if report.get("planning_expert_in_optimizer") is not False:
        failures.append("planning_expert_optimizer_flag")

    scope = report.get("scope", {})
    if int(scope.get("projector_trainable_parameter_count", -1)) != 1_330_734:
        failures.append("projector_parameter_count")
    if int(scope.get("model_trainable_parameter_count", -1)) != 393_216:
        failures.append("lora_parameter_count")
    if int(scope.get("vision_trainable_parameter_count", -1)) != 0:
        failures.append("vision_not_frozen")
    if int(scope.get("planning_expert_trainable_parameter_count", -1)) != 0:
        failures.append("planning_expert_not_frozen")
    if scope.get("embedding_trainable") is not False:
        failures.append("embedding_not_frozen")
    if scope.get("lm_head_trainable") is not False:
        failures.append("lm_head_not_frozen")

    history = report.get("history", [])
    if len(history) != 15 or [row.get("optimizer_step") for row in history] != list(
        range(1, 16)
    ):
        failures.append("optimizer_step_sequence")
    if Counter(row.get("sample_id") for row in history) != Counter(
        {sample_id: 3 for sample_id in EXPECTED_SAMPLE_IDS}
    ):
        failures.append("optimizer_sample_balance")
    for index, row in enumerate(history):
        numeric = (
            "loss",
            "projector_gradient_norm_before_clip",
            "lora_gradient_norm_before_clip",
            "projector_update_norm",
            "lora_update_norm",
            "forward_seconds",
            "backward_seconds",
            "optimizer_step_seconds",
        )
        if any(
            not math.isfinite(float(row.get(field, float("nan"))))
            for field in numeric
        ):
            failures.append("nonfinite_history_%d" % index)
            continue
        if float(row["projector_update_norm"]) <= 0.0:
            failures.append("zero_projector_update_%d" % index)
        if float(row["lora_update_norm"]) <= 0.0:
            failures.append("zero_lora_update_%d" % index)
        if int(row.get("projector_nonzero_gradient_tensors", 0)) <= 0:
            failures.append("zero_projector_gradient_%d" % index)
        if int(row.get("lora_nonzero_gradient_tensors", 0)) <= 0:
            failures.append("zero_lora_gradient_%d" % index)

    checkpoint = report.get("checkpoint", {})
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if not checkpoint_path.is_file():
        failures.append("checkpoint_missing")
    elif _sha256(checkpoint_path) != checkpoint.get("sha256"):
        failures.append("checkpoint_sha256")
    if checkpoint.get("contains_optimizer_state") is not False:
        failures.append("checkpoint_optimizer_state")
    if checkpoint.get("contains_base_model_weights") is not False:
        failures.append("checkpoint_base_weights")
    if int(checkpoint.get("projector_tensor_count", -1)) != 7:
        failures.append("checkpoint_projector_tensors")
    if int(checkpoint.get("lora_tensor_count", -1)) != 16:
        failures.append("checkpoint_lora_tensors")

    pre_rows = report.get("pre_training_evaluations", [])
    evaluations = report.get("evaluations", [])
    expected_pairs = {
        (sample_id, control)
        for sample_id in EXPECTED_SAMPLE_IDS
        for control in EXPECTED_CONTROLS
    }
    actual_pairs = [
        (row.get("sample_id"), row.get("control")) for row in evaluations
    ]
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
        failures.append("post_evaluation_coverage")
    expected_pre = {(sample_id, "true_u") for sample_id in EXPECTED_SAMPLE_IDS}
    actual_pre = [(row.get("sample_id"), row.get("control")) for row in pre_rows]
    if len(actual_pre) != len(expected_pre) or set(actual_pre) != expected_pre:
        failures.append("pre_evaluation_coverage")
    for row in pre_rows + evaluations:
        fields = row.get("field_correct")
        if not isinstance(fields, dict) or set(fields) != set(TARGET_FIELDS):
            failures.append("evaluation_field_contract")
            break

    pre_metrics = _rates(pre_rows)
    metrics = _metrics_by_control(evaluations)
    true_metrics = metrics["true_u"]
    control_exact_ceiling = max(
        metrics["zero_u"]["canonical_exact"],
        metrics["spatial_shuffle"]["canonical_exact"],
    )
    control_action_ceiling = max(
        metrics["zero_u"]["field_accuracy"]["action"],
        metrics["spatial_shuffle"]["field_accuracy"]["action"],
    )
    exact_gap = true_metrics["canonical_exact"] - control_exact_ceiling
    action_gap = (
        true_metrics["field_accuracy"]["action"] - control_action_ceiling
    )
    causal_capacity_checks = {
        "true_exact_at_least_four_of_five": true_metrics["canonical_exact"]
        >= 0.8,
        "every_true_field_at_least_four_of_five": all(
            true_metrics["field_accuracy"][field] >= 0.8
            for field in TARGET_FIELDS
        ),
        "true_exact_control_gap_at_least_two_of_five": exact_gap >= 0.4,
        "true_action_control_gap_at_least_two_of_five": action_gap >= 0.4,
    }
    protocol_valid = not failures
    causal_capacity_passed = protocol_valid and all(causal_capacity_checks.values())
    status = (
        "causal_plumbing_overfit_pass"
        if causal_capacity_passed
        else (
            "valid_run_without_causal_grounding"
            if protocol_valid
            else "invalid_run"
        )
    )
    audit = {
        "schema": GROUNDING_OVERFIT_AUDIT_SCHEMA,
        "status": status,
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "protocol_valid": protocol_valid,
        "protocol_failures": failures,
        "causal_capacity_passed": causal_capacity_passed,
        "causal_capacity_checks": causal_capacity_checks,
        "pre_training_true_metrics": pre_metrics,
        "post_training_metrics": metrics,
        "true_minus_control_ceiling": {
            "canonical_exact": exact_gap,
            "action": action_gap,
        },
        "loss_first": float(history[0]["loss"]) if history else None,
        "loss_last": float(history[-1]["loss"]) if history else None,
        "claim_boundary": {
            "plumbing_overfit_only": True,
            "reportable_generalization": False,
            "safety_claim_allowed": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


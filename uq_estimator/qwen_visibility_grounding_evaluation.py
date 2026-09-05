"""Fail-closed evaluation of the bounded Qwen visibility grounding overfit."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping


GROUNDING_OVERFIT_AUDIT_SCHEMA = "orion.qwen-visibility-grounding-overfit-audit/v1"
FACTORIZED_GROUNDING_OVERFIT_AUDIT_SCHEMA = (
    "orion.qwen-visibility-grounding-factorized-overfit-audit/v1"
)
FACTORIZED_CONFIG_SCHEMA = "orion.qwen-visibility-grounding-factorized-config/v1"
EXPECTED_SAMPLE_IDS = (
    "route151-step-000000",
    "route151-step-000200",
    "route151-step-000260",
    "route151-step-000280",
    "route151-step-000300",
)
EXPECTED_CONTROLS = ("true_u", "zero_u", "spatial_shuffle")
TARGET_FIELDS = ("frontier", "route", "margin", "action")
FACTORIZED_ALLOWED_ANSWERS = {
    "frontier": tuple("F%02d" % index for index in range(32)),
    "route": ("ON_ROUTE", "OFF_ROUTE"),
    "margin": ("INSIDE", "NEAR", "CLEAR"),
    "action": ("KEEP", "SLOW", "STOP"),
}


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


def _factorized_rates(rows: Iterable[Mapping[str, object]]) -> dict:
    rows = list(rows)
    count = len(rows)
    return {
        "records": count,
        "exact_accuracy": (
            sum(bool(row.get("canonical_exact")) for row in rows) / count
            if count
            else 0.0
        ),
    }


def _factorized_metrics(rows: Iterable[Mapping[str, object]]) -> dict:
    rows = list(rows)
    return {
        field: {
            control: _factorized_rates(
                row
                for row in rows
                if row.get("task_field") == field
                and row.get("control") == control
            )
            for control in EXPECTED_CONTROLS
        }
        for field in TARGET_FIELDS
    }


def _check_recorded_file(report: Mapping[str, object], name: str, failures: list):
    path_value = report.get(name + "_path")
    digest = report.get(name + "_sha256")
    if not isinstance(path_value, str) or not path_value:
        failures.append(name + "_path")
        return None
    path = Path(path_value)
    if not path.is_file():
        failures.append(name + "_missing")
        return None
    if not isinstance(digest, str) or _sha256(path) != digest:
        failures.append(name + "_sha256")
        return None
    return path


def audit_factorized_grounding_overfit_report(
    report_path: Path, output_path: Path
) -> dict:
    """Audit the balanced V1d Route-151 capacity probe without overclaiming it."""

    report_path = Path(report_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(
            "refusing to overwrite factorized grounding audit: %s" % output_path
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures = []
    if report.get("schema") != "orion.qwen-visibility-grounding-smoke-report/v1":
        failures.append("report_schema")
    if report.get("status") != "complete":
        failures.append("report_status")
    if report.get("stage") != "V1d_route151_factorized_overfit":
        failures.append("report_stage")
    claim_boundary = {
        "plumbing_overfit_only": True,
        "reportable_generalization": False,
        "safety_claim_allowed": False,
    }
    if report.get("claim_boundary") != claim_boundary:
        failures.append("claim_boundary")
    objective = {
        "type": "factorized_fields",
        "fields": list(TARGET_FIELDS),
    }
    if report.get("objective") != objective:
        failures.append("objective")
    if tuple(report.get("sample_ids", ())) != EXPECTED_SAMPLE_IDS:
        failures.append("sample_ids")
    if report.get("optimizer_controls") != []:
        failures.append("optimizer_controls")
    if report.get("hidden_actor_labels_used") is not False:
        failures.append("hidden_actor_labels")
    if report.get("planning_expert_in_optimizer") is not False:
        failures.append("planning_expert_optimizer_flag")

    protocol_path = _check_recorded_file(report, "protocol", failures)
    _check_recorded_file(report, "manifest", failures)
    if protocol_path is not None:
        try:
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append("protocol_json")
        else:
            if protocol.get("schema") != FACTORIZED_CONFIG_SCHEMA:
                failures.append("protocol_schema")
            if protocol.get("stage") != "V1d_route151_factorized_overfit":
                failures.append("protocol_stage")
            if protocol.get("objective") != objective:
                failures.append("protocol_objective")
            if int(protocol.get("training", {}).get("optimizer_steps", -1)) != 60:
                failures.append("protocol_optimizer_steps")
            if protocol.get("training", {}).get("separate_gradient_clipping") is not True:
                failures.append("protocol_gradient_clipping")
            if protocol.get("evaluation", {}).get("pre_training_controls") != [
                "true_u"
            ]:
                failures.append("protocol_pre_controls")
            if protocol.get("evaluation", {}).get("controls") != list(
                EXPECTED_CONTROLS
            ):
                failures.append("protocol_controls")
            if tuple(protocol.get("sample_ids", ())) != EXPECTED_SAMPLE_IDS:
                failures.append("protocol_sample_ids")
            if protocol.get("claim_boundary") != claim_boundary:
                failures.append("protocol_claim_boundary")

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
    if len(history) != 60 or [row.get("optimizer_step") for row in history] != list(
        range(1, 61)
    ):
        failures.append("optimizer_step_sequence")
    expected_training_pairs = Counter(
        {
            (sample_id, field): 3
            for sample_id in EXPECTED_SAMPLE_IDS
            for field in TARGET_FIELDS
        }
    )
    actual_training_pairs = Counter(
        (row.get("sample_id"), row.get("task_field")) for row in history
    )
    if actual_training_pairs != expected_training_pairs:
        failures.append("optimizer_sample_field_balance")
    numeric_fields = (
        "loss",
        "projector_gradient_norm_before_clip",
        "lora_gradient_norm_before_clip",
        "projector_update_norm",
        "lora_update_norm",
        "forward_seconds",
        "backward_seconds",
        "optimizer_step_seconds",
    )
    for index, row in enumerate(history):
        try:
            numeric_values = [float(row.get(field, float("nan"))) for field in numeric_fields]
        except (TypeError, ValueError):
            numeric_values = [float("nan")]
        if any(not math.isfinite(value) for value in numeric_values):
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
    expected_pre = {
        (sample_id, field, "true_u")
        for sample_id in EXPECTED_SAMPLE_IDS
        for field in TARGET_FIELDS
    }
    actual_pre = [
        (row.get("sample_id"), row.get("task_field"), row.get("control"))
        for row in pre_rows
    ]
    if len(actual_pre) != len(expected_pre) or set(actual_pre) != expected_pre:
        failures.append("pre_evaluation_coverage")
    expected_post = {
        (sample_id, field, control)
        for sample_id in EXPECTED_SAMPLE_IDS
        for field in TARGET_FIELDS
        for control in EXPECTED_CONTROLS
    }
    actual_post = [
        (row.get("sample_id"), row.get("task_field"), row.get("control"))
        for row in evaluations
    ]
    if len(actual_post) != len(expected_post) or set(actual_post) != expected_post:
        failures.append("post_evaluation_coverage")
    for index, row in enumerate(pre_rows + evaluations):
        field = row.get("task_field")
        expected_answer = row.get("expected_answer")
        answer = row.get("answer")
        if field not in TARGET_FIELDS:
            failures.append("evaluation_task_field_%d" % index)
            continue
        if expected_answer not in FACTORIZED_ALLOWED_ANSWERS[field]:
            failures.append("evaluation_expected_answer_%d" % index)
            continue
        exact = isinstance(answer, str) and answer.strip() == expected_answer
        if row.get("canonical_exact") is not exact:
            failures.append("evaluation_exact_contract_%d" % index)
        if row.get("field_correct") != {field: exact}:
            failures.append("evaluation_field_contract_%d" % index)

    pre_metrics = {
        field: _factorized_rates(
            row for row in pre_rows if row.get("task_field") == field
        )
        for field in TARGET_FIELDS
    }
    metrics = _factorized_metrics(evaluations)
    true_accuracy = {
        field: metrics[field]["true_u"]["exact_accuracy"]
        for field in TARGET_FIELDS
    }
    frontier_control_ceiling = max(
        metrics["frontier"]["zero_u"]["exact_accuracy"],
        metrics["frontier"]["spatial_shuffle"]["exact_accuracy"],
    )
    gaps = {
        "frontier_minus_control_ceiling": (
            true_accuracy["frontier"] - frontier_control_ceiling
        ),
        "margin_minus_zero_u": (
            true_accuracy["margin"]
            - metrics["margin"]["zero_u"]["exact_accuracy"]
        ),
        "action_minus_zero_u": (
            true_accuracy["action"]
            - metrics["action"]["zero_u"]["exact_accuracy"]
        ),
    }
    causal_capacity_checks = {
        "each_true_field_at_least_four_of_five": all(
            value >= 0.8 for value in true_accuracy.values()
        ),
        "frontier_true_control_gap_at_least_two_of_five": (
            gaps["frontier_minus_control_ceiling"] >= 0.4
        ),
        "margin_true_zero_gap_at_least_two_of_five": (
            gaps["margin_minus_zero_u"] >= 0.4
        ),
        "action_true_zero_gap_at_least_two_of_five": (
            gaps["action_minus_zero_u"] >= 0.4
        ),
    }
    protocol_valid = not failures
    causal_capacity_passed = protocol_valid and all(causal_capacity_checks.values())
    status = (
        "causal_factorized_plumbing_overfit_pass"
        if causal_capacity_passed
        else (
            "valid_run_without_causal_grounding"
            if protocol_valid
            else "invalid_run"
        )
    )
    audit = {
        "schema": FACTORIZED_GROUNDING_OVERFIT_AUDIT_SCHEMA,
        "status": status,
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "protocol_valid": protocol_valid,
        "protocol_failures": failures,
        "causal_capacity_passed": causal_capacity_passed,
        "causal_capacity_checks": causal_capacity_checks,
        "pre_training_true_metrics": pre_metrics,
        "post_training_metrics": metrics,
        "true_minus_controls": gaps,
        "control_semantics": {
            "frontier": "spatial shuffle changes which metric slot owns the maximum-score record",
            "route": "no causal gap required because all five plumbing labels are ON_ROUTE",
            "margin": (
                "spatial shuffle preserves the selected record content; "
                "zero-U is the causal control"
            ),
            "action": (
                "spatial shuffle preserves the selected record content; "
                "zero-U is the causal control"
            ),
        },
        "loss_first": float(history[0]["loss"]) if history else None,
        "loss_last": float(history[-1]["loss"]) if history else None,
        "claim_boundary": claim_boundary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def audit_visibility_grounding_report(report_path: Path, output_path: Path) -> dict:
    """Dispatch to the stage-specific fail-closed grounding audit."""

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("stage") == "V1d_route151_factorized_overfit":
        return audit_factorized_grounding_overfit_report(report_path, output_path)
    return audit_grounding_overfit_report(report_path, output_path)

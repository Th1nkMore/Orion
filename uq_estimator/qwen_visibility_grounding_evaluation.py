"""Fail-closed evaluation of the bounded Qwen visibility grounding overfit."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping


GROUNDING_OVERFIT_AUDIT_SCHEMA = "orion.qwen-visibility-grounding-overfit-audit/v1"
FACTORIZED_GROUNDING_OVERFIT_AUDIT_SCHEMA = (
    "orion.qwen-visibility-grounding-factorized-overfit-audit/v1"
)
FACTORIZED_CONFIG_SCHEMA = "orion.qwen-visibility-grounding-factorized-config/v1"
ROW_GROUNDING_OVERFIT_AUDIT_SCHEMA = (
    "orion.qwen-visibility-row-grounding-overfit-audit/v1"
)
ROW_CONFIG_SCHEMA = "orion.qwen-visibility-row-grounding-config/v1"
ROW_CURRICULUM_SCHEMA = "orion.qwen-visibility-row-grounding-curriculum/v1"
ROUTE_READOUT_OVERFIT_AUDIT_SCHEMA = (
    "orion.qwen-visibility-route-readout-overfit-audit/v1"
)
ROUTE_READOUT_CONFIG_SCHEMA = "orion.qwen-visibility-route-readout-config/v1"
TYPED_ROUTE_READOUT_CONFIG_SCHEMA = (
    "orion.qwen-visibility-typed-route-readout-config/v1"
)
SLOT_TYPED_ROUTE_READOUT_CONFIG_SCHEMA = (
    "orion.qwen-visibility-slot-typed-route-readout-config/v1"
)
FULL_ATTENTION_SLOT_TYPED_ROUTE_READOUT_CONFIG_SCHEMA = (
    "orion.qwen-visibility-slot-typed-full-attention-route-readout-config/v1"
)
TARGET_ROW_ROUTE_READOUT_CONFIG_SCHEMA = (
    "orion.qwen-visibility-target-row-route-readout-config/v1"
)
ROUTE_READOUT_CURRICULUM_SCHEMA = (
    "orion.qwen-visibility-route-readout-curriculum/v1"
)
TARGET_ROW_ROUTE_READOUT_CURRICULUM_SCHEMA = (
    "orion.qwen-visibility-target-row-route-readout-curriculum/v1"
)
V1I_FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
V1I_LORA_MODULE_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
V1J_FULLY_HELD_OUT_FRAME = "route151-step-000200"
V1J_TARGET_ROW_PAIRS = {
    "route151-step-000000": {
        "train": ((6, 24), (14, 28)),
        "held_out_target_rows": ((1, 20),),
    },
    "route151-step-000200": {
        "train": (),
        "held_out_target_rows": ((11, 31),),
    },
    "route151-step-000260": {
        "train": ((4, 15), (10, 28), (17, 31)),
        "held_out_target_rows": ((0, 8),),
    },
    "route151-step-000280": {
        "train": ((9, 3), (11, 5), (15, 20), (29, 23), (31, 25)),
        "held_out_target_rows": ((0, 2),),
    },
    "route151-step-000300": {
        "train": ((6, 24), (9, 26), (11, 30)),
        "held_out_target_rows": ((5, 16),),
    },
}
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


def _row_rates(rows: Iterable[Mapping[str, object]]) -> dict:
    rows = list(rows)
    count = len(rows)
    return {
        "records": count,
        "true_target_accuracy": (
            sum(bool(row.get("canonical_exact")) for row in rows) / count
            if count
            else 0.0
        ),
        "control_target_accuracy": (
            sum(bool(row.get("control_semantic_exact")) for row in rows) / count
            if count
            else 0.0
        ),
    }


def audit_row_addressed_grounding_overfit_report(
    report_path: Path, output_path: Path
) -> dict:
    """Audit V1e's real-row, within-image anti-shortcut grounding probe."""

    report_path = Path(report_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(
            "refusing to overwrite row-addressed grounding audit: %s"
            % output_path
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures = []
    claim_boundary = {
        "plumbing_overfit_only": True,
        "reportable_generalization": False,
        "safety_claim_allowed": False,
    }
    objective = {
        "type": "row_addressed_fields",
        "fields": list(TARGET_FIELDS),
    }
    if report.get("schema") != "orion.qwen-visibility-grounding-smoke-report/v1":
        failures.append("report_schema")
    if report.get("status") != "complete":
        failures.append("report_status")
    if report.get("stage") != "V1e_route151_row_addressed_overfit":
        failures.append("report_stage")
    if report.get("claim_boundary") != claim_boundary:
        failures.append("claim_boundary")
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
    manifest_path = _check_recorded_file(report, "manifest", failures)
    curriculum_path = _check_recorded_file(report, "curriculum", failures)
    protocol = {}
    if protocol_path is not None:
        try:
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append("protocol_json")
        else:
            if protocol.get("schema") != ROW_CONFIG_SCHEMA:
                failures.append("protocol_schema")
            if protocol.get("stage") != "V1e_route151_row_addressed_overfit":
                failures.append("protocol_stage")
            if protocol.get("objective") != objective:
                failures.append("protocol_objective")
            if protocol.get("training", {}).get("optimizer_steps") != 360:
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
            if curriculum_path is not None and Path(
                str(protocol.get("curriculum", ""))
            ).resolve() != curriculum_path:
                failures.append("protocol_curriculum_path")

    curriculum = {}
    examples = []
    schedule = []
    if curriculum_path is not None:
        try:
            curriculum = json.loads(
                curriculum_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            failures.append("curriculum_json")
        else:
            if curriculum.get("schema") != ROW_CURRICULUM_SCHEMA:
                failures.append("curriculum_schema")
            if curriculum.get("reportable_generalization") is not False:
                failures.append("curriculum_reportable")
            if curriculum.get("controls_used_for_optimizer") is not False:
                failures.append("curriculum_optimizer_controls")
            if curriculum.get("hidden_actor_labels_used") is not False:
                failures.append("curriculum_hidden_actor_labels")
            if curriculum.get("planning_expert_used_for_optimizer") is not False:
                failures.append("curriculum_planning_expert")
            if curriculum.get("complete_row_permutations_only") is not True:
                failures.append("curriculum_row_permutations")
            if (
                curriculum.get("spatial_shuffle_target_changed_for_every_example")
                is not True
            ):
                failures.append("curriculum_shuffle_changes")
            if curriculum.get("example_count") != 43:
                failures.append("curriculum_example_count")
            if report.get("curriculum_example_count") != 43:
                failures.append("report_curriculum_example_count")
            if curriculum.get("steps_per_field") != 90:
                failures.append("curriculum_steps_per_field")
            if curriculum.get("optimizer_steps") != 360:
                failures.append("curriculum_optimizer_steps")
            if curriculum.get("field_counts") != {
                "action": 6,
                "frontier": 15,
                "margin": 12,
                "route": 10,
            }:
                failures.append("curriculum_field_counts")
            if curriculum.get("label_counts") != {
                "action": {"KEEP": 2, "SLOW": 2, "STOP": 2},
                "frontier": {"F03": 5, "F13": 5, "F23": 5},
                "margin": {"CLEAR": 4, "INSIDE": 4, "NEAR": 4},
                "route": {"OFF_ROUTE": 5, "ON_ROUTE": 5},
            }:
                failures.append("curriculum_label_counts")
            if manifest_path is not None:
                if Path(
                    str(curriculum.get("base_manifest_path", ""))
                ).resolve() != manifest_path:
                    failures.append("curriculum_manifest_path")
                if curriculum.get("base_manifest_sha256") != report.get(
                    "manifest_sha256"
                ):
                    failures.append("curriculum_manifest_sha256")
            examples = curriculum.get("examples", [])
            schedule = curriculum.get("training_schedule", [])

    scope = report.get("scope", {})
    expected_scope = {
        "projector_trainable_parameter_count": 1_330_734,
        "model_trainable_parameter_count": 393_216,
        "vision_trainable_parameter_count": 0,
        "planning_expert_trainable_parameter_count": 0,
        "embedding_trainable": False,
        "lm_head_trainable": False,
    }
    for name, expected in expected_scope.items():
        if scope.get(name) != expected:
            failures.append("scope_" + name)

    history = report.get("history", [])
    if len(history) != 360 or [row.get("optimizer_step") for row in history] != list(
        range(1, 361)
    ):
        failures.append("optimizer_step_sequence")
    if len(schedule) != 360 or [row.get("example_id") for row in history] != schedule:
        failures.append("optimizer_schedule")
    history_fields = Counter(row.get("task_field") for row in history)
    if history_fields != Counter({field: 90 for field in TARGET_FIELDS}):
        failures.append("optimizer_field_balance")
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
            values = [
                float(row.get(field, float("nan"))) for field in numeric_fields
            ]
        except (TypeError, ValueError):
            values = [float("nan")]
        if any(not math.isfinite(value) for value in values):
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
    if checkpoint.get("projector_tensor_count") != 7:
        failures.append("checkpoint_projector_tensors")
    if checkpoint.get("lora_tensor_count") != 16:
        failures.append("checkpoint_lora_tensors")

    example_by_id = {
        example.get("example_id"): example
        for example in examples
        if isinstance(example, dict)
    }
    if len(example_by_id) != 43 or None in example_by_id:
        failures.append("curriculum_example_ids")
    for example_id, example in example_by_id.items():
        field = example.get("task_field")
        expected_answer = example.get("expected_answer")
        controls = example.get("control_expected_answers", {})
        if field not in TARGET_FIELDS:
            failures.append("curriculum_example_field_" + str(example_id))
            continue
        if expected_answer not in FACTORIZED_ALLOWED_ANSWERS[field]:
            failures.append("curriculum_example_answer_" + str(example_id))
        if controls.get("true_u") != expected_answer:
            failures.append("curriculum_true_target_" + str(example_id))
        if controls.get("spatial_shuffle") == expected_answer:
            failures.append("curriculum_shuffle_target_" + str(example_id))
        permutation = example.get("sequence_permutation_new_to_manifest", [])
        if sorted(permutation) != list(range(32)):
            failures.append("curriculum_permutation_" + str(example_id))

    pre_rows = report.get("pre_training_evaluations", [])
    post_rows = report.get("evaluations", [])
    expected_pre = {(example_id, "true_u") for example_id in example_by_id}
    actual_pre = [
        (row.get("example_id"), row.get("control")) for row in pre_rows
    ]
    if len(actual_pre) != len(expected_pre) or set(actual_pre) != expected_pre:
        failures.append("pre_evaluation_coverage")
    expected_post = {
        (example_id, control)
        for example_id in example_by_id
        for control in EXPECTED_CONTROLS
    }
    actual_post = [
        (row.get("example_id"), row.get("control")) for row in post_rows
    ]
    if len(actual_post) != len(expected_post) or set(actual_post) != expected_post:
        failures.append("post_evaluation_coverage")
    for index, row in enumerate(pre_rows + post_rows):
        example = example_by_id.get(row.get("example_id"))
        if example is None:
            failures.append("evaluation_example_%d" % index)
            continue
        field = example["task_field"]
        control = row.get("control")
        if control not in EXPECTED_CONTROLS:
            failures.append("evaluation_control_%d" % index)
            continue
        if row.get("sample_id") != example.get("sample_id"):
            failures.append("evaluation_sample_%d" % index)
        if row.get("task_field") != field:
            failures.append("evaluation_field_%d" % index)
        if row.get("expected_answer") != example.get("expected_answer"):
            failures.append("evaluation_true_answer_%d" % index)
        control_answer = example["control_expected_answers"].get(control)
        if row.get("control_expected_answer") != control_answer:
            failures.append("evaluation_control_answer_%d" % index)
        answer = row.get("answer")
        true_exact = (
            isinstance(answer, str)
            and answer.strip() == example.get("expected_answer")
        )
        control_exact = isinstance(answer, str) and answer.strip() == control_answer
        if row.get("canonical_exact") is not true_exact:
            failures.append("evaluation_exact_contract_%d" % index)
        if row.get("control_semantic_exact") is not control_exact:
            failures.append("evaluation_control_contract_%d" % index)
        if row.get("field_correct") != {field: true_exact}:
            failures.append("evaluation_field_contract_%d" % index)

    pre_metrics = {
        field: _row_rates(
            row for row in pre_rows if row.get("task_field") == field
        )
        for field in TARGET_FIELDS
    }
    post_metrics = {
        field: {
            control: _row_rates(
                row
                for row in post_rows
                if row.get("task_field") == field
                and row.get("control") == control
            )
            for control in EXPECTED_CONTROLS
        }
        for field in TARGET_FIELDS
    }
    true_label_metrics = {
        field: {
            label: _row_rates(
                row
                for row in post_rows
                if row.get("task_field") == field
                and row.get("control") == "true_u"
                and row.get("expected_answer") == label
            )
            for label in sorted(
                value
                for value in {
                    example.get("expected_answer")
                    for example in examples
                    if example.get("task_field") == field
                }
                if isinstance(value, str)
            )
        }
        for field in TARGET_FIELDS
    }
    gaps = {}
    for field in TARGET_FIELDS:
        true_accuracy = post_metrics[field]["true_u"]["true_target_accuracy"]
        control_ceiling = max(
            post_metrics[field]["zero_u"]["true_target_accuracy"],
            post_metrics[field]["spatial_shuffle"]["true_target_accuracy"],
        )
        gaps[field] = true_accuracy - control_ceiling
    causal_capacity_checks = {
        "each_true_field_at_least_eighty_percent": all(
            post_metrics[field]["true_u"]["true_target_accuracy"] >= 0.8
            for field in TARGET_FIELDS
        ),
        "every_true_label_at_least_eighty_percent": all(
            metric["true_target_accuracy"] >= 0.8
            for labels in true_label_metrics.values()
            for metric in labels.values()
        ),
        "each_true_control_gap_at_least_thirty_percent": all(
            gaps[field] >= 0.3 for field in TARGET_FIELDS
        ),
    }
    protocol_valid = not failures
    causal_capacity_passed = protocol_valid and all(causal_capacity_checks.values())
    status = (
        "causal_row_addressed_plumbing_overfit_pass"
        if causal_capacity_passed
        else (
            "valid_run_without_causal_grounding"
            if protocol_valid
            else "invalid_run"
        )
    )
    audit = {
        "schema": ROW_GROUNDING_OVERFIT_AUDIT_SCHEMA,
        "status": status,
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "protocol_valid": protocol_valid,
        "protocol_failures": failures,
        "causal_capacity_passed": causal_capacity_passed,
        "causal_capacity_checks": causal_capacity_checks,
        "pre_training_true_metrics": pre_metrics,
        "post_training_metrics": post_metrics,
        "post_training_true_label_metrics": true_label_metrics,
        "true_minus_control_ceiling": gaps,
        "loss_first": float(history[0]["loss"]) if history else None,
        "loss_last": float(history[-1]["loss"]) if history else None,
        "claim_boundary": claim_boundary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def audit_route_readout_overfit_report(
    report_path: Path, output_path: Path
) -> dict:
    """Audit V1f's matched-pair route-scalar readout diagnostic."""

    report_path = Path(report_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(
            "refusing to overwrite route-readout audit: %s" % output_path
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures = []
    stage = report.get("stage")
    stage_specs = {
        "V1f_route151_route_readout_overfit": {
            "config_schema": ROUTE_READOUT_CONFIG_SCHEMA,
            "projector_parameter_count": 1_330_734,
            "projector_type": "generic_mlp",
            "lora_parameter_count": 393_216,
            "lora_tensor_count": 16,
            "pass_status": "causal_route_readout_plumbing_pass",
            "negative_status": "valid_run_without_causal_route_readout",
        },
        "V1g_route151_typed_route_readout_overfit": {
            "config_schema": TYPED_ROUTE_READOUT_CONFIG_SCHEMA,
            "projector_parameter_count": 1_367_040,
            "projector_type": "typed_scalar_basis",
            "lora_parameter_count": 393_216,
            "lora_tensor_count": 16,
            "pass_status": "causal_typed_route_readout_plumbing_pass",
            "negative_status": "valid_run_without_causal_typed_route_readout",
        },
        "V1h_route151_slot_typed_route_readout_overfit": {
            "config_schema": SLOT_TYPED_ROUTE_READOUT_CONFIG_SCHEMA,
            "projector_parameter_count": 1_391_616,
            "projector_type": "slot_typed_scalar_basis",
            "lora_parameter_count": 393_216,
            "lora_tensor_count": 16,
            "pass_status": "causal_slot_typed_route_readout_plumbing_pass",
            "negative_status": (
                "valid_run_without_causal_slot_typed_route_readout"
            ),
        },
        "V1i_route151_slot_typed_full_attention_route_readout_overfit": {
            "config_schema": FULL_ATTENTION_SLOT_TYPED_ROUTE_READOUT_CONFIG_SCHEMA,
            "projector_parameter_count": 1_391_616,
            "projector_type": "slot_typed_scalar_basis",
            "lora_parameter_count": 1_572_864,
            "lora_tensor_count": 64,
            "pass_status": (
                "causal_slot_typed_full_attention_route_readout_plumbing_pass"
            ),
            "negative_status": (
                "valid_run_without_causal_slot_typed_full_attention_route_readout"
            ),
        },
        "V1j_route151_target_row_route_readout_overfit": {
            "config_schema": TARGET_ROW_ROUTE_READOUT_CONFIG_SCHEMA,
            "projector_parameter_count": 1_391_616,
            "projector_type": "slot_typed_scalar_basis",
            "lora_parameter_count": 1_572_864,
            "lora_tensor_count": 64,
            "pass_status": "target_row_disjoint_route_readout_plumbing_pass",
            "negative_status": (
                "valid_run_without_target_row_disjoint_route_readout"
            ),
        },
    }
    stage_spec = stage_specs.get(stage)
    if stage_spec is None:
        failures.append("report_stage")
        stage_spec = stage_specs["V1f_route151_route_readout_overfit"]
    is_v1j = stage == "V1j_route151_target_row_route_readout_overfit"
    claim_boundary = {
        "plumbing_overfit_only": True,
        "reportable_generalization": False,
        "safety_claim_allowed": False,
    }
    objective = {"type": "route_readout_pairs", "fields": ["route"]}
    if report.get("schema") != "orion.qwen-visibility-grounding-smoke-report/v1":
        failures.append("report_schema")
    if report.get("status") != "complete":
        failures.append("report_status")
    if report.get("claim_boundary") != claim_boundary:
        failures.append("claim_boundary")
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
    manifest_path = _check_recorded_file(report, "manifest", failures)
    curriculum_path = _check_recorded_file(report, "curriculum", failures)
    if protocol_path is not None:
        try:
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append("protocol_json")
        else:
            if protocol.get("schema") != stage_spec["config_schema"]:
                failures.append("protocol_schema")
            if protocol.get("stage") != stage:
                failures.append("protocol_stage")
            if protocol.get("projector", {}).get(
                "type", "generic_mlp"
            ) != stage_spec["projector_type"]:
                failures.append("protocol_projector_type")
            if protocol.get("objective") != objective:
                failures.append("protocol_objective")
            if protocol.get("training", {}).get("optimizer_steps") != 240:
                failures.append("protocol_optimizer_steps")
            if protocol.get("training", {}).get("separate_gradient_clipping") is not True:
                failures.append("protocol_gradient_clipping")
            expected_evaluation_split = (
                "held_out_target_rows"
                if stage == "V1j_route151_target_row_route_readout_overfit"
                else "held_out_order"
            )
            if protocol.get("evaluation", {}).get("split") != expected_evaluation_split:
                failures.append("protocol_evaluation_split")
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
            if curriculum_path is not None and Path(
                str(protocol.get("curriculum", ""))
            ).resolve() != curriculum_path:
                failures.append("protocol_curriculum_path")
            if stage in {
                "V1i_route151_slot_typed_full_attention_route_readout_overfit",
                "V1j_route151_target_row_route_readout_overfit",
            }:
                expected_projector = {
                    "type": "slot_typed_scalar_basis",
                    "feature_dim": 23,
                    "scalar_basis_dim": 4,
                    "maximum_token_slots": 48,
                    "hidden_dim": 512,
                    "vlm_hidden_dim": 2560,
                }
                expected_lora = {
                    "layer_indices": list(V1I_FULL_ATTENTION_LAYERS),
                    "module_names": list(V1I_LORA_MODULE_NAMES),
                    "rank": 8,
                    "alpha": 16.0,
                    "dropout": 0.0,
                }
                expected_training = {
                    "optimizer_steps": 240,
                    "projector_learning_rate": 0.0001,
                    "lora_learning_rate": 0.0002,
                    "weight_decay": 0.0,
                    "maximum_gradient_norm": 1.0,
                    "projector_maximum_gradient_norm": 1.0,
                    "lora_maximum_gradient_norm": 1.0,
                    "separate_gradient_clipping": True,
                    "gradient_checkpointing": True,
                    "seed": 42,
                }
                expected_evaluation = {
                    "max_new_tokens": 16,
                    "split": expected_evaluation_split,
                    "pre_training_controls": ["true_u"],
                    "controls": list(EXPECTED_CONTROLS),
                }
                if stage == "V1j_route151_target_row_route_readout_overfit":
                    expected_evaluation["spatial_shuffle_role"] = (
                        "reported_diagnostic_not_hard_gate"
                    )
                if protocol.get("base_bridge_config") != (
                    "configs/qwen_drive_b2d_agent_oracle_visibility_sft_v1.json"
                ):
                    failures.append("protocol_base_bridge_config")
                if protocol.get("projector") != expected_projector:
                    failures.append("protocol_projector_config")
                if protocol.get("lora") != expected_lora:
                    failures.append("protocol_lora_config")
                if protocol.get("training") != expected_training:
                    failures.append("protocol_training_config")
                if protocol.get("evaluation") != expected_evaluation:
                    failures.append("protocol_evaluation_config")

    curriculum = {}
    examples = []
    schedule = []
    training_ids = set()
    evaluation_ids = set()
    if curriculum_path is not None:
        try:
            curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append("curriculum_json")
        else:
            expected_curriculum_schema = (
                TARGET_ROW_ROUTE_READOUT_CURRICULUM_SCHEMA
                if is_v1j
                else ROUTE_READOUT_CURRICULUM_SCHEMA
            )
            if curriculum.get("schema") != expected_curriculum_schema:
                failures.append("curriculum_schema")
            false_flags = [
                "reportable_generalization",
                "controls_used_for_optimizer",
                "hidden_actor_labels_used",
                "planning_expert_used_for_optimizer",
            ]
            if is_v1j:
                false_flags.append("spatial_shuffle_examples_used_for_optimizer")
            if any(curriculum.get(name) is not False for name in false_flags):
                failures.append("curriculum_false_flags")
            true_flags = [
                "complete_row_permutations_only",
                "matched_pairs_differ_only_by_query_row_swap",
                "non_query_order_randomized",
                "spatial_shuffle_target_changed_for_every_example",
            ]
            true_flags.extend(
                [
                    "held_out_target_row_evaluation",
                    "queried_target_rows_disjoint_verified",
                ]
                if is_v1j
                else [
                    "held_out_order_evaluation",
                    "held_out_order_disjoint_verified",
                ]
            )
            if any(curriculum.get(name) is not True for name in true_flags):
                failures.append("curriculum_true_flags")
            if curriculum.get("query_frontier") != "F00":
                failures.append("curriculum_query_frontier")
            expected_example_count = 98 if is_v1j else 50
            if curriculum.get("example_count") != expected_example_count:
                failures.append("curriculum_example_count")
            if report.get("curriculum_example_count") != expected_example_count:
                failures.append("report_curriculum_example_count")
            train_variant_field = (
                "train_pair_variants" if is_v1j else "train_pair_variants_per_sample"
            )
            evaluation_variant_field = (
                "evaluation_pair_variants"
                if is_v1j
                else "evaluation_pair_variants_per_sample"
            )
            if curriculum.get(train_variant_field) != 3:
                failures.append("curriculum_train_variants")
            if curriculum.get(evaluation_variant_field) != 2:
                failures.append("curriculum_evaluation_variants")
            expected_training_labels = (
                {"OFF_ROUTE": 39, "ON_ROUTE": 39}
                if is_v1j
                else {"OFF_ROUTE": 15, "ON_ROUTE": 15}
            )
            if curriculum.get("training_label_counts") != expected_training_labels:
                failures.append("curriculum_training_labels")
            if curriculum.get("evaluation_label_counts") != {
                "OFF_ROUTE": 10,
                "ON_ROUTE": 10,
            }:
                failures.append("curriculum_evaluation_labels")
            if curriculum.get("optimizer_label_counts") != {
                "OFF_ROUTE": 120,
                "ON_ROUTE": 120,
            }:
                failures.append("curriculum_optimizer_labels")
            if curriculum.get("optimizer_steps") != 240:
                failures.append("curriculum_optimizer_steps")
            if is_v1j:
                if curriculum.get("distinct_training_target_pairs") != 13:
                    failures.append("curriculum_training_target_pairs")
                if curriculum.get("distinct_evaluation_target_pairs") != 5:
                    failures.append("curriculum_evaluation_target_pairs")
                if curriculum.get("fully_held_out_frame") != V1J_FULLY_HELD_OUT_FRAME:
                    failures.append("curriculum_fully_held_out_frame")
                if (
                    curriculum.get("spatial_shuffle_evaluation_role")
                    != "reported_diagnostic_not_hard_gate"
                ):
                    failures.append("curriculum_spatial_shuffle_role")
                expected_target_split = {
                    sample_id: {
                        split: [
                            {
                                "on_route_row": int(on_index),
                                "off_route_row": int(off_index),
                            }
                            for on_index, off_index in V1J_TARGET_ROW_PAIRS[
                                sample_id
                            ][split]
                        ]
                        for split in ("train", "held_out_target_rows")
                    }
                    for sample_id in sorted(V1J_TARGET_ROW_PAIRS)
                }
                if curriculum.get("target_row_split") != expected_target_split:
                    failures.append("curriculum_target_row_split")
            if manifest_path is not None:
                if Path(str(curriculum.get("base_manifest_path", ""))).resolve() != manifest_path:
                    failures.append("curriculum_manifest_path")
                if curriculum.get("base_manifest_sha256") != report.get(
                    "manifest_sha256"
                ):
                    failures.append("curriculum_manifest_sha256")
            examples = curriculum.get("examples", [])
            schedule = curriculum.get("training_schedule", [])
            training_ids = set(curriculum.get("training_example_ids", []))
            evaluation_ids = set(curriculum.get("evaluation_example_ids", []))

    example_by_id = {
        example.get("example_id"): example
        for example in examples
        if isinstance(example, dict)
    }
    expected_example_count = 98 if is_v1j else 50
    expected_training_count = 78 if is_v1j else 30
    expected_evaluation_count = 20
    if len(example_by_id) != expected_example_count or None in example_by_id:
        failures.append("curriculum_example_ids")
    if (
        len(training_ids) != expected_training_count
        or len(evaluation_ids) != expected_evaluation_count
        or training_ids & evaluation_ids
        or training_ids | evaluation_ids != set(example_by_id)
    ):
        failures.append("curriculum_split")
    if len(schedule) != 240 or any(value not in training_ids for value in schedule):
        failures.append("curriculum_schedule")
    if is_v1j:
        if Counter(
            example_by_id[value].get("expected_answer") for value in schedule
        ) != Counter({"OFF_ROUTE": 120, "ON_ROUTE": 120}):
            failures.append("curriculum_schedule_balance")
    elif Counter(schedule) != Counter({value: 8 for value in training_ids}):
        failures.append("curriculum_schedule_balance")
    pairs = {}
    target_rows = {
        "train": {},
        "held_out_target_rows": {},
    }
    for example_id, example in example_by_id.items():
        expected_split = (
            "train"
            if example_id in training_ids
            else ("held_out_target_rows" if is_v1j else "held_out_order")
        )
        if example.get("split") != expected_split:
            failures.append("curriculum_example_split_" + str(example_id))
        if example.get("task_field") != "route":
            failures.append("curriculum_example_field_" + str(example_id))
        answer = example.get("expected_answer")
        if answer not in {"ON_ROUTE", "OFF_ROUTE"}:
            failures.append("curriculum_example_answer_" + str(example_id))
        controls = example.get("control_expected_answers", {})
        if controls.get("true_u") != answer:
            failures.append("curriculum_true_target_" + str(example_id))
        if controls.get("spatial_shuffle") == answer:
            failures.append("curriculum_shuffle_target_" + str(example_id))
        permutation = example.get("sequence_permutation_new_to_manifest", [])
        if sorted(permutation) != list(range(32)):
            failures.append("curriculum_permutation_" + str(example_id))
        pairs.setdefault(example.get("pair_id"), []).append(example)
        if is_v1j:
            sample_rows = target_rows[expected_split].setdefault(
                example.get("sample_id"), set()
            )
            try:
                sample_rows.add(int(str(example.get("source_manifest_frontier"))[1:]))
            except (TypeError, ValueError):
                failures.append("curriculum_source_row_" + str(example_id))
    if not is_v1j:
        for sample_id in EXPECTED_SAMPLE_IDS:
            for label in ("ON_ROUTE", "OFF_ROUTE"):
                train_orders = {
                    tuple(example_by_id[value].get("sequence_permutation_new_to_manifest", []))
                    for value in training_ids
                    if example_by_id.get(value, {}).get("sample_id") == sample_id
                    and example_by_id.get(value, {}).get("expected_answer") == label
                }
                evaluation_orders = {
                    tuple(example_by_id[value].get("sequence_permutation_new_to_manifest", []))
                    for value in evaluation_ids
                    if example_by_id.get(value, {}).get("sample_id") == sample_id
                    and example_by_id.get(value, {}).get("expected_answer") == label
                }
                if len(train_orders) != 3 or len(evaluation_orders) != 2:
                    failures.append("curriculum_order_variants_%s_%s" % (sample_id, label))
                if train_orders & evaluation_orders:
                    failures.append("curriculum_order_leak_%s_%s" % (sample_id, label))
    else:
        for sample_id in EXPECTED_SAMPLE_IDS:
            train_rows = target_rows["train"].get(sample_id, set())
            evaluation_rows = target_rows["held_out_target_rows"].get(sample_id, set())
            if train_rows & evaluation_rows:
                failures.append("curriculum_target_row_leak_" + sample_id)
        if target_rows["train"].get(V1J_FULLY_HELD_OUT_FRAME, set()):
            failures.append("curriculum_held_out_frame_leak")
    expected_pair_count = 49 if is_v1j else 25
    if len(pairs) != expected_pair_count or any(len(pair) != 2 for pair in pairs.values()):
        failures.append("curriculum_pairs")
    else:
        for pair_id, pair in pairs.items():
            if {example.get("expected_answer") for example in pair} != {
                "ON_ROUTE",
                "OFF_ROUTE",
            }:
                failures.append("curriculum_pair_labels_" + str(pair_id))
                continue
            if pair[0].get("sample_id") != pair[1].get("sample_id"):
                failures.append("curriculum_pair_sample_" + str(pair_id))
            first = pair[0].get("sequence_permutation_new_to_manifest", [])
            second = pair[1].get("sequence_permutation_new_to_manifest", [])
            if len(first) != 32 or len(second) != 32:
                continue
            changed = [
                index
                for index, values in enumerate(zip(first, second))
                if values[0] != values[1]
            ]
            if len(changed) != 2 or 0 not in changed:
                failures.append("curriculum_pair_swap_" + str(pair_id))

    scope = report.get("scope", {})
    expected_scope = {
        "projector_trainable_parameter_count": stage_spec[
            "projector_parameter_count"
        ],
        "model_trainable_parameter_count": stage_spec["lora_parameter_count"],
        "vision_trainable_parameter_count": 0,
        "planning_expert_trainable_parameter_count": 0,
        "embedding_trainable": False,
        "lm_head_trainable": False,
    }
    for name, expected in expected_scope.items():
        if scope.get(name) != expected:
            failures.append("scope_" + name)
    if stage in {
        "V1i_route151_slot_typed_full_attention_route_readout_overfit",
        "V1j_route151_target_row_route_readout_overfit",
    }:
        expected_lora = {
            "layer_indices": list(V1I_FULL_ATTENTION_LAYERS),
            "module_names": list(V1I_LORA_MODULE_NAMES),
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.0,
        }
        expected_modules = [
            "vlm.model.language_model.layers.%d.self_attn.%s" % (layer, module)
            for layer in V1I_FULL_ATTENTION_LAYERS
            for module in V1I_LORA_MODULE_NAMES
        ]
        expected_trainable_names = [
            module + suffix
            for module in expected_modules
            for suffix in (".lora_a", ".lora_b")
        ]
        if report.get("lora") != expected_lora:
            failures.append("report_lora_config")
        if report.get("installed_lora_modules") != expected_modules:
            failures.append("installed_lora_modules")
        if scope.get("model_trainable_names") != expected_trainable_names:
            failures.append("scope_model_trainable_names")

    history = report.get("history", [])
    if len(history) != 240 or [row.get("optimizer_step") for row in history] != list(
        range(1, 241)
    ):
        failures.append("optimizer_step_sequence")
    if len(schedule) != 240 or [row.get("example_id") for row in history] != schedule:
        failures.append("optimizer_schedule")
    if any(row.get("task_field") != "route" for row in history):
        failures.append("optimizer_task_field")
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
            values = [float(row.get(field, float("nan"))) for field in numeric_fields]
        except (TypeError, ValueError):
            values = [float("nan")]
        if any(not math.isfinite(value) for value in values):
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
    if checkpoint.get("projector_tensor_count") != 7:
        failures.append("checkpoint_projector_tensors")
    if checkpoint.get("lora_tensor_count") != stage_spec["lora_tensor_count"]:
        failures.append("checkpoint_lora_tensors")

    pre_rows = report.get("pre_training_evaluations", [])
    post_rows = report.get("evaluations", [])
    expected_pre = {(example_id, "true_u") for example_id in evaluation_ids}
    actual_pre = [(row.get("example_id"), row.get("control")) for row in pre_rows]
    if len(actual_pre) != len(expected_pre) or set(actual_pre) != expected_pre:
        failures.append("pre_evaluation_coverage")
    expected_post = {
        (example_id, control)
        for example_id in evaluation_ids
        for control in EXPECTED_CONTROLS
    }
    actual_post = [(row.get("example_id"), row.get("control")) for row in post_rows]
    if len(actual_post) != len(expected_post) or set(actual_post) != expected_post:
        failures.append("post_evaluation_coverage")
    for index, row in enumerate(pre_rows + post_rows):
        example = example_by_id.get(row.get("example_id"))
        if example is None:
            failures.append("evaluation_example_%d" % index)
            continue
        control = row.get("control")
        if control not in EXPECTED_CONTROLS:
            failures.append("evaluation_control_%d" % index)
            continue
        if row.get("sample_id") != example.get("sample_id"):
            failures.append("evaluation_sample_%d" % index)
        if row.get("task_field") != "route":
            failures.append("evaluation_field_%d" % index)
        expected_row_split = "held_out_target_rows" if is_v1j else "held_out_order"
        if row.get("split") != expected_row_split:
            failures.append("evaluation_split_%d" % index)
        if row.get("pair_id") != example.get("pair_id"):
            failures.append("evaluation_pair_%d" % index)
        true_answer = example.get("expected_answer")
        control_answer = example.get("control_expected_answers", {}).get(control)
        answer = row.get("answer")
        true_exact = isinstance(answer, str) and answer.strip() == true_answer
        control_exact = isinstance(answer, str) and answer.strip() == control_answer
        if row.get("expected_answer") != true_answer:
            failures.append("evaluation_true_answer_%d" % index)
        if row.get("control_expected_answer") != control_answer:
            failures.append("evaluation_control_answer_%d" % index)
        if row.get("canonical_exact") is not true_exact:
            failures.append("evaluation_exact_contract_%d" % index)
        if row.get("control_semantic_exact") is not control_exact:
            failures.append("evaluation_control_contract_%d" % index)
        if row.get("field_correct") != {"route": true_exact}:
            failures.append("evaluation_field_contract_%d" % index)

    pre_metrics = _row_rates(pre_rows)
    post_metrics = {
        control: _row_rates(
            row for row in post_rows if row.get("control") == control
        )
        for control in EXPECTED_CONTROLS
    }
    true_label_metrics = {
        label: _row_rates(
            row
            for row in post_rows
            if row.get("control") == "true_u"
            and row.get("expected_answer") == label
        )
        for label in ("ON_ROUTE", "OFF_ROUTE")
    }
    true_rows_by_pair = {
        row.get("pair_id"): [] for row in post_rows if row.get("control") == "true_u"
    }
    for row in post_rows:
        if row.get("control") == "true_u":
            true_rows_by_pair.setdefault(row.get("pair_id"), []).append(row)
    held_out_pair_rows = list(true_rows_by_pair.values())
    matched_pair_accuracy = (
        sum(
            len(rows) == 2 and all(bool(row.get("canonical_exact")) for row in rows)
            for rows in held_out_pair_rows
        )
        / len(held_out_pair_rows)
        if held_out_pair_rows
        else 0.0
    )
    true_accuracy = post_metrics["true_u"]["true_target_accuracy"]
    control_ceiling = max(
        post_metrics["zero_u"]["true_target_accuracy"],
        post_metrics["spatial_shuffle"]["true_target_accuracy"],
    )
    true_control_gap = true_accuracy - control_ceiling
    causal_capacity_checks = {
        "held_out_true_at_least_ninety_percent": true_accuracy >= 0.9,
        "each_held_out_label_at_least_ninety_percent": all(
            metric["true_target_accuracy"] >= 0.9
            for metric in true_label_metrics.values()
        ),
        "matched_pair_flip_at_least_eighty_percent": matched_pair_accuracy >= 0.8,
        "true_control_gap_at_least_thirty_percent": true_control_gap >= 0.3,
    }
    if not is_v1j:
        causal_capacity_checks[
            "spatial_shuffle_control_target_at_least_eighty_percent"
        ] = post_metrics["spatial_shuffle"]["control_target_accuracy"] >= 0.8
    protocol_valid = not failures
    causal_capacity_passed = protocol_valid and all(causal_capacity_checks.values())
    status = (
        stage_spec["pass_status"]
        if causal_capacity_passed
        else (
            stage_spec["negative_status"]
            if protocol_valid
            else "invalid_run"
        )
    )
    audit = {
        "schema": ROUTE_READOUT_OVERFIT_AUDIT_SCHEMA,
        "status": status,
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "protocol_valid": protocol_valid,
        "protocol_failures": failures,
        "causal_capacity_passed": causal_capacity_passed,
        "projector_type": stage_spec["projector_type"],
        "causal_capacity_checks": causal_capacity_checks,
        "pre_training_held_out_metrics": pre_metrics,
        "post_training_held_out_metrics": post_metrics,
        "post_training_true_label_metrics": true_label_metrics,
        "held_out_matched_pair_accuracy": matched_pair_accuracy,
        "true_minus_control_ceiling": true_control_gap,
        "spatial_shuffle_role": (
            "reported_diagnostic_not_hard_gate" if is_v1j else "hard_gate"
        ),
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
    if report.get("stage") in {
        "V1f_route151_route_readout_overfit",
        "V1g_route151_typed_route_readout_overfit",
        "V1h_route151_slot_typed_route_readout_overfit",
        "V1i_route151_slot_typed_full_attention_route_readout_overfit",
        "V1j_route151_target_row_route_readout_overfit",
    }:
        return audit_route_readout_overfit_report(report_path, output_path)
    if report.get("stage") == "V1e_route151_row_addressed_overfit":
        return audit_row_addressed_grounding_overfit_report(
            report_path, output_path
        )
    if report.get("stage") == "V1d_route151_factorized_overfit":
        return audit_factorized_grounding_overfit_report(report_path, output_path)
    return audit_grounding_overfit_report(report_path, output_path)

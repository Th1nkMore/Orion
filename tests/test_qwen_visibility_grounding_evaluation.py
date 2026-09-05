import hashlib
import importlib.util
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_grounding_evaluation.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_qwen_visibility_grounding_evaluation_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluation = _load_module()


def _row(sample_id, control, exact, correct):
    return {
        "sample_id": sample_id,
        "control": control,
        "answer": "{}",
        "parsed": {} if correct else None,
        "canonical_exact": exact,
        "field_correct": {field: correct for field in evaluation.TARGET_FIELDS},
    }


def _report(tmp_path, causal=True):
    checkpoint = tmp_path / "adaptation.pt"
    checkpoint.write_bytes(b"small-adaptation")
    import hashlib

    history = []
    for step in range(15):
        history.append(
            {
                "optimizer_step": step + 1,
                "sample_id": evaluation.EXPECTED_SAMPLE_IDS[step % 5],
                "loss": 2.0 - step / 20,
                "projector_gradient_norm_before_clip": 10.0,
                "lora_gradient_norm_before_clip": 0.2,
                "projector_update_norm": 0.1,
                "lora_update_norm": 0.01,
                "projector_nonzero_gradient_tensors": 3,
                "lora_nonzero_gradient_tensors": 8,
                "forward_seconds": 1.0,
                "backward_seconds": 2.0,
                "optimizer_step_seconds": 3.0,
            }
        )
    post = []
    for sample_id in evaluation.EXPECTED_SAMPLE_IDS:
        post.append(_row(sample_id, "true_u", causal, causal))
        post.append(_row(sample_id, "zero_u", False, False))
        post.append(_row(sample_id, "spatial_shuffle", False, False))
    return {
        "schema": "orion.qwen-visibility-grounding-smoke-report/v1",
        "status": "complete",
        "stage": "V1c_route151_plumbing_overfit",
        "claim_boundary": {
            "plumbing_overfit_only": True,
            "reportable_generalization": False,
            "safety_claim_allowed": False,
        },
        "sample_ids": list(evaluation.EXPECTED_SAMPLE_IDS),
        "optimizer_controls": [],
        "hidden_actor_labels_used": False,
        "planning_expert_in_optimizer": False,
        "scope": {
            "projector_trainable_parameter_count": 1_330_734,
            "model_trainable_parameter_count": 393_216,
            "vision_trainable_parameter_count": 0,
            "planning_expert_trainable_parameter_count": 0,
            "embedding_trainable": False,
            "lm_head_trainable": False,
        },
        "history": history,
        "pre_training_evaluations": [
            _row(sample_id, "true_u", False, False)
            for sample_id in evaluation.EXPECTED_SAMPLE_IDS
        ],
        "evaluations": post,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "contains_optimizer_state": False,
            "contains_base_model_weights": False,
            "projector_tensor_count": 7,
            "lora_tensor_count": 16,
        },
    }


def _factorized_row(sample_id, field, control, correct):
    expected = {
        "frontier": "F07",
        "route": "ON_ROUTE",
        "margin": "NEAR",
        "action": "SLOW",
    }[field]
    incorrect = {
        "frontier": "F00",
        "route": "OFF_ROUTE",
        "margin": "CLEAR",
        "action": "KEEP",
    }[field]
    answer = expected if correct else incorrect
    return {
        "sample_id": sample_id,
        "task_field": field,
        "control": control,
        "answer": answer,
        "parsed": answer,
        "expected_answer": expected,
        "canonical_exact": correct,
        "field_correct": {field: correct},
    }


def _factorized_report(tmp_path, causal=True):
    checkpoint = tmp_path / "factorized-adaptation.pt"
    checkpoint.write_bytes(b"factorized-small-adaptation")
    protocol = {
        "schema": "orion.qwen-visibility-grounding-factorized-config/v1",
        "stage": "V1d_route151_factorized_overfit",
        "sample_ids": list(evaluation.EXPECTED_SAMPLE_IDS),
        "objective": {
            "type": "factorized_fields",
            "fields": list(evaluation.TARGET_FIELDS),
        },
        "training": {
            "optimizer_steps": 60,
            "separate_gradient_clipping": True,
        },
        "evaluation": {
            "pre_training_controls": ["true_u"],
            "controls": list(evaluation.EXPECTED_CONTROLS),
        },
        "claim_boundary": {
            "plumbing_overfit_only": True,
            "reportable_generalization": False,
            "safety_claim_allowed": False,
        },
    }
    protocol_path = tmp_path / "factorized-protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    manifest_path = tmp_path / "factorized-manifest.json"
    manifest_path.write_text('{"immutable":"synthetic-test"}')
    history = []
    step = 0
    for _epoch in range(3):
        for sample_id in evaluation.EXPECTED_SAMPLE_IDS:
            for field in evaluation.TARGET_FIELDS:
                step += 1
                history.append(
                    {
                        "optimizer_step": step,
                        "sample_id": sample_id,
                        "task_field": field,
                        "loss": 2.0 - step / 100,
                        "projector_gradient_norm_before_clip": 10.0,
                        "lora_gradient_norm_before_clip": 0.2,
                        "projector_update_norm": 0.1,
                        "lora_update_norm": 0.01,
                        "projector_nonzero_gradient_tensors": 3,
                        "lora_nonzero_gradient_tensors": 8,
                        "forward_seconds": 1.0,
                        "backward_seconds": 2.0,
                        "optimizer_step_seconds": 3.0,
                    }
                )
    pre = []
    post = []
    for sample_id in evaluation.EXPECTED_SAMPLE_IDS:
        for field in evaluation.TARGET_FIELDS:
            pre.append(_factorized_row(sample_id, field, "true_u", False))
            for control in evaluation.EXPECTED_CONTROLS:
                if control == "true_u":
                    correct = True
                elif not causal:
                    correct = True
                elif field == "route":
                    correct = True
                elif control == "spatial_shuffle" and field in {"margin", "action"}:
                    correct = True
                else:
                    correct = False
                post.append(_factorized_row(sample_id, field, control, correct))
    return {
        "schema": "orion.qwen-visibility-grounding-smoke-report/v1",
        "status": "complete",
        "stage": "V1d_route151_factorized_overfit",
        "objective": protocol["objective"],
        "claim_boundary": protocol["claim_boundary"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "sample_ids": list(evaluation.EXPECTED_SAMPLE_IDS),
        "optimizer_controls": [],
        "hidden_actor_labels_used": False,
        "planning_expert_in_optimizer": False,
        "scope": {
            "projector_trainable_parameter_count": 1_330_734,
            "model_trainable_parameter_count": 393_216,
            "vision_trainable_parameter_count": 0,
            "planning_expert_trainable_parameter_count": 0,
            "embedding_trainable": False,
            "lm_head_trainable": False,
        },
        "history": history,
        "pre_training_evaluations": pre,
        "evaluations": post,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "contains_optimizer_state": False,
            "contains_base_model_weights": False,
            "projector_tensor_count": 7,
            "lora_tensor_count": 16,
        },
    }


def test_causal_overfit_passes_only_as_nonreportable_plumbing(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report(tmp_path, causal=True)))
    output = tmp_path / "audit.json"
    audit = evaluation.audit_grounding_overfit_report(report_path, output)
    assert audit["protocol_valid"] is True
    assert audit["causal_capacity_passed"] is True
    assert audit["status"] == "causal_plumbing_overfit_pass"
    assert audit["claim_boundary"]["reportable_generalization"] is False


def test_valid_but_noncausal_training_is_not_accepted(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report(tmp_path, causal=False)))
    audit = evaluation.audit_grounding_overfit_report(
        report_path, tmp_path / "audit.json"
    )
    assert audit["protocol_valid"] is True
    assert audit["causal_capacity_passed"] is False
    assert audit["status"] == "valid_run_without_causal_grounding"


def test_scope_violation_invalidates_report(tmp_path):
    report = _report(tmp_path, causal=True)
    report["scope"]["planning_expert_trainable_parameter_count"] = 1
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    audit = evaluation.audit_grounding_overfit_report(
        report_path, tmp_path / "audit.json"
    )
    assert audit["protocol_valid"] is False
    assert audit["causal_capacity_passed"] is False
    assert "planning_expert_not_frozen" in audit["protocol_failures"]


def test_factorized_causal_overfit_accepts_control_semantics(tmp_path):
    report_path = tmp_path / "factorized-report.json"
    report_path.write_text(json.dumps(_factorized_report(tmp_path, causal=True)))
    audit = evaluation.audit_factorized_grounding_overfit_report(
        report_path, tmp_path / "factorized-audit.json"
    )
    assert audit["protocol_valid"] is True
    assert audit["causal_capacity_passed"] is True
    assert audit["status"] == "causal_factorized_plumbing_overfit_pass"
    assert audit["post_training_metrics"]["route"]["spatial_shuffle"][
        "exact_accuracy"
    ] == 1.0
    assert audit["post_training_metrics"]["margin"]["spatial_shuffle"][
        "exact_accuracy"
    ] == 1.0
    assert audit["claim_boundary"]["reportable_generalization"] is False


def test_factorized_true_without_causal_gaps_is_not_accepted(tmp_path):
    report_path = tmp_path / "factorized-report.json"
    report_path.write_text(json.dumps(_factorized_report(tmp_path, causal=False)))
    audit = evaluation.audit_visibility_grounding_report(
        report_path, tmp_path / "factorized-audit.json"
    )
    assert audit["protocol_valid"] is True
    assert audit["causal_capacity_passed"] is False
    assert audit["status"] == "valid_run_without_causal_grounding"


def test_factorized_unbalanced_optimizer_invalidates_report(tmp_path):
    report = _factorized_report(tmp_path, causal=True)
    report["history"][0]["task_field"] = "action"
    report_path = tmp_path / "factorized-report.json"
    report_path.write_text(json.dumps(report))
    audit = evaluation.audit_factorized_grounding_overfit_report(
        report_path, tmp_path / "factorized-audit.json"
    )
    assert audit["protocol_valid"] is False
    assert audit["causal_capacity_passed"] is False
    assert "optimizer_sample_field_balance" in audit["protocol_failures"]

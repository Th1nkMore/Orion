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


def _alternate_label(field, label):
    labels = {
        "frontier": ("F03", "F13", "F23"),
        "route": ("ON_ROUTE", "OFF_ROUTE"),
        "margin": ("INSIDE", "NEAR", "CLEAR"),
        "action": ("KEEP", "SLOW", "STOP"),
    }[field]
    return labels[(labels.index(label) + 1) % len(labels)]


def _row_addressed_report(tmp_path, causal=True):
    manifest_path = tmp_path / "row-base-manifest.json"
    manifest_path.write_text('{"immutable":"synthetic-row-test"}')
    examples = []
    label_repetitions = {
        "frontier": {"F03": 5, "F13": 5, "F23": 5},
        "route": {"OFF_ROUTE": 5, "ON_ROUTE": 5},
        "margin": {"CLEAR": 4, "INSIDE": 4, "NEAR": 4},
        "action": {"KEEP": 2, "SLOW": 2, "STOP": 2},
    }
    sample_index = 0
    for field in evaluation.TARGET_FIELDS:
        for label, repetitions in label_repetitions[field].items():
            for repetition in range(repetitions):
                sample_id = evaluation.EXPECTED_SAMPLE_IDS[
                    sample_index % len(evaluation.EXPECTED_SAMPLE_IDS)
                ]
                sample_index += 1
                example_id = "%s-%s-%d" % (field, label, repetition)
                examples.append(
                    {
                        "example_id": example_id,
                        "sample_id": sample_id,
                        "task_field": field,
                        "expected_answer": label,
                        "control_expected_answers": {
                            "true_u": label,
                            "zero_u": _alternate_label(field, label),
                            "spatial_shuffle": _alternate_label(field, label),
                        },
                        "sequence_permutation_new_to_manifest": list(range(32)),
                    }
                )
    pools = {}
    for example in examples:
        pools.setdefault(
            (example["task_field"], example["expected_answer"]), []
        ).append(example["example_id"])
    for pool in pools.values():
        pool.sort()
    label_order = {
        "frontier": ("F03", "F13", "F23"),
        "route": ("ON_ROUTE", "OFF_ROUTE"),
        "margin": ("INSIDE", "NEAR", "CLEAR"),
        "action": ("KEEP", "SLOW", "STOP"),
    }
    schedule = []
    for round_index in range(90):
        for field in evaluation.TARGET_FIELDS:
            labels = label_order[field]
            label_index = round_index % len(labels)
            label = labels[label_index]
            occurrence = round_index // len(labels)
            pool = pools[(field, label)]
            schedule.append(pool[occurrence % len(pool)])
    curriculum = {
        "schema": "orion.qwen-visibility-row-grounding-curriculum/v1",
        "base_manifest_path": str(manifest_path),
        "base_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "reportable_generalization": False,
        "controls_used_for_optimizer": False,
        "hidden_actor_labels_used": False,
        "planning_expert_used_for_optimizer": False,
        "complete_row_permutations_only": True,
        "spatial_shuffle_target_changed_for_every_example": True,
        "example_count": 43,
        "steps_per_field": 90,
        "optimizer_steps": 360,
        "field_counts": {
            "action": 6,
            "frontier": 15,
            "margin": 12,
            "route": 10,
        },
        "label_counts": label_repetitions,
        "training_schedule": schedule,
        "examples": examples,
    }
    curriculum_path = tmp_path / "row-curriculum.json"
    curriculum_path.write_text(json.dumps(curriculum))
    claim_boundary = {
        "plumbing_overfit_only": True,
        "reportable_generalization": False,
        "safety_claim_allowed": False,
    }
    objective = {
        "type": "row_addressed_fields",
        "fields": list(evaluation.TARGET_FIELDS),
    }
    protocol = {
        "schema": "orion.qwen-visibility-row-grounding-config/v1",
        "stage": "V1e_route151_row_addressed_overfit",
        "sample_ids": list(evaluation.EXPECTED_SAMPLE_IDS),
        "objective": objective,
        "curriculum": str(curriculum_path),
        "training": {
            "optimizer_steps": 360,
            "separate_gradient_clipping": True,
        },
        "evaluation": {
            "pre_training_controls": ["true_u"],
            "controls": list(evaluation.EXPECTED_CONTROLS),
        },
        "claim_boundary": claim_boundary,
    }
    protocol_path = tmp_path / "row-protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    by_id = {example["example_id"]: example for example in examples}
    history = []
    for step, example_id in enumerate(schedule, start=1):
        example = by_id[example_id]
        history.append(
            {
                "optimizer_step": step,
                "example_id": example_id,
                "sample_id": example["sample_id"],
                "task_field": example["task_field"],
                "loss": 2.0 - step / 500,
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

    def row(example, control, correct):
        true_answer = example["expected_answer"]
        control_answer = example["control_expected_answers"][control]
        answer = true_answer if correct else control_answer
        true_exact = answer == true_answer
        return {
            "example_id": example["example_id"],
            "sample_id": example["sample_id"],
            "task_field": example["task_field"],
            "control": control,
            "answer": answer,
            "parsed": answer,
            "expected_answer": true_answer,
            "control_expected_answer": control_answer,
            "canonical_exact": true_exact,
            "control_semantic_exact": answer == control_answer,
            "field_correct": {example["task_field"]: true_exact},
        }

    pre = [row(example, "true_u", False) for example in examples]
    post = []
    for example in examples:
        for control in evaluation.EXPECTED_CONTROLS:
            post.append(row(example, control, control == "true_u" or not causal))
    checkpoint = tmp_path / "row-adaptation.pt"
    checkpoint.write_bytes(b"row-small-adaptation")
    return {
        "schema": "orion.qwen-visibility-grounding-smoke-report/v1",
        "status": "complete",
        "stage": "V1e_route151_row_addressed_overfit",
        "objective": objective,
        "claim_boundary": claim_boundary,
        "protocol_path": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "curriculum_path": str(curriculum_path),
        "curriculum_sha256": hashlib.sha256(
            curriculum_path.read_bytes()
        ).hexdigest(),
        "curriculum_example_count": 43,
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


def test_row_addressed_causal_overfit_passes_only_as_plumbing(tmp_path):
    report_path = tmp_path / "row-report.json"
    report_path.write_text(json.dumps(_row_addressed_report(tmp_path, causal=True)))
    audit = evaluation.audit_visibility_grounding_report(
        report_path, tmp_path / "row-audit.json"
    )
    assert audit["protocol_valid"] is True
    assert audit["causal_capacity_passed"] is True
    assert audit["status"] == "causal_row_addressed_plumbing_overfit_pass"
    assert all(
        gap == 1.0 for gap in audit["true_minus_control_ceiling"].values()
    )
    assert audit["claim_boundary"]["reportable_generalization"] is False


def test_row_addressed_noncausal_outputs_do_not_pass(tmp_path):
    report_path = tmp_path / "row-report.json"
    report_path.write_text(json.dumps(_row_addressed_report(tmp_path, causal=False)))
    audit = evaluation.audit_row_addressed_grounding_overfit_report(
        report_path, tmp_path / "row-audit.json"
    )
    assert audit["protocol_valid"] is True
    assert audit["causal_capacity_passed"] is False
    assert audit["status"] == "valid_run_without_causal_grounding"


def test_row_addressed_schedule_mismatch_invalidates_report(tmp_path):
    report = _row_addressed_report(tmp_path, causal=True)
    report["history"][0]["example_id"] = report["history"][1]["example_id"]
    report_path = tmp_path / "row-report.json"
    report_path.write_text(json.dumps(report))
    audit = evaluation.audit_row_addressed_grounding_overfit_report(
        report_path, tmp_path / "row-audit.json"
    )
    assert audit["protocol_valid"] is False
    assert "optimizer_schedule" in audit["protocol_failures"]


def _route_readout_report(tmp_path, causal=True):
    manifest_path = tmp_path / "route-readout-manifest.json"
    manifest_path.write_text('{"immutable":"synthetic-route-readout"}')
    examples = []
    training_ids = []
    evaluation_ids = []
    for sample_index, sample_id in enumerate(evaluation.EXPECTED_SAMPLE_IDS):
        for split, variants in (("train", 3), ("held_out_order", 2)):
            for variant in range(variants):
                pair_id = "%s-%s-%d" % (sample_id, split, variant)
                on_order = list(range(32))
                order_variant = variant + (0 if split == "train" else 3)
                on_source = 1 + (sample_index * 5 + order_variant) % 15
                off_source = 16 + (sample_index * 3 + order_variant) % 15
                on_position = on_order.index(on_source)
                on_order[0], on_order[on_position] = (
                    on_order[on_position],
                    on_order[0],
                )
                off_position = on_order.index(off_source)
                off_order = list(on_order)
                off_order[0], off_order[off_position] = (
                    off_order[off_position],
                    off_order[0],
                )
                for label, order in (
                    ("ON_ROUTE", on_order),
                    ("OFF_ROUTE", off_order),
                ):
                    alternate = "OFF_ROUTE" if label == "ON_ROUTE" else "ON_ROUTE"
                    example_id = "%s-%s" % (pair_id, label.lower())
                    examples.append(
                        {
                            "example_id": example_id,
                            "sample_id": sample_id,
                            "task_field": "route",
                            "split": split,
                            "pair_id": pair_id,
                            "expected_answer": label,
                            "control_expected_answers": {
                                "true_u": label,
                                "zero_u": alternate,
                                "spatial_shuffle": alternate,
                            },
                            "sequence_permutation_new_to_manifest": order,
                        }
                    )
                    (training_ids if split == "train" else evaluation_ids).append(
                        example_id
                    )
    schedule = []
    for _ in range(8):
        schedule.extend(sorted(training_ids))
    curriculum = {
        "schema": evaluation.ROUTE_READOUT_CURRICULUM_SCHEMA,
        "base_manifest_path": str(manifest_path),
        "base_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "reportable_generalization": False,
        "controls_used_for_optimizer": False,
        "hidden_actor_labels_used": False,
        "planning_expert_used_for_optimizer": False,
        "complete_row_permutations_only": True,
        "matched_pairs_differ_only_by_query_row_swap": True,
        "non_query_order_randomized": True,
        "held_out_order_evaluation": True,
        "held_out_order_disjoint_verified": True,
        "spatial_shuffle_target_changed_for_every_example": True,
        "query_frontier": "F00",
        "example_count": 50,
        "train_pair_variants_per_sample": 3,
        "evaluation_pair_variants_per_sample": 2,
        "training_label_counts": {"OFF_ROUTE": 15, "ON_ROUTE": 15},
        "evaluation_label_counts": {"OFF_ROUTE": 10, "ON_ROUTE": 10},
        "optimizer_steps": 240,
        "optimizer_label_counts": {"OFF_ROUTE": 120, "ON_ROUTE": 120},
        "training_example_ids": sorted(training_ids),
        "evaluation_example_ids": sorted(evaluation_ids),
        "training_schedule": schedule,
        "examples": examples,
    }
    curriculum_path = tmp_path / "route-readout-curriculum.json"
    curriculum_path.write_text(json.dumps(curriculum))
    claim_boundary = {
        "plumbing_overfit_only": True,
        "reportable_generalization": False,
        "safety_claim_allowed": False,
    }
    objective = {"type": "route_readout_pairs", "fields": ["route"]}
    protocol = {
        "schema": evaluation.ROUTE_READOUT_CONFIG_SCHEMA,
        "stage": "V1f_route151_route_readout_overfit",
        "sample_ids": list(evaluation.EXPECTED_SAMPLE_IDS),
        "objective": objective,
        "curriculum": str(curriculum_path),
        "training": {
            "optimizer_steps": 240,
            "separate_gradient_clipping": True,
        },
        "evaluation": {
            "split": "held_out_order",
            "pre_training_controls": ["true_u"],
            "controls": list(evaluation.EXPECTED_CONTROLS),
        },
        "claim_boundary": claim_boundary,
    }
    protocol_path = tmp_path / "route-readout-protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    by_id = {example["example_id"]: example for example in examples}
    history = []
    for step, example_id in enumerate(schedule, start=1):
        example = by_id[example_id]
        history.append(
            {
                "optimizer_step": step,
                "example_id": example_id,
                "sample_id": example["sample_id"],
                "task_field": "route",
                "loss": 2.0 - step / 500,
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

    def result_row(example, control):
        true_answer = example["expected_answer"]
        control_answer = example["control_expected_answers"][control]
        answer = (
            true_answer
            if control == "true_u" or not causal
            else control_answer
        )
        true_exact = answer == true_answer
        return {
            "example_id": example["example_id"],
            "sample_id": example["sample_id"],
            "task_field": "route",
            "split": "held_out_order",
            "pair_id": example["pair_id"],
            "control": control,
            "answer": answer,
            "parsed": answer,
            "expected_answer": true_answer,
            "control_expected_answer": control_answer,
            "control_semantic_exact": answer == control_answer,
            "canonical_exact": true_exact,
            "field_correct": {"route": true_exact},
        }

    held_out_examples = [by_id[example_id] for example_id in evaluation_ids]
    pre = [result_row(example, "true_u") for example in held_out_examples]
    for row in pre:
        row["answer"] = ""
        row["parsed"] = ""
        row["canonical_exact"] = False
        row["control_semantic_exact"] = False
        row["field_correct"] = {"route": False}
    post = [
        result_row(example, control)
        for example in held_out_examples
        for control in evaluation.EXPECTED_CONTROLS
    ]
    checkpoint = tmp_path / "route-readout-adaptation.pt"
    checkpoint.write_bytes(b"route-readout-small-adaptation")
    return {
        "schema": "orion.qwen-visibility-grounding-smoke-report/v1",
        "status": "complete",
        "stage": "V1f_route151_route_readout_overfit",
        "objective": objective,
        "claim_boundary": claim_boundary,
        "protocol_path": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "curriculum_path": str(curriculum_path),
        "curriculum_sha256": hashlib.sha256(
            curriculum_path.read_bytes()
        ).hexdigest(),
        "curriculum_example_count": 50,
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


def _promote_route_readout_report_to_v1i(report):
    stage = "V1i_route151_slot_typed_full_attention_route_readout_overfit"
    layers = list(evaluation.V1I_FULL_ATTENTION_LAYERS)
    modules = list(evaluation.V1I_LORA_MODULE_NAMES)
    lora = {
        "layer_indices": layers,
        "module_names": modules,
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.0,
    }
    installed = [
        "vlm.model.language_model.layers.%d.self_attn.%s" % (layer, module)
        for layer in layers
        for module in modules
    ]
    report["stage"] = stage
    report["lora"] = lora
    report["installed_lora_modules"] = installed
    report["scope"].update(
        {
            "projector_trainable_parameter_count": 1_391_616,
            "model_trainable_parameter_count": 1_572_864,
            "model_trainable_names": [
                module + suffix
                for module in installed
                for suffix in (".lora_a", ".lora_b")
            ],
        }
    )
    report["checkpoint"]["lora_tensor_count"] = 64
    protocol_path = Path(report["protocol_path"])
    protocol = json.loads(protocol_path.read_text())
    protocol.update(
        {
            "schema": evaluation.FULL_ATTENTION_SLOT_TYPED_ROUTE_READOUT_CONFIG_SCHEMA,
            "stage": stage,
            "base_bridge_config": (
                "configs/qwen_drive_b2d_agent_oracle_visibility_sft_v1.json"
            ),
            "projector": {
                "type": "slot_typed_scalar_basis",
                "feature_dim": 23,
                "scalar_basis_dim": 4,
                "maximum_token_slots": 48,
                "hidden_dim": 512,
                "vlm_hidden_dim": 2560,
            },
            "lora": lora,
            "training": {
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
            },
            "evaluation": {
                "max_new_tokens": 16,
                "split": "held_out_order",
                "pre_training_controls": ["true_u"],
                "controls": list(evaluation.EXPECTED_CONTROLS),
            },
        }
    )
    protocol_path.write_text(json.dumps(protocol))
    report["protocol_sha256"] = hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()
    return report


def test_route_readout_causal_pairs_pass_only_as_plumbing(tmp_path):
    report_path = tmp_path / "route-readout-report.json"
    report_path.write_text(json.dumps(_route_readout_report(tmp_path, causal=True)))
    audit = evaluation.audit_visibility_grounding_report(
        report_path, tmp_path / "route-readout-audit.json"
    )
    assert audit["protocol_valid"] is True
    assert audit["causal_capacity_passed"] is True
    assert audit["status"] == "causal_route_readout_plumbing_pass"
    assert audit["held_out_matched_pair_accuracy"] == 1.0
    assert audit["post_training_held_out_metrics"]["spatial_shuffle"][
        "control_target_accuracy"
    ] == 1.0
    assert audit["claim_boundary"]["reportable_generalization"] is False


def test_route_readout_noncausal_pairs_do_not_pass(tmp_path):
    report_path = tmp_path / "route-readout-report.json"
    report_path.write_text(json.dumps(_route_readout_report(tmp_path, causal=False)))
    audit = evaluation.audit_route_readout_overfit_report(
        report_path, tmp_path / "route-readout-audit.json"
    )
    assert audit["protocol_valid"] is True
    assert audit["causal_capacity_passed"] is False
    assert audit["status"] == "valid_run_without_causal_route_readout"
    assert audit["true_minus_control_ceiling"] == 0.0


def test_typed_route_readout_uses_stage_specific_scope_and_status(tmp_path):
    report = _route_readout_report(tmp_path, causal=True)
    report["stage"] = "V1g_route151_typed_route_readout_overfit"
    report["scope"]["projector_trainable_parameter_count"] = 1_367_040
    protocol_path = Path(report["protocol_path"])
    protocol = json.loads(protocol_path.read_text())
    protocol["schema"] = evaluation.TYPED_ROUTE_READOUT_CONFIG_SCHEMA
    protocol["stage"] = "V1g_route151_typed_route_readout_overfit"
    protocol["projector"] = {"type": "typed_scalar_basis"}
    protocol_path.write_text(json.dumps(protocol))
    report["protocol_sha256"] = hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()
    report_path = tmp_path / "typed-route-readout-report.json"
    report_path.write_text(json.dumps(report))
    audit = evaluation.audit_visibility_grounding_report(
        report_path, tmp_path / "typed-route-readout-audit.json"
    )
    assert audit["protocol_valid"] is True
    assert audit["causal_capacity_passed"] is True
    assert audit["projector_type"] == "typed_scalar_basis"
    assert audit["status"] == "causal_typed_route_readout_plumbing_pass"


def test_slot_typed_route_readout_uses_its_own_scope_and_status(tmp_path):
    report = _route_readout_report(tmp_path, causal=True)
    report["stage"] = "V1h_route151_slot_typed_route_readout_overfit"
    report["scope"]["projector_trainable_parameter_count"] = 1_391_616
    protocol_path = Path(report["protocol_path"])
    protocol = json.loads(protocol_path.read_text())
    protocol["schema"] = evaluation.SLOT_TYPED_ROUTE_READOUT_CONFIG_SCHEMA
    protocol["stage"] = "V1h_route151_slot_typed_route_readout_overfit"
    protocol["projector"] = {"type": "slot_typed_scalar_basis"}
    protocol_path.write_text(json.dumps(protocol))
    report["protocol_sha256"] = hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()
    report_path = tmp_path / "slot-typed-route-readout-report.json"
    report_path.write_text(json.dumps(report))
    audit = evaluation.audit_visibility_grounding_report(
        report_path, tmp_path / "slot-typed-route-readout-audit.json"
    )
    assert audit["protocol_valid"] is True
    assert audit["causal_capacity_passed"] is True
    assert audit["projector_type"] == "slot_typed_scalar_basis"
    assert audit["status"] == "causal_slot_typed_route_readout_plumbing_pass"


def test_full_attention_slot_typed_route_readout_has_exact_scope_and_status(
    tmp_path,
):
    report = _promote_route_readout_report_to_v1i(
        _route_readout_report(tmp_path, causal=True)
    )
    report_path = tmp_path / "full-attention-route-readout-report.json"
    report_path.write_text(json.dumps(report))
    audit = evaluation.audit_visibility_grounding_report(
        report_path, tmp_path / "full-attention-route-readout-audit.json"
    )
    assert audit["protocol_valid"] is True
    assert audit["causal_capacity_passed"] is True
    assert audit["projector_type"] == "slot_typed_scalar_basis"
    assert audit["status"] == (
        "causal_slot_typed_full_attention_route_readout_plumbing_pass"
    )


def test_full_attention_slot_typed_route_readout_rejects_reduced_scope(tmp_path):
    report = _promote_route_readout_report_to_v1i(
        _route_readout_report(tmp_path, causal=True)
    )
    protocol_path = Path(report["protocol_path"])
    protocol = json.loads(protocol_path.read_text())
    protocol["lora"]["layer_indices"] = [27, 31]
    protocol_path.write_text(json.dumps(protocol))
    report["protocol_sha256"] = hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()
    report_path = tmp_path / "reduced-attention-route-readout-report.json"
    report_path.write_text(json.dumps(report))
    audit = evaluation.audit_visibility_grounding_report(
        report_path, tmp_path / "reduced-attention-route-readout-audit.json"
    )
    assert audit["protocol_valid"] is False
    assert "protocol_lora_config" in audit["protocol_failures"]


def test_route_readout_held_out_example_in_optimizer_invalidates_report(tmp_path):
    report = _route_readout_report(tmp_path, causal=True)
    curriculum_path = Path(report["curriculum_path"])
    curriculum = json.loads(curriculum_path.read_text())
    held_out_id = curriculum["evaluation_example_ids"][0]
    curriculum["training_schedule"][0] = held_out_id
    curriculum_path.write_text(json.dumps(curriculum))
    report["curriculum_sha256"] = hashlib.sha256(
        curriculum_path.read_bytes()
    ).hexdigest()
    report["history"][0]["example_id"] = held_out_id
    report_path = tmp_path / "route-readout-report.json"
    report_path.write_text(json.dumps(report))
    audit = evaluation.audit_route_readout_overfit_report(
        report_path, tmp_path / "route-readout-audit.json"
    )
    assert audit["protocol_valid"] is False
    assert "curriculum_schedule" in audit["protocol_failures"]

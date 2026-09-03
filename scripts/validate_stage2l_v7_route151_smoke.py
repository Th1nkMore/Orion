#!/usr/bin/env python3
"""Independently validate the bounded Route151 Stage2-L v7 smoke report."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from scenario_factory_lib import sha256_file


REPORT_SCHEMA = "orion.stage2l_v7_calibrated_matched_smoke.v1"
ATTESTATION_SCHEMA = "orion.stage2l_v7_smoke_submission_attestation.v1"
PROTOCOL_SCHEMA = "orion.stage2l_calibrated_training_protocol.v1"
QA_CONFIG_SCHEMA = "orion.uq_relevance_qa_factory_config.v3"
AMENDMENT_SCHEMA = "orion.scenario_factory.amendment.v1"
EXPECTED_EVENT_ID = "route151_step218"
EXPECTED_STEPS = 20
EXPECTED_CHECKS = {
    "exactly_one_optimizer_step_per_complete_group",
    "language_nll_decreases",
    "same_family_preference_improves_and_gte_0_8",
    "relevance_foreground_recall",
    "relevance_background_fpr",
    "all_groups_positive_on_off_order",
    "all_groups_attain_oracle_fraction",
    "all_stance_variants_correct",
    "all_stance_classes_recalled",
    "minimum_stance_target_probability",
    "generated_family_tags_match",
    "generated_text_nonrepeating",
    "generated_driving_stances_parse",
    "generated_driving_stances_agree",
    "diagnostic_driving_targets_excluded",
    "trajectory_and_control_remain_disabled",
}


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _require_finite(value: Any, path: str = "report") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("non-finite numeric value at %s" % path)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite(child, "%s.%s" % (path, key))
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite(child, "%s[%d]" % (path, index))


def _recompute_checks(
    report: Mapping[str, Any],
    protocol: Mapping[str, Any],
    qa_config: Mapping[str, Any],
) -> Dict[str, bool]:
    before = report["before"]
    after = report["after"]
    history = report["history"]
    relevance_gates = qa_config["task_relevance_gates"]
    stance_gates = qa_config["stance_gates"]
    generation_gates = qa_config["generation_gates"]
    required_fraction = float(
        protocol["losses"]["geometry_normalized_on_off_ranking"][
            "required_oracle_fraction"
        ]
    )
    return {
        "exactly_one_optimizer_step_per_complete_group": (
            int(report["optimizer_steps"]) == EXPECTED_STEPS
            and len(history) == EXPECTED_STEPS
            and all(
                int(row["records_in_optimizer_unit"]) == 20
                and int(row["optimizer_steps_inside_group"]) == 0
                for row in history
            )
        ),
        "language_nll_decreases": (
            float(after["first_group_mean_hard_language_nll"])
            < float(before["first_group_mean_hard_language_nll"])
        ),
        "same_family_preference_improves_and_gte_0_8": (
            float(after["first_group_same_family_margin_pass_fraction"])
            > float(before["first_group_same_family_margin_pass_fraction"])
            and float(after["first_group_same_family_margin_pass_fraction"]) >= 0.8
        ),
        "relevance_foreground_recall": (
            float(after["relevance_support"]["foreground_recall"])
            >= float(relevance_gates["minimum_foreground_recall"])
        ),
        "relevance_background_fpr": (
            float(after["relevance_support"]["background_false_positive_rate"])
            <= float(relevance_gates["maximum_background_false_positive_rate"])
        ),
        "all_groups_positive_on_off_order": (
            float(after["ranking"]["positive_order_fraction"]) == 1.0
        ),
        "all_groups_attain_oracle_fraction": (
            float(after["ranking"]["minimum_attained_fraction"])
            >= required_fraction
        ),
        "all_stance_variants_correct": all(
            float(value) >= float(stance_gates["minimum_per_variant_accuracy"])
            for value in after["stance"]["per_variant_accuracy"].values()
        ),
        "all_stance_classes_recalled": all(
            float(value)
            >= float(stance_gates["minimum_per_target_class_recall"])
            for value in after["stance"]["per_target_class_recall"].values()
        ),
        "minimum_stance_target_probability": (
            float(after["stance"]["minimum_target_probability"])
            >= float(generation_gates["minimum_hard_stance_target_probability"])
        ),
        "generated_family_tags_match": (
            float(after["generation_contract"]["family_tag_parse_and_accuracy"])
            >= float(generation_gates["family_tag_accuracy"])
        ),
        "generated_text_nonrepeating": (
            float(after["generation_contract"]["nonrepeating_text_fraction"])
            == 1.0
        ),
        "generated_driving_stances_parse": (
            float(after["generation_contract"]["hard_driving_stance_parse_rate"])
            >= float(generation_gates["hard_driving_stance_parse_rate"])
        ),
        "generated_driving_stances_agree": (
            float(after["generation_contract"]["hard_driving_stance_agreement"])
            >= float(generation_gates["hard_driving_structured_stance_agreement"])
        ),
        "diagnostic_driving_targets_excluded": (
            int(report["qa_audit"]["hard_language_record_count"]) == 90
            and int(report["qa_audit"]["hard_stance_record_count"]) == 15
        ),
        "trajectory_and_control_remain_disabled": True,
    }


def _validate_source_integrity(
    *, attestation: Mapping[str, Any], project_root: Path
) -> None:
    paths = {
        "submitter": "scripts/submit_stage2l_v7_route151_smoke.sh",
        "trainer": "scripts/train_stage2l_v7_route151_smoke.py",
        "training_protocol": "configs/scenario_factory/stage2l_training_v7_calibrated_matched_semantics.json",
        "launch_amendment": "configs/scenario_factory/amendments/20260830_stage2l_v7_route151_calibrated_smoke_v1.json",
        "qa_factory_config": "configs/scenario_factory/qa_factory_v3_calibrated_semantics.json",
        "calibrated_objective": "uq_estimator/stage2l_calibrated_objective.py",
        "qa_contract": "uq_estimator/stage2l_qa_contract_v3.py",
        "semantic_bottleneck": "uq_estimator/stage2l_semantic_bottleneck_v2.py",
        "semantic_runtime": "uq_estimator/stage2l_semantic_runtime_v2.py",
    }
    expected = attestation["source_sha256"]
    for name, relative in paths.items():
        path = project_root / relative
        if not path.is_file() or sha256_file(path) != expected.get(name):
            raise ValueError("attested source hash mismatch: %s" % name)


def validate_report(
    *,
    report_path: Path,
    checkpoint_path: Path,
    attestation_path: Path,
    protocol_path: Path,
    qa_config_path: Path,
    amendment_path: Path,
    project_root: Path,
) -> Dict[str, Any]:
    report_path = report_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    attestation_path = attestation_path.resolve()
    protocol_path = protocol_path.resolve()
    qa_config_path = qa_config_path.resolve()
    amendment_path = amendment_path.resolve()
    project_root = project_root.resolve()
    report = _load(report_path)
    attestation = _load(attestation_path)
    protocol = _load(protocol_path)
    qa_config = _load(qa_config_path)
    amendment = _load(amendment_path)
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported Route151 v7 smoke report schema")
    if attestation.get("schema") != ATTESTATION_SCHEMA:
        raise ValueError("unsupported Route151 v7 submission attestation schema")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported Stage2-L v7 protocol schema")
    if qa_config.get("schema") != QA_CONFIG_SCHEMA:
        raise ValueError("unsupported Stage2-L v3 QA config schema")
    if amendment.get("schema") != AMENDMENT_SCHEMA:
        raise ValueError("unsupported v7 launch amendment schema")
    if (
        protocol.get("launch_locks", {}).get("stage2l_pilot_training_allowed")
        is not False
        or amendment.get("launch_locks", {}).get(
            "stage2l_pilot_training_allowed"
        )
        is not False
    ):
        raise ValueError("v7 smoke inputs unexpectedly unlock pilot training")
    _validate_source_integrity(attestation=attestation, project_root=project_root)
    _require_finite(report)

    bounds = attestation["training_bounds"]
    if (
        int(bounds.get("maximum_submissions", -1)) != 1
        or int(bounds.get("maximum_optimizer_steps", -1)) != EXPECTED_STEPS
        or bounds.get("formal_training") is not False
        or bounds.get("stage2p_training") is not False
        or bounds.get("trajectory_or_control_loss") is not False
        or bounds.get("automatic_retry_or_extension") is not False
    ):
        raise ValueError("submission attestation violates bounded v7 smoke")
    if (
        report.get("event_id") != EXPECTED_EVENT_ID
        or int(report.get("optimizer_steps", -1)) != EXPECTED_STEPS
        or int(report.get("record_equivalent_presentations", -1)) != 400
    ):
        raise ValueError("report violates fixed event or presentation bounds")
    audit = report["qa_audit"]
    if (
        audit.get("passed") is not True
        or int(audit.get("record_count", -1)) != 100
        or int(audit.get("matched_group_count", -1)) != 5
        or int(audit.get("hard_language_record_count", -1)) != 90
        or int(audit.get("hard_stance_record_count", -1)) != 15
        or int(audit.get("same_family_preference_anchor_count", -1)) != 90
        or int(audit.get("distinct_counterfactual_negative_count", -1)) != 270
        or not all(audit.get("checks", {}).values())
    ):
        raise ValueError("report QA audit differs from frozen Route151 v3 data")
    history = report["history"]
    if [int(row["optimizer_step"]) for row in history] != list(
        range(1, EXPECTED_STEPS + 1)
    ):
        raise ValueError("optimizer-step history is incomplete or reordered")
    group_counts = Counter(str(row["group_id"]) for row in history)
    if len(group_counts) != 5 or set(group_counts.values()) != {4}:
        raise ValueError("the four smoke epochs do not each cover all five groups")
    if any(
        int(row["hard_language_anchors"]) != 18
        or float(row["gradient_norm_before_clip"]) <= 0.0
        for row in history
    ):
        raise ValueError("history violates anchor or gradient invariants")

    expected_architecture = {
        "stage1_adapter_frozen_and_task_agnostic": True,
        "task_relevance_owned_by_vlm": True,
        "raw_k_magnitude_side_channel": True,
        "one_optimizer_step_per_20_record_group": True,
        "cross_family_answers_used_as_negatives": False,
        "exact_duplicate_answers_used_as_negatives": False,
        "overall_stance_accuracy_used_as_gate": False,
        "ground_truth_stance_enters_forward": False,
        "legacy_density_uq_used": False,
        "hard_governor_used": False,
        "trajectory_or_control_loss": False,
    }
    if report.get("architecture") != expected_architecture:
        raise ValueError("report architecture differs from frozen v7 boundary")
    if any(
        report.get(name) is not False
        for name in (
            "formal_training_ready",
            "stage2l_pilot_training_ready",
            "stage2p_ready",
        )
    ) or report.get("engineering_smoke_only") is not True:
        raise ValueError("one-event v7 smoke improperly expands its claim")

    hashes = attestation["source_sha256"]
    provenance = report["provenance"]
    expected_provenance = {
        "records": hashes["records"],
        "visual_cache": hashes["visual_cache"],
        "qa_config": hashes["qa_factory_config"],
        "training_protocol": hashes["training_protocol"],
        "objective_diagnostic": hashes["objective_diagnostic"],
        "launch_amendment": hashes["launch_amendment"],
        "trainer": hashes["trainer"],
        "base_orion_checkpoint": hashes["base_orion_checkpoint"],
    }
    for name, expected_hash in expected_provenance.items():
        if provenance.get(name, {}).get("sha256") != expected_hash:
            raise ValueError("report provenance differs from attestation: %s" % name)
    if sha256_file(protocol_path) != hashes["training_protocol"]:
        raise ValueError("supplied v7 protocol differs from attestation")
    if sha256_file(qa_config_path) != hashes["qa_factory_config"]:
        raise ValueError("supplied v3 QA config differs from attestation")
    if sha256_file(amendment_path) != hashes["launch_amendment"]:
        raise ValueError("supplied v7 amendment differs from attestation")
    checkpoint_hash = sha256_file(checkpoint_path)
    if provenance.get("checkpoint", {}).get("sha256") != checkpoint_hash:
        raise ValueError("downloaded v7 checkpoint is absent or hash-mismatched")

    recorded_checks = report.get("checks", {})
    if set(recorded_checks) != EXPECTED_CHECKS:
        raise ValueError("report check set differs from frozen v7 gates")
    recomputed = _recompute_checks(report, protocol, qa_config)
    if recorded_checks != recomputed:
        raise ValueError("trainer checks differ from independent recomputation")
    passed = all(recomputed.values())
    expected_status = (
        "engineering_v7_calibrated_smoke_pass"
        if passed
        else "engineering_v7_calibrated_smoke_failed_gate"
    )
    if report.get("status") != expected_status:
        raise ValueError("report status differs from independent v7 outcome")
    failed = sorted(name for name, value in recomputed.items() if not value)
    return {
        "schema": "orion.stage2l_v7_route151_independent_validation.v1",
        "status": "validated_pass" if passed else "validated_failed_gate",
        "integrity_valid": True,
        "smoke_passed": passed,
        "failed_checks": failed,
        "checks": recomputed,
        "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_hash},
        "submission_attestation": {
            "path": str(attestation_path),
            "sha256": sha256_file(attestation_path),
        },
        "next_action": (
            "keep formal pilot locked pending human review and a separate authorization"
            if passed
            else "keep pilot locked and diagnose the failed v7 semantic gates"
        ),
        "claim_boundary": (
            "Independent Route151 v7 engineering-smoke integrity and gate "
            "validation only; no held-out, trajectory, closed-loop, "
            "generalization or safety evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--submission-attestation", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--qa-config", type=Path, required=True)
    parser.add_argument("--launch-amendment", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite independent v7 validation")
    result = validate_report(
        report_path=args.report,
        checkpoint_path=args.checkpoint,
        attestation_path=args.submission_attestation,
        protocol_path=args.training_protocol,
        qa_config_path=args.qa_config,
        amendment_path=args.launch_amendment,
        project_root=args.project_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

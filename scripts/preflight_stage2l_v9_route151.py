#!/usr/bin/env python3
"""Fail-closed CPU preflight for the locked Route151 Stage2-L v9 design."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping


SCHEMA = "orion.stage2l_v9_route151_preflight.v1"


def _read(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must contain an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(
    *,
    protocol: Mapping[str, Any],
    qa_config: Mapping[str, Any],
    dataset_audit: Mapping[str, Any],
    reference_audit: Mapping[str, Any],
    v8_report: Mapping[str, Any],
    v8_validation: Mapping[str, Any],
    v8_diagnosis: Mapping[str, Any],
    prompt_alignment: Mapping[str, Any],
    project_root: Path,
    records_path: Path,
    protocol_path: Path,
    qa_config_path: Path,
    dataset_audit_path: Path,
    reference_audit_path: Path,
    v8_report_path: Path,
    v8_validation_path: Path,
    v8_diagnosis_path: Path,
    prompt_alignment_path: Path,
) -> Dict[str, Any]:
    expected_sources = protocol["implementation_sources"]
    source_hashes = {}
    missing_sources = []
    mismatched_sources = []
    for relative, expected in sorted(expected_sources.items()):
        path = project_root / relative
        if not path.is_file():
            missing_sources.append(relative)
            continue
        actual = _sha256(path)
        source_hashes[relative] = actual
        if actual != expected:
            mismatched_sources.append(relative)

    dataset = protocol["route151_v9_dataset"]
    trigger = protocol["triggering_v8_failure"]
    locks = protocol["launch_locks"]
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    zero_rows = [
        row
        for row in records
        if row["counterfactual"]["variant"] == "zero_uq"
    ]
    off_path_task_rows = [
        row
        for row in records
        if row["counterfactual"]["variant"] == "off_path_uq"
        and row["question_family"] == "task_relevance"
    ]
    task_field_rows = [
        row
        for row in records
        if row["target"].get("vlm_task_field_targets")
    ]
    checks = {
        "protocol_schema": protocol.get("schema")
        == "orion.stage2l_vlm_task_field_training_protocol.v1",
        "qa_config_schema": qa_config.get("schema")
        == "orion.uq_relevance_qa_factory_config.v5",
        "v8_is_independently_validated_gate_failure": (
            v8_report.get("status")
            == "engineering_v8_gradient_routed_smoke_failed_gate"
            and v8_validation.get("status") == "validated_failed_gate"
            and v8_validation.get("integrity_valid") is True
            and v8_validation.get("smoke_passed") is False
            and v8_diagnosis.get("status") == "diagnosed_v8_gate_failure"
        ),
        "v8_prompt_alignment_was_not_the_failure": (
            prompt_alignment.get("status") == "alignment_pass"
            and prompt_alignment.get("anchor_count") == 90
            and all(prompt_alignment.get("checks", {}).values())
        ),
        "v8_trigger_hashes_match": (
            _sha256(v8_report_path) == trigger["report_sha256"]
            and _sha256(v8_validation_path)
            == trigger["independent_validation_sha256"]
            and _sha256(v8_diagnosis_path) == trigger["diagnosis_sha256"]
            and _sha256(prompt_alignment_path)
            == trigger["prompt_alignment_sha256"]
        ),
        "implementation_sources_present_and_hash_matched": (
            not missing_sources and not mismatched_sources
        ),
        "qa_config_hash_matches_protocol": (
            _sha256(qa_config_path)
            == expected_sources[
                "configs/scenario_factory/qa_factory_v5_vlm_task_fields.json"
            ]
        ),
        "records_hash_matches": _sha256(records_path)
        == dataset["records_sha256"],
        "dataset_audit_hash_and_status": (
            _sha256(dataset_audit_path) == dataset["audit_sha256"]
            and dataset_audit.get("passed") is True
        ),
        "reference_audit_hash_and_status": (
            _sha256(reference_audit_path)
            == dataset["reference_audit_sha256"]
            and reference_audit.get("passed") is True
            and reference_audit.get("verified_sha256_count") == 90
        ),
        "dataset_counts_match": (
            len(records) == dataset["record_count"] == 100
            and dataset_audit.get("matched_group_count")
            == dataset["matched_group_count"] == 5
            and dataset_audit.get("task_relevance_field_record_count")
            == dataset["task_relevance_field_record_count"] == 25
            and dataset_audit.get("hard_stance_field_record_count")
            == dataset["hard_stance_field_record_count"] == 15
        ),
        "records_use_v5_semantic_contract": all(
            row.get("schema") == "orion.uq_relevance_qa_record.v5"
            and row.get("target", {}).get("qa_contract_schema")
            == "orion.stage2l_qa_contract.v5"
            for row in records
        ),
        "zero_uq_has_explicit_absence_and_maintain": (
            len(zero_rows) == 20
            and all(
                row["target"]["structured_summary"]["observation_uncertainty"][
                    "peak_view"
                ]
                == "none"
                and row["target"]["structured_summary"]["task_risk"][
                    "peak_view"
                ]
                == "none"
                and row["target"]["structured_summary"]["planning_implication"][
                    "stance"
                ]
                == "maintain"
                for row in zero_rows
            )
        ),
        "u_and_k_are_semantically_distinguishable": (
            len(off_path_task_rows) == 5
            and all(
                row["target"]["vlm_task_field_targets"] == {
                    "relevance_level": "low",
                    "risk_level": "none",
                    "risk_view": "none",
                    "risk_region": "none",
                }
                for row in off_path_task_rows
            )
            and all(
                row["target"]["semantic_fields"].get("relevance_level")
                == "not_applicable"
                for row in zero_rows
                if row["question_family"] == "task_relevance"
            )
        ),
        "field_targets_are_partial_and_vlm_owned": (
            len(task_field_rows) == 40
            and all(
                set(row["target"]["vlm_task_field_targets"])
                in (
                    {"relevance_level", "risk_level", "risk_view", "risk_region"},
                    {"stance"},
                )
                for row in task_field_rows
            )
        ),
        "gradient_ownership_is_explicit": (
            protocol["gradient_ownership"][
                "qa_language_loss_to_relevance_logits"
            ]
            is False
            and protocol["gradient_ownership"][
                "qa_language_loss_to_task_risk_bridge"
            ]
            is False
            and protocol["gradient_ownership"][
                "qa_language_loss_to_task_field_classifiers"
            ]
            is False
            and protocol["gradient_ownership"][
                "task_field_loss_to_relevance_path"
            ]
            is False
            and protocol["gradient_ownership"][
                "dense_relevance_and_ranking_losses_to_relevance_path"
            ]
            is True
        ),
        "primary_optimizer_unit_covers_all_groups": (
            protocol["optimizer_unit"][
                "primary_groups_per_optimizer_step"
            ]
            == 5
            and protocol["optimizer_unit"][
                "primary_records_per_optimizer_step"
            ]
            == 100
            and protocol["optimizer_unit"][
                "language_groups_per_optimizer_step"
            ]
            == 1
            and protocol["optimizer_unit"][
                "primary_loss_is_averaged_before_step"
            ]
            is True
        ),
        "trajectory_control_and_legacy_paths_disabled": (
            protocol["architecture"]["stage1_adapter_trainable"] is False
            and protocol["architecture"]["trajectory_training_enabled"] is False
            and protocol["architecture"]["direct_control_training_enabled"]
            is False
            and protocol["architecture"]["legacy_density_uq_used"] is False
            and protocol["architecture"]["hard_governor_used"] is False
        ),
        "language_is_auxiliary_not_release_evidence": all(
            row["loss_policy"]["language_is_release_evidence"] is False
            for row in records
        )
        and qa_config["response_contract"][
            "free_generation_is_release_evidence"
        ]
        is False
        and qa_config["release_gates_for_a_future_bounded_smoke"][
            "free_language_nll_decrease"
        ]
        == "diagnostic_only"
        and qa_config["release_gates_for_a_future_bounded_smoke"][
            "free_generation_parse_and_style"
        ]
        == "diagnostic_only"
        and protocol["losses_for_future_bounded_smoke"][
            "auxiliary_structured_qa_causal_language_modeling"
        ]["release_evidence"]
        is False,
        "all_launches_remain_locked": (
            locks["real_orion_smoke_allowed"] is False
            and locks["stage2l_pilot_training_allowed"] is False
            and locks["stage2p_allowed"] is False
            and locks["new_immutable_amendment_required"] is True
        ),
        "future_probe_is_not_authorized": (
            protocol["future_probe_not_authorized"][
                "automatic_retry_or_extension"
            ]
            is False
        ),
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "status": (
            "v9_architecture_data_preflight_pass_training_locked"
            if passed
            else "v9_architecture_data_preflight_failed"
        ),
        "passed": passed,
        "checks": checks,
        "failed_checks": sorted(
            key for key, value in checks.items() if not value
        ),
        "source_hashes": source_hashes,
        "missing_sources": missing_sources,
        "mismatched_sources": mismatched_sources,
        "protocol_sha256": _sha256(protocol_path),
        "record_count": len(records),
        "selected_cpu_test_result": protocol["cpu_preflight_tests"][
            "selected_test_result"
        ],
        "training_started": False,
        "real_orion_smoke_authorized": False,
        "stage2l_pilot_authorized": False,
        "stage2p_authorized": False,
        "next_action": (
            "run and review the bounded trainer's CPU-only preflight; require "
            "a separate immutable amendment before any A800 launch"
        ),
        "claim_boundary": (
            "V9 architecture, semantic data, gradient ownership and artifact "
            "integrity preflight only; no v9 model, held-out, trajectory, "
            "closed-loop, generalization or safety evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--qa-config", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--v8-report", type=Path, required=True)
    parser.add_argument("--v8-validation", type=Path, required=True)
    parser.add_argument("--v8-diagnosis", type=Path, required=True)
    parser.add_argument("--prompt-alignment", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = preflight(
        protocol=_read(args.protocol),
        qa_config=_read(args.qa_config),
        dataset_audit=_read(args.dataset_audit),
        reference_audit=_read(args.reference_audit),
        v8_report=_read(args.v8_report),
        v8_validation=_read(args.v8_validation),
        v8_diagnosis=_read(args.v8_diagnosis),
        prompt_alignment=_read(args.prompt_alignment),
        project_root=args.project_root.resolve(),
        records_path=args.records.resolve(),
        protocol_path=args.protocol.resolve(),
        qa_config_path=args.qa_config.resolve(),
        dataset_audit_path=args.dataset_audit.resolve(),
        reference_audit_path=args.reference_audit.resolve(),
        v8_report_path=args.v8_report.resolve(),
        v8_validation_path=args.v8_validation.resolve(),
        v8_diagnosis_path=args.v8_diagnosis.resolve(),
        prompt_alignment_path=args.prompt_alignment.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

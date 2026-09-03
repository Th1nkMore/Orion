#!/usr/bin/env python3
"""Fail-closed CPU preflight for the locked Route151 Stage2-L v8 repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping


SCHEMA = "orion.stage2l_v8_route151_preflight.v1"


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
    v7_report: Mapping[str, Any],
    v7_validation: Mapping[str, Any],
    v7_diagnosis: Mapping[str, Any],
    project_root: Path,
    records_path: Path,
    protocol_path: Path,
    qa_config_path: Path,
    dataset_audit_path: Path,
    reference_audit_path: Path,
    v7_report_path: Path,
    v7_validation_path: Path,
    v7_diagnosis_path: Path,
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

    dataset = protocol["route151_v8_dataset"]
    trigger = protocol["triggering_v7_failure"]
    locks = protocol["launch_locks"]
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checks = {
        "protocol_schema": protocol.get("schema")
        == "orion.stage2l_gradient_routed_training_protocol.v1",
        "qa_config_schema": qa_config.get("schema")
        == "orion.uq_relevance_qa_factory_config.v4",
        "v7_is_independently_validated_gate_failure": (
            v7_report.get("status") == "engineering_v7_calibrated_smoke_failed_gate"
            and v7_validation.get("status") == "validated_failed_gate"
            and v7_validation.get("integrity_valid") is True
            and v7_validation.get("smoke_passed") is False
            and v7_diagnosis.get("status") == "diagnosed_v7_gate_failure"
        ),
        "v7_trigger_hashes_match": (
            _sha256(v7_report_path) == trigger["report_sha256"]
            and _sha256(v7_validation_path)
            == trigger["independent_validation_sha256"]
            and _sha256(v7_diagnosis_path) == trigger["diagnosis_sha256"]
        ),
        "implementation_sources_present_and_hash_matched": (
            not missing_sources and not mismatched_sources
        ),
        "qa_config_hash_matches_protocol": (
            _sha256(qa_config_path)
            == expected_sources[
                "configs/scenario_factory/qa_factory_v4_structured_semantics.json"
            ]
        ),
        "records_hash_matches": _sha256(records_path) == dataset["records_sha256"],
        "dataset_audit_hash_and_status": (
            _sha256(dataset_audit_path) == dataset["audit_sha256"]
            and dataset_audit.get("passed") is True
        ),
        "reference_audit_hash_and_status": (
            _sha256(reference_audit_path) == dataset["reference_audit_sha256"]
            and reference_audit.get("passed") is True
        ),
        "dataset_counts_match": (
            len(records) == dataset["record_count"] == 100
            and dataset_audit.get("matched_group_count")
            == dataset["matched_group_count"]
            and dataset_audit.get("hard_language_record_count")
            == dataset["hard_language_record_count"]
            and dataset_audit.get("hard_stance_record_count")
            == dataset["hard_stance_record_count"]
            and dataset_audit.get("same_family_preference_anchor_count")
            == dataset["same_family_preference_anchor_count"]
            and dataset_audit.get("distinct_counterfactual_negative_count")
            == dataset["distinct_counterfactual_negative_count"]
        ),
        "records_use_v4_semantic_contract": all(
            row.get("schema") == "orion.uq_relevance_qa_record.v4"
            and row.get("target", {}).get("qa_contract_schema")
            == "orion.stage2l_qa_contract.v4"
            for row in records
        ),
        "gradient_ownership_is_explicit": (
            protocol["gradient_ownership"]["qa_language_loss_to_relevance_logits"]
            is False
            and protocol["gradient_ownership"]["qa_language_loss_to_stance_classifier"]
            is False
            and protocol["gradient_ownership"]["ground_truth_stance_enters_forward"]
            is False
        ),
        "trajectory_control_and_legacy_paths_disabled": (
            protocol["architecture"]["trajectory_training_enabled"] is False
            and protocol["architecture"]["direct_control_training_enabled"] is False
            and protocol["architecture"]["legacy_density_uq_used"] is False
            and protocol["architecture"]["hard_governor_used"] is False
        ),
        "all_launches_remain_locked": (
            locks["real_orion_smoke_allowed"] is False
            and locks["stage2l_pilot_training_allowed"] is False
            and locks["stage2p_allowed"] is False
            and locks["new_immutable_amendment_required"] is True
        ),
        "future_probe_is_not_authorized": (
            protocol["future_probe_bound_not_authorized"]["automatic_retry_or_extension"]
            is False
        ),
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "status": (
            "v8_objective_data_preflight_pass_training_locked"
            if passed
            else "v8_objective_data_preflight_failed"
        ),
        "passed": passed,
        "checks": checks,
        "failed_checks": sorted(key for key, value in checks.items() if not value),
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
            "review the bounded v8 trainer and require a separate immutable amendment before any A800 launch"
        ),
        "claim_boundary": (
            "Objective, gradient-routing, dataset and artifact-reference preflight only; "
            "no v8 model result, held-out result, trajectory, closed-loop, generalization or safety evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--qa-config", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--v7-report", type=Path, required=True)
    parser.add_argument("--v7-validation", type=Path, required=True)
    parser.add_argument("--v7-diagnosis", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = preflight(
        protocol=_read(args.protocol),
        qa_config=_read(args.qa_config),
        dataset_audit=_read(args.dataset_audit),
        reference_audit=_read(args.reference_audit),
        v7_report=_read(args.v7_report),
        v7_validation=_read(args.v7_validation),
        v7_diagnosis=_read(args.v7_diagnosis),
        project_root=args.project_root.resolve(),
        records_path=args.records.resolve(),
        protocol_path=args.protocol.resolve(),
        qa_config_path=args.qa_config.resolve(),
        dataset_audit_path=args.dataset_audit.resolve(),
        reference_audit_path=args.reference_audit.resolve(),
        v7_report_path=args.v7_report.resolve(),
        v7_validation_path=args.v7_validation.resolve(),
        v7_diagnosis_path=args.v7_diagnosis.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

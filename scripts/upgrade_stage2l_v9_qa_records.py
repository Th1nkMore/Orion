#!/usr/bin/env python3
"""Upgrade Stage2-L records to the semantically closed v5 QA contract.

This is a CPU-only data transformation.  It does not start training.  Stage1
observation uncertainty remains unchanged; only derived semantic summaries,
QA answers, and VLM-owned task-field labels are made explicit and auditable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "uq_estimator"
    / "stage2l_qa_contract_v5.py"
)
_CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "stage2l_qa_contract_v5_pure", _CONTRACT_PATH
)
_CONTRACT = importlib.util.module_from_spec(_CONTRACT_SPEC)
if _CONTRACT_SPEC.loader is None:
    raise RuntimeError("cannot load Stage2-L QA contract v5")
_CONTRACT_SPEC.loader.exec_module(_CONTRACT)
NO_SIGNAL_EPSILON = _CONTRACT.NO_SIGNAL_EPSILON
QUESTION_FAMILIES = _CONTRACT.QUESTION_FAMILIES
QA_CONTRACT_SCHEMA = _CONTRACT.SCHEMA
expected_semantic_fields = _CONTRACT.expected_semantic_fields
expected_task_field_targets = _CONTRACT.expected_task_field_targets
parse_semantic_fields = _CONTRACT.parse_semantic_fields
render_structured_answer = _CONTRACT.render_structured_answer
TASK_FIELD_VOCABULARIES = {
    field: (
        _CONTRACT.FIELD_VOCABULARIES["uq_view"]
        if field == "risk_view"
        else _CONTRACT.FIELD_VOCABULARIES["uq_region"]
        if field == "risk_region"
        else _CONTRACT.FIELD_VOCABULARIES[field]
    )
    for field in _CONTRACT.TASK_FIELD_KEYS
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


RECORD_SCHEMA = "orion.uq_relevance_qa_record.v5"
AUDIT_SCHEMA = "orion.uq_relevance_qa_audit.v5"
MATCHED_VARIANTS = (
    "observed",
    "zero_uq",
    "on_path_uq",
    "off_path_uq",
    "view_shuffled_uq",
)
HARD_STANCE_VARIANTS = ("zero_uq", "on_path_uq", "off_path_uq")
TASK_RELEVANCE_FIELDS = (
    "relevance_level",
    "risk_level",
    "risk_view",
    "risk_region",
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _has_observation_signal(summary: Mapping[str, Any]) -> bool:
    return (
        float(summary["observation_uncertainty"].get("peak_score", 0.0))
        > NO_SIGNAL_EPSILON
    )


def _has_task_risk(summary: Mapping[str, Any]) -> bool:
    return (
        float(summary["task_risk"].get("peak_score", 0.0))
        > NO_SIGNAL_EPSILON
    )


def normalize_structured_summary(
    source: Mapping[str, Any]
) -> Dict[str, Any]:
    """Replace argmax tie artifacts with explicit absence sentinels."""

    summary = copy.deepcopy(source)
    observation = summary["observation_uncertainty"]
    relevance = summary["relevance_at_most_uncertain_region"]
    risk = summary["task_risk"]
    planning = summary["planning_implication"]
    if not _has_observation_signal(summary):
        observation.update({
            "level": "low",
            "peak_score": 0.0,
            "peak_view": "none",
            "peak_region": "none",
            "temporal_trend": "stable",
        })
        if "temporal_peak_region_delta" in observation:
            observation["temporal_peak_region_delta"] = 0.0
        relevance.update({"level": "not_applicable", "score": 0.0})
        risk.update({
            "level": "none",
            "peak_score": 0.0,
            "peak_view": "none",
            "peak_region": "none",
        })
        planning["stance"] = "maintain"
        if "risk_bearing" in planning:
            planning["risk_bearing"] = "none"
    elif not _has_task_risk(summary):
        risk.update({
            "level": "none",
            "peak_score": 0.0,
            "peak_view": "none",
            "peak_region": "none",
        })
        if "risk_bearing" in planning:
            planning["risk_bearing"] = "none"
    planning["is_direct_control_command"] = False
    return summary


def _groups(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Mapping[str, Any]]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in records:
        group_id = str(row["counterfactual"]["group_id"])
        groups.setdefault(group_id, []).append(row)
    expected = {
        (variant, family)
        for variant in MATCHED_VARIANTS
        for family in QUESTION_FAMILIES
    }
    for group_id, rows in groups.items():
        keys = {
            (
                str(row["counterfactual"]["variant"]),
                str(row["question_family"]),
            )
            for row in rows
        }
        if len(rows) != len(expected) or keys != expected:
            raise ValueError("incomplete matched group: %s" % group_id)
    return groups


def language_auxiliary_allowed(variant: str, family: str) -> bool:
    return not (
        family == "driving_implication"
        and variant in {"observed", "view_shuffled_uq"}
    )


def _field_targets(
    family: str, variant: str, summary: Mapping[str, Any]
) -> Dict[str, str]:
    task_fields = expected_task_field_targets(summary)
    if family == "task_relevance":
        return {key: task_fields[key] for key in TASK_RELEVANCE_FIELDS}
    if family == "driving_implication" and variant in HARD_STANCE_VARIANTS:
        return {"stance": task_fields["stance"]}
    return {}


def upgrade_records(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    upgraded: List[Dict[str, Any]] = []
    for source in records:
        row = copy.deepcopy(source)
        family = str(row["question_family"])
        variant = str(row["counterfactual"]["variant"])
        source_sha256 = _canonical_sha256(source)
        summary = normalize_structured_summary(
            row["target"]["structured_summary"]
        )
        answer = render_structured_answer(family, summary)
        field_targets = _field_targets(family, variant, summary)
        row["schema"] = RECORD_SCHEMA
        row["target"]["structured_summary"] = summary
        row["target"]["semantic_fields"] = expected_semantic_fields(
            family, summary
        )
        row["target"]["vlm_task_field_targets"] = field_targets
        row["target"]["rendered_answer"] = answer
        row["target"]["qa_contract_schema"] = QA_CONTRACT_SCHEMA
        row["conversation"][1]["value"] = answer
        row.setdefault("provenance", {})["v5_qa_upgrade"] = {
            "source_record_schema": str(source.get("schema", "unknown")),
            "source_record_sha256": source_sha256,
            "stage1_observation_uq_unchanged": True,
            "route_context_unchanged": True,
            "arbitrary_no_signal_argmax_locations_removed": True,
        }
        auxiliary = language_auxiliary_allowed(variant, family)
        row["loss_policy"] = {
            "contract": "vlm_task_fields_with_auxiliary_language_v5",
            "dense_relevance_target": True,
            "vlm_task_field_target": bool(field_targets),
            "vlm_task_field_names": sorted(field_targets),
            "language_auxiliary_target": auxiliary,
            "language_is_release_evidence": False,
            "deterministic_renderer_is_release_interface": True,
            "stage1_adapter_trainable": False,
            "trajectory_or_control_loss_enabled": False,
            "optimizer_group_complete_before_step": True,
        }
        upgraded.append(row)
    return upgraded, audit_records(upgraded)


def audit_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups = _groups(records)
    errors: List[Dict[str, str]] = []
    zero_rows = 0
    no_risk_rows = 0
    task_field_rows = 0
    stance_rows = 0
    language_auxiliary_rows = 0
    class_counts = {
        field: {value: 0 for value in vocabulary}
        for field, vocabulary in TASK_FIELD_VOCABULARIES.items()
    }
    for rows in groups.values():
        for row in rows:
            sample_id = str(row.get("sample_id", ""))
            family = str(row["question_family"])
            variant = str(row["counterfactual"]["variant"])
            summary = row["target"]["structured_summary"]
            policy = row.get("loss_policy", {})
            fields = expected_semantic_fields(family, summary)
            expected_answer = render_structured_answer(family, summary)
            expected_targets = _field_targets(family, variant, summary)
            answer = str(row["conversation"][1]["value"])
            auxiliary = language_auxiliary_allowed(variant, family)
            language_auxiliary_rows += int(auxiliary)
            task_field_rows += int(family == "task_relevance")
            stance_rows += int(
                family == "driving_implication"
                and variant in HARD_STANCE_VARIANTS
            )
            for field, value in expected_targets.items():
                class_counts[field][value] += 1
            if variant == "zero_uq":
                zero_rows += 1
                observation = summary["observation_uncertainty"]
                relevance = summary["relevance_at_most_uncertain_region"]
                planning = summary["planning_implication"]
                if (
                    observation.get("peak_view") != "none"
                    or observation.get("peak_region") != "none"
                    or relevance.get("level") != "not_applicable"
                    or planning.get("stance") != "maintain"
                ):
                    errors.append({
                        "sample_id": sample_id,
                        "error": "zero-UQ summary has non-absence semantics",
                    })
            if not _has_task_risk(summary):
                no_risk_rows += 1
                risk = summary["task_risk"]
                if (
                    risk.get("level") != "none"
                    or risk.get("peak_view") != "none"
                    or risk.get("peak_region") != "none"
                ):
                    errors.append({
                        "sample_id": sample_id,
                        "error": "zero task-risk summary has an arbitrary location",
                    })
            try:
                parsed = parse_semantic_fields(answer, family)
            except ValueError as error:
                parsed = {}
                errors.append({"sample_id": sample_id, "error": str(error)})
            if (
                row.get("schema") != RECORD_SCHEMA
                or row["target"].get("qa_contract_schema")
                != QA_CONTRACT_SCHEMA
                or row["target"].get("semantic_fields") != fields
                or row["target"].get("vlm_task_field_targets")
                != expected_targets
                or row["target"].get("rendered_answer") != expected_answer
                or answer != expected_answer
                or parsed != fields
                or policy.get("contract")
                != "vlm_task_fields_with_auxiliary_language_v5"
                or policy.get("dense_relevance_target") is not True
                or policy.get("vlm_task_field_target")
                is not bool(expected_targets)
                or policy.get("vlm_task_field_names")
                != sorted(expected_targets)
                or policy.get("language_auxiliary_target") is not auxiliary
                or policy.get("language_is_release_evidence") is not False
                or policy.get("deterministic_renderer_is_release_interface")
                is not True
                or policy.get("stage1_adapter_trainable") is not False
                or policy.get("trajectory_or_control_loss_enabled") is not False
            ):
                errors.append({
                    "sample_id": sample_id,
                    "error": "record contract mismatch",
                })
    checks = {
        "complete_matched_groups": bool(groups),
        "record_count_matches_groups": len(records) == 20 * len(groups),
        "zero_uq_rows_have_explicit_absence_semantics": zero_rows
        == 4 * len(groups),
        "zero_task_risk_rows_have_no_arbitrary_location": no_risk_rows > 0,
        "task_relevance_fields_cover_all_variants": task_field_rows
        == 5 * len(groups),
        "stance_fields_use_only_controlled_variants": stance_rows
        == 3 * len(groups),
        "language_is_auxiliary_not_release_evidence": language_auxiliary_rows
        == 18 * len(groups),
        "all_records_match_v5_contract": not errors,
        "stage1_is_frozen": all(
            row["loss_policy"]["stage1_adapter_trainable"] is False
            for row in records
        ),
        "trajectory_and_control_losses_are_disabled": all(
            row["loss_policy"]["trajectory_or_control_loss_enabled"] is False
            for row in records
        ),
    }
    return {
        "schema": AUDIT_SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "record_count": len(records),
        "matched_group_count": len(groups),
        "zero_uq_record_count": zero_rows,
        "zero_task_risk_record_count": no_risk_rows,
        "task_relevance_field_record_count": task_field_rows,
        "hard_stance_field_record_count": stance_rows,
        "language_auxiliary_record_count": language_auxiliary_rows,
        "task_field_class_counts": class_counts,
        "class_coverage_is_a_dataset_diagnostic_not_a_smoke_gate": True,
        "errors": errors,
    }


def _load_records(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records, audit = upgrade_records(_load_records(args.input_records))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records_path = args.output_dir / "records.jsonl"
    audit_path = args.output_dir / "audit.json"
    records_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in records
        ),
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "orion.uq_relevance_qa_dataset_manifest.v5",
        "status": (
            "prepared_training_locked" if audit["passed"] else "failed_audit"
        ),
        "source_records": {
            "path": str(args.input_records.resolve()),
            "sha256": sha256_file(args.input_records),
        },
        "records": {
            "path": str(records_path.resolve()),
            "sha256": sha256_file(records_path),
        },
        "audit": {
            "path": str(audit_path.resolve()),
            "sha256": sha256_file(audit_path),
        },
        "qa_contract": QA_CONTRACT_SCHEMA,
        "training_started": False,
        "gpu_job_authorized": False,
        "stage2l_pilot_authorized": False,
        "stage2p_authorized": False,
        "claim_boundary": (
            "CPU-only semantic target repair; no model learning, task-field "
            "accuracy, or generalization evidence."
        ),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"manifest_path": str(manifest_path.resolve()), "audit": audit},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

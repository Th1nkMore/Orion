#!/usr/bin/env python3
"""Upgrade matched Stage2-L records to semantic-field QA contract v4."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from scripts.scenario_factory_lib import sha256_file


_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "uq_estimator"
    / "stage2l_qa_contract_v4.py"
)
_CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "stage2l_qa_contract_v4_pure", _CONTRACT_PATH
)
_CONTRACT = importlib.util.module_from_spec(_CONTRACT_SPEC)
if _CONTRACT_SPEC.loader is None:
    raise RuntimeError("cannot load Stage2-L QA contract v4")
_CONTRACT_SPEC.loader.exec_module(_CONTRACT)
HARD_STANCE_VARIANTS = _CONTRACT.HARD_STANCE_VARIANTS
QUESTION_FAMILIES = _CONTRACT.QUESTION_FAMILIES
QA_CONTRACT_SCHEMA = _CONTRACT.SCHEMA
parse_semantic_fields = _CONTRACT.parse_semantic_fields
render_structured_answer = _CONTRACT.render_structured_answer
same_family_unique_structured_answers = (
    _CONTRACT.same_family_unique_structured_answers
)


RECORD_SCHEMA = "orion.uq_relevance_qa_record.v4"
AUDIT_SCHEMA = "orion.uq_relevance_qa_audit.v4"
MATCHED_VARIANTS = (
    "observed",
    "zero_uq",
    "on_path_uq",
    "off_path_uq",
    "view_shuffled_uq",
)


def hard_language_allowed(variant: str, family: str) -> bool:
    return not (
        family == "driving_implication"
        and variant in {"observed", "view_shuffled_uq"}
    )


def _groups(records: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in records:
        group_id = str(row["counterfactual"]["group_id"])
        groups.setdefault(group_id, []).append(row)
    expected = {(variant, family) for variant in MATCHED_VARIANTS for family in QUESTION_FAMILIES}
    for group_id, rows in groups.items():
        keys = {
            (str(row["counterfactual"]["variant"]), str(row["question_family"]))
            for row in rows
        }
        if len(rows) != 20 or keys != expected:
            raise ValueError("incomplete matched group: %s" % group_id)
    return groups


def upgrade_records(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    upgraded: List[Dict[str, Any]] = []
    for source in records:
        row = copy.deepcopy(source)
        family = str(row["question_family"])
        variant = str(row["counterfactual"]["variant"])
        answer = render_structured_answer(
            family, row["target"]["structured_summary"]
        )
        row["schema"] = RECORD_SCHEMA
        row["conversation"][1]["value"] = answer
        row["target"]["rendered_answer"] = answer
        row["target"]["qa_contract_schema"] = QA_CONTRACT_SCHEMA
        row["loss_policy"] = {
            "contract": "gradient_routed_semantic_fields_v4",
            "hard_language_target": hard_language_allowed(variant, family),
            "semantic_field_target": hard_language_allowed(variant, family),
            "hard_stance_target": (
                family == "driving_implication"
                and variant in HARD_STANCE_VARIANTS
            ),
            "dense_relevance_target": True,
            "cross_family_preference_anchor": False,
            "same_family_counterfactual_preference_anchor": False,
            "counterfactual_preference_negative_count": 0,
            "deterministic_format_is_release_gate": False,
            "optimizer_group_complete_before_step": True,
        }
        upgraded.append(row)

    groups = _groups(upgraded)
    for rows in groups.values():
        for row in rows:
            policy = row["loss_policy"]
            if not policy["hard_language_target"]:
                continue
            try:
                candidates = same_family_unique_structured_answers(rows, row)
            except ValueError:
                candidates = ()
            negative_count = max(0, len(candidates) - 1)
            policy["same_family_counterfactual_preference_anchor"] = (
                negative_count > 0
            )
            policy["counterfactual_preference_negative_count"] = negative_count
    return upgraded, audit_records(upgraded)


def audit_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups = _groups(records)
    errors = []
    hard_language = 0
    hard_stance = 0
    preference_anchors = 0
    distinct_negatives = 0
    stance_counts = {name: 0 for name in ("maintain", "caution", "prepare_to_yield")}
    for group_id, rows in sorted(groups.items()):
        for row in rows:
            sample_id = str(row.get("sample_id", ""))
            family = str(row["question_family"])
            variant = str(row["counterfactual"]["variant"])
            expected = render_structured_answer(
                family, row["target"]["structured_summary"]
            )
            answer = str(row["conversation"][1]["value"])
            hard = hard_language_allowed(variant, family)
            stance_hard = (
                family == "driving_implication"
                and variant in HARD_STANCE_VARIANTS
            )
            hard_language += int(hard)
            hard_stance += int(stance_hard)
            if stance_hard:
                stance = str(
                    row["target"]["structured_summary"]["planning_implication"]["stance"]
                )
                stance_counts[stance] += 1
            policy = row.get("loss_policy", {})
            try:
                parsed = parse_semantic_fields(answer, family)
                candidates = same_family_unique_structured_answers(rows, row)
                negative_count = len(candidates) - 1
            except ValueError as error:
                parsed = {}
                negative_count = 0
                if hard:
                    errors.append({"sample_id": sample_id, "error": str(error)})
            anchor = hard and negative_count > 0
            preference_anchors += int(anchor)
            distinct_negatives += negative_count if anchor else 0
            if (
                row.get("schema") != RECORD_SCHEMA
                or row["target"].get("qa_contract_schema") != QA_CONTRACT_SCHEMA
                or row["target"].get("rendered_answer") != expected
                or answer != expected
                or not parsed
                or "<" in answer
                or ">" in answer
                or policy.get("contract") != "gradient_routed_semantic_fields_v4"
                or policy.get("hard_language_target") is not hard
                or policy.get("semantic_field_target") is not hard
                or policy.get("hard_stance_target") is not stance_hard
                or policy.get("dense_relevance_target") is not True
                or policy.get("cross_family_preference_anchor") is not False
                or policy.get("same_family_counterfactual_preference_anchor") is not anchor
                or int(policy.get("counterfactual_preference_negative_count", -1)) != (
                    negative_count if hard else 0
                )
                or policy.get("deterministic_format_is_release_gate") is not False
                or policy.get("optimizer_group_complete_before_step") is not True
            ):
                errors.append({"sample_id": sample_id, "error": "record contract mismatch"})
    checks = {
        "five_complete_matched_groups": len(groups) == 5,
        "one_hundred_records": len(records) == 100,
        "ninety_hard_language_targets": hard_language == 90,
        "fifteen_hard_stance_targets": hard_stance == 15,
        "all_stance_classes_present": all(value > 0 for value in stance_counts.values()),
        "structured_answers_parse_and_match": not errors,
        "semantic_preference_anchors_exist": preference_anchors > 0,
        "format_is_not_a_release_target": all(
            row["loss_policy"].get("deterministic_format_is_release_gate") is False
            for row in records
        ),
    }
    return {
        "schema": AUDIT_SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "record_count": len(records),
        "matched_group_count": len(groups),
        "hard_language_record_count": hard_language,
        "hard_stance_record_count": hard_stance,
        "stance_class_counts": stance_counts,
        "same_family_preference_anchor_count": preference_anchors,
        "distinct_counterfactual_negative_count": distinct_negatives,
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
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "orion.uq_relevance_qa_dataset_manifest.v4",
        "status": "prepared_training_locked" if audit["passed"] else "failed_audit",
        "source_records": {
            "path": str(args.input_records.resolve()),
            "sha256": sha256_file(args.input_records),
        },
        "records": {"path": str(records_path.resolve()), "sha256": sha256_file(records_path)},
        "audit": {"path": str(audit_path.resolve()), "sha256": sha256_file(audit_path)},
        "qa_contract": QA_CONTRACT_SCHEMA,
        "training_started": False,
        "claim_boundary": "Structured QA preparation only; no model learning or generalization evidence.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "audit": audit}, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

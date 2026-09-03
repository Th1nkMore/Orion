"""Upgrade and audit v2 Stage2-L QA records for the v3 semantic contract."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from uq_estimator.stage2l_matched_objective import (
    HARD_STANCE_VARIANTS,
    QUESTION_FAMILIES,
    hard_language_supervision_allowed,
    partition_complete_matched_groups,
)
from uq_estimator.stage2l_qa_contract_v3 import (
    FAMILY_RESPONSE_TAGS,
    SCHEMA as QA_CONTRACT_SCHEMA,
    canonical_tagged_answer,
    same_family_unique_counterfactual_answers,
    tagged_answer_family,
)


CONFIG_SCHEMA = "orion.uq_relevance_qa_factory_config.v3"
RECORD_SCHEMA = "orion.uq_relevance_qa_record.v2"
AUDIT_SCHEMA = "orion.uq_relevance_qa_v3_audit.v1"


def _stable_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported v3 QA factory config")
    if config.get("response_tags") != FAMILY_RESPONSE_TAGS:
        raise ValueError("v3 response tags differ from the code contract")
    policy = config.get("supervision_policy", {})
    if tuple(policy.get("hard_stance_variants", [])) != HARD_STANCE_VARIANTS:
        raise ValueError("v3 hard stance variants differ from the matched contract")
    exclusions = {
        (str(row.get("variant")), str(row.get("question_family")))
        for row in policy.get("hard_language_exclusions", [])
    }
    if exclusions != {
        ("observed", "driving_implication"),
        ("view_shuffled_uq", "driving_implication"),
    }:
        raise ValueError("v3 hard language exclusions are malformed")
    preference = policy.get("answer_preference", {})
    if (
        preference.get("kind") != "same_family_unique_counterfactual"
        or preference.get("cross_family_answers_are_negatives") is not False
        or preference.get("exact_duplicate_answers_are_negatives") is not False
        or int(preference.get("minimum_distinct_negative_count_for_anchor", -1))
        != 1
    ):
        raise ValueError("v3 answer preference contract is malformed")
    optimizer = policy.get("optimizer_group", {})
    if (
        int(optimizer.get("records", -1)) != 20
        or int(optimizer.get("optimizer_steps_inside_group", -1)) != 0
        or int(optimizer.get("optimizer_steps_after_group", -1)) != 1
    ):
        raise ValueError("v3 optimizer-group contract is malformed")


def _base_loss_policy(variant: str, family: str) -> Dict[str, Any]:
    hard_language = hard_language_supervision_allowed(variant, family)
    return {
        "contract": "calibrated_same_family_counterfactual_v3",
        "hard_language_target": hard_language,
        "qa_family_tag_target": hard_language,
        "hard_stance_target": (
            family == "driving_implication" and variant in HARD_STANCE_VARIANTS
        ),
        "dense_relevance_target": True,
        "cross_family_preference_anchor": False,
        "same_family_counterfactual_preference_anchor": False,
        "counterfactual_preference_negative_count": 0,
        "optimizer_group_complete_before_step": True,
    }


def upgrade_records(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Create v3 records from structured v2 targets without changing model inputs."""

    _validate_config(config)
    source_groups = partition_complete_matched_groups(records)
    upgraded: List[Dict[str, Any]] = []
    for source_group in source_groups:
        current_group: List[Dict[str, Any]] = []
        for source in source_group:
            row = deepcopy(dict(source))
            family = str(row["question_family"])
            variant = str(row["counterfactual"]["variant"])
            original_answer = str(row["conversation"][1]["value"])
            answer = canonical_tagged_answer(family, original_answer)
            row["schema"] = RECORD_SCHEMA
            row["conversation"][1]["value"] = answer
            row["target"]["rendered_answer"] = answer
            row["target"]["qa_contract_schema"] = QA_CONTRACT_SCHEMA
            row["loss_policy"] = _base_loss_policy(variant, family)
            provenance = dict(row.get("provenance", {}))
            provenance["v3_qa_upgrade"] = {
                "source_record_sha256": _stable_sha256(source),
                "source_record_schema": str(source.get("schema", "")),
                "model_input_unchanged": True,
                "structured_target_unchanged": True,
            }
            row["provenance"] = provenance
            current_group.append(row)

        for row in current_group:
            if not row["loss_policy"]["hard_language_target"]:
                continue
            try:
                answers = same_family_unique_counterfactual_answers(
                    current_group, row
                )
            except ValueError as error:
                if "distinct answer" not in str(error):
                    raise
                answers = (str(row["conversation"][1]["value"]),)
            negative_count = len(answers) - 1
            row["loss_policy"][
                "counterfactual_preference_negative_count"
            ] = negative_count
            row["loss_policy"][
                "same_family_counterfactual_preference_anchor"
            ] = negative_count >= 1
        upgraded.extend(current_group)

    audit = audit_v3_records(upgraded, config=config)
    if not all(audit["checks"].values()):
        raise ValueError("upgraded v3 QA records failed their contract audit")
    return upgraded, audit


def audit_v3_records(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    _validate_config(config)
    groups = partition_complete_matched_groups(records)
    tag_errors = []
    answer_errors = []
    policy_errors = []
    stance_errors = []
    preference_anchor_count = 0
    distinct_negative_count = 0
    hard_language_count = 0
    hard_stance_count = 0
    for group in groups:
        for row in group:
            sample_id = str(row.get("sample_id", ""))
            family = str(row["question_family"])
            variant = str(row["counterfactual"]["variant"])
            answer = str(row["conversation"][1]["value"])
            expected_hard_language = hard_language_supervision_allowed(
                variant, family
            )
            expected_hard_stance = (
                family == "driving_implication" and variant in HARD_STANCE_VARIANTS
            )
            hard_language_count += int(expected_hard_language)
            hard_stance_count += int(expected_hard_stance)
            try:
                parsed_family = tagged_answer_family(answer)
            except ValueError:
                parsed_family = None
            if parsed_family != family:
                tag_errors.append(sample_id)
            if (
                row.get("schema") != RECORD_SCHEMA
                or row["target"].get("rendered_answer") != answer
                or row["target"].get("qa_contract_schema") != QA_CONTRACT_SCHEMA
            ):
                answer_errors.append(sample_id)
            policy = row.get("loss_policy", {})
            try:
                answers = same_family_unique_counterfactual_answers(group, row)
                expected_negative_count = len(answers) - 1
            except ValueError as error:
                if "distinct answer" not in str(error):
                    raise
                expected_negative_count = 0
            expected_anchor = expected_hard_language and expected_negative_count >= 1
            preference_anchor_count += int(expected_anchor)
            distinct_negative_count += expected_negative_count if expected_anchor else 0
            if (
                policy.get("contract")
                != "calibrated_same_family_counterfactual_v3"
                or policy.get("hard_language_target") is not expected_hard_language
                or policy.get("qa_family_tag_target") is not expected_hard_language
                or policy.get("hard_stance_target") is not expected_hard_stance
                or policy.get("dense_relevance_target") is not True
                or policy.get("cross_family_preference_anchor") is not False
                or policy.get("same_family_counterfactual_preference_anchor")
                is not expected_anchor
                or int(policy.get("counterfactual_preference_negative_count", -1))
                != (expected_negative_count if expected_hard_language else 0)
                or policy.get("optimizer_group_complete_before_step") is not True
            ):
                policy_errors.append(sample_id)
            if family == "driving_implication":
                stance = str(
                    row["target"]["structured_summary"]["planning_implication"][
                        "stance"
                    ]
                )
                if (
                    not answer.startswith(FAMILY_RESPONSE_TAGS[family])
                    or ("planning stance is %s." % stance) not in answer
                ):
                    stance_errors.append(sample_id)

    checks = {
        "all_family_tags_parse_and_match": not tag_errors,
        "rendered_answers_are_hashable_v3_targets": not answer_errors,
        "loss_policies_match_unique_counterfactual_contract": not policy_errors,
        "driving_answers_match_structured_stance": not stance_errors,
        "cross_family_preference_is_fully_disabled": all(
            row["loss_policy"].get("cross_family_preference_anchor") is False
            for row in records
        ),
        "at_least_one_semantic_preference_anchor_per_group": (
            preference_anchor_count >= len(groups)
        ),
    }
    return {
        "schema": AUDIT_SCHEMA,
        "record_count": len(records),
        "matched_group_count": len(groups),
        "hard_language_record_count": hard_language_count,
        "hard_stance_record_count": hard_stance_count,
        "same_family_preference_anchor_count": preference_anchor_count,
        "distinct_counterfactual_negative_count": distinct_negative_count,
        "errors": {
            "family_tags": tag_errors,
            "answers": answer_errors,
            "loss_policy": policy_errors,
            "driving_stance": stance_errors,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "claim_boundary": (
            "Static QA contract audit only; no language learnability, generation, "
            "model understanding, planning or safety evidence."
        ),
    }


__all__ = [
    "AUDIT_SCHEMA",
    "CONFIG_SCHEMA",
    "RECORD_SCHEMA",
    "audit_v3_records",
    "upgrade_records",
]

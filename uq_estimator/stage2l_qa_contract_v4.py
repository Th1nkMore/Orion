"""Structured, common-vocabulary QA contract for Stage2-L.

Formatting is kept human-readable but is not treated as evidence of semantic
understanding.  Release metrics parse and score the value-bearing fields.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Dict, Mapping, Sequence, Tuple


SCHEMA = "orion.stage2l_qa_contract.v4"
QUESTION_FAMILIES = (
    "observation_semantics",
    "epistemic_limitation",
    "task_relevance",
    "driving_implication",
)
HARD_STANCE_VARIANTS = ("zero_uq", "off_path_uq", "on_path_uq")
FIELD_PREFIXES = {
    "observation_semantics": "Observation uncertainty:",
    "epistemic_limitation": "Epistemic limitation:",
    "task_relevance": "Task relevance map:",
    "driving_implication": "Planning stance:",
}
FAMILY_FIELD_KEYS = {
    "observation_semantics": ("uq_level", "uq_view", "uq_region", "uq_trend"),
    "epistemic_limitation": (
        "evidence",
        "evidence_view",
        "evidence_region",
        "hidden_content",
        "task_relevance",
    ),
    "task_relevance": (
        "relevance_level",
        "risk_level",
        "risk_view",
        "risk_region",
    ),
    "driving_implication": ("stance", "direct_control"),
}
_FIELD_PATTERN = re.compile(r"\b([a-z][a-z0-9_]*)=([A-Za-z0-9_]+)\b")


def expected_semantic_fields(
    family: str, summary: Mapping[str, Any]
) -> Dict[str, str]:
    family = str(family)
    observation = summary["observation_uncertainty"]
    relevance = summary["relevance_at_most_uncertain_region"]
    risk = summary["task_risk"]
    planning = summary["planning_implication"]
    if family == "observation_semantics":
        return {
            "uq_level": str(observation["level"]),
            "uq_view": str(observation["peak_view"]),
            "uq_region": str(observation["peak_region"]),
            "uq_trend": str(observation["temporal_trend"]),
        }
    if family == "epistemic_limitation":
        return {
            "evidence": "unreliable",
            "evidence_view": str(observation["peak_view"]),
            "evidence_region": str(observation["peak_region"]),
            "hidden_content": "unknown",
            "task_relevance": "separate",
        }
    if family == "task_relevance":
        return {
            "relevance_level": str(relevance["level"]),
            "risk_level": str(risk["level"]),
            "risk_view": str(risk["peak_view"]),
            "risk_region": str(risk["peak_region"]),
        }
    if family == "driving_implication":
        return {
            "stance": str(planning["stance"]),
            "direct_control": (
                "yes" if bool(planning["is_direct_control_command"]) else "no"
            ),
        }
    raise ValueError("unsupported QA family: %s" % family)


def render_structured_answer(family: str, summary: Mapping[str, Any]) -> str:
    fields = expected_semantic_fields(family, summary)
    ordered = FAMILY_FIELD_KEYS[str(family)]
    body = "; ".join("%s=%s" % (key, fields[key]) for key in ordered)
    return "%s %s." % (FIELD_PREFIXES[str(family)], body)


def parse_semantic_fields(answer: str, family: str) -> Dict[str, str]:
    family = str(family)
    if family not in FAMILY_FIELD_KEYS:
        raise ValueError("unsupported QA family: %s" % family)
    pairs = _FIELD_PATTERN.findall(str(answer))
    parsed: Dict[str, str] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError("duplicate semantic field: %s" % key)
        parsed[key] = value
    expected_keys = set(FAMILY_FIELD_KEYS[family])
    if set(parsed) != expected_keys:
        raise ValueError("semantic fields do not match the requested QA family")
    return parsed


def has_expected_prefix(answer: str, family: str) -> bool:
    return str(answer).strip().startswith(FIELD_PREFIXES[str(family)])


def semantic_text_is_nonrepeating(answer: str) -> bool:
    tokens = re.findall(r"[A-Za-z0-9_]+", str(answer).lower())
    if len(tokens) < 4:
        return False
    counts = Counter(tokens)
    return (
        len(counts) / len(tokens) >= 0.6
        and max(counts.values()) / len(tokens) <= 0.3
    )


def generation_semantic_metrics(
    generated: Mapping[str, Mapping[str, str]],
    target_summaries: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    if set(generated) != set(HARD_STANCE_VARIANTS):
        raise ValueError("generated variants are incomplete")
    if set(target_summaries) != set(HARD_STANCE_VARIANTS):
        raise ValueError("target summaries are incomplete")
    answer_count = 0
    parsed_answers = 0
    exact_answers = 0
    prefix_matches = 0
    nonrepeating = 0
    field_total = 0
    field_correct = 0
    diagnostics = {}
    for variant in HARD_STANCE_VARIANTS:
        if set(generated[variant]) != set(QUESTION_FAMILIES):
            raise ValueError("generated QA families are incomplete")
        diagnostics[variant] = {}
        for family in QUESTION_FAMILIES:
            text = str(generated[variant][family])
            target = expected_semantic_fields(family, target_summaries[variant])
            answer_count += 1
            prefix = has_expected_prefix(text, family)
            prefix_matches += int(prefix)
            good_text = semantic_text_is_nonrepeating(text)
            nonrepeating += int(good_text)
            try:
                parsed = parse_semantic_fields(text, family)
            except ValueError:
                parsed = {}
            parsed_answers += int(bool(parsed))
            exact = parsed == target
            exact_answers += int(exact)
            for key, expected in target.items():
                field_total += 1
                field_correct += int(parsed.get(key) == expected)
            diagnostics[variant][family] = {
                "parsed": bool(parsed),
                "semantic_exact_match": exact,
                "prefix_match_diagnostic": prefix,
                "nonrepeating": good_text,
                "parsed_fields": parsed,
                "target_fields": target,
            }
    return {
        "semantic_parse_rate": parsed_answers / answer_count,
        "semantic_answer_exact_match": exact_answers / answer_count,
        "semantic_field_accuracy": field_correct / field_total,
        "format_prefix_accuracy_diagnostic": prefix_matches / answer_count,
        "nonrepeating_text_fraction": nonrepeating / answer_count,
        "diagnostics": diagnostics,
    }


def same_family_unique_structured_answers(
    records: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any]
) -> Tuple[str, ...]:
    group_id = str(anchor["counterfactual"]["group_id"])
    family = str(anchor["question_family"])
    target = render_structured_answer(
        family, anchor["target"]["structured_summary"]
    )
    anchor_id = str(anchor.get("sample_id", ""))
    answers = [target]
    seen = {target}
    for row in records:
        if (
            str(row["counterfactual"]["group_id"]) != group_id
            or str(row["question_family"]) != family
            or row is anchor
            or (anchor_id and str(row.get("sample_id", "")) == anchor_id)
        ):
            continue
        candidate = render_structured_answer(
            family, row["target"]["structured_summary"]
        )
        if candidate not in seen:
            seen.add(candidate)
            answers.append(candidate)
    if len(answers) < 2:
        raise ValueError("structured preference requires a distinct answer")
    return tuple(answers)


__all__ = [
    "FIELD_PREFIXES",
    "FAMILY_FIELD_KEYS",
    "HARD_STANCE_VARIANTS",
    "QUESTION_FAMILIES",
    "SCHEMA",
    "expected_semantic_fields",
    "generation_semantic_metrics",
    "has_expected_prefix",
    "parse_semantic_fields",
    "render_structured_answer",
    "same_family_unique_structured_answers",
    "semantic_text_is_nonrepeating",
]

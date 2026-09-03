"""Semantically closed structured QA contract for Stage2-L.

Stage1 observation uncertainty remains task agnostic.  Dense task relevance,
task risk and the uncertainty-response stance remain VLM owned.  Text is a
deterministic rendering of value-bearing fields, not a substitute for those
predictions.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Dict, Mapping, Tuple


SCHEMA = "orion.stage2l_qa_contract.v5"
NO_SIGNAL_EPSILON = 1e-8
QUESTION_FAMILIES = (
    "observation_semantics",
    "epistemic_limitation",
    "task_relevance",
    "driving_implication",
)
TASK_FIELD_KEYS = (
    "relevance_level",
    "risk_level",
    "risk_view",
    "risk_region",
    "stance",
)
FIELD_PREFIXES = {
    "observation_semantics": "Observation uncertainty:",
    "epistemic_limitation": "Epistemic limitation:",
    "task_relevance": "Task relevance map:",
    "driving_implication": "Uncertainty response:",
}
FAMILY_FIELD_KEYS = {
    "observation_semantics": (
        "uq_level",
        "uq_view",
        "uq_region",
        "uq_trend",
    ),
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
    "driving_implication": (
        "stance",
        "direct_control",
        "response_basis",
    ),
}
FIELD_VOCABULARIES = {
    "uq_level": ("low", "medium", "high"),
    "uq_view": (
        "none",
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ),
    "uq_region": (
        "none",
        "upper_left",
        "upper_center",
        "upper_right",
        "middle_left",
        "middle_center",
        "middle_right",
        "lower_left",
        "lower_center",
        "lower_right",
    ),
    "uq_trend": ("stable", "rising", "falling"),
    "evidence": ("not_flagged", "unreliable"),
    "evidence_view": (),
    "evidence_region": (),
    "hidden_content": ("not_applicable", "unknown"),
    "task_relevance": ("separate",),
    "relevance_level": ("not_applicable", "low", "high"),
    "risk_level": ("none", "low", "medium", "high"),
    "risk_view": (),
    "risk_region": (),
    "stance": ("maintain", "caution", "prepare_to_yield"),
    "direct_control": ("no",),
    "response_basis": ("observation_uncertainty",),
}
_FIELD_PATTERN = re.compile(r"\b([a-z][a-z0-9_]*)=([A-Za-z0-9_]+)\b")


def _has_observation_signal(summary: Mapping[str, Any]) -> bool:
    observation = summary["observation_uncertainty"]
    return float(observation.get("peak_score", 0.0)) > NO_SIGNAL_EPSILON


def _has_task_risk(summary: Mapping[str, Any]) -> bool:
    risk = summary["task_risk"]
    return float(risk.get("peak_score", 0.0)) > NO_SIGNAL_EPSILON


def _validate_field_value(key: str, value: str) -> None:
    if key in {"evidence_view", "risk_view"}:
        vocabulary = FIELD_VOCABULARIES["uq_view"]
    elif key in {"evidence_region", "risk_region"}:
        vocabulary = FIELD_VOCABULARIES["uq_region"]
    else:
        vocabulary = FIELD_VOCABULARIES[key]
    if str(value) not in vocabulary:
        raise ValueError("unsupported semantic field value: %s=%s" % (key, value))


def expected_semantic_fields(
    family: str, summary: Mapping[str, Any]
) -> Dict[str, str]:
    """Produce target fields with explicit no-signal semantics."""

    family = str(family)
    observation = summary["observation_uncertainty"]
    relevance = summary["relevance_at_most_uncertain_region"]
    risk = summary["task_risk"]
    planning = summary["planning_implication"]
    has_signal = _has_observation_signal(summary)
    has_task_risk = _has_task_risk(summary)
    if family == "observation_semantics":
        fields = {
            "uq_level": str(observation["level"]),
            "uq_view": str(observation["peak_view"]) if has_signal else "none",
            "uq_region": (
                str(observation["peak_region"]) if has_signal else "none"
            ),
            "uq_trend": str(observation["temporal_trend"]),
        }
    elif family == "epistemic_limitation":
        fields = {
            "evidence": "unreliable" if has_signal else "not_flagged",
            "evidence_view": (
                str(observation["peak_view"]) if has_signal else "none"
            ),
            "evidence_region": (
                str(observation["peak_region"]) if has_signal else "none"
            ),
            "hidden_content": "unknown" if has_signal else "not_applicable",
            "task_relevance": "separate",
        }
    elif family == "task_relevance":
        fields = {
            "relevance_level": (
                str(relevance["level"]) if has_signal else "not_applicable"
            ),
            "risk_level": (
                str(risk["level"]) if has_task_risk else "none"
            ),
            "risk_view": (
                str(risk["peak_view"]) if has_task_risk else "none"
            ),
            "risk_region": (
                str(risk["peak_region"]) if has_task_risk else "none"
            ),
        }
    elif family == "driving_implication":
        fields = {
            "stance": str(planning["stance"]),
            "direct_control": "no",
            "response_basis": "observation_uncertainty",
        }
    else:
        raise ValueError("unsupported QA family: %s" % family)
    for key, value in fields.items():
        _validate_field_value(key, value)
    return fields


def expected_task_field_targets(
    summary: Mapping[str, Any]
) -> Dict[str, str]:
    """Return the VLM-owned task fields without duplicating label logic."""

    fields = expected_semantic_fields("task_relevance", summary)
    fields["stance"] = expected_semantic_fields(
        "driving_implication", summary
    )["stance"]
    if tuple(fields) != TASK_FIELD_KEYS:
        raise RuntimeError("task-field target order differs from contract")
    return fields


def render_semantic_fields(family: str, fields: Mapping[str, Any]) -> str:
    family = str(family)
    if family not in FAMILY_FIELD_KEYS:
        raise ValueError("unsupported QA family: %s" % family)
    ordered = FAMILY_FIELD_KEYS[family]
    if set(fields) != set(ordered):
        raise ValueError("semantic fields do not match the requested QA family")
    values = {key: str(fields[key]) for key in ordered}
    for key, value in values.items():
        _validate_field_value(key, value)
    body = "; ".join("%s=%s" % (key, values[key]) for key in ordered)
    return "%s %s." % (FIELD_PREFIXES[family], body)


def render_structured_answer(family: str, summary: Mapping[str, Any]) -> str:
    return render_semantic_fields(
        family, expected_semantic_fields(family, summary)
    )


def parse_semantic_fields(answer: str, family: str) -> Dict[str, str]:
    family = str(family)
    if family not in FAMILY_FIELD_KEYS:
        raise ValueError("unsupported QA family: %s" % family)
    parsed: Dict[str, str] = {}
    for key, value in _FIELD_PATTERN.findall(str(answer)):
        if key in parsed:
            raise ValueError("duplicate semantic field: %s" % key)
        parsed[key] = value
    if set(parsed) != set(FAMILY_FIELD_KEYS[family]):
        raise ValueError("semantic fields do not match the requested QA family")
    for key, value in parsed.items():
        _validate_field_value(key, value)
    return parsed


def semantic_text_is_nonrepeating(answer: str) -> bool:
    tokens = re.findall(r"[A-Za-z0-9_]+", str(answer).lower())
    if len(tokens) < 4:
        return False
    counts = Counter(tokens)
    return (
        len(counts) / len(tokens) >= 0.6
        and max(counts.values()) / len(tokens) <= 0.3
    )


def deterministic_render_metrics(
    predicted_fields: Mapping[str, Mapping[str, Mapping[str, str]]],
    target_summaries: Mapping[str, Mapping[str, Any]],
) -> Dict[str, float]:
    """Score predicted fields and their deterministic text rendering."""

    answer_count = 0
    exact_answers = 0
    field_count = 0
    field_correct = 0
    parse_count = 0
    nonrepeating = 0
    for variant, family_fields in predicted_fields.items():
        if variant not in target_summaries:
            raise ValueError("predicted variant has no target summary")
        if set(family_fields) != set(QUESTION_FAMILIES):
            raise ValueError("predicted QA families are incomplete")
        for family, fields in family_fields.items():
            answer_count += 1
            expected = expected_semantic_fields(
                family, target_summaries[variant]
            )
            text = render_semantic_fields(family, fields)
            parsed = parse_semantic_fields(text, family)
            parse_count += 1
            exact_answers += int(parsed == expected)
            nonrepeating += int(semantic_text_is_nonrepeating(text))
            for key, value in expected.items():
                field_count += 1
                field_correct += int(parsed.get(key) == value)
    if answer_count == 0 or field_count == 0:
        raise ValueError("semantic metric input is empty")
    return {
        "semantic_parse_rate": parse_count / answer_count,
        "semantic_answer_exact_match": exact_answers / answer_count,
        "semantic_field_accuracy": field_correct / field_count,
        "nonrepeating_text_fraction": nonrepeating / answer_count,
    }


__all__ = [
    "FAMILY_FIELD_KEYS",
    "FIELD_PREFIXES",
    "FIELD_VOCABULARIES",
    "NO_SIGNAL_EPSILON",
    "QUESTION_FAMILIES",
    "TASK_FIELD_KEYS",
    "SCHEMA",
    "deterministic_render_metrics",
    "expected_semantic_fields",
    "expected_task_field_targets",
    "parse_semantic_fields",
    "render_semantic_fields",
    "render_structured_answer",
    "semantic_text_is_nonrepeating",
]

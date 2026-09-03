"""Explicit, machine-parseable prompts for the frozen Stage2-L1 U concepts.

This module is deliberately versioned separately from the completed v14.1
training contract.  It does not change U, labels, routes, risk, actions, or
model weights.  It only makes the already-frozen six-field answer language
literal and exposes finite candidate sets for constrained diagnostic decode.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from uq_estimator.stage2l_u_concept_qa_v14 import (
    TAG_ORDER,
    UConceptSummary,
    render_u_answer,
)


SCHEMA = "orion.stage2l-u-concept-explicit-schema/v1"
FIELD_VOCABULARIES: Mapping[str, tuple[str, ...]] = {
    "U_PRESENT": ("yes", "no"),
    "U_VIEW": (
        "front",
        "front_left",
        "front_right",
        "rear",
        "rear_left",
        "rear_right",
        "none",
    ),
    "U_REGION": (
        "upper_left",
        "upper_center",
        "upper_right",
        "middle_left",
        "middle_center",
        "middle_right",
        "lower_left",
        "lower_center",
        "lower_right",
        "none",
    ),
    "U_LEVEL": ("low", "medium", "high", "none"),
    "U_TREND": ("rising", "stable", "falling"),
    "U_COMPONENT": (
        "persistent_direction",
        "persistent_magnitude",
        "transient_inconsistency",
        "mixed",
        "none",
    ),
}


def _choice_line(tag: str) -> str:
    if tag not in FIELD_VOCABULARIES:
        raise ValueError("unknown U concept tag")
    return "<%s> [%s]" % (tag, "|".join(FIELD_VOCABULARIES[tag]))


def explicit_u_request(tag: str | None = None) -> str:
    """Return a literal output schema with all legal values in the question."""

    boundary = (
        "Read only the supplied observation-uncertainty tokens. Do not infer "
        "task relevance, driving risk, an action, a trajectory, or control. "
    )
    restrictions = (
        "Replace every bracketed choice with exactly one listed value. "
        "Copy every field name exactly. Do not shorten or rename a field. "
        "Do not repeat a field. Output no explanation, prefix, suffix, blank "
        "line, Markdown, or code fence."
    )
    if tag is not None:
        if tag not in TAG_ORDER:
            raise ValueError("unknown U concept tag")
        return "%sOutput exactly this one-line schema:\n%s\n%s" % (
            boundary,
            _choice_line(tag),
            restrictions,
        )
    lines = "\n".join(_choice_line(current) for current in TAG_ORDER)
    return (
        "%sOutput exactly six lines in the following order:\n%s\n%s"
        % (boundary, lines, restrictions)
    )


def build_explicit_u_qa_row(
    summary: UConceptSummary, tag: str | None = None
) -> dict:
    if tag is not None and tag not in TAG_ORDER:
        raise ValueError("unknown U concept tag")
    return {
        "schema": SCHEMA,
        "question_family": (
            "u_explicit_full_state"
            if tag is None
            else "u_explicit_field_%s" % tag.lower()
        ),
        "conversation": (
            {"from": "human", "value": explicit_u_request(tag)},
            {"from": "gpt", "value": render_u_answer(summary, tag)},
        ),
    }


def candidate_answers(tag: str) -> tuple[str, ...]:
    if tag not in FIELD_VOCABULARIES:
        raise ValueError("unknown U concept tag")
    return tuple("<%s> %s" % (tag, value) for value in FIELD_VOCABULARIES[tag])


def decode_candidate_nlls(tag: str, nlls: Sequence[float]) -> str:
    vocabulary = FIELD_VOCABULARIES.get(tag)
    if vocabulary is None:
        raise ValueError("unknown U concept tag")
    if len(nlls) != len(vocabulary):
        raise ValueError("candidate NLL count differs from field vocabulary")
    values = [float(value) for value in nlls]
    if any(value != value or value in (float("inf"), float("-inf")) for value in values):
        raise ValueError("candidate NLLs must be finite")
    return vocabulary[min(range(len(values)), key=values.__getitem__)]


def parse_strict_u_answer(text: str, tag: str | None = None) -> dict[str, str]:
    """Parse only the exact requested grammar; reject repaired/partial text."""

    requested = TAG_ORDER if tag is None else (tag,)
    if tag is not None and tag not in TAG_ORDER:
        raise ValueError("unknown U concept tag")
    value = str(text)
    if value != value.strip():
        raise ValueError("U answer contains leading or trailing whitespace")
    lines = value.splitlines()
    if len(lines) != len(requested):
        raise ValueError("U answer line count differs from the schema")
    parsed: dict[str, str] = {}
    for expected_tag, line in zip(requested, lines):
        match = re.fullmatch(r"<([A-Z_]+)> ([a-z_]+)", line)
        if match is None or match.group(1) != expected_tag:
            raise ValueError("U answer field name or syntax differs")
        field_value = match.group(2)
        if field_value not in FIELD_VOCABULARIES[expected_tag]:
            raise ValueError("U answer contains an out-of-vocabulary value")
        parsed[expected_tag] = field_value
    return parsed


__all__ = [
    "FIELD_VOCABULARIES",
    "SCHEMA",
    "build_explicit_u_qa_row",
    "candidate_answers",
    "decode_candidate_nlls",
    "explicit_u_request",
    "parse_strict_u_answer",
]

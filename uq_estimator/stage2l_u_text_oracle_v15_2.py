"""Task-free natural-language oracle for localizing the U interface bottleneck.

The oracle does not estimate uncertainty and it is not an input modality for
training.  It renders the already frozen Stage-1 U summary as authoritative
natural language so that the same ORION language model can be evaluated with
the continuous U-token channel removed.  Route, relevance, risk, action,
planning, trajectory, and control concepts are intentionally absent.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from uq_estimator.stage2l_u_concept_explicit_schema_v14_2 import (
    FIELD_VOCABULARIES,
)
from uq_estimator.stage2l_u_concept_qa_v14 import TAG_ORDER, UConceptSummary


SCHEMA = "orion.stage2l-u-text-oracle/v1"


FIELD_QUESTIONS: Mapping[str, str] = {
    "U_PRESENT": "Is observation uncertainty present?",
    "U_VIEW": "Which camera view contains the strongest observation uncertainty?",
    "U_REGION": "Which image region contains its strongest location?",
    "U_LEVEL": "What is its overall uncertainty level?",
    "U_TREND": "What is its temporal trend from the first frame to the latest frame?",
    "U_COMPONENT": "Which uncertainty component is dominant?",
}


def _words(value: str) -> str:
    return str(value).replace("_", "-")


def render_text_oracle_summary(summary: UConceptSummary) -> str:
    """Render every canonical U field as non-driving natural-language facts."""

    fields = summary.fields()
    if fields["U_PRESENT"] == "no":
        return (
            "No observation uncertainty is present. Consequently no camera "
            "view, image region, uncertainty level, or uncertainty component "
            "applies. The temporal trend is stable."
        )
    return (
        "Observation uncertainty is present. Its strongest location is in the "
        f"{_words(fields['U_VIEW'])} camera view and the "
        f"{_words(fields['U_REGION'])} image region. Its overall uncertainty "
        f"level is {fields['U_LEVEL']}. From the first frame to the latest "
        f"frame, its temporal trend is {fields['U_TREND']}. Its dominant "
        f"uncertainty component is {_words(fields['U_COMPONENT'])}."
    )


def text_oracle_field_row(summary: UConceptSummary, tag: str) -> dict:
    """Build one extraction question whose only U source is literal text."""

    if tag not in TAG_ORDER:
        raise ValueError("unknown U field")
    choices = ", ".join(FIELD_VOCABULARIES[tag])
    prompt = (
        "The following is an exact, authoritative, task-free description of "
        "observation uncertainty. Read the description literally. Do not infer "
        "task relevance, driving risk, an action, a trajectory, or control.\n"
        f"Observation-uncertainty description: {render_text_oracle_summary(summary)}\n"
        f"Question: {FIELD_QUESTIONS[tag]}\n"
        f"Answer with exactly one canonical value from this list: {choices}. "
        "Output only that value, with no explanation or punctuation."
    )
    return {
        "schema": SCHEMA,
        "question_family": "u_text_oracle_%s" % tag.lower(),
        "conversation": (
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": summary.fields()[tag]},
        ),
    }


def text_oracle_full_row(summary: UConceptSummary) -> dict:
    """Build the secondary free-generation diagnostic prompt."""

    prompt = (
        "The following is an exact, authoritative, task-free description of "
        "observation uncertainty. Read it literally and do not infer driving "
        "relevance or an action.\n"
        f"Observation-uncertainty description: {render_text_oracle_summary(summary)}\n"
        "Return exactly six lines, in this order, using canonical values:\n"
        "present=<value>\nview=<value>\nregion=<value>\nlevel=<value>\n"
        "trend=<value>\ncomponent=<value>\n"
        "Output no explanation, Markdown, or extra line."
    )
    return {
        "schema": SCHEMA,
        "question_family": "u_text_oracle_full_state",
        "conversation": (
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": render_text_oracle_answer(summary)},
        ),
    }


def text_oracle_candidates(tag: str) -> tuple[str, ...]:
    if tag not in TAG_ORDER:
        raise ValueError("unknown U field")
    return tuple(FIELD_VOCABULARIES[tag])


def render_text_oracle_answer(summary: UConceptSummary) -> str:
    fields = summary.fields()
    keys = ("present", "view", "region", "level", "trend", "component")
    return "\n".join(
        "%s=%s" % (key, fields[tag]) for key, tag in zip(keys, TAG_ORDER)
    )


def parse_text_oracle_answer(text: str) -> dict[str, str]:
    """Strictly parse the secondary six-line rendering diagnostic."""

    value = str(text)
    if value != value.strip():
        raise ValueError("text-oracle answer contains boundary whitespace")
    keys = ("present", "view", "region", "level", "trend", "component")
    lines = value.splitlines()
    if len(lines) != len(TAG_ORDER):
        raise ValueError("text-oracle answer line count differs")
    parsed = {}
    for key, tag, line in zip(keys, TAG_ORDER, lines):
        match = re.fullmatch(r"([a-z]+)=([a-z_]+)", line)
        if match is None or match.group(1) != key:
            raise ValueError("text-oracle answer syntax differs")
        field_value = match.group(2)
        if field_value not in FIELD_VOCABULARIES[tag]:
            raise ValueError("text-oracle answer value is not canonical")
        parsed[tag] = field_value
    return parsed


def decode_candidate_nlls(tag: str, nlls: Sequence[float]) -> str:
    candidates = text_oracle_candidates(tag)
    values = [float(value) for value in nlls]
    if len(values) != len(candidates):
        raise ValueError("candidate NLL count differs")
    if any(value != value or value in (float("inf"), float("-inf")) for value in values):
        raise ValueError("candidate NLLs must be finite")
    return candidates[min(range(len(values)), key=values.__getitem__)]


__all__ = [
    "FIELD_QUESTIONS",
    "SCHEMA",
    "decode_candidate_nlls",
    "parse_text_oracle_answer",
    "render_text_oracle_answer",
    "render_text_oracle_summary",
    "text_oracle_candidates",
    "text_oracle_field_row",
    "text_oracle_full_row",
]

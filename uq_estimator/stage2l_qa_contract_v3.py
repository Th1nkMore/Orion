"""Pure-Python Stage2-L v3 QA tags and matched-answer contracts."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Mapping, Sequence, Tuple


SCHEMA = "orion.stage2l_qa_contract.v3"
FAMILY_RESPONSE_TAGS = {
    "observation_semantics": "<observation_uncertainty>",
    "epistemic_limitation": "<epistemic_limitation>",
    "task_relevance": "<task_relevance_map>",
    "driving_implication": "<planning_stance>",
}
QUESTION_FAMILIES = tuple(FAMILY_RESPONSE_TAGS)
HARD_STANCE_VARIANTS = ("zero_uq", "off_path_uq", "on_path_uq")


def canonical_tagged_answer(question_family: str, answer: str) -> str:
    """Return a deterministic family-tagged QA answer for generation checks."""

    family = str(question_family)
    if family not in FAMILY_RESPONSE_TAGS:
        raise ValueError("unsupported QA family: %s" % family)
    value = str(answer).strip()
    if not value:
        raise ValueError("QA answer must be non-empty")
    for tag in FAMILY_RESPONSE_TAGS.values():
        if value.startswith(tag):
            value = value[len(tag) :].lstrip()
            break
    # v2 reused the R-map tag for the driving answer.  Strip it even if a
    # caller supplied a partial/legacy tag set.
    if value.startswith("<task_relevance_map>"):
        value = value[len("<task_relevance_map>") :].lstrip()
    return "%s %s" % (FAMILY_RESPONSE_TAGS[family], value)


def tagged_answer_family(answer: str) -> str:
    """Parse the exact leading family tag or fail closed."""

    value = str(answer).strip()
    matches = [
        family for family, tag in FAMILY_RESPONSE_TAGS.items() if value.startswith(tag)
    ]
    if len(matches) != 1:
        raise ValueError("answer does not start with exactly one QA-family tag")
    return matches[0]


def parse_planning_stance(answer: str) -> str:
    """Parse the explicit generated planning stance or fail closed."""

    match = re.search(
        r"\bplanning\s+stance\s+is\s+"
        r"(maintain|caution|prepare_to_yield)\b",
        str(answer),
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError("generated answer lacks an explicit planning stance")
    return match.group(1).lower()


def generation_is_nonrepeating(answer: str) -> bool:
    """Reject empty, very short or cyclic text such as the failed v6 output."""

    tokens = re.findall(r"[A-Za-z0-9_]+", str(answer).lower())
    if len(tokens) < 4:
        return False
    counts = Counter(tokens)
    if len(counts) / len(tokens) < 0.5:
        return False
    if max(counts.values()) / len(tokens) > 0.35:
        return False
    bigrams = list(zip(tokens, tokens[1:]))
    if bigrams and max(Counter(bigrams).values()) / len(bigrams) > 0.3:
        return False
    return True


def generation_contract_metrics(
    generated: Mapping[str, Mapping[str, str]],
    target_stance_by_variant: Mapping[str, str],
) -> Mapping[str, Any]:
    """Aggregate fail-closed family, repetition and stance generation checks."""

    if set(generated) != set(HARD_STANCE_VARIANTS):
        raise ValueError("generated answers do not cover all hard stance variants")
    if set(target_stance_by_variant) != set(HARD_STANCE_VARIANTS):
        raise ValueError("generation targets do not cover all hard stance variants")
    family_total = 0
    family_correct = 0
    nonrepeating = 0
    stance_total = 0
    stance_parsed = 0
    stance_agreement = 0
    diagnostics = {}
    for variant in HARD_STANCE_VARIANTS:
        if set(generated[variant]) != set(QUESTION_FAMILIES):
            raise ValueError("generated answers do not cover all QA families")
        diagnostics[variant] = {}
        for family in QUESTION_FAMILIES:
            text = str(generated[variant][family])
            family_total += 1
            try:
                parsed_family = tagged_answer_family(text)
            except ValueError:
                parsed_family = None
            family_correct += int(parsed_family == family)
            is_nonrepeating = generation_is_nonrepeating(text)
            nonrepeating += int(is_nonrepeating)
            item = {
                "parsed_family": parsed_family,
                "nonrepeating": is_nonrepeating,
            }
            if family == "driving_implication":
                stance_total += 1
                try:
                    parsed_stance = parse_planning_stance(text)
                except ValueError:
                    parsed_stance = None
                stance_parsed += int(parsed_stance is not None)
                stance_agreement += int(
                    parsed_stance == target_stance_by_variant[variant]
                )
                item["parsed_stance"] = parsed_stance
                item["target_stance"] = target_stance_by_variant[variant]
            diagnostics[variant][family] = item
    return {
        "family_tag_parse_and_accuracy": family_correct / family_total,
        "nonrepeating_text_fraction": nonrepeating / family_total,
        "hard_driving_stance_parse_rate": stance_parsed / stance_total,
        "hard_driving_stance_agreement": stance_agreement / stance_total,
        "diagnostics": diagnostics,
    }


def same_family_unique_counterfactual_answers(
    records: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Any],
) -> Tuple[str, ...]:
    """Return target plus unique, genuinely different same-family answers."""

    group_id = str(anchor["counterfactual"]["group_id"])
    family = str(anchor["question_family"])
    target = canonical_tagged_answer(
        family, str(anchor["conversation"][1]["value"])
    )
    anchor_id = str(anchor.get("sample_id", ""))
    negatives = []
    seen = {target}
    for row in records:
        if (
            str(row["counterfactual"]["group_id"]) != group_id
            or str(row["question_family"]) != family
            or row is anchor
            or (anchor_id and str(row.get("sample_id", "")) == anchor_id)
        ):
            continue
        candidate = canonical_tagged_answer(
            family, str(row["conversation"][1]["value"])
        )
        if candidate not in seen:
            seen.add(candidate)
            negatives.append(candidate)
    if not negatives:
        raise ValueError("counterfactual preference requires a distinct answer")
    return (target, *negatives)


__all__ = [
    "FAMILY_RESPONSE_TAGS",
    "HARD_STANCE_VARIANTS",
    "QUESTION_FAMILIES",
    "SCHEMA",
    "canonical_tagged_answer",
    "generation_is_nonrepeating",
    "generation_contract_metrics",
    "parse_planning_stance",
    "same_family_unique_counterfactual_answers",
    "tagged_answer_family",
]

"""Task-free U-to-language alignment contracts for Stage2-L1 v15.

The frozen observation-uncertainty estimator remains the sole source of U.
This module only defines balanced, auditable language supervision over the
existing U fields.  It contains no route, actor, risk, action, trajectory, or
control concepts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import random
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from uq_estimator.stage2l_u_concept_explicit_schema_v14_2 import (
    FIELD_VOCABULARIES,
    build_explicit_u_qa_row,
    candidate_answers,
)
from uq_estimator.stage2l_u_concept_qa_v14 import (
    TAG_ORDER,
    U_VARIANTS,
    UConceptSummary,
)


SCHEMA = "orion.stage2l-u-language-alignment/v1"


@dataclass(frozen=True)
class ULanguageExample:
    group_id: str
    variant: str
    tag: str
    target: str
    schema: str = SCHEMA

    def to_dict(self) -> dict:
        return asdict(self)


def build_balanced_field_schedule(
    *,
    group_ids: Sequence[str],
    summaries: Mapping[tuple[str, str], UConceptSummary],
    optimizer_steps: int,
    seed: int,
) -> tuple[ULanguageExample, ...]:
    """Balance fields and their target values without inventing U samples.

    Fields rotate every step.  Within each field every observed canonical
    value rotates before reuse.  Examples inside a value bucket are shuffled
    deterministically and reused only after the complete bucket is consumed.
    """

    groups = tuple(sorted(str(value) for value in group_ids))
    if not groups or optimizer_steps < len(TAG_ORDER):
        raise ValueError("balanced schedule requires groups and every U field")
    buckets: dict[str, dict[str, list[tuple[str, str]]]] = {
        tag: defaultdict(list) for tag in TAG_ORDER
    }
    for group_id in groups:
        for variant in U_VARIANTS:
            summary = summaries.get((group_id, variant))
            if summary is None:
                raise ValueError("balanced schedule is missing a U summary")
            for tag, target in summary.fields().items():
                if target not in FIELD_VOCABULARIES[tag]:
                    raise ValueError("U target lies outside the canonical vocabulary")
                buckets[tag][target].append((group_id, variant))

    rng = random.Random(int(seed))
    ordered_values: dict[str, tuple[str, ...]] = {}
    cursors: dict[tuple[str, str], int] = {}
    for tag in TAG_ORDER:
        available = tuple(
            value
            for value in FIELD_VOCABULARIES[tag]
            if buckets[tag].get(value)
        )
        if set(available) != set(FIELD_VOCABULARIES[tag]):
            raise ValueError("training split does not cover every canonical value")
        ordered_values[tag] = available
        for value in available:
            rng.shuffle(buckets[tag][value])
            cursors[(tag, value)] = 0

    field_presentations = {tag: 0 for tag in TAG_ORDER}
    schedule = []
    for index in range(int(optimizer_steps)):
        tag = TAG_ORDER[index % len(TAG_ORDER)]
        values = ordered_values[tag]
        field_index = field_presentations[tag]
        target = values[field_index % len(values)]
        candidates = buckets[tag][target]
        cursor_key = (tag, target)
        cursor = cursors[cursor_key]
        if cursor and cursor % len(candidates) == 0:
            rng.shuffle(candidates)
        group_id, variant = candidates[cursor % len(candidates)]
        cursors[cursor_key] = cursor + 1
        field_presentations[tag] += 1
        schedule.append(
            ULanguageExample(
                group_id=group_id,
                variant=variant,
                tag=tag,
                target=target,
            )
        )
    return tuple(schedule)


def field_qa_and_candidates(
    summary: UConceptSummary, tag: str
) -> tuple[dict, tuple[str, ...], int]:
    """Return a literal prompt, all legal answers, and the target index."""

    if tag not in TAG_ORDER:
        raise ValueError("unknown U field")
    row = build_explicit_u_qa_row(summary, tag)
    answers = candidate_answers(tag)
    target = summary.fields()[tag]
    target_index = FIELD_VOCABULARIES[tag].index(target)
    if row["conversation"][1]["value"] != answers[target_index]:
        raise RuntimeError("prompt target and canonical candidate order differ")
    return row, answers, target_index


def all_candidate_cross_entropy(
    candidate_nlls: torch.Tensor, target_index: int
) -> torch.Tensor:
    """Cross entropy over every legal answer, using negative NLL as a score."""

    if candidate_nlls.ndim != 1 or candidate_nlls.numel() < 2:
        raise ValueError("candidate NLLs must be a one-dimensional finite vector")
    if not bool(torch.isfinite(candidate_nlls).all()):
        raise ValueError("candidate NLLs must be finite")
    if target_index < 0 or target_index >= candidate_nlls.numel():
        raise ValueError("target candidate index is out of range")
    target = torch.tensor(
        [int(target_index)], dtype=torch.long, device=candidate_nlls.device
    )
    return F.cross_entropy((-candidate_nlls)[None], target)


def exact_nll_gradient_coefficients(
    candidate_nlls: torch.Tensor, target_index: int
) -> torch.Tensor:
    """Derivative of all-candidate CE with respect to each candidate NLL.

    The trainer uses these detached coefficients to replay one candidate at a
    time, avoiding retention of all 7B-model activation graphs at once while
    preserving the exact local gradient of the multiclass objective.
    """

    _ = all_candidate_cross_entropy(candidate_nlls, target_index)
    coefficients = -torch.softmax(-candidate_nlls, dim=0)
    coefficients = coefficients.clone()
    coefficients[int(target_index)] += 1.0
    return coefficients


def target_margin(candidate_nlls: Sequence[float], target_index: int) -> float:
    values = [float(value) for value in candidate_nlls]
    if len(values) < 2 or any(not torch.isfinite(torch.tensor(value)) for value in values):
        raise ValueError("target margin requires finite candidates")
    if target_index < 0 or target_index >= len(values):
        raise ValueError("target candidate index is out of range")
    wrong = [value for index, value in enumerate(values) if index != target_index]
    return float(min(wrong) - values[target_index])


__all__ = [
    "SCHEMA",
    "ULanguageExample",
    "all_candidate_cross_entropy",
    "build_balanced_field_schedule",
    "exact_nll_gradient_coefficients",
    "field_qa_and_candidates",
    "target_margin",
]

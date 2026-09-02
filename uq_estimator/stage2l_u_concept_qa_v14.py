"""Pure observation-uncertainty QA contracts for Stage2-L1.

This module intentionally knows nothing about routes, actors, task relevance,
driving risk, trajectories, or control.  It derives auditable language targets
only from frozen Stage-1 uncertainty components and constructs
distribution-preserving counterfactuals that force a language consumer to use
U rather than scene or prompt-template priors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Mapping

import torch


SCHEMA = "orion.stage2l-u-concept-qa/v1"
CAMERA_NAMES = (
    "front",
    "front_left",
    "front_right",
    "rear",
    "rear_left",
    "rear_right",
)
COMPONENT_NAMES = (
    "persistent_direction",
    "persistent_magnitude",
    "transient_inconsistency",
)
U_VARIANTS = (
    "zero_u",
    "observed_u",
    "view_shifted_u",
    "spatial_shifted_u",
    "component_shifted_u",
    "temporal_reversed_u",
)
TAG_ORDER = (
    "U_PRESENT",
    "U_VIEW",
    "U_REGION",
    "U_LEVEL",
    "U_TREND",
    "U_COMPONENT",
)


@dataclass(frozen=True)
class UConceptSummary:
    present: str
    view: str
    region: str
    level: str
    trend: str
    component: str
    peak: float
    first_peak: float
    latest_peak: float
    schema: str = SCHEMA

    def fields(self) -> Mapping[str, str]:
        return {
            "U_PRESENT": self.present,
            "U_VIEW": self.view,
            "U_REGION": self.region,
            "U_LEVEL": self.level,
            "U_TREND": self.trend,
            "U_COMPONENT": self.component,
        }

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_components(components: torch.Tensor) -> None:
    if (
        components.ndim != 6
        or components.shape[0] != 1
        or components.shape[2:] != (6, 40, 40, 3)
        or components.shape[1] < 2
        or not components.is_floating_point()
    ):
        raise ValueError("U components must have shape [1,T,6,40,40,3]")
    if not bool(torch.isfinite(components).all()):
        raise ValueError("U components must be finite")
    if bool((components < 0).any()) or bool((components > 1).any()):
        raise ValueError("normalized U components must lie in [0,1]")


def _third(index: int, size: int, names: tuple[str, str, str]) -> str:
    bucket = min(2, (3 * int(index)) // int(size))
    return names[bucket]


def summarize_u_components(components: torch.Tensor) -> UConceptSummary:
    """Convert one frozen U history into six task-free semantic fields."""

    _validate_components(components)
    value = components.detach().float()[0]
    scalar = value.mean(dim=-1)
    first_peak = float(scalar[0].max())
    latest = scalar[-1]
    latest_peak = float(latest.max())
    peak = float(scalar.max())
    if peak <= 1e-6:
        return UConceptSummary(
            present="no",
            view="none",
            region="none",
            level="none",
            trend="stable",
            component="none",
            peak=peak,
            first_peak=first_peak,
            latest_peak=latest_peak,
        )

    flat_index = int(latest.reshape(-1).argmax())
    height, width = latest.shape[-2:]
    view_index = flat_index // (height * width)
    cell_index = flat_index % (height * width)
    row = cell_index // width
    column = cell_index % width
    vertical = _third(row, height, ("upper", "middle", "lower"))
    horizontal = _third(column, width, ("left", "center", "right"))
    level = "low" if peak < 0.35 else "medium" if peak < 0.70 else "high"
    difference = latest_peak - first_peak
    trend = "rising" if difference > 0.05 else "falling" if difference < -0.05 else "stable"

    component_values = value[-1, view_index, row, column]
    ordered = torch.argsort(component_values, descending=True)
    top = float(component_values[ordered[0]])
    second = float(component_values[ordered[1]])
    if top - second <= 0.05:
        component = "mixed"
    else:
        component = COMPONENT_NAMES[int(ordered[0])]
    return UConceptSummary(
        present="yes",
        view=CAMERA_NAMES[view_index],
        region="%s_%s" % (vertical, horizontal),
        level=level,
        trend=trend,
        component=component,
        peak=peak,
        first_peak=first_peak,
        latest_peak=latest_peak,
    )


def render_u_answer(summary: UConceptSummary, tag: str | None = None) -> str:
    fields = summary.fields()
    if tag is not None:
        if tag not in fields:
            raise ValueError("unknown U concept tag")
        return "<%s> %s" % (tag, fields[tag])
    return "\n".join("<%s> %s" % (name, fields[name]) for name in TAG_ORDER)


def build_u_qa_row(summary: UConceptSummary, tag: str | None = None) -> dict:
    if tag is None:
        request = (
            "Read only the supplied observation-uncertainty tokens. Do not infer "
            "task relevance, driving risk, or an action. Report exactly six "
            "lines using <U_PRESENT>, <U_VIEW>, <U_REGION>, <U_LEVEL>, "
            "<U_TREND>, and <U_COMPONENT>."
        )
        family = "u_full_state"
    else:
        if tag not in TAG_ORDER:
            raise ValueError("unknown U concept tag")
        request = (
            "Read only the supplied observation-uncertainty tokens. Do not infer "
            "driving relevance or an action. Return exactly one line for <%s>." % tag
        )
        family = "u_field_%s" % tag.lower()
    return {
        "schema": SCHEMA,
        "question_family": family,
        "conversation": (
            {"from": "human", "value": request},
            {"from": "gpt", "value": render_u_answer(summary, tag)},
        ),
    }


def _token_aligned_spatial_shift(
    observed: torch.Tensor, group_id: str
) -> torch.Tensor:
    original = summarize_u_components(observed)
    digest = hashlib.sha256(str(group_id).encode("utf-8")).digest()
    candidates = (
        (4 * (1 + digest[0] % 8), 4 * (1 + digest[1] % 8)),
        (12, 20),
        (20, 12),
        (24, 24),
        (32, 16),
    )
    fallback = None
    for shift in candidates:
        candidate = torch.roll(observed, shifts=shift, dims=(3, 4))
        fallback = candidate
        summary = summarize_u_components(candidate)
        if (summary.view, summary.region) != (original.view, original.region):
            return candidate
    assert fallback is not None
    return fallback


def build_distribution_preserving_u_variants(
    observed: torch.Tensor, group_id: str
) -> dict[str, torch.Tensor]:
    """Build task-free U counterfactuals without inventing new amplitudes.

    Except for the definition-level all-zero anchor, every transformation is a
    permutation of the exact observed Stage-1 values.  Therefore the value
    multiset, range, and marginal amplitude distribution are preserved.
    """

    _validate_components(observed)
    base = observed.detach().float().clone()
    variants = {
        "zero_u": torch.zeros_like(base),
        "observed_u": base,
        "view_shifted_u": torch.roll(base, shifts=1, dims=2),
        "spatial_shifted_u": _token_aligned_spatial_shift(base, group_id),
        "component_shifted_u": torch.roll(base, shifts=1, dims=5),
        "temporal_reversed_u": torch.flip(base, dims=(1,)),
    }
    if tuple(variants) != U_VARIANTS:
        raise RuntimeError("U variant order differs from the contract")
    reference = torch.sort(base.flatten()).values
    for name in U_VARIANTS[2:]:
        if not torch.equal(torch.sort(variants[name].flatten()).values, reference):
            raise RuntimeError("distribution-preserving U transform changed values")
    return variants


def audit_u_variant_group(
    variants: Mapping[str, torch.Tensor],
) -> dict:
    if tuple(variants) != U_VARIANTS:
        raise ValueError("U variant group is incomplete or unordered")
    summaries = {
        name: summarize_u_components(value) for name, value in variants.items()
    }
    answers = {name: render_u_answer(value) for name, value in summaries.items()}
    observed = variants["observed_u"]
    reference = torch.sort(observed.flatten()).values
    checks = {
        "zero_is_exact": int(torch.count_nonzero(variants["zero_u"])) == 0,
        "observed_is_nonzero": int(torch.count_nonzero(observed)) > 0,
        "permutations_preserve_value_multiset": all(
            torch.equal(torch.sort(variants[name].flatten()).values, reference)
            for name in U_VARIANTS[2:]
        ),
        "zero_answer_reports_absence": summaries["zero_u"].present == "no",
        "observed_answer_reports_presence": summaries["observed_u"].present == "yes",
        "view_shift_changes_view_answer": (
            summaries["view_shifted_u"].view != summaries["observed_u"].view
        ),
        "spatial_shift_changes_location_answer": (
            (
                summaries["spatial_shifted_u"].view,
                summaries["spatial_shifted_u"].region,
            )
            != (summaries["observed_u"].view, summaries["observed_u"].region)
        ),
        "at_least_four_distinct_full_answers": len(set(answers.values())) >= 4,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "distinct_answer_count": len(set(answers.values())),
        "summaries": {name: value.to_dict() for name, value in summaries.items()},
        "answers": answers,
    }


__all__ = [
    "CAMERA_NAMES",
    "COMPONENT_NAMES",
    "SCHEMA",
    "TAG_ORDER",
    "U_VARIANTS",
    "UConceptSummary",
    "audit_u_variant_group",
    "build_distribution_preserving_u_variants",
    "build_u_qa_row",
    "render_u_answer",
    "summarize_u_components",
]

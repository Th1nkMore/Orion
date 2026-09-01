"""Identifiability audits for the factorized Stage2-L U/R interface.

The active interface keeps contextual task relevance ``R`` independent of
observation uncertainty ``U`` and combines them only as ``K = U * sigmoid(R)``.
These helpers test the matched counterfactual evidence needed to show that a
task-risk or language result changes because U changed, rather than because
the visual scene, route, ego state, or R target changed.

The module is deliberately model-agnostic.  It does not create labels, train a
planner, or claim that a synthetic U map is uncertainty ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from uq_estimator.uq_relevance_tokenizer import fixed_task_risk


SCHEMA = "orion.stage2l_identifiability.v1"
REQUIRED_RISK_VARIANTS = (
    "zero_uq",
    "on_path_uq",
    "off_path_uq",
    "view_shuffled_uq",
)


@dataclass(frozen=True)
class RelevanceInvarianceAudit:
    reference_variant: str
    maximum_absolute_drift: Mapping[str, float]
    invariant_by_variant: Mapping[str, bool]
    all_invariant: bool
    schema: str = SCHEMA


@dataclass(frozen=True)
class MatchedTaskRiskAudit:
    risk_peak_by_variant: Mapping[str, torch.Tensor]
    uncertainty_mass_by_variant: Mapping[str, torch.Tensor]
    on_over_off_margin: torch.Tensor
    on_over_shuffled_margin: torch.Tensor
    per_sample_gates: Mapping[str, torch.Tensor]
    aggregate_gates: Mapping[str, bool]
    on_over_off_fraction: float
    on_over_shuffled_fraction: float
    schema: str = SCHEMA


@dataclass(frozen=True)
class AnswerPreferenceAudit:
    preference_by_variant: Mapping[str, torch.Tensor]
    preference_fraction_by_variant: Mapping[str, float]
    gates: Mapping[str, bool]
    all_passed: bool
    schema: str = SCHEMA


def _validate_variant_tensor_mapping(
    values: Mapping[str, torch.Tensor],
    *,
    required_variants: Sequence[str],
    name: str,
) -> Tuple[str, ...]:
    required = tuple(str(value) for value in required_variants)
    if not required or len(set(required)) != len(required):
        raise ValueError("required variants must be non-empty and unique")
    missing = sorted(set(required) - set(values))
    if missing:
        raise ValueError("%s is missing variants: %s" % (name, missing))
    first = values[required[0]]
    if not isinstance(first, torch.Tensor):
        raise TypeError("%s values must be tensors" % name)
    for variant in required:
        tensor = values[variant]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("%s values must be tensors" % name)
        if tensor.shape != first.shape:
            raise ValueError("%s variant shapes differ" % name)
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError("%s values must be finite floating tensors" % name)
    return required


def audit_relevance_invariance(
    relevance_logits_by_variant: Mapping[str, torch.Tensor],
    *,
    required_variants: Sequence[str] = REQUIRED_RISK_VARIANTS,
    reference_variant: str = "zero_uq",
    absolute_tolerance: float = 0.0,
) -> RelevanceInvarianceAudit:
    """Verify that matched U variants reuse the same contextual R map.

    Exact equality is the default because the factorized implementation should
    run the first visual/route/ego pass once and reuse its R logits.  A positive
    tolerance is available only for audits of separately serialized replays.
    """

    if absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be non-negative")
    required = _validate_variant_tensor_mapping(
        relevance_logits_by_variant,
        required_variants=required_variants,
        name="relevance logits",
    )
    if reference_variant not in required:
        raise ValueError("reference variant is not required by the audit")
    reference = relevance_logits_by_variant[reference_variant]
    maximum_drift: Dict[str, float] = {}
    invariant: Dict[str, bool] = {}
    for variant in required:
        drift = float(
            (relevance_logits_by_variant[variant] - reference)
            .abs()
            .amax()
            .item()
        )
        maximum_drift[variant] = drift
        invariant[variant] = drift <= float(absolute_tolerance)
    return RelevanceInvarianceAudit(
        reference_variant=str(reference_variant),
        maximum_absolute_drift=maximum_drift,
        invariant_by_variant=invariant,
        all_invariant=all(invariant.values()),
    )


def audit_matched_task_risk(
    relevance_logits: torch.Tensor,
    uncertainty_by_variant: Mapping[str, torch.Tensor],
    *,
    required_on_over_off_margin: float = 0.0,
    required_on_over_shuffled_margin: float = 0.0,
    maximum_off_path_risk: float = 0.25,
    minimum_fraction: float = 0.8,
    minimum_shuffled_fraction: Optional[float] = None,
    matched_mass_rtol: float = 1e-5,
    matched_mass_atol: float = 1e-6,
) -> MatchedTaskRiskAudit:
    """Audit K under a fixed R map and matched counterfactual U variants.

    Only on-path and off-path U are required to have equal total mass and peak
    magnitude.  The view-shuffled variant is derived from the observed Stage-1
    map in the current data factory and therefore is not falsely treated as a
    magnitude-matched copy of the controlled on/off pair.
    """

    thresholds = (
        required_on_over_off_margin,
        required_on_over_shuffled_margin,
        maximum_off_path_risk,
        minimum_fraction,
        matched_mass_rtol,
        matched_mass_atol,
    )
    if any(float(value) < 0.0 for value in thresholds):
        raise ValueError("risk audit thresholds must be non-negative")
    if minimum_fraction > 1.0:
        raise ValueError("minimum_fraction must lie in [0,1]")
    if minimum_shuffled_fraction is not None and not (
        0.0 <= minimum_shuffled_fraction <= 1.0
    ):
        raise ValueError("minimum_shuffled_fraction must lie in [0,1]")
    required = _validate_variant_tensor_mapping(
        uncertainty_by_variant,
        required_variants=REQUIRED_RISK_VARIANTS,
        name="uncertainty",
    )
    first = uncertainty_by_variant[required[0]]
    if relevance_logits.shape != first.shape or relevance_logits.ndim != 4:
        raise ValueError("R and matched U must share shape [B,V,H,W]")
    if not relevance_logits.is_floating_point() or not bool(
        torch.isfinite(relevance_logits).all()
    ):
        raise ValueError("relevance logits must be finite floating tensors")
    for variant in required:
        value = uncertainty_by_variant[variant]
        if bool((value < 0.0).any()) or bool((value > 1.0).any()):
            raise ValueError("uncertainty must lie in [0,1]")

    zero = uncertainty_by_variant["zero_uq"]
    on = uncertainty_by_variant["on_path_uq"]
    off = uncertainty_by_variant["off_path_uq"]
    risk = {
        variant: fixed_task_risk(uncertainty_by_variant[variant], relevance_logits)
        for variant in required
    }
    risk_peak = {
        variant: value.flatten(1).amax(dim=1) for variant, value in risk.items()
    }
    mass = {
        variant: uncertainty_by_variant[variant].flatten(1).sum(dim=1)
        for variant in required
    }
    peaks = {
        variant: uncertainty_by_variant[variant].flatten(1).amax(dim=1)
        for variant in required
    }
    support_counts = {
        variant: uncertainty_by_variant[variant].flatten(1).gt(0.0).sum(dim=1)
        for variant in required
    }
    zero_exact = zero.flatten(1).abs().amax(dim=1).eq(0.0)
    mass_matched = torch.isclose(
        mass["on_path_uq"],
        mass["off_path_uq"],
        rtol=float(matched_mass_rtol),
        atol=float(matched_mass_atol),
    )
    peak_matched = torch.isclose(
        peaks["on_path_uq"],
        peaks["off_path_uq"],
        rtol=float(matched_mass_rtol),
        atol=float(matched_mass_atol),
    )
    support_count_matched = support_counts["on_path_uq"].eq(
        support_counts["off_path_uq"]
    )
    spatially_distinct = on.flatten(1).ne(off.flatten(1)).any(dim=1)
    on_over_off = risk_peak["on_path_uq"] - risk_peak["off_path_uq"]
    on_over_shuffled = risk_peak["on_path_uq"] - risk_peak["view_shuffled_uq"]
    on_off_pass = on_over_off >= float(required_on_over_off_margin)
    on_shuffle_pass = on_over_shuffled >= float(required_on_over_shuffled_margin)
    off_low = risk_peak["off_path_uq"] <= float(maximum_off_path_risk)
    zero_risk_exact = risk_peak["zero_uq"].eq(0.0)
    per_sample = {
        "zero_u_exact": zero_exact,
        "zero_task_risk_exact": zero_risk_exact,
        "on_off_u_mass_matched": mass_matched,
        "on_off_u_peak_matched": peak_matched,
        "on_off_u_support_count_matched": support_count_matched,
        "on_off_support_spatially_distinct": spatially_distinct,
        "on_over_off_margin": on_off_pass,
        "on_over_view_shuffled_margin": on_shuffle_pass,
        "off_path_risk_below_ceiling": off_low,
    }

    def fraction(value: torch.Tensor) -> float:
        return float(value.float().mean().item())

    aggregate = {
        "zero_u_exact": bool(zero_exact.all()),
        "zero_task_risk_exact": bool(zero_risk_exact.all()),
        "on_off_u_mass_matched": bool(mass_matched.all()),
        "on_off_u_peak_matched": bool(peak_matched.all()),
        "on_off_u_support_count_matched": bool(support_count_matched.all()),
        "on_off_support_spatially_distinct": bool(spatially_distinct.all()),
        "on_over_off_fraction": fraction(on_off_pass) >= float(minimum_fraction),
        "off_path_low_risk_fraction": fraction(off_low) >= float(minimum_fraction),
    }
    if minimum_shuffled_fraction is not None:
        # Current view-shuffled U is derived from the observed adapter map, not
        # from the controlled on/off pair.  It is therefore a release gate only
        # when a protocol explicitly freezes that comparison and threshold.
        aggregate["on_over_view_shuffled_fraction"] = fraction(
            on_shuffle_pass
        ) >= float(minimum_shuffled_fraction)
    return MatchedTaskRiskAudit(
        risk_peak_by_variant=risk_peak,
        uncertainty_mass_by_variant=mass,
        on_over_off_margin=on_over_off,
        on_over_shuffled_margin=on_over_shuffled,
        per_sample_gates=per_sample,
        aggregate_gates=aggregate,
        on_over_off_fraction=fraction(on_off_pass),
        on_over_shuffled_fraction=fraction(on_shuffle_pass),
    )


def audit_answer_preferences(
    target_nll_by_variant: Mapping[str, torch.Tensor],
    counterfactual_nll_by_variant: Mapping[str, torch.Tensor],
    *,
    required_variants: Sequence[str] = REQUIRED_RISK_VARIANTS,
    minimum_fraction: float = 0.8,
) -> AnswerPreferenceAudit:
    """Check whether each matched input prefers its own frozen target answer."""

    if not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError("minimum_fraction must lie in [0,1]")
    required = _validate_variant_tensor_mapping(
        target_nll_by_variant,
        required_variants=required_variants,
        name="target NLL",
    )
    _validate_variant_tensor_mapping(
        counterfactual_nll_by_variant,
        required_variants=required,
        name="counterfactual NLL",
    )
    preferences: Dict[str, torch.Tensor] = {}
    fractions: Dict[str, float] = {}
    gates: Dict[str, bool] = {}
    for variant in required:
        target = target_nll_by_variant[variant]
        alternative = counterfactual_nll_by_variant[variant]
        if target.shape != alternative.shape or target.ndim != 1:
            raise ValueError("answer NLL variants must have shape [B]")
        preferred = target < alternative
        fraction = float(preferred.float().mean().item())
        preferences[variant] = preferred
        fractions[variant] = fraction
        gates[variant] = fraction >= float(minimum_fraction)
    return AnswerPreferenceAudit(
        preference_by_variant=preferences,
        preference_fraction_by_variant=fractions,
        gates=gates,
        all_passed=all(gates.values()),
    )


__all__ = [
    "AnswerPreferenceAudit",
    "MatchedTaskRiskAudit",
    "REQUIRED_RISK_VARIANTS",
    "RelevanceInvarianceAudit",
    "SCHEMA",
    "audit_answer_preferences",
    "audit_matched_task_risk",
    "audit_relevance_invariance",
]

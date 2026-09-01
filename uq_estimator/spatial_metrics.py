"""Dependency-free calibration and selective-risk metrics for spatial UQ."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _binary_inputs(
    score: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    score, target = torch.broadcast_tensors(score.float(), target.float())
    score = score.reshape(-1)
    target = target.reshape(-1)
    if score.numel() == 0:
        raise ValueError("metrics require at least one element")
    if not torch.isfinite(score).all() or not torch.isfinite(target).all():
        raise ValueError("metric inputs must be finite")
    if torch.any((target != 0) & (target != 1)):
        raise ValueError("binary targets must contain only zero and one")
    return score, target


@dataclass(frozen=True)
class BinarySpatialMetrics:
    """Binary failure-detection metrics flattened over valid spatial cells."""

    average_precision: float
    auroc: float
    fpr_at_95_tpr: float
    brier: float
    ece: float
    aurc: float
    positives: int
    negatives: int


def average_precision(score: torch.Tensor, target: torch.Tensor) -> float:
    """Average precision for a binary spatial error mask."""
    score, target = _binary_inputs(score, target)
    positives = int(target.sum().item())
    if positives == 0:
        return float("nan")
    order = torch.argsort(score, descending=True, stable=True)
    ordered = target[order]
    true_positive = ordered.cumsum(0)
    precision = true_positive / torch.arange(
        1, ordered.numel() + 1, device=ordered.device, dtype=ordered.dtype
    )
    return float((precision * ordered).sum().item() / positives)


def auroc(score: torch.Tensor, target: torch.Tensor) -> float:
    """ROC area computed from the descending threshold sweep."""
    score, target = _binary_inputs(score, target)
    positives = target.sum()
    negatives = (1.0 - target).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(score, descending=True, stable=True)
    ordered = target[order]
    tpr = torch.cat((ordered.new_zeros(1), ordered.cumsum(0) / positives))
    fpr = torch.cat(
        (ordered.new_zeros(1), (1.0 - ordered).cumsum(0) / negatives)
    )
    return float(torch.trapezoid(tpr, fpr).item())


def fpr_at_tpr(
    score: torch.Tensor,
    target: torch.Tensor,
    requested_tpr: float = 0.95,
) -> float:
    """Minimum false-positive rate whose threshold reaches requested TPR."""
    if not 0.0 < requested_tpr <= 1.0:
        raise ValueError("requested_tpr must lie in (0, 1]")
    score, target = _binary_inputs(score, target)
    positives = target.sum()
    negatives = (1.0 - target).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(score, descending=True, stable=True)
    ordered = target[order]
    tpr = ordered.cumsum(0) / positives
    fpr = (1.0 - ordered).cumsum(0) / negatives
    eligible = fpr[tpr >= requested_tpr]
    return float(eligible.min().item()) if eligible.numel() else float("nan")


def expected_calibration_error(
    probability: torch.Tensor,
    target: torch.Tensor,
    bins: int = 15,
) -> float:
    """Equal-width ECE for calibrated failure probabilities."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    probability, target = _binary_inputs(probability, target)
    if torch.any((probability < 0) | (probability > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    boundaries = torch.linspace(
        0.0, 1.0, bins + 1, device=probability.device
    )
    result = probability.new_zeros(())
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        if index == bins - 1:
            member = (probability >= lower) & (probability <= upper)
        else:
            member = (probability >= lower) & (probability < upper)
        if member.any():
            weight = member.float().mean()
            result = result + weight * (
                probability[member].mean() - target[member].mean()
            ).abs()
    return float(result.item())


def area_under_risk_coverage(
    uncertainty: torch.Tensor,
    error: torch.Tensor,
) -> float:
    """AURC when retaining cells from lowest to highest uncertainty.

    Lower is better: a useful uncertainty score removes error-prone cells late,
    keeping the cumulative error of accepted cells low at each coverage level.
    """
    uncertainty, error = torch.broadcast_tensors(
        uncertainty.float(), error.float()
    )
    uncertainty = uncertainty.reshape(-1)
    error = error.reshape(-1)
    if uncertainty.numel() == 0:
        raise ValueError("AURC requires at least one element")
    if not torch.isfinite(uncertainty).all() or not torch.isfinite(error).all():
        raise ValueError("AURC inputs must be finite")
    if torch.any(error < 0):
        raise ValueError("error must be non-negative")
    order = torch.argsort(uncertainty, descending=False, stable=True)
    accepted_error = error[order].cumsum(0)
    accepted_count = torch.arange(
        1, error.numel() + 1, device=error.device, dtype=error.dtype
    )
    selective_risk = accepted_error / accepted_count
    return float(selective_risk.mean().item())


def binary_spatial_metrics(
    failure_probability: torch.Tensor,
    failure_target: torch.Tensor,
    bins: int = 15,
) -> BinarySpatialMetrics:
    """Compute the Stage-1 binary spatial gate metrics in one pass."""
    probability, target = _binary_inputs(failure_probability, failure_target)
    if torch.any((probability < 0) | (probability > 1)):
        raise ValueError("failure_probability must lie in [0, 1]")
    positives = int(target.sum().item())
    negatives = int(target.numel() - positives)
    return BinarySpatialMetrics(
        average_precision=average_precision(probability, target),
        auroc=auroc(probability, target),
        fpr_at_95_tpr=fpr_at_tpr(probability, target, requested_tpr=0.95),
        brier=float((probability - target).square().mean().item()),
        ece=expected_calibration_error(probability, target, bins=bins),
        aurc=area_under_risk_coverage(probability, target),
        positives=positives,
        negatives=negatives,
    )


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return one-based average ranks with exact tie handling."""
    values = values.reshape(-1)
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values, dtype=torch.float64)
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and bool(
            sorted_values[end] == sorted_values[start]
        ):
            end += 1
        average = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average
        start = end
    return ranks


def spearman_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    """Spearman rank correlation with average tie ranks and no SciPy dependency."""
    left, right = torch.broadcast_tensors(left.float(), right.float())
    left = left.reshape(-1)
    right = right.reshape(-1)
    if left.numel() < 2:
        return float("nan")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("Spearman inputs must be finite")
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = torch.sqrt(
        left_centered.square().sum() * right_centered.square().sum()
    )
    if denominator == 0:
        return float("nan")
    return float(
        (left_centered * right_centered).sum().div(denominator).item()
    )


@dataclass(frozen=True)
class TemporalEventMetrics:
    precision: float
    recall: float
    f1: float
    onset_latency_seconds: float
    recovery_latency_seconds: float
    false_trigger_seconds: float


def temporal_event_metrics(
    score: torch.Tensor,
    event_active: torch.Tensor,
    threshold: float,
    step_seconds: float,
) -> TemporalEventMetrics:
    """Evaluate one contiguous corruption event's onset and recovery response."""
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    score, active_float = _binary_inputs(score, event_active)
    active = active_float.bool()
    active_indices = torch.nonzero(active, as_tuple=False).flatten()
    if active_indices.numel() == 0:
        raise ValueError("event_active must contain an event window")
    start = int(active_indices[0])
    end = int(active_indices[-1])
    expected = torch.arange(start, end + 1, device=active_indices.device)
    if not torch.equal(active_indices, expected):
        raise ValueError("event_active must contain one contiguous window")

    predicted = score >= threshold
    true_positive = int((predicted & active).sum())
    false_positive = int((predicted & ~active).sum())
    false_negative = int((~predicted & active).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    onset_candidates = torch.nonzero(predicted[start : end + 1], as_tuple=False)
    onset_latency = float("inf")
    if onset_candidates.numel():
        onset_latency = int(onset_candidates[0]) * step_seconds

    recovery_latency = float("inf")
    if end + 1 < predicted.numel():
        recovery_candidates = torch.nonzero(
            ~predicted[end + 1 :], as_tuple=False
        )
        if recovery_candidates.numel():
            recovery_latency = int(recovery_candidates[0]) * step_seconds

    false_trigger_seconds = false_positive * step_seconds
    return TemporalEventMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        onset_latency_seconds=float(onset_latency),
        recovery_latency_seconds=float(recovery_latency),
        false_trigger_seconds=float(false_trigger_seconds),
    )


__all__ = [
    "BinarySpatialMetrics",
    "TemporalEventMetrics",
    "average_precision",
    "area_under_risk_coverage",
    "auroc",
    "binary_spatial_metrics",
    "expected_calibration_error",
    "fpr_at_tpr",
    "spearman_correlation",
    "temporal_event_metrics",
]

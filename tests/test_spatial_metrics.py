import math

import pytest
import torch

from uq_estimator.spatial_metrics import (
    area_under_risk_coverage,
    binary_spatial_metrics,
    spearman_correlation,
    temporal_event_metrics,
)


def test_perfect_binary_failure_detector_metrics():
    probability = torch.tensor([0.01, 0.02, 0.98, 0.99])
    target = torch.tensor([0, 0, 1, 1])
    metrics = binary_spatial_metrics(probability, target, bins=2)
    assert metrics.average_precision == pytest.approx(1.0)
    assert metrics.auroc == pytest.approx(1.0)
    assert metrics.fpr_at_95_tpr == pytest.approx(0.0)
    assert metrics.brier < 0.001
    assert metrics.positives == 2
    assert metrics.negatives == 2


def test_aurc_rewards_uncertainty_that_orders_errors_last():
    error = torch.tensor([0.0, 0.0, 1.0, 1.0])
    useful = area_under_risk_coverage(torch.tensor([0.1, 0.2, 0.8, 0.9]), error)
    reversed_score = area_under_risk_coverage(
        torch.tensor([0.9, 0.8, 0.2, 0.1]), error
    )
    assert useful < reversed_score


def test_spearman_handles_ties_and_constant_input():
    assert spearman_correlation(
        torch.tensor([1.0, 2.0, 2.0, 3.0]),
        torch.tensor([10.0, 20.0, 20.0, 30.0]),
    ) == pytest.approx(1.0)
    assert math.isnan(
        spearman_correlation(torch.ones(3), torch.arange(3).float())
    )


def test_temporal_event_onset_recovery_and_false_trigger():
    score = torch.tensor([0.7, 0.1, 0.2, 0.8, 0.9, 0.8, 0.7, 0.2, 0.1])
    event = torch.tensor([0, 0, 1, 1, 1, 1, 1, 0, 0])
    metrics = temporal_event_metrics(score, event, threshold=0.5, step_seconds=0.1)
    assert metrics.onset_latency_seconds == pytest.approx(0.1)
    assert metrics.recovery_latency_seconds == pytest.approx(0.0)
    assert metrics.false_trigger_seconds == pytest.approx(0.1)
    assert metrics.recall == pytest.approx(4 / 5)


def test_temporal_event_rejects_disjoint_windows():
    with pytest.raises(ValueError, match="contiguous"):
        temporal_event_metrics(
            torch.ones(5), torch.tensor([0, 1, 0, 1, 0]), 0.5, 0.1
        )

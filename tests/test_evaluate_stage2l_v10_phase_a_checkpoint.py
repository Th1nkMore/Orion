import numpy as np
import pytest

from scripts.evaluate_stage2l_v10_phase_a_checkpoint import (
    _average_precision,
    _support_mask,
    spatial_diagnostics,
)


def test_average_precision_is_one_for_perfect_ordering():
    assert _average_precision(
        np.array([0.9, 0.8, 0.2, 0.1]),
        np.array([True, True, False, False]),
    ) == pytest.approx(1.0)


def test_support_mask_is_target_peak_relative_per_sample():
    target = np.array([
        [[1.0, 0.09], [0.11, 0.0]],
        [[0.5, 0.049], [0.051, 0.0]],
    ])
    support = _support_mask(target, 0.1)
    assert support.tolist() == [
        [[True, False], [True, False]],
        [[True, False], [True, False]],
    ]


def test_spatial_diagnostics_exports_threshold_free_and_frozen_metrics():
    target = np.array([[[1.0, 0.2], [0.0, 0.0]]])
    probabilities = np.array([[[0.9, 0.8], [0.2, 0.1]]])
    result = spatial_diagnostics(
        probabilities,
        target,
        support_fraction=0.1,
        thresholds=(0.0, 0.5, 1.0),
    )
    assert result["average_precision"] == pytest.approx(1.0)
    assert result["foreground_cell_count"] == 2
    assert result["frozen_target_relative_threshold"]["recall"] == pytest.approx(1.0)
    assert result["frozen_target_relative_threshold"]["background_fpr"] == pytest.approx(1.0)
    assert [row["threshold"] for row in result["absolute_threshold_sweep"]] == [0.0, 0.5, 1.0]

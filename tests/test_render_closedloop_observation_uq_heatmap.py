import numpy as np
import pytest

from scripts.render_closedloop_observation_uq_heatmap import (
    calibrate_spatial_grid,
    fit_spatial_baseline,
    resolve_time_window,
)


def test_spatial_heatmap_uses_position_specific_causal_baseline():
    baseline = np.zeros((50, 2, 2), dtype=np.float64)
    baseline[:, 0, 0] = 10.0
    median, scale = fit_spatial_baseline(baseline)
    clean = calibrate_spatial_grid(baseline[0], median, scale)
    event_grid = baseline[0].copy()
    event_grid[1, 1] = 1.0
    event = calibrate_spatial_grid(event_grid, median, scale)
    assert clean.max() == pytest.approx(1.0 / (1.0 + np.exp(4.0)))
    assert event[1, 1] > 0.99
    assert event[0, 0] == pytest.approx(clean[0, 0])


def test_spatial_heatmap_rejects_shape_mismatch():
    baseline = np.zeros((40, 2, 2), dtype=np.float64)
    median, scale = fit_spatial_baseline(baseline)
    with pytest.raises(ValueError, match="shapes differ"):
        calibrate_spatial_grid(np.zeros((3, 3)), median, scale)


def test_explicit_event_center_controls_window_without_using_ttc_minimum():
    rows = [
        {
            "step": 1,
            "sim_time_seconds": 1.0,
            "closedloop_safety": {"min_obb_collision_ttc_seconds": 0.1},
        }
    ]
    start, end, basis, center = resolve_time_window(
        rows,
        first_time=0.0,
        last_time=30.0,
        pre_seconds=3.0,
        post_seconds=4.0,
        full_route=False,
        center_time_seconds=20.0,
    )
    assert (start, end, center) == (17.0, 24.0, 20.0)
    assert basis == "explicit_preregistered_or_evaluator_event_time"


def test_full_route_rejects_explicit_center():
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_time_window(
            [],
            first_time=0.0,
            last_time=1.0,
            pre_seconds=1.0,
            post_seconds=1.0,
            full_route=True,
            center_time_seconds=0.5,
        )

import numpy as np
import pytest

from scripts.upgrade_stage2l_v11_consumer_grid_controls import (
    build_consumer_grid_pair,
    normalize_v5_structured_summary,
    pool_40_to_10,
    select_consumer_grid_centers,
)


def _expanded_relevance():
    consumer = np.zeros((6, 10, 10), dtype=np.float32)
    consumer[2, 4, 5] = 1.0
    consumer[2, 3, 5] = 0.8
    consumer[2, 5, 5] = 0.7
    consumer[2, 4, 4] = 0.6
    consumer[2, 4, 6] = 0.5
    return np.repeat(np.repeat(consumer, 4, axis=1), 4, axis=2)


def test_consumer_grid_pair_is_matched_after_actual_pooling():
    pair = build_consumer_grid_pair(
        _expanded_relevance(), time_steps=4, component_count=3, peak=0.9
    )
    on = pair["on_path_uq"]
    off = pair["off_path_uq"]
    assert on["uq"].shape == (4, 6, 40, 40)
    assert on["components"].shape == (4, 6, 40, 40, 3)
    assert np.allclose(on["uq"], on["components"].mean(axis=-1), atol=1e-7)
    assert np.allclose(off["uq"], off["components"].mean(axis=-1), atol=1e-7)
    on_latest = pool_40_to_10(on["uq"][-1])
    off_latest = pool_40_to_10(off["uq"][-1])
    assert not np.array_equal(on_latest, off_latest)
    assert np.isclose(on_latest.max(), off_latest.max())
    assert np.isclose(on_latest.sum(), off_latest.sum())
    assert np.count_nonzero(on_latest) == np.count_nonzero(off_latest)
    assert (
        on["support"]["support_weighted_relevance"]
        > off["support"]["support_weighted_relevance"]
    )
    assert on["support"]["construction"] == ("consumer_grid_then_exact_block_expand")


def test_center_selection_rejects_unidentifiable_uniform_relevance():
    relevance = np.ones((6, 10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="does not distinguish"):
        select_consumer_grid_centers(relevance)


def test_v5_summary_maps_medium_relevance_to_binary_low():
    source = {
        "observation_uncertainty": {
            "level": "high",
            "peak_score": 0.9,
            "peak_view": "CAM_FRONT",
            "peak_region": "middle_center",
            "temporal_trend": "rising",
        },
        "relevance_at_most_uncertain_region": {
            "level": "medium",
            "score": 0.5,
        },
        "task_risk": {
            "level": "medium",
            "peak_score": 0.45,
            "peak_view": "CAM_FRONT",
            "peak_region": "middle_center",
        },
        "planning_implication": {
            "stance": "caution",
            "is_direct_control_command": False,
        },
    }
    summary = normalize_v5_structured_summary(source, relevance_high_threshold=0.66)
    assert summary["relevance_at_most_uncertain_region"]["level"] == "low"

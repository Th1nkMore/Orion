import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "preflight_qwen_visibility_offline_depth.py"
)
SPEC = importlib.util.spec_from_file_location("qwen_offline_depth_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


FEATURES = (
    "token_is_global",
    "token_is_frontier",
    "center_x_normalized",
    "center_y_normalized",
    "extent_x_normalized",
    "extent_y_normalized",
    "visible_free_height_ratio",
    "route_weight_mean",
)


def _snapshot(centers, route_weights):
    frontier = np.zeros((3, len(FEATURES)), dtype=np.float32)
    for index, ((x, y), route_weight) in enumerate(zip(centers, route_weights)):
        frontier[index, 1] = 1.0
        frontier[index, 2] = x / 10.0
        frontier[index, 3] = y / 10.0
        frontier[index, 7] = route_weight
    return MODULE.TokenSnapshot(
        global_tokens=np.zeros((1, len(FEATURES)), dtype=np.float32),
        frontier_tokens=frontier,
        frontier_mask=np.ones(3, dtype=bool),
        feature_names=FEATURES,
        x_bounds_m=(-10.0, 10.0),
        y_bounds_m=(-10.0, 10.0),
    )


def test_select_centered_stride_window_respects_gaps_and_route_center():
    frames = list(range(30))
    assert MODULE.select_centered_stride_window(frames, stride=5, length=4) == [7, 12, 17, 22]
    frames.remove(17)
    assert MODULE.select_centered_stride_window(frames, stride=5, length=4) == [6, 11, 16, 21]


def test_depth_hypothesis_preserves_far_saturation_and_applies_endpoint():
    stored = {"CAM_FRONT": np.asarray([[4, 255]], dtype=np.uint8)}
    value = MODULE.depth_hypothesis(
        stored,
        offset_m=-0.5,
        saturated_value=255,
        saturated_replacement_m=1000.0,
    )["CAM_FRONT"]
    np.testing.assert_allclose(value, [[3.5, 1000.0]])


def test_snapshot_comparison_separates_physical_match_from_slot_order():
    left = _snapshot([(0, 0), (2, 0), (4, 0)], [0.4, 0.1, 0.6])
    right = _snapshot([(4, 0), (0, 0), (2, 0)], [0.6, 0.4, 0.1])
    result = MODULE.compare_snapshots(
        left, right, radius_m=0.5, route_threshold=0.2
    )
    assert result["bidirectional_nearest_within_radius"] == 6
    assert result["nearest_route_label_agree"] == 6
    assert result["same_slot_within_radius"] == 0
    assert result["same_slot_route_label_agree"] == 1
    assert result["nearest_label_source_on_total"] == 4
    assert result["nearest_label_source_on_agree"] == 4
    assert result["nearest_label_source_off_total"] == 2
    assert result["nearest_label_source_off_agree"] == 2
    assert result["same_slot_left_on_right_on"] == 1
    assert result["same_slot_left_on_right_off"] == 1
    assert result["same_slot_left_off_right_on"] == 1
    assert result["same_slot_left_off_right_off"] == 0

    aggregate = MODULE._aggregate_comparison([result])
    assert aggregate[
        "nearest_matched_route_label_agreement_conditioned_on_source_ON"
    ] == 1.0
    assert aggregate[
        "nearest_matched_route_label_agreement_conditioned_on_source_OFF"
    ] == 1.0
    assert aggregate["nearest_matched_route_label_balanced_agreement"] == 1.0
    assert aggregate[
        "same_slot_route_label_agreement_conditioned_on_left_ON"
    ] == 0.5
    assert aggregate[
        "same_slot_route_label_agreement_conditioned_on_left_OFF"
    ] == 0.0
    assert aggregate["same_slot_route_label_balanced_agreement"] == 0.25

"""Dependency-light regression tests for traffic-state/box filtering."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "mmcv"
    / "datasets"
    / "pipelines"
    / "traffic_state_alignment.py"
)
SPEC = importlib.util.spec_from_file_location("traffic_state_alignment_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_filters_state_and_validity_from_the_same_original_object_axis():
    state = np.array([[0, 1], [2, 0], [1, 1]], dtype=np.int64)
    state_valid = np.array([True, False, True], dtype=np.bool_)
    box_valid = np.array([True, False, True], dtype=np.bool_)

    filtered_state, filtered_valid = MODULE.filter_traffic_state_by_box_mask(
        state, state_valid, box_valid
    )

    np.testing.assert_array_equal(filtered_state, np.array([[0, 1], [1, 1]]))
    np.testing.assert_array_equal(filtered_valid, np.array([True, True]))
    assert filtered_state.shape == (2, 2)
    assert filtered_valid.shape == (2,)


@pytest.mark.parametrize(
    "state,state_valid,box_valid,message",
    [
        (np.zeros((3,), dtype=np.int64), np.ones(3, bool), np.ones(3, bool), r"\[N,2\]"),
        (np.zeros((3, 2), dtype=np.int64), np.ones(2, bool), np.ones(3, bool), "traffic_state_mask"),
        (np.zeros((3, 2), dtype=np.int64), np.ones(3, bool), np.ones(2, bool), "gt_bboxes_3d_mask"),
        (np.zeros((3, 2), dtype=np.int64), np.ones(3, np.int64), np.ones(3, bool), "boolean dtype"),
    ],
)
def test_rejects_shape_or_dtype_misalignment(state, state_valid, box_valid, message):
    with pytest.raises(MODULE.TrafficStateAlignmentError, match=message):
        MODULE.filter_traffic_state_by_box_mask(state, state_valid, box_valid)

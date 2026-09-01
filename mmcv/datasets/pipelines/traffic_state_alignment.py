"""Fail-closed traffic-state filtering on the 3D-box object axis."""

from __future__ import annotations

import numpy as np


class TrafficStateAlignmentError(ValueError):
    """Raised when traffic labels cannot be aligned with filtered GT boxes."""


def filter_traffic_state_by_box_mask(
    traffic_state: np.ndarray,
    traffic_state_mask: np.ndarray,
    gt_bboxes_3d_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter `[N,2]` state and `[N]` validity with one original `[N]` mask."""

    state = np.asarray(traffic_state)
    state_valid = np.asarray(traffic_state_mask)
    box_valid = np.asarray(gt_bboxes_3d_mask)
    if state.ndim != 2 or state.shape[1] != 2:
        raise TrafficStateAlignmentError(
            "traffic_state must have shape [N,2] before GT-box filtering"
        )
    count = state.shape[0]
    if state_valid.shape != (count,):
        raise TrafficStateAlignmentError(
            "traffic_state_mask must have shape [N] matching traffic_state"
        )
    if box_valid.shape != (count,):
        raise TrafficStateAlignmentError(
            "gt_bboxes_3d_mask must have shape [N] matching traffic_state"
        )
    if state_valid.dtype != np.bool_ or box_valid.dtype != np.bool_:
        raise TrafficStateAlignmentError(
            "traffic and GT-box masks must use boolean dtype"
        )
    return state[box_valid], state_valid[box_valid]


__all__ = [
    "TrafficStateAlignmentError",
    "filter_traffic_state_by_box_mask",
]

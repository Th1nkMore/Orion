"""Dependency-light actor-ID axis validation for target-export pipelines."""

from __future__ import annotations

import numpy as np


def normalize_actor_ids(actor_ids, expected_count):
    """Return an owned int64 ``[N]`` array or fail on an ambiguous axis."""

    ids = np.asarray(actor_ids)
    if ids.ndim != 1 or ids.shape[0] != expected_count:
        raise ValueError('gt_actor_ids must be an integer [N] axis')
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('gt_actor_ids must use an integer dtype')
    normalized = ids.astype(np.int64, copy=True)
    if np.unique(normalized).shape[0] != normalized.shape[0]:
        raise ValueError('gt_actor_ids must be unique within a frame')
    return normalized


def filter_actor_ids_by_box_mask(actor_ids, box_mask):
    """Filter actor IDs using the same original object-axis mask as boxes."""

    ids = np.asarray(actor_ids)
    mask = np.asarray(box_mask)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError('object filter mask must be boolean [N]')
    normalized = normalize_actor_ids(ids, mask.shape[0])
    return normalized[mask]


__all__ = ['filter_actor_ids_by_box_mask', 'normalize_actor_ids']

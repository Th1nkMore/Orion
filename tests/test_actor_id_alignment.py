import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'mmcv/datasets/pipelines/actor_id_alignment.py'
)
SPEC = importlib.util.spec_from_file_location('actor_id_alignment', MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_actor_ids_follow_the_exact_original_object_mask_without_mutation():
    actor_ids = np.asarray([41, 7, 99, 12], dtype=np.int32)
    before = actor_ids.copy()
    selected = MODULE.filter_actor_ids_by_box_mask(
        actor_ids, np.asarray([True, False, True, False], dtype=np.bool_)
    )
    assert selected.dtype == np.int64
    assert selected.tolist() == [41, 99]
    np.testing.assert_array_equal(actor_ids, before)


@pytest.mark.parametrize(
    'actor_ids,mask,message',
    [
        ([1, 1], np.asarray([True, False], dtype=np.bool_), 'unique'),
        ([1, 2], np.asarray([1, 0], dtype=np.int64), 'boolean'),
        ([1, 2, 3], np.asarray([True, False], dtype=np.bool_), r'\[N\]'),
        (['a', 'b'], np.asarray([True, False], dtype=np.bool_), 'integer'),
    ],
)
def test_actor_id_axis_fails_closed(actor_ids, mask, message):
    with pytest.raises(ValueError, match=message):
        MODULE.filter_actor_ids_by_box_mask(actor_ids, mask)

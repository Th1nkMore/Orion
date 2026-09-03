import pytest

from scripts.render_corruption_hardcase_bakeoff import (
    nearest_progress_indices,
    stale_source_indices,
)


def test_stale_selection_uses_newest_frame_at_or_before_target_time():
    timestamps = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
    assert stale_source_indices(timestamps, 100) == [0, 0, 0, 1, 2, 3]
    assert stale_source_indices(timestamps, 200) == [0, 0, 0, 0, 0, 1]


def test_stale_selection_rejects_nonmonotonic_time():
    with pytest.raises(ValueError, match="strictly increasing"):
        stale_source_indices([0.0, 0.1, 0.1], 100)


def test_nearest_progress_match_is_deterministic():
    assert nearest_progress_indices([0.1, 0.2, 0.4], [0.11, 0.29, 0.39]) == [0, 1, 2]

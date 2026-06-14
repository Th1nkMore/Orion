"""Tests for frame vs clip aggregation in openloop_aggregate."""
import pytest

from uq_estimator.openloop_aggregate import (
    build_clip_records,
    compute_aggregate_stats,
    compute_clip_aggregate_stats,
    run_aggregation,
)


def _noop_uq_corr(records):
    return {}


def _frame(folder, l2_2s, is_adverse, col_3s=0.0, uq=0.5):
    return {
        'folder': folder,
        'frame_idx': 0,
        'weather_id': 7 if is_adverse else 0,
        'weather_name': 'test',
        'is_adverse': is_adverse,
        'fut_valid': True,
        'plan_L2_1s': l2_2s * 0.5,
        'plan_L2_2s': l2_2s,
        'plan_L2_3s': l2_2s * 1.5,
        'plan_obj_col_1s': col_3s,
        'plan_obj_col_2s': col_3s,
        'plan_obj_col_3s': col_3s,
        'plan_obj_box_col_1s': 0.0,
        'plan_obj_box_col_2s': 0.0,
        'plan_obj_box_col_3s': 0.0,
        'uq_score': uq,
    }


def test_clip_macro_differs_from_frame_micro():
    """Long adverse clip should not dominate clip-macro L2 like frame-micro."""
    records = [
        _frame('v1/ClipA_Weather0', 2.0, False),
        _frame('v1/ClipA_Weather0', 2.0, False),
        _frame('v1/ClipB_Weather7', 0.5, True),
        _frame('v1/ClipB_Weather7', 0.5, True),
        _frame('v1/ClipB_Weather7', 0.5, True),
        _frame('v1/ClipB_Weather7', 0.5, True),
    ]
    stats_frame, stats_clip, clip_records, _, _ = run_aggregation(records, _noop_uq_corr)

    assert len(clip_records) == 2
    # frame micro: (2+2+0.5*4)/6 = 1.0
    assert stats_frame['all']['plan_L2_2s_mean'] == pytest.approx(1.0)
    # clip macro: (2.0 + 0.5) / 2 = 1.25
    assert stats_clip['all']['avg_l2_2s'] == pytest.approx(1.25)
    assert stats_clip['all']['n_clips'] == 2
    assert stats_clip['all']['n_frames'] == 6


def test_clip_groups_normal_adverse():
    records = [
        _frame('v1/Normal_Weather0', 1.0, False),
        _frame('v1/Adverse_Weather7', 3.0, True),
    ]
    clip_records = build_clip_records(records)
    stats_clip = compute_clip_aggregate_stats(clip_records)

    assert stats_clip['normal']['n_clips'] == 1
    assert stats_clip['adverse']['n_clips'] == 1
    assert stats_clip['normal']['avg_l2_2s'] == pytest.approx(1.0)
    assert stats_clip['adverse']['avg_l2_2s'] == pytest.approx(3.0)


def test_backward_compat_stats_key():
    records = [_frame('v1/Only_Weather0', 0.68, False)]
    stats = compute_aggregate_stats(records)
    assert 'plan_L2_2s_mean' in stats['all']

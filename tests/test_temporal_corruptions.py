import pytest
import torch

from uq_estimator.temporal_corruptions import (
    StaleFrameBuffer,
    stale_delay_ms_for_severity,
)


def _frame(value):
    result = torch.zeros(1, 6, 3, 2, 2)
    result[:, 0] = value
    result[:, 1:] = value + 100
    return result


@pytest.mark.parametrize(
    ("severity", "delay_ms"), [(1, 100), (2, 200), (3, 400)]
)
def test_frozen_stale_severity_mapping(severity, delay_ms):
    assert stale_delay_ms_for_severity(severity) == delay_ms


def test_stale_frame_uses_simulation_time_and_only_selected_view():
    buffer = StaleFrameBuffer(delay_ms=200)
    for timestamp, value in [(0.0, 0), (0.05, 1), (0.11, 2), (0.21, 3)]:
        result = buffer.apply(
            _frame(value),
            timestamp_seconds=timestamp,
            active=timestamp >= 0.21,
            view_indices=[0],
        )
    assert torch.equal(result.images[:, 0], _frame(0)[:, 0])
    assert torch.equal(result.images[:, 1], _frame(3)[:, 1])
    assert result.metadata["source_timestamp_seconds"] == pytest.approx(0.0)
    assert result.metadata["effective_delay_ms"] == pytest.approx(210.0)
    assert result.metadata["applied"] is True


def test_stale_frame_primes_history_while_schedule_is_inactive():
    buffer = StaleFrameBuffer(delay_ms=100)
    buffer.apply(_frame(0), timestamp_seconds=0.0, active=False, view_indices=[0])
    buffer.apply(_frame(1), timestamp_seconds=0.05, active=False, view_indices=[0])
    result = buffer.apply(
        _frame(2), timestamp_seconds=0.10, active=True, view_indices=[0]
    )
    assert result.metadata["history_warm"] is True
    assert result.metadata["source_timestamp_seconds"] == pytest.approx(0.0)
    assert torch.equal(result.images[:, 0], _frame(0)[:, 0])


def test_stale_frame_discloses_warmup_without_fabricating_delay():
    buffer = StaleFrameBuffer(delay_ms=400)
    result = buffer.apply(
        _frame(0), timestamp_seconds=0.0, active=True, view_indices=[0]
    )
    assert torch.equal(result.images, _frame(0))
    assert result.metadata["schedule_active"] is True
    assert result.metadata["history_warm"] is False
    assert result.metadata["applied"] is False
    assert result.metadata["source_timestamp_seconds"] is None


def test_timestamp_regression_resets_history_but_duplicate_fails():
    buffer = StaleFrameBuffer(delay_ms=100)
    buffer.apply(_frame(0), timestamp_seconds=1.0, active=False, view_indices=[0])
    with pytest.raises(ValueError, match="strictly increasing"):
        buffer.apply(
            _frame(1), timestamp_seconds=1.0, active=False, view_indices=[0]
        )
    result = buffer.apply(
        _frame(2), timestamp_seconds=0.0, active=True, view_indices=[0]
    )
    assert result.metadata["history_warm"] is False
    assert buffer.history_length == 1

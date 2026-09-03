"""Stateful, simulation-time camera corruptions for closed-loop experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch


TEMPORAL_CORRUPTION_SCHEMA_V1 = "orion.temporal_corruption.v1"
STALE_FRAME_DELAYS_MS = (100, 200, 400)


@dataclass(frozen=True)
class TemporalCorruptionResultV1:
    """One model-input tensor plus an auditable temporal intervention record."""

    images: torch.Tensor
    metadata: dict[str, Any]


def stale_delay_ms_for_severity(severity: int) -> int:
    """Map the frozen three-level severity scale to physical delay."""
    if severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2, or 3")
    return STALE_FRAME_DELAYS_MS[severity - 1]


def _validate_images(images: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not torch.is_tensor(images):
        raise TypeError("images must be a torch.Tensor")
    squeeze_batch = images.ndim == 4
    batched = images.unsqueeze(0) if squeeze_batch else images
    if batched.ndim != 5 or batched.shape[0] != 1 or batched.shape[2] != 3:
        raise ValueError("images must have shape [V,3,H,W] or [1,V,3,H,W]")
    if not batched.is_floating_point():
        raise TypeError("stale-frame corruption requires floating-point images")
    return batched, squeeze_batch


def _validate_view_indices(
    view_indices: Sequence[int], n_views: int
) -> tuple[int, ...]:
    indices = tuple(int(index) for index in view_indices)
    if not indices:
        raise ValueError("view_indices must not be empty")
    if len(indices) != len(set(indices)):
        raise ValueError("view_indices must not contain duplicates")
    if any(index < 0 or index >= n_views for index in indices):
        raise ValueError("view_indices contains an out-of-range view")
    return indices


class StaleFrameBuffer:
    """Replace selected views with the newest frame old enough for a delay.

    History is populated on every call, including before and after the active
    corruption window.  Selection uses simulation timestamps rather than loop
    counts, so sensor/runtime jitter does not silently change the requested
    delay.  Only selected views are retained.
    """

    def __init__(self, delay_ms: int):
        delay_ms = int(delay_ms)
        if delay_ms <= 0:
            raise ValueError("delay_ms must be positive")
        self.delay_ms = delay_ms
        self.delay_seconds = delay_ms / 1000.0
        self._history: deque[tuple[float, torch.Tensor]] = deque()
        self._last_timestamp: float | None = None

    def reset(self) -> None:
        self._history.clear()
        self._last_timestamp = None

    @property
    def history_length(self) -> int:
        return len(self._history)

    def apply(
        self,
        images: torch.Tensor,
        *,
        timestamp_seconds: float,
        active: bool,
        view_indices: Sequence[int],
    ) -> TemporalCorruptionResultV1:
        batched, squeeze_batch = _validate_images(images)
        timestamp = float(timestamp_seconds)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp_seconds must be finite and non-negative")
        indices = _validate_view_indices(view_indices, batched.shape[1])
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            if timestamp < self._last_timestamp:
                self.reset()
            else:
                raise ValueError("stale-frame timestamps must be strictly increasing")

        target_timestamp = timestamp - self.delay_seconds
        # Discard frames that are older than the newest frame satisfying the
        # target.  Retaining that boundary frame plus newer frames bounds GPU
        # memory to approximately one requested delay window.
        while (
            len(self._history) > 1
            and self._history[1][0] <= target_timestamp + 1e-9
        ):
            self._history.popleft()
        selected = (
            self._history[0]
            if self._history and self._history[0][0] <= target_timestamp + 1e-9
            else None
        )

        output = batched
        applied = bool(active and selected is not None)
        source_timestamp = None
        effective_delay_ms = None
        if applied:
            source_timestamp, stale_views = selected
            output = batched.clone()
            output[:, list(indices)] = stale_views
            effective_delay_ms = (timestamp - source_timestamp) * 1000.0

        # Snapshot after selection so the current frame can never satisfy its
        # own delay.  detach/clone prevents later in-place model operations from
        # mutating history.
        self._history.append(
            (timestamp, batched[:, list(indices)].detach().clone())
        )
        self._last_timestamp = timestamp
        metadata = {
            "schema_version": TEMPORAL_CORRUPTION_SCHEMA_V1,
            "corruption": "front_stale",
            "requested_delay_ms": self.delay_ms,
            "timestamp_seconds": timestamp,
            "target_source_timestamp_seconds": target_timestamp,
            "source_timestamp_seconds": source_timestamp,
            "effective_delay_ms": effective_delay_ms,
            "view_indices": list(indices),
            "schedule_active": bool(active),
            "applied": applied,
            "history_warm": selected is not None,
            "history_length_after_observe": len(self._history),
            "selection_policy": "newest_source_at_or_before_target_simulation_time",
        }
        result_images = output.squeeze(0) if squeeze_batch else output
        return TemporalCorruptionResultV1(images=result_images, metadata=metadata)

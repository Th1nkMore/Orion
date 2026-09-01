"""Frozen pairwise observation-UQ inference and causal online calibration.

The pairwise Stage-1 objective identifies score increments, not an absolute
zero point.  Closed-loop use therefore calibrates the front-view scalar from a
fixed pre-event prefix.  Neither the calibrator nor the adapter receives the
known corruption state, route progress, hazard state, or paired reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from uq_estimator.counterfactual_evidence import (
    EVIDENCE_COMPONENTS,
    ObservationEvidenceHurdleAdapter,
)


PAIRWISE_CHECKPOINT_SCHEMA = (
    "orion.counterfactual-evidence-pairwise-native-checkpoint/v1"
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_pairwise_adapter(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[ObservationEvidenceHurdleAdapter, dict[str, Any]]:
    """Load and attest the frozen pairwise adapter used by closed loop."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("pairwise observation-UQ checkpoint is missing: %s" % path)
    observed_sha256 = _sha256(checkpoint_path)
    if expected_sha256 and observed_sha256 != expected_sha256:
        raise RuntimeError("pairwise observation-UQ checkpoint hash differs")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError("pairwise observation-UQ checkpoint must be a mapping")
    if payload.get("schema_version") != PAIRWISE_CHECKPOINT_SCHEMA:
        raise RuntimeError("pairwise observation-UQ checkpoint schema differs")
    if payload.get("requires_inference_baseline_calibration") is not True:
        raise RuntimeError("pairwise checkpoint does not require baseline calibration")
    model_config = payload.get("model_config")
    state = payload.get("student_state")
    if not isinstance(model_config, Mapping) or not isinstance(state, Mapping):
        raise RuntimeError("pairwise checkpoint lacks model config or student state")
    model = ObservationEvidenceHurdleAdapter(**dict(model_config))
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).to(device).eval()
    metadata = {
        "path": str(checkpoint_path.resolve()),
        "sha256": observed_sha256,
        "schema_version": payload["schema_version"],
        "model_config": dict(model_config),
        "best_epoch": payload.get("best_epoch"),
        "requires_inference_baseline_calibration": True,
    }
    return model, metadata


@dataclass(frozen=True)
class ObservationEvidenceAggregate:
    """JSON-safe scalar summaries of one spatial adapter prediction."""

    front_raw_score: float
    view_raw_scores: tuple[float, ...]
    front_component_scores: tuple[float, ...]
    components: tuple[str, ...] = EVIDENCE_COMPONENTS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpatialObservationEvidenceSummary:
    """Compact task-agnostic localization evidence for one front-view region."""

    front_view_index: int
    normalized_region: tuple[float, float, float, float]
    feature_region: tuple[int, int, int, int]
    region_mean_score: float
    outside_mean_score: float
    region_minus_outside: float
    region_component_scores: tuple[float, ...]
    outside_component_scores: tuple[float, ...]
    pooled_front_grid: tuple[tuple[float, ...], ...]
    components: tuple[str, ...] = EVIDENCE_COMPONENTS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_observation_evidence(
    score_map: torch.Tensor,
    *,
    front_view_index: int = 0,
) -> ObservationEvidenceAggregate:
    """Mean-pool components and patches while retaining camera selectivity."""

    if (
        score_map.ndim != 5
        or score_map.shape[0] != 1
        or score_map.shape[-1] != len(EVIDENCE_COMPONENTS)
        or not score_map.is_floating_point()
    ):
        raise ValueError("score_map must have floating [1,V,H,W,3] shape")
    views = int(score_map.shape[1])
    if not 0 <= int(front_view_index) < views:
        raise ValueError("front view index is outside the score map")
    value = score_map.detach().float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("score_map must be finite")
    per_view = value.mean(dim=(0, 2, 3, 4))
    front_components = value[0, int(front_view_index)].mean(dim=(0, 1))
    return ObservationEvidenceAggregate(
        front_raw_score=float(per_view[int(front_view_index)].cpu()),
        view_raw_scores=tuple(float(item) for item in per_view.cpu()),
        front_component_scores=tuple(float(item) for item in front_components.cpu()),
    )


def pool_observation_evidence_grid(
    score_map: torch.Tensor,
    *,
    view_index: int = 0,
    grid_size: int = 10,
) -> tuple[tuple[float, ...], ...]:
    """Return a compact task-agnostic scalar grid for one camera view.

    The adapter emits three evidence components per feature location.  This
    helper averages only those components and spatially pools to a fixed grid;
    it does not read route progress, actors, the planned path, or corruption
    state.  The JSON-safe output is intended for auditable closed-loop heatmap
    rendering rather than for control.
    """

    if (
        score_map.ndim != 5
        or score_map.shape[0] != 1
        or score_map.shape[-1] != len(EVIDENCE_COMPONENTS)
        or not score_map.is_floating_point()
    ):
        raise ValueError("score_map must have floating [1,V,H,W,3] shape")
    views = int(score_map.shape[1])
    if not 0 <= int(view_index) < views:
        raise ValueError("view index is outside the score map")
    if isinstance(grid_size, bool) or int(grid_size) <= 0:
        raise ValueError("grid_size must be a positive integer")
    selected = score_map.detach().float()[0, int(view_index)]
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("score_map must be finite")
    scalar = selected.mean(dim=-1)[None, None]
    pooled = F.adaptive_avg_pool2d(
        scalar, (int(grid_size), int(grid_size))
    )[0, 0]
    return tuple(
        tuple(float(value) for value in row.cpu()) for row in pooled
    )


def pool_observation_evidence_grids(
    score_map: torch.Tensor,
    *,
    grid_size: int = 10,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Return one compact task-agnostic scalar grid for every camera view."""

    if score_map.ndim != 5:
        raise ValueError("score_map must have floating [1,V,H,W,3] shape")
    return tuple(
        pool_observation_evidence_grid(
            score_map, view_index=view_index, grid_size=grid_size
        )
        for view_index in range(int(score_map.shape[1]))
    )


def summarize_spatial_observation_evidence(
    score_map: torch.Tensor,
    region: tuple[float, float, float, float],
    *,
    front_view_index: int = 0,
    grid_size: int = 10,
) -> SpatialObservationEvidenceSummary:
    """Summarize whether UQ localizes a specified observation-space region.

    This summary deliberately contains no path, actor, hazard, or route input.
    An on-path and an equal-strength off-path degradation should both be
    localized at their respective image regions; task relevance belongs to the
    downstream driving model rather than this Stage-1 adapter diagnostic.
    """

    if (
        score_map.ndim != 5
        or score_map.shape[0] != 1
        or score_map.shape[-1] != len(EVIDENCE_COMPONENTS)
        or not score_map.is_floating_point()
    ):
        raise ValueError("score_map must have floating [1,V,H,W,3] shape")
    views = int(score_map.shape[1])
    if not 0 <= int(front_view_index) < views:
        raise ValueError("front view index is outside the score map")
    if len(region) != 4:
        raise ValueError("region must contain top,left,bottom,right")
    top, left, bottom, right = (float(value) for value in region)
    if not all(math.isfinite(value) for value in (top, left, bottom, right)):
        raise ValueError("region coordinates must be finite")
    if not (0.0 <= top < bottom <= 1.0 and 0.0 <= left < right <= 1.0):
        raise ValueError("region must be normalized and non-empty")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    front = score_map.detach().float()[0, int(front_view_index)]
    if not bool(torch.isfinite(front).all()):
        raise ValueError("score_map must be finite")
    height, width, _ = front.shape
    top_index = max(0, min(height, int(math.floor(top * height))))
    left_index = max(0, min(width, int(math.floor(left * width))))
    bottom_index = max(0, min(height, int(math.ceil(bottom * height))))
    right_index = max(0, min(width, int(math.ceil(right * width))))
    mask = torch.zeros((height, width), dtype=torch.bool, device=front.device)
    mask[top_index:bottom_index, left_index:right_index] = True
    if not bool(mask.any()) or bool(mask.all()):
        raise ValueError("region must cover some but not all feature cells")

    region_values = front[mask]
    outside_values = front[~mask]
    region_components = region_values.mean(dim=0)
    outside_components = outside_values.mean(dim=0)
    region_mean = region_values.mean()
    outside_mean = outside_values.mean()
    scalar_map = front.mean(dim=-1)[None, None]
    pooled = F.adaptive_avg_pool2d(scalar_map, (grid_size, grid_size))[0, 0]
    pooled_grid = tuple(
        tuple(float(value) for value in row.cpu()) for row in pooled
    )
    return SpatialObservationEvidenceSummary(
        front_view_index=int(front_view_index),
        normalized_region=(top, left, bottom, right),
        feature_region=(top_index, left_index, bottom_index, right_index),
        region_mean_score=float(region_mean.cpu()),
        outside_mean_score=float(outside_mean.cpu()),
        region_minus_outside=float((region_mean - outside_mean).cpu()),
        region_component_scores=tuple(
            float(value) for value in region_components.cpu()
        ),
        outside_component_scores=tuple(
            float(value) for value in outside_components.cpu()
        ),
        pooled_front_grid=pooled_grid,
    )


@dataclass(frozen=True)
class OnlineCalibrationOutput:
    raw_score: float
    z_score: float | None
    calibrated_score: float
    filtered_score: float
    baseline_count: int
    baseline_frozen: bool
    baseline_median: float | None
    baseline_scale: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RobustPreEventCalibrator:
    """Calibrate one scalar from a fixed simulation-time prefix.

    The frozen p3 contract collects raw scores during ``[1,4)`` seconds,
    estimates a robust location/scale once, and never updates it again.  A
    fast-attack/slow-release causal filter makes a persistent outage stable
    without reading its hidden intervention state.
    """

    def __init__(
        self,
        *,
        baseline_start_seconds: float = 1.0,
        baseline_end_seconds: float = 4.0,
        minimum_baseline_frames: int = 40,
        relative_scale_floor: float = 0.05,
        absolute_scale_floor: float = 0.001,
        z_center: float = 4.0,
        attack_alpha: float = 0.8,
        release_alpha: float = 0.2,
    ) -> None:
        if not 0.0 <= baseline_start_seconds < baseline_end_seconds:
            raise ValueError("baseline times must satisfy 0 <= start < end")
        if minimum_baseline_frames <= 0:
            raise ValueError("minimum baseline frames must be positive")
        if relative_scale_floor < 0.0 or absolute_scale_floor <= 0.0:
            raise ValueError("calibration scale floors are invalid")
        if not 0.0 < attack_alpha <= 1.0 or not 0.0 < release_alpha <= 1.0:
            raise ValueError("filter alphas must lie in (0,1]")
        self.baseline_start_seconds = float(baseline_start_seconds)
        self.baseline_end_seconds = float(baseline_end_seconds)
        self.minimum_baseline_frames = int(minimum_baseline_frames)
        self.relative_scale_floor = float(relative_scale_floor)
        self.absolute_scale_floor = float(absolute_scale_floor)
        self.z_center = float(z_center)
        self.attack_alpha = float(attack_alpha)
        self.release_alpha = float(release_alpha)
        self._baseline: list[float] = []
        self._median: float | None = None
        self._scale: float | None = None
        self._filtered: float = 0.0

    @property
    def frozen(self) -> bool:
        return self._median is not None

    @property
    def baseline_count(self) -> int:
        return len(self._baseline)

    def _freeze(self) -> None:
        if self.frozen:
            return
        if len(self._baseline) < self.minimum_baseline_frames:
            raise RuntimeError(
                "insufficient pre-event baseline frames: %d < %d"
                % (len(self._baseline), self.minimum_baseline_frames)
            )
        median = float(statistics.median(self._baseline))
        mad = float(statistics.median(abs(value - median) for value in self._baseline))
        self._median = median
        self._scale = max(
            1.4826 * mad,
            self.relative_scale_floor * abs(median),
            self.absolute_scale_floor,
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0.0:
            return 1.0 / (1.0 + math.exp(-value))
        exponential = math.exp(value)
        return exponential / (1.0 + exponential)

    def update(self, raw_score: float, simulation_time_seconds: float) -> OnlineCalibrationOutput:
        raw = float(raw_score)
        current_time = float(simulation_time_seconds)
        if not math.isfinite(raw) or not math.isfinite(current_time) or current_time < 0.0:
            raise ValueError("online calibration inputs must be finite and time non-negative")
        if (
            not self.frozen
            and self.baseline_start_seconds <= current_time < self.baseline_end_seconds
        ):
            self._baseline.append(raw)
        if not self.frozen and current_time >= self.baseline_end_seconds:
            self._freeze()

        if not self.frozen:
            z_score = None
            calibrated = 0.0
            self._filtered = 0.0
        else:
            z_score = (raw - self._median) / self._scale
            calibrated = self._sigmoid(z_score - self.z_center)
            alpha = (
                self.attack_alpha
                if calibrated >= self._filtered
                else self.release_alpha
            )
            self._filtered = alpha * calibrated + (1.0 - alpha) * self._filtered
        return OnlineCalibrationOutput(
            raw_score=raw,
            z_score=z_score,
            calibrated_score=calibrated,
            filtered_score=self._filtered,
            baseline_count=len(self._baseline),
            baseline_frozen=self.frozen,
            baseline_median=self._median,
            baseline_scale=self._scale,
        )


__all__ = [
    "ObservationEvidenceAggregate",
    "SpatialObservationEvidenceSummary",
    "OnlineCalibrationOutput",
    "PAIRWISE_CHECKPOINT_SCHEMA",
    "RobustPreEventCalibrator",
    "aggregate_observation_evidence",
    "load_frozen_pairwise_adapter",
    "summarize_spatial_observation_evidence",
]

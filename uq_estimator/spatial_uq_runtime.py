"""Online runtime for the frozen Stage-1 spatial observation-UQ adapter.

The pairwise Stage-1 checkpoint identifies *increments* in observation
evidence loss.  Its raw output therefore has no globally meaningful zero.  A
closed-loop consumer must first estimate a causal, route-local baseline from a
fixed clean prefix.  This module performs that calibration per view, spatial
cell, and evidence component; it never reads route progress, actors, hazards,
corruption metadata, or control state.

The runtime is deliberately separate from legacy Density UQ.  It accepts the
full frozen ORION image feature grid and returns ``[B,V,H,W,3]`` maps.  Stage-2
losses cannot update the Stage-1 adapter because inference is always executed
under ``torch.no_grad`` and every returned map is detached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from uq_estimator.online_observation_uq import load_frozen_pairwise_adapter


SPATIAL_UQ_RUNTIME_SCHEMA = "orion.spatial-observation-uq-runtime/v1"


@dataclass(frozen=True)
class SpatialObservationUQRuntimeOutput:
    """One causal Stage-1 inference result."""

    raw_score: torch.Tensor
    calibrated_score: torch.Tensor
    previous_valid: bool
    baseline_ready: bool
    baseline_count: int
    checkpoint_sha256: str
    schema_version: str = SPATIAL_UQ_RUNTIME_SCHEMA


class CausalSpatialEvidenceCalibrator:
    """Robustly calibrate every spatial evidence component from a fixed prefix.

    The first ``warmup_frames`` raw maps are retained on CPU.  At the boundary,
    the per-cell median and MAD are frozen for the rest of the route.  Before
    the boundary the calibrated map is exactly zero, which makes the Stage-2
    fusion path an exact identity while its baseline is unknown.
    """

    def __init__(
        self,
        *,
        warmup_frames: int = 60,
        relative_scale_floor: float = 0.05,
        absolute_scale_floor: float = 1e-3,
        z_center: float = 4.0,
        attack_alpha: float = 0.8,
        release_alpha: float = 0.2,
    ) -> None:
        if isinstance(warmup_frames, bool) or int(warmup_frames) <= 0:
            raise ValueError("warmup_frames must be a positive integer")
        if relative_scale_floor < 0.0 or absolute_scale_floor <= 0.0:
            raise ValueError("calibration scale floors are invalid")
        if not 0.0 < attack_alpha <= 1.0 or not 0.0 < release_alpha <= 1.0:
            raise ValueError("filter alphas must lie in (0,1]")
        self.warmup_frames = int(warmup_frames)
        self.relative_scale_floor = float(relative_scale_floor)
        self.absolute_scale_floor = float(absolute_scale_floor)
        self.z_center = float(z_center)
        self.attack_alpha = float(attack_alpha)
        self.release_alpha = float(release_alpha)
        self.reset()

    def reset(self) -> None:
        self._samples: list[torch.Tensor] = []
        self._median: torch.Tensor | None = None
        self._scale: torch.Tensor | None = None
        self._filtered: torch.Tensor | None = None
        self._shape: tuple[int, ...] | None = None

    @property
    def ready(self) -> bool:
        return self._median is not None

    @property
    def count(self) -> int:
        return len(self._samples) if not self.ready else self.warmup_frames

    def _validate(self, raw_score: torch.Tensor) -> None:
        if (
            raw_score.ndim != 5
            or raw_score.shape[-1] != 3
            or not raw_score.is_floating_point()
        ):
            raise ValueError("raw spatial evidence must have floating [B,V,H,W,3] shape")
        if not bool(torch.isfinite(raw_score).all()):
            raise ValueError("raw spatial evidence must be finite")
        if bool((raw_score < 0).any()):
            raise ValueError("raw spatial evidence must be non-negative")
        shape = tuple(int(value) for value in raw_score.shape)
        if self._shape is None:
            self._shape = shape
        elif shape != self._shape:
            raise ValueError("spatial evidence shape changed within one route")

    def _freeze(self) -> None:
        if self.ready:
            return
        if len(self._samples) != self.warmup_frames:
            raise RuntimeError("cannot freeze an incomplete spatial baseline")
        stacked = torch.stack(self._samples, dim=0).float()
        median = stacked.median(dim=0).values
        mad = (stacked - median.unsqueeze(0)).abs().median(dim=0).values
        scale = torch.maximum(1.4826 * mad, self.relative_scale_floor * median.abs())
        scale = scale.clamp_min(self.absolute_scale_floor)
        self._median = median
        self._scale = scale
        self._filtered = torch.zeros_like(median)
        # The frozen sufficient statistics replace the warm-up samples.
        self._samples.clear()

    def update(self, raw_score: torch.Tensor) -> torch.Tensor:
        self._validate(raw_score)
        detached = raw_score.detach().float()
        if not self.ready:
            # Float16 is sufficient for a robust baseline and bounds host memory
            # to roughly 3.5 MiB for 60 x 6 x 40 x 40 x 3 values.
            self._samples.append(detached.cpu().half())
            if len(self._samples) == self.warmup_frames:
                self._freeze()
            return torch.zeros_like(raw_score).detach()

        median = self._median.to(device=detached.device, dtype=detached.dtype)
        scale = self._scale.to(device=detached.device, dtype=detached.dtype)
        calibrated = torch.sigmoid((detached - median) / scale - self.z_center)
        previous = self._filtered.to(device=detached.device, dtype=detached.dtype)
        alpha = torch.where(
            calibrated >= previous,
            torch.full_like(calibrated, self.attack_alpha),
            torch.full_like(calibrated, self.release_alpha),
        )
        filtered = alpha * calibrated + (1.0 - alpha) * previous
        self._filtered = filtered.detach().cpu()
        return filtered.to(dtype=raw_score.dtype).detach()


class FrozenSpatialObservationUQRuntime(nn.Module):
    """Run the attested Stage-1 adapter on the current ORION feature grid."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        expected_sha256: str | None = None,
        warmup_frames: int = 60,
        relative_scale_floor: float = 0.05,
        absolute_scale_floor: float = 1e-3,
        z_center: float = 4.0,
        attack_alpha: float = 0.8,
        release_alpha: float = 0.2,
    ) -> None:
        super().__init__()
        adapter, metadata = load_frozen_pairwise_adapter(
            checkpoint_path,
            expected_sha256=expected_sha256,
            device="cpu",
        )
        self.adapter = adapter
        self.metadata: dict[str, Any] = metadata
        self.calibrator = CausalSpatialEvidenceCalibrator(
            warmup_frames=warmup_frames,
            relative_scale_floor=relative_scale_floor,
            absolute_scale_floor=absolute_scale_floor,
            z_center=z_center,
            attack_alpha=attack_alpha,
            release_alpha=release_alpha,
        )
        self._previous_features: torch.Tensor | None = None
        self._previous_valid = False
        self.requires_grad_(False)

    def train(self, mode: bool = True):
        # The Stage-1 boundary is frozen even when the surrounding ORION model
        # enters train mode for Stage 2.
        super().train(False)
        self.adapter.eval()
        return self

    def reset(self) -> None:
        self._previous_features = None
        self._previous_valid = False
        self.calibrator.reset()

    def forward(self, image_features: torch.Tensor) -> SpatialObservationUQRuntimeOutput:
        if (
            image_features.ndim != 5
            or not image_features.is_floating_point()
            or min(image_features.shape) <= 0
        ):
            raise ValueError("image_features must have floating [B,V,C,H,W] shape")
        if not bool(torch.isfinite(image_features).all()):
            raise ValueError("image_features must be finite")
        current = image_features.detach().permute(0, 1, 3, 4, 2).contiguous()
        if self._previous_features is not None and self._previous_features.shape != current.shape:
            raise ValueError("ORION feature shape changed within one route")
        valid = torch.full(
            (current.shape[0],),
            self._previous_valid,
            dtype=torch.bool,
            device=current.device,
        )
        with torch.no_grad():
            raw = self.adapter(current, self._previous_features, valid).detach()
            calibrated = self.calibrator.update(raw)
        was_valid = self._previous_valid
        self._previous_features = current.detach()
        self._previous_valid = True
        return SpatialObservationUQRuntimeOutput(
            raw_score=raw,
            calibrated_score=calibrated,
            previous_valid=was_valid,
            baseline_ready=self.calibrator.ready,
            baseline_count=self.calibrator.count,
            checkpoint_sha256=str(self.metadata["sha256"]),
        )


__all__ = [
    "SPATIAL_UQ_RUNTIME_SCHEMA",
    "CausalSpatialEvidenceCalibrator",
    "FrozenSpatialObservationUQRuntime",
    "SpatialObservationUQRuntimeOutput",
]

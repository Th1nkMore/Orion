"""Deterministic, temporally coherent lens-space waterdrop corruption."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from uq_estimator.corruptions import IMAGENET_MEAN, IMAGENET_STD


SCHEMA = "orion.lens_waterdrop.v1"
PROFILES = {
    1: {
        "count": 4,
        "radius_range": (0.035, 0.065),
        "distortion": 0.018,
        "blur_kernel": 3,
        "alpha": 0.55,
        "drift_per_second": 0.006,
    },
    2: {
        "count": 7,
        "radius_range": (0.045, 0.085),
        "distortion": 0.028,
        "blur_kernel": 5,
        "alpha": 0.70,
        "drift_per_second": 0.010,
    },
    3: {
        "count": 11,
        "radius_range": (0.055, 0.110),
        "distortion": 0.045,
        "blur_kernel": 7,
        "alpha": 0.82,
        "drift_per_second": 0.015,
    },
}


@dataclass(frozen=True)
class LensWaterdropResultV1:
    images: torch.Tensor
    mask: torch.Tensor
    metadata: dict[str, Any]


def _validate(
    images: torch.Tensor, severity: int, view_indices: Sequence[int]
) -> tuple[torch.Tensor, bool, tuple[int, ...]]:
    if severity not in PROFILES:
        raise ValueError("severity must be 1, 2, or 3")
    if not torch.is_tensor(images):
        raise TypeError("images must be a torch.Tensor")
    squeeze_batch = images.ndim == 4
    batched = images.unsqueeze(0) if squeeze_batch else images
    if batched.ndim != 5 or batched.shape[2] != 3:
        raise ValueError("images must have shape [V,3,H,W] or [B,V,3,H,W]")
    if not batched.is_floating_point():
        raise TypeError("waterdrop corruption requires floating-point images")
    indices = tuple(int(index) for index in view_indices)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("view_indices must be non-empty and unique")
    if any(index < 0 or index >= batched.shape[1] for index in indices):
        raise ValueError("view_indices contains an out-of-range view")
    return batched, squeeze_batch, indices


def _droplets_for_view(
    *, seed: int, view_index: int, severity: int, elapsed_seconds: float
) -> list[dict[str, float]]:
    profile = PROFILES[severity]
    rng = random.Random(int(seed) + 1009 * int(view_index))
    result = []
    radius_min, radius_max = profile["radius_range"]
    for _ in range(profile["count"]):
        radius = rng.uniform(radius_min, radius_max)
        aspect = rng.uniform(0.72, 1.18)
        center_x = rng.uniform(radius, 1.0 - radius)
        initial_y = rng.uniform(-0.10, 1.0 - radius)
        drift = profile["drift_per_second"] * rng.uniform(0.75, 1.25)
        span = 1.20 + 2.0 * radius
        center_y = ((initial_y + drift * elapsed_seconds + radius) % span) - radius
        result.append(
            {
                "center_x": center_x,
                "center_y": center_y,
                "radius": radius,
                "aspect": aspect,
                "drift_per_second": drift,
            }
        )
    return result


def apply_lens_waterdrop(
    images: torch.Tensor,
    *,
    severity: int,
    view_indices: Sequence[int],
    seed: int,
    elapsed_seconds: float,
) -> LensWaterdropResultV1:
    """Apply an actor-independent lens-space field to selected cameras.

    A fixed seed determines droplet geometry.  The only temporal degree of
    freedom is slow downward drift as a deterministic function of elapsed
    simulation time; no per-frame random sampling or actor bounding boxes are
    read.
    """
    batched, squeeze_batch, indices = _validate(images, severity, view_indices)
    elapsed = float(elapsed_seconds)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("elapsed_seconds must be finite and non-negative")
    profile = PROFILES[severity]
    batch, views, _, height, width = batched.shape
    mean = batched.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = batched.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    rgb = (batched * std + mean).div(255.0).clamp(0.0, 1.0)
    output_rgb = rgb.clone()
    exact_mask = torch.zeros(
        (batch, views, 1, height, width),
        dtype=torch.bool,
        device=batched.device,
    )
    y = torch.linspace(0.0, 1.0, height, device=batched.device, dtype=batched.dtype)
    x = torch.linspace(0.0, 1.0, width, device=batched.device, dtype=batched.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    base_grid = torch.stack((xx.mul(2.0).sub(1.0), yy.mul(2.0).sub(1.0)), dim=-1)
    metadata_views: dict[str, Any] = {}

    for view_index in indices:
        droplets = _droplets_for_view(
            seed=seed,
            view_index=view_index,
            severity=severity,
            elapsed_seconds=elapsed,
        )
        alpha = torch.zeros_like(xx)
        offset_x = torch.zeros_like(xx)
        offset_y = torch.zeros_like(yy)
        highlight = torch.zeros_like(xx)
        for droplet in droplets:
            radius = droplet["radius"]
            dx = (xx - droplet["center_x"]) / (radius * droplet["aspect"])
            dy = (yy - droplet["center_y"]) / radius
            distance = torch.sqrt(dx.square() + dy.square() + 1e-8)
            core = (1.0 - distance).clamp(0.0, 1.0)
            soft_body = torch.sigmoid((1.0 - distance) * 28.0)
            rim = torch.exp(-((distance - 0.88) / 0.10).square())
            local_alpha = profile["alpha"] * torch.maximum(
                soft_body * 0.72, rim * 0.50
            )
            alpha = torch.maximum(alpha, local_alpha)
            radial = core.square() * profile["distortion"]
            offset_x = offset_x + dx * radial
            offset_y = offset_y + dy * radial
            specular = torch.exp(
                -(
                    ((dx + 0.33) / 0.18).square()
                    + ((dy + 0.36) / 0.14).square()
                )
            )
            highlight = torch.maximum(highlight, specular * soft_body)

        grid = base_grid.clone()
        grid[..., 0] = (grid[..., 0] + 2.0 * offset_x).clamp(-1.0, 1.0)
        grid[..., 1] = (grid[..., 1] + 2.0 * offset_y).clamp(-1.0, 1.0)
        grid = grid.unsqueeze(0).expand(batch, -1, -1, -1)
        original = rgb[:, view_index]
        refracted = F.grid_sample(
            original,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        kernel = int(profile["blur_kernel"])
        blurred = F.avg_pool2d(
            original, kernel_size=kernel, stride=1, padding=kernel // 2
        )
        lens_content = 0.72 * refracted + 0.28 * blurred
        alpha_b = alpha.view(1, 1, height, width)
        composed = original * (1.0 - alpha_b) + lens_content * alpha_b
        composed = (composed + 0.24 * highlight.view(1, 1, height, width)).clamp(
            0.0, 1.0
        )
        output_rgb[:, view_index] = composed
        exact_mask[:, view_index] = alpha_b > 0.05
        metadata_views[str(view_index)] = droplets

    # Preserve unselected camera tensors bit-for-bit; only selected cameras
    # undergo the RGB round trip.
    normalized = batched.clone()
    normalized[:, list(indices)] = (
        output_rgb[:, list(indices)].mul(255.0) - mean
    ) / std
    result_images = normalized.squeeze(0) if squeeze_batch else normalized
    return LensWaterdropResultV1(
        images=result_images,
        mask=exact_mask,
        metadata={
            "schema_version": SCHEMA,
            "corruption": "lens_waterdrop",
            "severity": severity,
            "seed": int(seed),
            "elapsed_seconds": elapsed,
            "view_indices": list(indices),
            "placement_policy": "frozen_lens_space_seed_actor_independent",
            "temporal_policy": "deterministic_slow_vertical_drift",
            "profile": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in profile.items()
            },
            "droplets_by_view": metadata_views,
            "mask_fraction": float(exact_mask.float().mean().detach().cpu()),
        },
    )

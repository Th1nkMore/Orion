"""Deterministic multi-view image corruptions for paired UQ experiments."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)
CORRUPTION_METADATA_SCHEMA_V1 = "orion.spatial_corruption.v1"


@dataclass(frozen=True)
class CorruptionMetadataV1:
    """Serializable description of one spatial corruption operation.

    ``normalized_region`` uses ``(top, left, bottom, right)`` coordinates in
    ``[0, 1]``.  The returned region is aligned to the realised pixel mask, so
    it can be projected or replayed without ambiguity.
    """

    schema_version: str
    corruption: str
    seed: int
    severity: int
    view_indices: tuple[int, ...]
    normalized_region: tuple[float, float, float, float]
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metadata dictionary."""
        return {
            "schema_version": self.schema_version,
            "corruption": self.corruption,
            "seed": self.seed,
            "severity": self.severity,
            "view_indices": list(self.view_indices),
            "normalized_region": list(self.normalized_region),
            "parameters": copy.deepcopy(self.parameters),
        }


@dataclass(frozen=True)
class CorruptionResultV1:
    """Corrupted images, an exact spatial mask, and versioned metadata."""

    images: torch.Tensor
    mask: torch.Tensor
    metadata: CorruptionMetadataV1


@dataclass(frozen=True)
class BatchCorruptionResultV1:
    """Deep-copied dataloader batch plus its corruption supervision."""

    batch: dict
    mask: torch.Tensor
    metadata: CorruptionMetadataV1


def corrupt_multiview_images(
    images: torch.Tensor,
    corruption: str,
    severity: int = 2,
    view_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Corrupt normalized images shaped [B, V, C, H, W] or [V, C, H, W]."""
    if severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2, or 3")
    squeeze_batch = images.ndim == 4
    if squeeze_batch:
        images = images.unsqueeze(0)
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError("images must have shape [B, V, 3, H, W]")

    output = images.clone()
    if corruption == "blur":
        kernels = (3, 5, 9)
        kernel = kernels[severity - 1]
        flat = output.flatten(0, 1)
        flat = F.avg_pool2d(
            flat, kernel_size=kernel, stride=1, padding=kernel // 2
        )
        output = flat.reshape_as(output)
    elif corruption == "dark":
        factors = (0.75, 0.5, 0.3)
        factor = factors[severity - 1]
        mean = output.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = output.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
        output = factor * output + (factor - 1.0) * mean / std
    elif corruption == "camera_dropout":
        drop_counts = (1, 2, 3)
        if view_indices is None:
            candidates = list(range(output.shape[1]))
        else:
            candidates = [int(index) for index in view_indices]
            if len(set(candidates)) != len(candidates):
                raise ValueError("view_indices must not contain duplicates")
            if any(index < 0 or index >= output.shape[1] for index in candidates):
                raise ValueError("view_indices contains an out-of-range view")
            if not candidates:
                raise ValueError("view_indices must not be empty")
        count = min(drop_counts[severity - 1], len(candidates))
        selected = candidates[:count]
        mean = output.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = output.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
        output[:, selected] = -mean / std
    else:
        raise ValueError(f"Unsupported corruption: {corruption}")
    return output.squeeze(0) if squeeze_batch else output


def normalized_front_tensor_to_bgr(images: torch.Tensor) -> torch.Tensor:
    """Render the exact normalized front-view tensor as uint8 BGR pixels.

    The closed-loop batch may be shaped either ``[V,C,H,W]`` or
    ``[B,V,C,H,W]``.  This helper selects batch item zero and view zero,
    reverses the frozen ImageNet RGB normalization, and returns a CPU-free
    tensor suitable for a single, explicitly requested diagnostic transfer.
    It intentionally performs no resize or reconstruction from raw sensor
    pixels.
    """

    if not torch.is_tensor(images):
        raise TypeError("images must be a torch.Tensor")
    if images.ndim == 5:
        if images.shape[0] != 1:
            raise ValueError("exact front preview requires a single-item batch")
        front = images[0, 0]
    elif images.ndim == 4:
        front = images[0]
    else:
        raise ValueError("images must have shape [V,3,H,W] or [1,V,3,H,W]")
    if front.ndim != 3 or front.shape[0] != 3:
        raise ValueError("front view must have shape [3,H,W]")
    if not front.is_floating_point():
        raise TypeError("front view must contain normalized floating-point values")
    mean = front.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = front.new_tensor(IMAGENET_STD).view(3, 1, 1)
    rgb = front * std + mean
    bgr_hwc = rgb[[2, 1, 0]].permute(1, 2, 0)
    return bgr_hwc.round().clamp(0, 255).to(dtype=torch.uint8)


def corrupt_batch_images(
    batch: dict,
    corruption: str,
    severity: int = 2,
    view_indices: Sequence[int] | None = None,
) -> dict:
    """Deep-copy a dataloader batch and corrupt only its image tensor."""
    result = copy.deepcopy(batch)
    images = result["img"]
    if not isinstance(images, list) or not images:
        raise ValueError("Expected batch['img'] to be a non-empty list")
    images[0] = corrupt_multiview_images(
        images[0],
        corruption=corruption,
        severity=severity,
        view_indices=view_indices,
    )
    return result


def _validate_spatial_images(images: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Normalize supported image ranks without changing the caller's tensor."""
    if not torch.is_tensor(images):
        raise TypeError("images must be a torch.Tensor")
    squeeze_batch = images.ndim == 4
    batched = images.unsqueeze(0) if squeeze_batch else images
    if batched.ndim != 5 or batched.shape[2] != 3:
        raise ValueError("images must have shape [B, V, 3, H, W]")
    if batched.shape[0] <= 0 or batched.shape[1] <= 0:
        raise ValueError("images must contain at least one batch item and view")
    if batched.shape[-2] <= 0 or batched.shape[-1] <= 0:
        raise ValueError("images must have non-empty spatial dimensions")
    if not batched.is_floating_point():
        raise TypeError("spatial corruptions require floating-point normalized images")
    return batched, squeeze_batch


def _validate_view_indices(
    view_indices: Sequence[int] | None,
    n_views: int,
) -> tuple[int, ...]:
    if view_indices is None:
        return tuple(range(n_views))
    indices = tuple(int(index) for index in view_indices)
    if not indices:
        raise ValueError("view_indices must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("view_indices must not contain duplicates")
    if any(index < 0 or index >= n_views for index in indices):
        raise ValueError("view_indices contains an out-of-range view")
    return indices


def _validate_region(
    region: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(region) != 4:
        raise ValueError("region must be (top, left, bottom, right)")
    top, left, bottom, right = (float(value) for value in region)
    if not all(math.isfinite(value) for value in (top, left, bottom, right)):
        raise ValueError("region coordinates must be finite")
    if not (0.0 <= top < bottom <= 1.0):
        raise ValueError("region must satisfy 0 <= top < bottom <= 1")
    if not (0.0 <= left < right <= 1.0):
        raise ValueError("region must satisfy 0 <= left < right <= 1")
    return top, left, bottom, right


def _sample_region(seed: int, severity: int) -> tuple[float, float, float, float]:
    """Sample a deterministic local box; higher severity covers more area."""
    rng = random.Random(seed)
    base_fraction = (0.22, 0.34, 0.46)[severity - 1]
    height = min(base_fraction * (0.85 + 0.30 * rng.random()), 0.95)
    width = min(base_fraction * (0.85 + 0.30 * rng.random()), 0.95)
    top = rng.random() * (1.0 - height)
    left = rng.random() * (1.0 - width)
    return top, left, top + height, left + width


def _region_to_pixels(
    region: tuple[float, float, float, float],
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    top, left, bottom, right = region
    top_px = int(math.floor(top * height))
    left_px = int(math.floor(left * width))
    bottom_px = int(math.ceil(bottom * height))
    right_px = int(math.ceil(right * width))
    top_px = min(max(top_px, 0), height)
    left_px = min(max(left_px, 0), width)
    bottom_px = min(max(bottom_px, 0), height)
    right_px = min(max(right_px, 0), width)
    if top_px >= bottom_px or left_px >= right_px:
        raise ValueError("region produces an empty pixel mask")
    return top_px, left_px, bottom_px, right_px


def _pixel_aligned_region(
    pixel_region: tuple[int, int, int, int],
    height: int,
    width: int,
) -> tuple[float, float, float, float]:
    top, left, bottom, right = pixel_region
    return top / height, left / width, bottom / height, right / width


def _build_mask(
    images: torch.Tensor,
    view_indices: tuple[int, ...],
    pixel_region: tuple[int, int, int, int],
) -> torch.Tensor:
    batch, views, _, height, width = images.shape
    top, left, bottom, right = pixel_region
    mask = torch.zeros(
        (batch, views, 1, height, width),
        dtype=torch.bool,
        device=images.device,
    )
    mask[:, list(view_indices), :, top:bottom, left:right] = True
    if not bool(mask.any().item()):
        raise ValueError("corruption mask must not be empty")
    return mask


def _normalized_constant(
    images: torch.Tensor,
    rgb_value: float,
) -> torch.Tensor:
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    return (rgb_value - mean) / std


def corrupt_multiview_images_with_metadata(
    images: torch.Tensor,
    corruption: str,
    severity: int = 2,
    view_indices: Sequence[int] | None = None,
    *,
    seed: int = 0,
    region: Sequence[float] | None = None,
) -> CorruptionResultV1:
    """Apply a replayable spatial corruption and return its exact supervision.

    Args:
        images: Normalized images shaped ``[B,V,3,H,W]`` or ``[V,3,H,W]``.
        corruption: One of ``local_blur``, ``local_dark``, ``local_glare``,
            ``local_occlusion``, or diagnostic ``camera_dropout``.
        severity: Integer 1-3 controlling effect strength and, for a sampled
            local region, its approximate area.
        view_indices: Views to corrupt. ``None`` selects every view.  For
            camera dropout, severity selects the first 1/2/3 candidate views,
            matching the legacy diagnostic behavior.
        seed: Seed for deterministic local-region placement.
        region: Optional normalized ``(top,left,bottom,right)`` region.  Passing
            it bypasses random placement and enables fixed on/off-path pairs.

    Returns:
        ``CorruptionResultV1``.  Images preserve the input rank; mask always has
        shape ``[B,V,1,H,W]`` and boolean dtype.
    """
    if severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2, or 3")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    supported = {
        "local_blur",
        "local_dark",
        "local_glare",
        "local_occlusion",
        "camera_dropout",
    }
    if corruption not in supported:
        raise ValueError(f"Unsupported metadata corruption: {corruption}")

    batched, squeeze_batch = _validate_spatial_images(images)
    _, n_views, _, height, width = batched.shape
    candidates = _validate_view_indices(view_indices, n_views)
    requested_region = None

    if corruption == "camera_dropout":
        if region is not None:
            requested = _validate_region(region)
            if requested != (0.0, 0.0, 1.0, 1.0):
                raise ValueError(
                    "camera_dropout covers complete views; region must be omitted "
                    "or equal (0, 0, 1, 1)"
                )
        drop_count = min((1, 2, 3)[severity - 1], len(candidates))
        selected_views = candidates[:drop_count]
        normalized_region = (0.0, 0.0, 1.0, 1.0)
        pixel_region = (0, 0, height, width)
    else:
        selected_views = candidates
        if region is None:
            requested_region = _sample_region(seed, severity)
        else:
            requested_region = _validate_region(region)
        pixel_region = _region_to_pixels(requested_region, height, width)
        normalized_region = _pixel_aligned_region(pixel_region, height, width)

    mask = _build_mask(batched, selected_views, pixel_region)
    output = batched.clone()
    channel_mask = mask.expand(-1, -1, output.shape[2], -1, -1)
    parameters: dict[str, Any]

    if corruption == "local_blur":
        kernel = (3, 5, 9)[severity - 1]
        blurred = F.avg_pool2d(
            output.flatten(0, 1),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        ).reshape_as(output)
        output = torch.where(channel_mask, blurred, output)
        parameters = {"kernel_size": kernel, "method": "average_pool"}
    elif corruption == "local_dark":
        factor = (0.75, 0.5, 0.3)[severity - 1]
        mean = output.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = output.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
        darkened = factor * output + (factor - 1.0) * mean / std
        output = torch.where(channel_mask, darkened, output)
        parameters = {"intensity_factor": factor}
    elif corruption == "local_glare":
        alpha = (0.35, 0.60, 0.85)[severity - 1]
        white = _normalized_constant(output, 255.0).expand_as(output)
        glare = (1.0 - alpha) * output + alpha * white
        output = torch.where(channel_mask, glare, output)
        parameters = {"alpha": alpha, "target_rgb": 255.0}
    elif corruption == "local_occlusion":
        black = _normalized_constant(output, 0.0).expand_as(output)
        output = torch.where(channel_mask, black, output)
        parameters = {"fill_rgb": 0.0}
    else:
        black = _normalized_constant(output, 0.0).expand_as(output)
        output = torch.where(channel_mask, black, output)
        parameters = {
            "diagnostic_only": True,
            "drop_count": len(selected_views),
            "candidate_view_indices": list(candidates),
            "fill_rgb": 0.0,
        }

    if requested_region is not None:
        parameters["requested_region"] = list(requested_region)
    parameters["pixel_region"] = list(pixel_region)

    metadata = CorruptionMetadataV1(
        schema_version=CORRUPTION_METADATA_SCHEMA_V1,
        corruption=corruption,
        seed=seed,
        severity=severity,
        view_indices=selected_views,
        normalized_region=normalized_region,
        parameters=parameters,
    )
    result_images = output.squeeze(0) if squeeze_batch else output
    return CorruptionResultV1(
        images=result_images,
        mask=mask,
        metadata=metadata,
    )


def corrupt_batch_images_with_metadata(
    batch: dict,
    corruption: str,
    severity: int = 2,
    view_indices: Sequence[int] | None = None,
    *,
    seed: int = 0,
    region: Sequence[float] | None = None,
) -> BatchCorruptionResultV1:
    """Deep-copy a dataloader batch and retain spatial corruption labels."""
    result_batch = copy.deepcopy(batch)
    images = result_batch.get("img")
    if not isinstance(images, list) or not images:
        raise ValueError("Expected batch['img'] to be a non-empty list")
    image_result = corrupt_multiview_images_with_metadata(
        images[0],
        corruption=corruption,
        severity=severity,
        view_indices=view_indices,
        seed=seed,
        region=region,
    )
    images[0] = image_result.images
    return BatchCorruptionResultV1(
        batch=result_batch,
        mask=image_result.mask,
        metadata=image_result.metadata,
    )

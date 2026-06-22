"""Deterministic multi-view image corruptions for paired UQ experiments."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F


IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)


def corrupt_multiview_images(
    images: torch.Tensor,
    corruption: str,
    severity: int = 2,
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
        count = min(drop_counts[severity - 1], output.shape[1])
        mean = output.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = output.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
        output[:, :count] = -mean / std
    else:
        raise ValueError(f"Unsupported corruption: {corruption}")
    return output.squeeze(0) if squeeze_batch else output


def corrupt_batch_images(
    batch: dict,
    corruption: str,
    severity: int = 2,
) -> dict:
    """Deep-copy a dataloader batch and corrupt only its image tensor."""
    result = copy.deepcopy(batch)
    images = result["img"]
    if not isinstance(images, list) or not images:
        raise ValueError("Expected batch['img'] to be a non-empty list")
    images[0] = corrupt_multiview_images(
        images[0], corruption=corruption, severity=severity
    )
    return result

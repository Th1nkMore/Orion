"""Deterministic feature compaction for counterfactual evidence data."""

from __future__ import annotations

import hashlib
import math
from typing import Optional

import torch

from uq_estimator.counterfactual_evidence import CounterfactualEvidenceError


COMPACT_PROJECTION_SCHEMA_VERSION = "orion.counterfactual-compact-projection/v1"


def deterministic_rademacher_projection(
    input_dim: int,
    output_dim: int,
    seed: int,
) -> torch.Tensor:
    """Return a frozen Johnson-Lindenstrauss sign projection on CPU.

    The ``1/sqrt(output_dim)`` scale preserves Euclidean norms in expectation.
    A sign matrix is used instead of a fitted projection so no validation or
    held-out corruption data can influence the representation.
    """

    if min(int(input_dim), int(output_dim)) <= 0 or int(output_dim) > int(input_dim):
        raise CounterfactualEvidenceError("invalid compact projection dimensions")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    signs = torch.randint(
        0,
        2,
        (int(input_dim), int(output_dim)),
        generator=generator,
        dtype=torch.int8,
    )
    return signs.float().mul_(2.0).sub_(1.0).div_(math.sqrt(int(output_dim)))


def projection_sha256(projection: torch.Tensor) -> str:
    if projection.ndim != 2 or not projection.is_floating_point():
        raise CounterfactualEvidenceError("projection must be a floating matrix")
    payload = projection.detach().cpu().float().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def project_feature_grid(
    feature_grid: torch.Tensor,
    projection: torch.Tensor,
    device: torch.device,
    output_dtype: torch.dtype = torch.float16,
    patch_batch_size: int = 65536,
) -> torch.Tensor:
    """Project ``[..., D]`` feature grids without changing their spatial grid."""

    if (
        feature_grid.ndim != 4
        or not feature_grid.is_floating_point()
        or projection.ndim != 2
        or feature_grid.shape[-1] != projection.shape[0]
        or patch_batch_size <= 0
    ):
        raise CounterfactualEvidenceError("feature grid and projection differ")
    if output_dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise CounterfactualEvidenceError("unsupported compact feature dtype")
    flat = feature_grid.reshape(-1, feature_grid.shape[-1])
    matrix = projection.to(device=device, dtype=torch.float32)
    chunks = []
    for start in range(0, flat.shape[0], int(patch_batch_size)):
        current = flat[start : start + int(patch_batch_size)].to(
            device=device, dtype=torch.float32
        )
        chunks.append((current @ matrix).to(dtype=output_dtype).cpu())
    return torch.cat(chunks).reshape(*feature_grid.shape[:-1], projection.shape[1])


def dynamic_symmetric_int8_quantize(
    feature_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each grid and feature channel with a symmetric dynamic scale.

    A 4D value is one ``[V,H,W,D]`` grid.  A 5D value is a batch of grids and
    receives an independent scale for every ``[sample, feature channel]``.
    """

    if (
        feature_grid.ndim not in (4, 5)
        or not feature_grid.is_floating_point()
        or not bool(torch.isfinite(feature_grid).all())
    ):
        raise CounterfactualEvidenceError("invalid feature grid for int8 quantization")
    if feature_grid.ndim == 4:
        reduce_dims = (0, 1, 2)
        broadcast_shape = (1, 1, 1, feature_grid.shape[-1])
    else:
        reduce_dims = (1, 2, 3)
        broadcast_shape = (
            feature_grid.shape[0],
            1,
            1,
            1,
            feature_grid.shape[-1],
        )
    maximum = feature_grid.float().abs().amax(dim=reduce_dims)
    scale = (maximum / 127.0).clamp_min(torch.finfo(torch.float32).eps)
    quantized = torch.round(feature_grid.float() / scale.reshape(broadcast_shape))
    return quantized.clamp(-127, 127).to(torch.int8), scale


def dynamic_symmetric_int8_dequantize(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    if quantized.dtype != torch.int8 or quantized.ndim not in (4, 5):
        raise CounterfactualEvidenceError("invalid int8 feature payload")
    expected_scale_shape = (
        (quantized.shape[-1],)
        if quantized.ndim == 4
        else (quantized.shape[0], quantized.shape[-1])
    )
    if scale.shape != expected_scale_shape or not bool(torch.isfinite(scale).all()):
        raise CounterfactualEvidenceError("int8 feature scale differs")
    broadcast_shape = (
        (1, 1, 1, quantized.shape[-1])
        if quantized.ndim == 4
        else (quantized.shape[0], 1, 1, 1, quantized.shape[-1])
    )
    return (quantized.float() * scale.float().reshape(broadcast_shape)).to(output_dtype)


def dynamic_symmetric_int8_roundtrip(feature_grid: torch.Tensor) -> torch.Tensor:
    quantized, scale = dynamic_symmetric_int8_quantize(feature_grid)
    return dynamic_symmetric_int8_dequantize(
        quantized, scale, output_dtype=feature_grid.dtype
    )


__all__ = [
    "COMPACT_PROJECTION_SCHEMA_VERSION",
    "deterministic_rademacher_projection",
    "dynamic_symmetric_int8_dequantize",
    "dynamic_symmetric_int8_quantize",
    "dynamic_symmetric_int8_roundtrip",
    "project_feature_grid",
    "projection_sha256",
]

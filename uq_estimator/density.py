"""Density-based uncertainty estimation for frozen EVAViT features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from uq_estimator.model import UQOutput


def compute_view_moments(patch_tokens: torch.Tensor) -> torch.Tensor:
    """Return per-view patch mean and standard deviation.

    Args:
        patch_tokens: Tensor shaped ``[..., N_views, N_patches, D]``.

    Returns:
        Tensor shaped ``[..., N_views * 2 * D]``.
    """
    if patch_tokens.ndim < 3:
        raise ValueError(
            "patch_tokens must have at least 3 dimensions "
            "[..., N_views, N_patches, D]"
        )
    tokens = patch_tokens.float()
    mean = tokens.mean(dim=-2)
    std = tokens.std(dim=-2, unbiased=False)
    return torch.cat((mean, std), dim=-1).flatten(start_dim=-2)


class DensityUQEstimator(nn.Module):
    """Fixed density model that preserves the existing UQ output contract."""

    def __init__(
        self,
        descriptor_mean: torch.Tensor,
        descriptor_scale: torch.Tensor,
        pca_mean: torch.Tensor,
        pca_components: torch.Tensor,
        latent_mean: torch.Tensor,
        whitening: torch.Tensor,
        calibration_distances: torch.Tensor,
        output_projection: torch.Tensor | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.register_buffer("descriptor_mean", descriptor_mean.float())
        self.register_buffer("descriptor_scale", descriptor_scale.float())
        self.register_buffer("pca_mean", pca_mean.float())
        self.register_buffer("pca_components", pca_components.float())
        self.register_buffer("latent_mean", latent_mean.float())
        self.register_buffer("whitening", whitening.float())
        self.register_buffer(
            "calibration_distances",
            calibration_distances.float().flatten().sort().values,
        )
        if output_projection is None:
            output_projection = torch.eye(pca_components.shape[0])
        self.register_buffer("output_projection", output_projection.float())
        self.eps = float(eps)

    @property
    def embedding_dim(self) -> int:
        return int(self.output_projection.shape[0])

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path | dict[str, Any],
        map_location: str | torch.device = "cpu",
    ) -> "DensityUQEstimator":
        if isinstance(checkpoint, (str, Path)):
            payload = torch.load(
                checkpoint, map_location=map_location, weights_only=True
            )
        else:
            payload = checkpoint
        state = payload.get("model_state", payload)
        required = {
            "descriptor_mean",
            "descriptor_scale",
            "pca_mean",
            "pca_components",
            "latent_mean",
            "whitening",
            "calibration_distances",
        }
        missing = required.difference(state)
        if missing:
            raise KeyError(f"Density UQ checkpoint missing keys: {sorted(missing)}")
        kwargs = {key: state[key] for key in required}
        kwargs["output_projection"] = state.get("output_projection")
        return cls(**kwargs)

    def encode_descriptor(
        self, descriptor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode descriptors into direction embeddings and calibrated scores."""
        descriptor = descriptor.float()
        descriptor_mean = self.descriptor_mean.float()
        descriptor_scale = self.descriptor_scale.float()
        pca_mean = self.pca_mean.float()
        pca_components = self.pca_components.float()
        latent_mean = self.latent_mean.float()
        whitening = self.whitening.float()
        output_projection = self.output_projection.float()
        calibration_distances = self.calibration_distances.float()
        standardized = (
            descriptor - descriptor_mean
        ) / descriptor_scale.clamp_min(self.eps)
        latent = (standardized - pca_mean) @ pca_components.T
        residual = latent - latent_mean
        whitened = residual @ whitening.T
        distance = torch.linalg.vector_norm(whitened, dim=-1)
        active_embedding = whitened / distance.unsqueeze(-1).clamp_min(self.eps)
        embedding = active_embedding @ output_projection.T

        flat_distance = distance.reshape(-1).contiguous()
        ranks = torch.searchsorted(
            calibration_distances, flat_distance, right=True
        )
        score = ranks.to(distance.dtype) / max(
            int(calibration_distances.numel()), 1
        )
        return (
            embedding,
            score.reshape(distance.shape).unsqueeze(-1),
            distance,
            active_embedding,
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        stat_features: torch.Tensor | None = None,
    ) -> UQOutput:
        del stat_features
        descriptor = compute_view_moments(patch_tokens)
        embedding, score, _, active_embedding = self.encode_descriptor(descriptor)
        return UQOutput(
            embedding=embedding,
            score=score,
            active_embedding=active_embedding,
        )


def get_uq_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Return model weights from either legacy or density UQ checkpoints."""
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "model_state" in checkpoint:
        return checkpoint["model_state"]
    raise KeyError("UQ checkpoint contains neither model_state_dict nor model_state")


def is_density_checkpoint(checkpoint: dict[str, Any]) -> bool:
    state = checkpoint.get("model_state", {})
    return "descriptor_mean" in state and "pca_components" in state

"""Mock feature file generators for testing."""

from __future__ import annotations

from pathlib import Path

import torch


def create_mock_feature_dir(
    tmp_path: Path,
    n_files: int = 20,
    n_views: int = 6,
    n_patches: int = 256,
    d_patch: int = 1152,
    img_h: int = 64,
    img_w: int = 64,
) -> Path:
    """Create a temporary directory of mock .pt feature files.

    First 1/3 → normal scenes  (high gradient, high cross-view consistency)
    Last  1/3 → adverse scenes (low gradient, low cross-view consistency)
    Middle 1/3 → random

    Args:
        tmp_path: base temporary directory (e.g. from pytest ``tmp_path``).
        n_files:  number of files to generate.

    Returns:
        Path to the created feature directory.
    """
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    n_normal = n_files // 3
    n_adverse = n_files // 3
    # remainder goes to random
    n_random = n_files - n_normal - n_adverse

    for i in range(n_files):
        if i < n_normal:
            data = _make_normal_sample(n_views, n_patches, d_patch, img_h, img_w)
        elif i < n_normal + n_random:
            data = _make_random_sample(n_views, n_patches, d_patch, img_h, img_w)
        else:
            data = _make_adverse_sample(n_views, n_patches, d_patch, img_h, img_w)

        torch.save(data, str(feature_dir / f"sample_{i:04d}.pt"))

    return feature_dir


def _make_normal_sample(
    n_views: int, n_patches: int, d_patch: int, img_h: int, img_w: int
) -> dict[str, torch.Tensor]:
    """Normal scene: high-contrast image, consistent tokens across views."""
    # High-variance image → strong gradients
    image = torch.randn(n_views, 3, img_h, img_w) * 2.0  # [N_views, 3, H, W]

    # Consistent tokens: shared base + small per-view noise
    base = torch.randn(1, n_patches, d_patch)  # [1, N_patches, D]
    noise = torch.randn(n_views, n_patches, d_patch) * 0.05
    tokens = base.expand(n_views, -1, -1) + noise  # [N_views, N_patches, D]

    return {"tokens": tokens, "image": image, "scene_type": "normal"}


def _make_adverse_sample(
    n_views: int, n_patches: int, d_patch: int, img_h: int, img_w: int
) -> dict[str, torch.Tensor]:
    """Adverse scene: blurry image, inconsistent tokens across views."""
    # Low-variance image → weak gradients
    image = torch.randn(n_views, 3, img_h, img_w) * 0.1  # [N_views, 3, H, W]

    # Inconsistent tokens: fully independent per view
    tokens = torch.randn(n_views, n_patches, d_patch)  # [N_views, N_patches, D]

    return {"tokens": tokens, "image": image, "scene_type": "adverse"}


def _make_random_sample(
    n_views: int, n_patches: int, d_patch: int, img_h: int, img_w: int
) -> dict[str, torch.Tensor]:
    """Random sample with moderate parameters."""
    image = torch.randn(n_views, 3, img_h, img_w)  # [N_views, 3, H, W]
    tokens = torch.randn(n_views, n_patches, d_patch) * 0.5  # [N_views, N_patches, D]
    return {"tokens": tokens, "image": image, "scene_type": "unknown"}

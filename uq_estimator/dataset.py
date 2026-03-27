"""Dataset for loading pre-extracted features and UQ labels."""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.data import Dataset


def compute_stat_features(
    tokens: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    """Compute 5-dim raw statistical features from a single sample.

    Args:
        tokens: [N_views, N_patches, D] patch tokens.
        image:  [N_views, 3, H, W] multi-view images.

    Returns:
        [5] raw statistical features.
    """
    # a. Image gradient magnitude — mean & variance (2 dims)
    # Use Sobel-like finite differences on the grayscale image
    gray = image.mean(dim=1, keepdim=True)  # [N_views, 1, H, W]
    dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]  # [N_views, 1, H, W-1]
    dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]  # [N_views, 1, H-1, W]
    grad_mag = (dx[:, :, :-1, :].pow(2) + dy[:, :, :, :-1].pow(2)).sqrt()  # [N_views, 1, H-1, W-1]
    grad_mean = grad_mag.mean()   # scalar
    grad_var = grad_mag.var()     # scalar

    # b. Patch token activation — mean & variance (2 dims)
    token_mean = tokens.mean()    # scalar
    token_var = tokens.var()      # scalar

    # c. Cross-view token similarity — mean cosine similarity (1 dim)
    # Average token per view, then compute mean pairwise cosine similarity
    view_feats = tokens.mean(dim=1)  # [N_views, D]
    view_feats_norm = torch.nn.functional.normalize(view_feats, dim=-1)  # [N_views, D]
    sim_matrix = view_feats_norm @ view_feats_norm.T  # [N_views, N_views]
    # Mean of upper triangle (excluding diagonal)
    n_views = sim_matrix.size(0)
    mask = torch.triu(torch.ones(n_views, n_views, dtype=torch.bool), diagonal=1)  # [N_views, N_views]
    cross_sim_mean = sim_matrix[mask].mean() if mask.sum() > 0 else torch.tensor(0.0)  # scalar

    return torch.stack([grad_mean, grad_var, token_mean, token_var, cross_sim_mean])  # [5]


class UQFeatureDataset(Dataset):
    """Dataset that loads pre-extracted patch tokens and UQ labels.

    Args:
        feature_dir: directory containing .pt feature files.
        label_file:  path to a .pt file mapping filename → score.
        split:       'train' or 'val'.
        val_ratio:   fraction of data reserved for validation.
        mock:        if True, generate random data instead of reading files.
        mock_size:   number of samples in mock mode.
        n_views:     number of camera views (for mock).
        n_patches:   patches per view (for mock).
        d_patch:     patch token dimension (for mock).
    """

    def __init__(
        self,
        feature_dir: str = "",
        label_file: str = "",
        split: str = "train",
        val_ratio: float = 0.1,
        mock: bool = False,
        mock_size: int = 32,
        n_views: int = 6,
        n_patches: int = 256,
        d_patch: int = 1152,
    ) -> None:
        super().__init__()
        self.mock = mock
        self.n_views = n_views
        self.n_patches = n_patches
        self.d_patch = d_patch

        if mock:
            self.samples = list(range(mock_size))
            n_val = max(1, int(mock_size * val_ratio))
            if split == "val":
                self.samples = self.samples[:n_val]
            else:
                self.samples = self.samples[n_val:]
            return

        # Real mode: load feature files and labels
        self.labels: dict[str, dict] = torch.load(label_file, weights_only=True)
        all_files = sorted(
            f for f in os.listdir(feature_dir) if f.endswith(".pt")
        )
        # Only keep files that have corresponding labels
        all_files = [f for f in all_files if f in self.labels]

        n_val = max(1, int(len(all_files) * val_ratio))
        if split == "val":
            self.file_list = all_files[:n_val]
        else:
            self.file_list = all_files[n_val:]

        self.feature_dir = feature_dir

    def __len__(self) -> int:
        if self.mock:
            return len(self.samples)
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns dict with patch_tokens, stat_features, label, scene_type."""
        if self.mock:
            tokens = torch.randn(self.n_views, self.n_patches, self.d_patch)  # [N_views, N_patches, D]
            image = torch.randn(self.n_views, 3, 224, 224)  # [N_views, 3, H, W]
            stat = compute_stat_features(tokens, image)  # [5]
            label = torch.rand(1)  # [1]
            scene_type = "unknown"
            return {
                "patch_tokens": tokens,
                "stat_features": stat,
                "label": label,
                "scene_type": scene_type,
            }

        fname = self.file_list[idx]
        data = torch.load(
            os.path.join(self.feature_dir, fname), weights_only=True
        )
        tokens = data["tokens"]  # [N_views, N_patches, D]
        image = data["image"]    # [N_views, 3, H, W]
        stat = compute_stat_features(tokens, image)  # [5]
        label = torch.tensor([self.labels[fname]["score"]], dtype=torch.float32)  # [1]
        scene_type = self.labels[fname].get("scene_type", "unknown")

        return {
            "patch_tokens": tokens,
            "stat_features": stat,
            "label": label,
            "scene_type": scene_type,
        }

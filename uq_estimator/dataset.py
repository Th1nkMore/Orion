"""Dataset for loading pre-extracted features and UQ labels."""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.data import Dataset


def compute_stat_features(
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Compute 5-dim raw statistical features from tokens only (no image needed).

    Args:
        tokens: [N_views, N_patches, D] patch tokens (fp16, will convert to fp32).

    Returns:
        [5] raw statistical features:
        a. token各视角激活值方差均值（1维）
        b. token跨patch softmax熵均值（1维）
        c. 跨视角余弦相似度均值（1维）
        d. token激活值绝对均值（1维）
        e. token激活值最大值均值（1维）
    """
    # Convert fp16 to fp32 for computation
    tokens = tokens.float()  # [N_views, N_patches, D]

    # a. 各视角激活值方差均值
    view_var = tokens.var(dim=2).mean(dim=1)  # [N_views] → scalar
    a = view_var.mean()

    # b. 跨patch softmax熵均值
    D = tokens.shape[-1]
    p = torch.softmax(tokens, dim=-1)  # [N_views, N_patches, D]
    entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)  # [N_views, N_patches]
    max_entropy = torch.log(torch.tensor(float(D)))
    b = (entropy.mean() / max_entropy).clamp(0.0, 1.0)

    # c. 跨视角余弦相似度均值
    view_feats = tokens.mean(dim=1)  # [N_views, D]
    view_feats_norm = torch.nn.functional.normalize(view_feats, dim=-1)
    sim_matrix = view_feats_norm @ view_feats_norm.T  # [N_views, N_views]
    n_views = sim_matrix.size(0)
    mask = torch.triu(torch.ones(n_views, n_views, dtype=torch.bool), diagonal=1)
    c = sim_matrix[mask].mean() if mask.sum() > 0 else torch.tensor(0.0)

    # d. token激活值绝对均值
    d = tokens.abs().mean()

    # e. token激活值最大值均值
    e = tokens.abs().amax(dim=-1).mean()  # max over D dimension

    return torch.stack([a, b, c, d, e])  # [5]


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
            stat = compute_stat_features(tokens)  # [5]
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
        tokens = data["tokens"]  # [N_views, N_patches, D] fp16
        stat = compute_stat_features(tokens)  # [5] fp32
        label = torch.tensor([self.labels[fname]["score"]], dtype=torch.float32)  # [1]
        scene_type = self.labels[fname].get("scene_type", "unknown")

        return {
            "patch_tokens": tokens,
            "stat_features": stat,
            "label": label,
            "scene_type": scene_type,
        }

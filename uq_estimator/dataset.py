"""Dataset for loading pre-extracted features and UQ labels."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
from torch.utils.data import Dataset


def compute_stat_features(
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Compute 5-dim raw statistical features from tokens.

    Handles both [N_views, N_patches, D] (single sample) and
    [B, N_views, N_patches, D] (batched) inputs.

    Args:
        tokens: patch tokens, either [N_views, N_patches, D] or [B, N_views, N_patches, D].

    Returns:
        If input is [N_views, N_patches, D]  → returns [5]
        If input is [B, N_views, N_patches, D] → returns [B, 5]
    """
    # Normalize to [B, N_views, N_patches, D] or [1, N_views, N_patches, D]
    tokens = tokens.float()
    single_sample = (tokens.dim() == 3)
    if single_sample:
        tokens = tokens.unsqueeze(0)  # [1, N_views, N_patches, D]
    B, N_v, N_p, D = tokens.shape

    # a. 各视角激活值方差均值
    a = tokens.var(dim=3).mean(dim=(1, 2))  # [B]

    # b. 跨patch softmax熵均值
    p = torch.softmax(tokens, dim=-1)  # [B, N_views, N_patches, D]
    entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)  # [B, N_views, N_patches]
    max_entropy = torch.log(torch.tensor(float(D), device=tokens.device))
    b = (entropy.mean(dim=(1, 2)) / max_entropy).clamp(0.0, 1.0)  # [B]

    # c. 跨视角余弦相似度均值
    view_feats = tokens.mean(dim=2)  # [B, N_views, D]
    view_feats_n = torch.nn.functional.normalize(view_feats, dim=-1)
    sim = torch.bmm(view_feats_n, view_feats_n.transpose(1, 2))  # [B, N_views, N_views]
    mask = torch.triu(torch.ones(N_v, N_v, dtype=torch.bool, device=sim.device), diagonal=1)
    c = sim[:, mask].mean(dim=-1)  # [B]

    # d. token激活值绝对均值
    d = tokens.abs().mean(dim=(1, 2, 3))  # [B]

    # e. token激活值最大值均值
    e = tokens.amax(dim=-1).mean(dim=(1, 2))  # [B]

    stats = torch.stack([a, b, c, d, e], dim=1)  # [B, 5]

    # Only squeeze batch dim if original input was truly [N_v, N_p, D] (3D),
    # NOT when it was [1, N_v, N_p, D] (single-sample batched).
    if single_sample:
        stats = stats.squeeze(0)  # [5]
    return stats


class FastTensorLoader:
    """Single-process batch iterator backed by pre-allocated pinned-memory buffers.

    Avoids Python-level per-sample copy overhead (≈80ms/call on this machine)
    by using a single ``torch.index_select(..., out=buf)`` call per batch,
    writing directly into pre-faulted pinned memory.

    Usage::

        loader = FastTensorLoader(dataset, batch_size=64, shuffle=True, device=device)
        for batch in loader:
            patch_tokens = batch["patch_tokens"]  # already on device
            stat_features = batch["stat_features"]
            labels = batch["label"]

    Args:
        dataset:    a ``UQFeatureDataset`` with ``preload=True``.
        batch_size: number of samples per batch.
        shuffle:    if True, shuffle sample order every epoch.
        device:     target device for GPU tensors.
    """

    def __init__(
        self,
        dataset: "UQFeatureDataset",
        batch_size: int,
        shuffle: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        assert dataset.all_tokens is not None, "FastTensorLoader requires preload=True"
        self.all_tokens = dataset.all_tokens  # [N, V, P_sub, D] fp32
        self.all_stats = dataset.all_stats    # [N, 5] fp32
        self.all_labels = dataset.all_labels  # [N, 1] fp32
        self.N = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        # Pre-allocate and pre-fault pinned output buffers (zero touches pages).
        # All subsequent index_select writes go to already-allocated physical pages
        # → no page faults, no heap pressure during training.
        V, P_sub, D = self.all_tokens.shape[1:]
        self._tok_buf = torch.zeros(batch_size, V, P_sub, D, pin_memory=True)  # [B, V, P_sub, D]
        self._stat_buf = torch.zeros(batch_size, 5, pin_memory=True)            # [B, 5]
        self._lbl_buf = torch.zeros(batch_size, 1, pin_memory=True)             # [B, 1]

    def __len__(self) -> int:
        return (self.N + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = torch.randperm(self.N) if self.shuffle else torch.arange(self.N)
        for start in range(0, self.N, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]  # [B]
            B = len(batch_idx)
            # Three index_select calls — each is a single dispatch (≈80ms overhead total
            # regardless of B) followed by contiguous memory write into pre-faulted buffer.
            torch.index_select(self.all_tokens, 0, batch_idx, out=self._tok_buf[:B])
            torch.index_select(self.all_stats, 0, batch_idx, out=self._stat_buf[:B])
            torch.index_select(self.all_labels, 0, batch_idx, out=self._lbl_buf[:B])
            # Non-blocking H2D transfer — overlaps with GPU compute of previous batch
            yield {
                "patch_tokens": self._tok_buf[:B].to(self.device, non_blocking=True),
                "stat_features": self._stat_buf[:B].to(self.device, non_blocking=True),
                "label": self._lbl_buf[:B].to(self.device, non_blocking=True),
            }


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
        stat_cache_file: path to pre-computed stat_features cache (.pt).
                         Dict mapping filename → [5] tensor.
        preload:     if True, preload all tokens into a single contiguous fp32
                     tensor in RAM. Required for ``FastTensorLoader``.
        preload_workers: parallel threads for filling the preload buffer.
        n_patches_subsample: if > 0, randomly subsample this many patches per view
                     during preload (reduces tensor from [V,1600,D] to [V,P_sub,D]).
                     Stat_features are unaffected (computed from all patches in cache).
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
        stat_cache_file: str = "",
        preload: bool = False,
        preload_workers: int = 16,
        n_patches_subsample: int = 0,
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
            # Expose dummy tensors for FastTensorLoader compatibility
            n = len(self.samples)
            self.all_tokens = torch.zeros(n, n_views, n_patches_subsample or n_patches, d_patch)
            self.all_stats = torch.zeros(n, 5)
            self.all_labels = torch.zeros(n, 1)
            self.scene_types = ["unknown"] * n
            return

        # Real mode: load feature files and labels
        self.labels: dict[str, dict] = torch.load(label_file, weights_only=True)
        all_files = sorted(
            f for f in os.listdir(feature_dir) if f.endswith(".pt")
        )
        all_files = [f for f in all_files if f in self.labels]

        n_val = max(1, int(len(all_files) * val_ratio))
        if split == "val":
            self.file_list = all_files[:n_val]
        else:
            self.file_list = all_files[n_val:]

        self.feature_dir = feature_dir
        N = len(self.file_list)

        # Pre-build scalar tensors
        self.all_labels = torch.tensor(
            [self.labels[f]["score"] for f in self.file_list], dtype=torch.float32
        ).unsqueeze(-1)  # [N, 1]
        self.scene_types = [self.labels[f].get("scene_type", "unknown") for f in self.file_list]

        # Build stat_features tensor from cache
        stat_cache_full: dict[str, torch.Tensor] = {}
        if stat_cache_file and os.path.isfile(stat_cache_file):
            stat_cache_full = torch.load(stat_cache_file, weights_only=True)
        has_cache = all(f in stat_cache_full for f in self.file_list)
        if has_cache:
            self.all_stats = torch.stack([stat_cache_full[f].float() for f in self.file_list])  # [N, 5]
            print(f"stat_cache loaded: {N} entries")
        else:
            self.all_stats = None
            n_miss = sum(1 for f in self.file_list if f not in stat_cache_full)
            print(f"[WARN] stat_cache missing {n_miss}/{N} entries, computing on-the-fly")

        # Preload tokens into single contiguous fp32 tensor
        self.all_tokens: Optional[torch.Tensor] = None
        if preload:
            p_sub = n_patches_subsample if n_patches_subsample > 0 else n_patches
            gb = N * n_views * p_sub * d_patch * 4 / 1e9
            print(f"Allocating preload buffer: {N}×{n_views}×{p_sub}×{d_patch} fp32 ({gb:.1f} GB)...")
            self.all_tokens = torch.empty(N, n_views, p_sub, d_patch, dtype=torch.float32)

            def fill_slot(args: tuple) -> None:
                i, fname = args
                data = torch.load(
                    os.path.join(feature_dir, fname), weights_only=True
                )
                tokens = data["tokens"]  # [N_views, N_patches_full, D] fp16
                if n_patches_subsample > 0 and tokens.shape[1] > n_patches_subsample:
                    # Fixed random subset (deterministic per file by using file index as seed)
                    rng = torch.Generator()
                    rng.manual_seed(i)
                    perm = torch.randperm(tokens.shape[1], generator=rng)[:n_patches_subsample]
                    tokens = tokens[:, perm, :]  # [V, P_sub, D] fp16
                # copy_ into pre-allocated slot (fp16 → fp32 implicitly)
                self.all_tokens[i].copy_(tokens)

            print(f"Filling preload buffer ({preload_workers} threads)...")
            with ThreadPoolExecutor(max_workers=preload_workers) as pool:
                for done, _ in enumerate(pool.map(fill_slot, enumerate(self.file_list))):
                    if (done + 1) % 2000 == 0:
                        print(f"  Filled {done+1}/{N}")
            print(f"Preload complete: {N} samples ({gb:.1f} GB fp32 in RAM)")

    def __len__(self) -> int:
        if self.mock:
            return len(self.samples)
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns dict with patch_tokens [V,P,D], stat_features [5], label [1], scene_type str.

        Note: when preload=True, use FastTensorLoader instead for optimal performance.
        """
        if self.mock:
            return {
                "patch_tokens": self.all_tokens[idx],
                "stat_features": self.all_stats[idx],
                "label": self.all_labels[idx],
                "scene_type": "unknown",
            }

        if self.all_tokens is not None:
            tokens = self.all_tokens[idx]  # view, no copy
        else:
            data = torch.load(
                os.path.join(self.feature_dir, self.file_list[idx]), weights_only=True
            )
            tokens = data["tokens"].float()

        if self.all_stats is not None:
            stat = self.all_stats[idx]
        else:
            stat = compute_stat_features(tokens)

        return {
            "patch_tokens": tokens,
            "stat_features": stat,
            "label": self.all_labels[idx],
            "scene_type": self.scene_types[idx],
        }

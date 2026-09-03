"""Extended tests for dataset.py — stat features, split correctness, dtype handling.

Covers gaps not in test_uq_model.py:
- compute_stat_features: 3D vs 4D input, fp16 input, value ranges
- UQFeatureDataset: val/train split, float32 output dtypes, stat_cache loading
- FastTensorLoader: covers all data, correct batch shapes
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from uq_estimator.dataset import UQFeatureDataset, FastTensorLoader, compute_stat_features
from tests.fixtures import create_mock_feature_dir


# ── compute_stat_features ────────────────────────────────────────────────────

class TestComputeStatFeatures:
    def test_single_sample_output_shape(self):
        """3D input [N_views, N_patches, D] → output [5]."""
        tokens = torch.randn(6, 32, 64)
        stats = compute_stat_features(tokens)
        assert stats.shape == (5,), f"Expected (5,), got {stats.shape}"

    def test_batched_output_shape(self):
        """4D input [B, N_views, N_patches, D] → output [B, 5]."""
        tokens = torch.randn(4, 6, 32, 64)
        stats = compute_stat_features(tokens)
        assert stats.shape == (4, 5), f"Expected (4, 5), got {stats.shape}"

    def test_entropy_feature_in_range(self):
        """Entropy feature (index 1) should be in [0, 1]."""
        tokens = torch.randn(6, 16, 32)
        stats = compute_stat_features(tokens)
        entropy = stats[1].item()
        assert 0.0 <= entropy <= 1.0, f"Entropy {entropy} out of [0, 1]"

    def test_entropy_batched_in_range(self):
        """All entropy values in batch should be in [0, 1]."""
        tokens = torch.randn(8, 6, 16, 32)
        stats = compute_stat_features(tokens)  # [8, 5]
        entropy_vals = stats[:, 1]
        assert (entropy_vals >= 0.0).all() and (entropy_vals <= 1.0).all(), \
            f"Entropy out of range: min={entropy_vals.min():.4f}, max={entropy_vals.max():.4f}"

    def test_fp16_input_returns_fp32(self):
        """fp16 token input should be auto-cast to fp32 internally and return fp32."""
        tokens = torch.randn(6, 16, 32).half()  # fp16
        stats = compute_stat_features(tokens)
        assert stats.dtype == torch.float32, f"Expected float32 output, got {stats.dtype}"

    def test_output_is_finite(self):
        """No NaN or Inf in stat features for normal random tokens."""
        tokens = torch.randn(4, 6, 16, 64)
        stats = compute_stat_features(tokens)
        assert torch.isfinite(stats).all(), "stat_features contain NaN or Inf"

    def test_different_views_affect_cosine_sim(self):
        """Cosine similarity (index 2) should be higher for identical views."""
        # Identical views → cosine_sim close to 1
        base = torch.randn(1, 8, 32)
        identical = base.expand(6, -1, -1)
        # Random views → lower cosine_sim
        random = torch.randn(6, 8, 32)

        stats_identical = compute_stat_features(identical)
        stats_random = compute_stat_features(random)
        # Identical views should have higher cosine similarity (index 2)
        assert stats_identical[2].item() > stats_random[2].item(), (
            f"Identical views should have higher cosine sim: "
            f"identical={stats_identical[2]:.3f}, random={stats_random[2]:.3f}"
        )


# ── UQFeatureDataset split correctness ──────────────────────────────────────

class TestDatasetSplit:
    def test_val_train_no_overlap(self, tmp_path: Path):
        """val and train sets must not share file indices."""
        feature_dir = create_mock_feature_dir(tmp_path, n_files=20)
        labels = {}
        for f in feature_dir.iterdir():
            if f.suffix == ".pt":
                labels[f.name] = {"score": 0.5, "scene_type": "unknown"}
        label_file = tmp_path / "labels.pt"
        torch.save(labels, str(label_file))

        val_ds = UQFeatureDataset(
            feature_dir=str(feature_dir),
            label_file=str(label_file),
            split="val",
            val_ratio=0.2,
        )
        train_ds = UQFeatureDataset(
            feature_dir=str(feature_dir),
            label_file=str(label_file),
            split="train",
            val_ratio=0.2,
        )
        val_files = set(val_ds.file_list)
        train_files = set(train_ds.file_list)
        assert val_files.isdisjoint(train_files), (
            f"Overlap between val and train: {val_files & train_files}"
        )

    def test_val_train_cover_all_files(self, tmp_path: Path):
        """val + train should cover all available feature files."""
        feature_dir = create_mock_feature_dir(tmp_path, n_files=20)
        labels = {}
        for f in feature_dir.iterdir():
            if f.suffix == ".pt":
                labels[f.name] = {"score": 0.5, "scene_type": "unknown"}
        label_file = tmp_path / "labels.pt"
        torch.save(labels, str(label_file))

        val_ds = UQFeatureDataset(
            feature_dir=str(feature_dir),
            label_file=str(label_file),
            split="val",
            val_ratio=0.2,
        )
        train_ds = UQFeatureDataset(
            feature_dir=str(feature_dir),
            label_file=str(label_file),
            split="train",
            val_ratio=0.2,
        )
        total = len(val_ds) + len(train_ds)
        assert total == 20, f"Expected 20 total samples, got {total}"

    def test_val_size_respects_ratio(self, tmp_path: Path):
        """val dataset size ≈ val_ratio * N (at least 1)."""
        n_files = 30
        feature_dir = create_mock_feature_dir(tmp_path, n_files=n_files)
        labels = {
            f.name: {"score": 0.5, "scene_type": "unknown"}
            for f in feature_dir.iterdir() if f.suffix == ".pt"
        }
        label_file = tmp_path / "labels.pt"
        torch.save(labels, str(label_file))

        val_ds = UQFeatureDataset(
            feature_dir=str(feature_dir),
            label_file=str(label_file),
            split="val",
            val_ratio=0.1,
        )
        expected_val = max(1, int(n_files * 0.1))
        assert len(val_ds) == expected_val, (
            f"Expected {expected_val} val samples (ratio=0.1), got {len(val_ds)}"
        )


# ── UQFeatureDataset __getitem__ dtype ───────────────────────────────────────

class TestDatasetDtype:
    def test_real_dataset_returns_float32_tokens(self, tmp_path: Path):
        """__getitem__ must return float32 patch_tokens regardless of on-disk dtype."""
        feature_dir = tmp_path / "feats"
        feature_dir.mkdir()
        # Save 5 fp16 token files (as they'd come from EVAViT)
        labels = {}
        for i in range(5):
            fname = f"sample_{i:04d}.pt"
            torch.save({"tokens": torch.randn(6, 32, 64).half()},
                       str(feature_dir / fname))
            labels[fname] = {"score": 0.5, "scene_type": "unknown"}

        label_file = tmp_path / "labels.pt"
        torch.save(labels, str(label_file))

        ds = UQFeatureDataset(
            feature_dir=str(feature_dir),
            label_file=str(label_file),
            split="train",
            val_ratio=0.2,  # 1 val, 4 train
        )
        assert len(ds) > 0, "Train dataset should be non-empty"
        item = ds[0]
        assert item["patch_tokens"].dtype == torch.float32, (
            f"Expected float32, got {item['patch_tokens'].dtype}"
        )

    def test_mock_dataset_returns_float32(self):
        """Mock dataset __getitem__ should return float32 tensors."""
        ds = UQFeatureDataset(mock=True, mock_size=8)
        item = ds[0]
        assert item["patch_tokens"].dtype == torch.float32
        assert item["stat_features"].dtype == torch.float32
        assert item["label"].dtype == torch.float32


# ── stat_cache_file ───────────────────────────────────────────────────────────

class TestStatCache:
    def test_cache_loaded_when_present(self, tmp_path: Path):
        """When stat_cache_file is valid and complete, all_stats is set from cache."""
        feature_dir = create_mock_feature_dir(tmp_path, n_files=10)
        labels = {
            f.name: {"score": 0.5, "scene_type": "unknown"}
            for f in sorted(feature_dir.iterdir()) if f.suffix == ".pt"
        }
        label_file = tmp_path / "labels.pt"
        torch.save(labels, str(label_file))

        # Build a stat cache with known values
        stat_cache = {fname: torch.ones(5) * 0.42 for fname in labels}
        cache_file = tmp_path / "stat_cache.pt"
        torch.save(stat_cache, str(cache_file))

        ds = UQFeatureDataset(
            feature_dir=str(feature_dir),
            label_file=str(label_file),
            split="train",
            val_ratio=0.1,
            stat_cache_file=str(cache_file),
        )
        assert ds.all_stats is not None, "all_stats should be loaded from cache"
        # All stat values should be 0.42 (from our synthetic cache)
        assert torch.allclose(ds.all_stats, torch.full_like(ds.all_stats, 0.42)), \
            "all_stats should match cache values"

    def test_missing_cache_computes_on_the_fly(self, tmp_path: Path):
        """When stat_cache_file is absent, all_stats should be None (computed per-batch)."""
        feature_dir = create_mock_feature_dir(tmp_path, n_files=5)
        labels = {
            f.name: {"score": 0.5, "scene_type": "unknown"}
            for f in sorted(feature_dir.iterdir()) if f.suffix == ".pt"
        }
        label_file = tmp_path / "labels.pt"
        torch.save(labels, str(label_file))

        ds = UQFeatureDataset(
            feature_dir=str(feature_dir),
            label_file=str(label_file),
            split="train",
            val_ratio=0.0,
            stat_cache_file="",  # no cache
        )
        # Without cache, all_stats is None → compute_stat_features called in __getitem__
        assert ds.all_stats is None, "all_stats should be None without cache"
        # __getitem__ should still work (computes on-the-fly)
        if len(ds) > 0:
            item = ds[0]
            assert item["stat_features"].shape == (5,)


# ── FastTensorLoader ─────────────────────────────────────────────────────────
# FastTensorLoader uses pin_memory which requires CUDA. Tests are skipped on CPU.
_CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="FastTensorLoader requires CUDA (pin_memory)")
class TestFastTensorLoader:
    def test_covers_all_samples(self):
        """All N samples should appear exactly once per epoch (shuffle=False)."""
        N = 20
        ds = UQFeatureDataset(mock=True, mock_size=N, split="train", val_ratio=0.0)
        # Ensure all_labels has distinct values for tracking
        ds.all_labels = torch.arange(N, dtype=torch.float32).unsqueeze(1)
        ds.all_stats = torch.zeros(N, 5)

        loader = FastTensorLoader(ds, batch_size=5, shuffle=False, device=None)
        seen_labels = []
        for batch in loader:
            seen_labels.extend(batch["label"].squeeze(-1).tolist())

        assert len(seen_labels) == N, f"Expected {N} samples, got {len(seen_labels)}"
        assert sorted(seen_labels) == list(range(N)), "Not all samples appeared exactly once"

    def test_batch_shapes(self):
        """Each batch has correct shapes for patch_tokens, stat_features, label."""
        N, B = 16, 4
        n_views, n_patches, d = 6, 32, 64
        ds = UQFeatureDataset(mock=True, mock_size=N, split="train", val_ratio=0.0,
                              n_views=n_views, n_patches=n_patches, d_patch=d)
        loader = FastTensorLoader(ds, batch_size=B, shuffle=False, device=None)
        batch = next(iter(loader))
        assert batch["patch_tokens"].shape == (B, n_views, n_patches, d), \
            f"Unexpected tokens shape: {batch['patch_tokens'].shape}"
        assert batch["stat_features"].shape == (B, 5), \
            f"Unexpected stat shape: {batch['stat_features'].shape}"
        assert batch["label"].shape == (B, 1), \
            f"Unexpected label shape: {batch['label'].shape}"

    def test_last_batch_smaller(self):
        """When N % batch_size != 0, last batch has fewer samples."""
        N, B = 10, 3  # 3 full batches + 1 partial (1 sample)
        ds = UQFeatureDataset(mock=True, mock_size=N, split="train", val_ratio=0.0)
        loader = FastTensorLoader(ds, batch_size=B, shuffle=False, device=None)
        batches = list(loader)
        last_batch_size = batches[-1]["label"].shape[0]
        expected_last = N % B if N % B != 0 else B
        assert last_batch_size == expected_last, (
            f"Expected last batch size {expected_last}, got {last_batch_size}"
        )

    def test_requires_preload(self):
        """FastTensorLoader raises if all_tokens is None (preload=False in real mode)."""
        ds = UQFeatureDataset(mock=True, mock_size=4, split="train", val_ratio=0.0)
        ds.all_tokens = None  # Simulate non-preloaded state
        with pytest.raises(AssertionError):
            FastTensorLoader(ds, batch_size=2)


@pytest.mark.skipif(_CUDA_AVAILABLE, reason="Only test on CPU (no CUDA)")
class TestFastTensorLoaderNoCuda:
    def test_requires_preload_cpu(self):
        """FastTensorLoader raises AssertionError if all_tokens is None."""
        ds = UQFeatureDataset(mock=True, mock_size=4, split="train", val_ratio=0.0)
        ds.all_tokens = None
        with pytest.raises(AssertionError):
            FastTensorLoader(ds, batch_size=2)

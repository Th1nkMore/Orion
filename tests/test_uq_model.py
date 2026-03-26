"""Tests for UQ Estimator model, losses, and dataset."""

import torch
import pytest

from uq_estimator.model import UQEstimator, UQOutput, D_STAT_RAW
from uq_estimator.losses import CombinedUQLoss
from uq_estimator.dataset import UQFeatureDataset


# Mock config matching configs/uq_train.yaml model section
MODEL_CFG = {
    "d_patch": 1152,
    "n_views": 6,
    "n_patches": 256,
    "n_query": 16,
    "d_stat": 64,
    "d_hidden": 512,
    "d_out": 256,
}

B = 2
N_VIEWS = MODEL_CFG["n_views"]
N_PATCHES = MODEL_CFG["n_patches"]
D = MODEL_CFG["d_patch"]
D_OUT = MODEL_CFG["d_out"]


@pytest.fixture
def model():
    return UQEstimator(MODEL_CFG)


@pytest.fixture
def mock_inputs():
    patch_tokens = torch.randn(B, N_VIEWS, N_PATCHES, D)  # [B, N_views, N_patches, D]
    stat_features = torch.randn(B, D_STAT_RAW)  # [B, 5]
    return patch_tokens, stat_features


def test_output_shape(model, mock_inputs):
    """Verify embedding [B, 256] and score [B, 1] shapes."""
    patch_tokens, stat_features = mock_inputs
    with torch.no_grad():
        out = model(patch_tokens, stat_features)
    assert out.embedding.shape == (B, D_OUT), f"Expected ({B}, {D_OUT}), got {out.embedding.shape}"
    assert out.score.shape == (B, 1), f"Expected ({B}, 1), got {out.score.shape}"


def test_score_range(model, mock_inputs):
    """Verify score values are in [0, 1]."""
    patch_tokens, stat_features = mock_inputs
    with torch.no_grad():
        out = model(patch_tokens, stat_features)
    assert out.score.min() >= 0.0, f"Score min {out.score.min()} < 0"
    assert out.score.max() <= 1.0, f"Score max {out.score.max()} > 1"


def test_multi_view(model, mock_inputs):
    """Verify the model works with N_views=6."""
    patch_tokens, stat_features = mock_inputs
    assert patch_tokens.shape[1] == 6
    with torch.no_grad():
        out = model(patch_tokens, stat_features)
    assert isinstance(out, UQOutput)
    assert out.embedding.shape == (B, D_OUT)


def test_loss_dict_keys():
    """CombinedUQLoss returns dict with all expected keys."""
    loss_fn = CombinedUQLoss()
    pred = torch.sigmoid(torch.randn(B, 1))  # [B, 1]
    target = torch.rand(B, 1)  # [B, 1]
    result = loss_fn(pred, target)
    expected_keys = {"total", "regression", "ranking", "calibration"}
    assert set(result.keys()) == expected_keys, f"Got keys {set(result.keys())}"
    for k, v in result.items():
        assert v.dim() == 0, f"Loss '{k}' should be scalar, got shape {v.shape}"


def test_dataset_mock():
    """UQFeatureDataset(mock=True) can produce items."""
    ds = UQFeatureDataset(mock=True, mock_size=8, val_ratio=0.1)
    sample = ds[0]
    assert "patch_tokens" in sample
    assert "stat_features" in sample
    assert "label" in sample
    assert sample["patch_tokens"].shape == (6, 256, 1152)
    assert sample["stat_features"].shape == (5,)
    assert sample["label"].shape == (1,)


def test_parameter_count(model):
    """UQEstimator must have fewer than 5M parameters."""
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 5_000_000, f"Parameter count {n_params:,} exceeds 5M limit"

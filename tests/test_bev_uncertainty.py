"""Tests for BEV uncertainty module.

Verifies:
- Patch quality shape, range, and blur sensitivity
- BEV uncertainty shape, range, and uniform-attention property
- Trajectory cost shape and spatial sensitivity
- Mode score adjustment identity and monotonicity
- Heatmap rendering output
"""

import torch
import pytest

from uq_estimator.bev_uncertainty import (
    compute_patch_quality,
    compute_bev_uncertainty,
    compute_trajectory_cost,
    adjust_mode_scores,
    render_bev_heatmap,
)

# ── Mock dimensions ──
B = 2
N_VIEWS = 6
N_PATCHES = 256  # mock patches (real = 1600)
N_Q = 600
N_MODES = 20
N_STEPS = 6


# ── Patch Quality Tests ─────────────────────────────────────────────


def test_patch_quality_shape():
    """Output shape matches [B, N_views, H_p * W_p]."""
    images = torch.rand(B, N_VIEWS, 3, 640, 640)
    pq = compute_patch_quality(images, patch_size=16)
    assert pq.shape == (B, N_VIEWS, 1600)


def test_patch_quality_range():
    """Output values in [0, 1] after normalisation."""
    images = torch.rand(B, N_VIEWS, 3, 128, 128)
    pq = compute_patch_quality(images, patch_size=16)
    assert pq.min() >= 0.0
    assert pq.max() <= 1.0 + 1e-6


def test_patch_quality_blur_sensitivity():
    """Blurred images should have lower mean quality than sharp ones."""
    # Sharp: high-frequency random noise
    sharp = torch.rand(1, 1, 3, 128, 128) * 255
    # Blurred: uniform color (zero texture/gradient/contrast)
    blurred = torch.ones(1, 1, 3, 128, 128) * 128

    pq_sharp = compute_patch_quality(sharp, patch_size=16)
    pq_blur = compute_patch_quality(blurred, patch_size=16)
    # Sharp should have higher quality on average
    assert pq_sharp.mean() > pq_blur.mean(), \
        f"Sharp quality ({pq_sharp.mean():.4f}) should > blur ({pq_blur.mean():.4f})"


# ── BEV Uncertainty Tests ───────────────────────────────────────────


@pytest.fixture
def mock_bev_inputs():
    pq = torch.rand(B, N_VIEWS, N_PATCHES)
    attn = torch.softmax(torch.randn(B, N_Q, N_VIEWS * N_PATCHES), dim=-1)
    return attn, pq


def test_bev_uncertainty_shape(mock_bev_inputs):
    attn, pq = mock_bev_inputs
    bev = compute_bev_uncertainty(attn, pq)
    assert bev.shape == (B, N_Q)


def test_bev_uncertainty_range(mock_bev_inputs):
    attn, pq = mock_bev_inputs
    bev = compute_bev_uncertainty(attn, pq)
    # bev_unc is a weighted average of values in [0, 1] → should be in [0, 1]
    assert bev.min() >= -1e-6
    assert bev.max() <= 1.0 + 1e-6


def test_bev_uncertainty_uniform_attn():
    """With uniform attention, bev_unc ≈ mean(1 - patch_quality)."""
    N_total = N_VIEWS * N_PATCHES
    pq = torch.rand(B, N_VIEWS, N_PATCHES)
    attn = torch.ones(B, N_Q, N_total) / N_total  # uniform

    bev = compute_bev_uncertainty(attn, pq)
    expected = (1.0 - pq.reshape(B, -1)).mean(dim=-1, keepdim=True)  # [B, 1]
    expected = expected.expand(B, N_Q)

    torch.testing.assert_close(bev, expected, atol=1e-5, rtol=1e-5)


# ── Trajectory Cost Tests ───────────────────────────────────────────


@pytest.fixture
def mock_traj_inputs():
    bev_unc = torch.rand(B, N_Q)
    ref_xy = torch.randn(N_Q, 2) * 30
    waypoints = torch.randn(B, N_MODES, N_STEPS, 2) * 20
    return bev_unc, ref_xy, waypoints


def test_trajectory_cost_shape(mock_traj_inputs):
    bev_unc, ref_xy, waypoints = mock_traj_inputs
    tc = compute_trajectory_cost(bev_unc, ref_xy, waypoints, k=5)
    assert tc.shape == (B, N_MODES)


def test_trajectory_cost_spatial_sensitivity():
    """Trajectory through high-uncertainty region should have higher cost."""
    N_q_small = 100
    bev_unc = torch.zeros(1, N_q_small)
    # Place a cluster of high-uncertainty queries at (10, 10)
    ref_xy = torch.randn(N_q_small, 2) * 30
    ref_xy[:10] = torch.tensor([10.0, 10.0])
    bev_unc[0, :10] = 1.0  # high uncertainty cluster

    # Mode 0: passes through (10, 10) — should have higher cost
    wp_high = torch.zeros(1, 2, N_STEPS, 2)
    wp_high[0, 0, :, 0] = torch.linspace(5, 15, N_STEPS)   # x: 5→15
    wp_high[0, 0, :, 1] = torch.linspace(5, 15, N_STEPS)   # y: 5→15
    # Mode 1: far from cluster at (-20, -20)
    wp_high[0, 1, :, 0] = torch.linspace(-25, -15, N_STEPS)
    wp_high[0, 1, :, 1] = torch.linspace(-25, -15, N_STEPS)

    tc = compute_trajectory_cost(bev_unc, ref_xy, wp_high, k=5)
    assert tc[0, 0] > tc[0, 1], \
        f"Mode through high-unc ({tc[0,0]:.4f}) should > mode far away ({tc[0,1]:.4f})"


# ── Adjust Mode Scores Tests ────────────────────────────────────────


def test_adjust_scores_zero_score_identity():
    """score=0 → adjusted == original (no modulation)."""
    poses_cls = torch.softmax(torch.randn(B, N_MODES), dim=-1)
    traj_cost = torch.rand(B, N_MODES)
    score = torch.zeros(B, 1)
    adjusted = adjust_mode_scores(poses_cls, traj_cost, score, lambda_val=0.5)
    torch.testing.assert_close(adjusted, poses_cls)


def test_adjust_scores_reduces_high_cost_mode():
    """High-cost mode should have lower score after adjustment."""
    poses_cls = torch.ones(B, N_MODES) * 0.5  # uniform scores
    traj_cost = torch.zeros(B, N_MODES)
    traj_cost[:, 0] = 1.0  # mode 0 is expensive
    score = torch.ones(B, 1)  # full uncertainty

    adjusted = adjust_mode_scores(poses_cls, traj_cost, score, lambda_val=0.5)
    # Mode 0 should be penalised more than others
    assert adjusted[0, 0] < adjusted[0, 1], \
        f"High-cost mode ({adjusted[0,0]:.4f}) should < low-cost ({adjusted[0,1]:.4f})"


def test_adjust_scores_preserves_order_uniform_cost():
    """With uniform cost, argmax should not change."""
    poses_cls = torch.tensor([[0.1, 0.3, 0.6]])
    traj_cost = torch.ones(1, 3) * 0.5
    score = torch.tensor([[0.5]])
    adjusted = adjust_mode_scores(poses_cls, traj_cost, score, lambda_val=0.3)
    assert adjusted.argmax(dim=-1) == poses_cls.argmax(dim=-1)


# ── Heatmap Rendering Test ──────────────────────────────────────────


def test_render_heatmap_returns_image():
    """Render should return a valid RGB numpy array."""
    bev_unc = torch.rand(N_Q)
    ref_xy = torch.randn(N_Q, 2) * 30
    img = render_bev_heatmap(bev_unc, ref_xy)
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.dtype == "uint8"

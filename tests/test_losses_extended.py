"""Extended tests for loss functions — edge cases and gradient flow.

Covers gaps not addressed in test_uq_model.py:
- Ranking loss with no valid pairs (uniform labels)
- Ranking loss pair ordering correctness
- Calibration term magnitude when std → 0
- Gradient flow through every loss component
- CombinedUQLoss weight parameters correctly scale components
"""

from __future__ import annotations

import torch
import pytest

from uq_estimator.losses import UQRegressionLoss, UQRankingLoss, CombinedUQLoss


# ── UQRegressionLoss ─────────────────────────────────────────────────────────

class TestUQRegressionLoss:
    def test_regression_zero_on_perfect_pred(self):
        """MSE = 0 when pred == target."""
        loss_fn = UQRegressionLoss(lambda_cal=0.0)
        target = torch.tensor([[0.3], [0.7]])
        reg, cal = loss_fn(target.clone(), target)
        assert reg.item() < 1e-6, f"Expected near-zero MSE, got {reg.item()}"

    def test_calibration_large_when_std_zero(self):
        """When all predictions are identical (std→0), calibration term blows up."""
        loss_fn = UQRegressionLoss(lambda_cal=1.0)
        pred = torch.full((8, 1), 0.5)  # zero std
        target = torch.rand(8, 1)
        _, cal = loss_fn(pred, target)
        # 1 / (0 + 1e-6) = 1_000_000 → lambda_cal * cal should be large
        assert cal.item() > 1.0, f"Calibration should be large when std=0, got {cal.item()}"

    def test_calibration_small_when_spread_high(self):
        """Calibration term is small when predictions are spread out."""
        loss_fn = UQRegressionLoss(lambda_cal=0.1)
        # Full range from 0 to 1 → std ≈ 0.29
        pred = torch.linspace(0.0, 1.0, 16).unsqueeze(1)
        target = torch.rand(16, 1)
        _, cal = loss_fn(pred, target)
        # 0.1 / 0.29 ≈ 0.34 — definitely < 1.0
        assert cal.item() < 1.0, f"Calibration should be small when spread is high, got {cal.item()}"

    def test_gradient_flows_through_regression(self):
        """backward() flows through regression loss to upstream parameters."""
        import torch.nn as nn
        loss_fn = UQRegressionLoss()
        linear = nn.Linear(5, 1)
        x = torch.randn(4, 5)
        pred = torch.sigmoid(linear(x))
        target = torch.rand(4, 1)
        reg, cal = loss_fn(pred, target)
        (reg + cal).backward()
        assert linear.weight.grad is not None
        assert not linear.weight.grad.isnan().any()

    def test_lambda_cal_zero_disables_calibration(self):
        """lambda_cal=0 makes calibration term exactly zero."""
        loss_fn = UQRegressionLoss(lambda_cal=0.0)
        pred = torch.full((4, 1), 0.5)  # std=0, would explode without lambda_cal=0
        target = torch.rand(4, 1)
        _, cal = loss_fn(pred, target)
        assert cal.item() == 0.0


# ── UQRankingLoss ────────────────────────────────────────────────────────────

class TestUQRankingLoss:
    def test_no_pairs_returns_zero_with_grad(self):
        """Uniform labels → no valid pairs → loss=0.0 but requires_grad=True."""
        loss_fn = UQRankingLoss()
        pred = torch.sigmoid(torch.randn(4, 1, requires_grad=True))
        target = torch.full((4, 1), 0.5)  # all equal → no pairs
        loss = loss_fn(pred, target)
        assert loss.item() == 0.0, f"Expected 0.0 with uniform labels, got {loss.item()}"
        assert loss.requires_grad, "Loss should have requires_grad=True even when 0"

    def test_correct_ordering_reduces_loss(self):
        """When predictions already match target ordering, loss should be near 0."""
        loss_fn = UQRankingLoss(margin=0.05)
        # pred_i > pred_j whenever target_i > target_j, with large margin
        target = torch.tensor([[0.1], [0.3], [0.7], [0.9]])
        pred = torch.tensor([[0.1], [0.3], [0.7], [0.9]])
        loss = loss_fn(pred, target)
        assert loss.item() < 0.1, f"Well-ordered predictions should have low loss, got {loss.item()}"

    def test_reversed_ordering_gives_high_loss(self):
        """When predictions are exactly reversed from target, loss is maximised."""
        loss_fn = UQRankingLoss(margin=0.1)
        target = torch.tensor([[0.1], [0.9]])  # i=1 should rank higher
        pred = torch.tensor([[0.9], [0.1]])     # but model predicts opposite
        loss = loss_fn(pred, target)
        assert loss.item() > 0.5, f"Reversed ordering should yield high loss, got {loss.item()}"

    def test_pair_count_correctness(self):
        """B=4 with distinct labels → 4*3/2 = 6 ordered pairs for strict ordering."""
        # Target [0.1, 0.4, 0.7, 0.9] has 6 pairs where t_i > t_j
        # We verify loss is finite and non-zero (since pred is random)
        loss_fn = UQRankingLoss()
        target = torch.tensor([[0.1], [0.4], [0.7], [0.9]])
        pred = torch.tensor([[0.5], [0.5], [0.5], [0.5]])  # tied pred → all pairs at margin
        loss = loss_fn(pred, target)
        assert loss.item() > 0.0
        assert torch.isfinite(loss)

    def test_gradient_flows_through_ranking(self):
        """backward() flows through ranking loss to upstream parameters."""
        import torch.nn as nn
        loss_fn = UQRankingLoss()
        linear = nn.Linear(5, 1)
        x = torch.randn(4, 5)
        pred = torch.sigmoid(linear(x))
        target = torch.tensor([[0.1], [0.3], [0.7], [0.9]])
        loss = loss_fn(pred, target)
        loss.backward()
        assert linear.weight.grad is not None
        assert not linear.weight.grad.isnan().any()

    def test_gradient_flows_when_no_pairs(self):
        """backward() works even when no pairs exist (loss=0 with requires_grad)."""
        loss_fn = UQRankingLoss()
        pred = torch.sigmoid(torch.randn(4, 1, requires_grad=True))
        target = torch.full((4, 1), 0.5)
        loss = loss_fn(pred, target)
        # Should not raise
        loss.backward()
        # pred.grad will be None (no actual computation graph), which is OK
        # The important thing is backward() doesn't throw


# ── CombinedUQLoss ───────────────────────────────────────────────────────────

class TestCombinedUQLoss:
    def test_all_keys_present_and_scalar(self):
        """Forward returns dict with correct keys, all scalar tensors."""
        loss_fn = CombinedUQLoss()
        pred = torch.sigmoid(torch.randn(4, 1))
        target = torch.rand(4, 1)
        result = loss_fn(pred, target)
        for key in ("total", "regression", "ranking", "calibration"):
            assert key in result
            assert result[key].dim() == 0, f"'{key}' should be scalar"

    def test_all_losses_finite(self):
        """No NaN or Inf in loss components with normal inputs."""
        loss_fn = CombinedUQLoss()
        pred = torch.sigmoid(torch.randn(8, 1))
        target = torch.rand(8, 1)
        result = loss_fn(pred, target)
        for key, val in result.items():
            assert torch.isfinite(val), f"'{key}' is not finite: {val.item()}"

    def test_lambda_rank_zero_zeroes_ranking(self):
        """lambda_rank=0 makes ranking component zero in total."""
        loss_fn_noreg = CombinedUQLoss(lambda_rank=0.0)
        loss_fn_reg = CombinedUQLoss(lambda_rank=1.0)
        pred = torch.sigmoid(torch.randn(4, 1))
        target = torch.tensor([[0.1], [0.4], [0.7], [0.9]])
        r_noreg = loss_fn_noreg(pred, target)
        r_reg = loss_fn_reg(pred, target)
        # Both should have same regression + calibration, different total
        assert abs(r_noreg["regression"].item() - r_reg["regression"].item()) < 1e-5
        # Noreg total == noreg_reg + noreg_cal
        expected = r_noreg["regression"] + r_noreg["calibration"]
        assert abs(r_noreg["total"].item() - expected.item()) < 1e-5

    def test_gradient_flows_through_combined_loss(self):
        """backward() flows through total loss to model parameters."""
        import torch.nn as nn
        loss_fn = CombinedUQLoss()
        linear = nn.Linear(5, 1)
        x = torch.randn(8, 5)
        pred = torch.sigmoid(linear(x))
        target = torch.rand(8, 1)
        result = loss_fn(pred, target)
        result["total"].backward()
        assert linear.weight.grad is not None
        assert not linear.weight.grad.isnan().any()

    def test_total_equals_sum_of_components(self):
        """total = regression + calibration + lambda_rank * ranking."""
        lambda_rank = 0.3
        loss_fn = CombinedUQLoss(lambda_rank=lambda_rank)
        pred = torch.sigmoid(torch.randn(8, 1))
        target = torch.tensor([[0.0], [0.1], [0.2], [0.5], [0.6], [0.7], [0.8], [0.9]])
        result = loss_fn(pred, target)
        expected_total = (
            result["regression"] + result["calibration"] + lambda_rank * result["ranking"]
        )
        assert abs(result["total"].item() - expected_total.item()) < 1e-5

    def test_batch_size_one_calibration_nan(self):
        """B=1 → torch.std() returns NaN (Bessel correction, n-1=0) → calibration=NaN.

        This is a known edge case: the training script uses batch_size>=4, so this
        never occurs in practice. This test documents the failure mode so it is not
        accidentally introduced.
        """
        import math
        loss_fn = CombinedUQLoss()
        pred = torch.sigmoid(torch.randn(1, 1))
        target = torch.rand(1, 1)
        result = loss_fn(pred, target)
        # calibration = lambda_cal / (std + eps). With B=1, std() = NaN in PyTorch.
        # Verify this is NaN (not silently passing), so we know to guard against B=1.
        assert math.isnan(result["calibration"].item()), (
            "Expected calibration NaN for B=1 (std() undefined). "
            "If this test fails, PyTorch behaviour changed — re-evaluate."
        )

"""Loss functions for UQ Estimator training."""

from __future__ import annotations

import torch
import torch.nn as nn


class UQRegressionLoss(nn.Module):
    """MSE loss with calibration regulariser.

    The calibration term penalises low standard deviation of predicted scores,
    preventing the model from collapsing all outputs to the batch mean.

    Args:
        lambda_cal: weight for the calibration regularisation term.
    """

    def __init__(self, lambda_cal: float = 0.1) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_cal = lambda_cal

    def forward(
        self,
        pred_score: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pred_score: [B, 1] predicted uncertainty scores.
            target:     [B, 1] pseudo-labels.

        Returns:
            (regression_loss, calibration_loss) — both scalar tensors.
        """
        reg_loss = self.mse(pred_score, target)  # scalar

        # Calibration: encourage spread — penalise 1/(std + eps)
        score_std = pred_score.std()  # scalar
        cal_loss = 1.0 / (score_std + 1e-6)  # scalar

        return reg_loss, self.lambda_cal * cal_loss


class UQRankingLoss(nn.Module):
    """Pairwise margin ranking loss.

    For every pair (i, j) in the batch where target_i > target_j,
    enforces pred_i > pred_j with a margin.

    Args:
        margin: minimum score gap enforced between ordered pairs.
    """

    def __init__(self, margin: float = 0.1) -> None:
        super().__init__()
        self.margin_loss = nn.MarginRankingLoss(margin=margin)

    def forward(
        self,
        pred_score: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_score: [B, 1] predicted uncertainty scores.
            target:     [B, 1] pseudo-labels.

        Returns:
            Scalar ranking loss.
        """
        pred_flat = pred_score.squeeze(-1)  # [B]
        target_flat = target.squeeze(-1)  # [B]

        # Build all (i, j) pairs where target_i > target_j
        # target_sign: +1 means first should rank higher
        idx_i, idx_j = [], []
        B = pred_flat.size(0)
        for i in range(B):
            for j in range(B):
                if target_flat[i] > target_flat[j]:
                    idx_i.append(i)
                    idx_j.append(j)

        if len(idx_i) == 0:
            return torch.tensor(0.0, device=pred_score.device, requires_grad=True)

        idx_i = torch.tensor(idx_i, device=pred_score.device)  # [N_pairs]
        idx_j = torch.tensor(idx_j, device=pred_score.device)  # [N_pairs]

        pred_i = pred_flat[idx_i]  # [N_pairs]
        pred_j = pred_flat[idx_j]  # [N_pairs]
        labels = torch.ones_like(pred_i)  # [N_pairs] — first should be larger

        return self.margin_loss(pred_i, pred_j, labels)  # scalar


class CombinedUQLoss(nn.Module):
    """Combined regression + ranking loss for UQ training.

    Args:
        lambda_cal:  calibration regularisation weight.
        lambda_rank: ranking loss weight.
    """

    def __init__(
        self,
        lambda_cal: float = 0.1,
        lambda_rank: float = 0.5,
    ) -> None:
        super().__init__()
        self.regression = UQRegressionLoss(lambda_cal=lambda_cal)
        self.ranking = UQRankingLoss()
        self.lambda_rank = lambda_rank

    def forward(
        self,
        pred_score: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred_score: [B, 1] predicted uncertainty scores.
            target:     [B, 1] pseudo-labels.

        Returns:
            Dict with keys: 'total', 'regression', 'ranking', 'calibration'.
        """
        reg_loss, cal_loss = self.regression(pred_score, target)
        rank_loss = self.ranking(pred_score, target)

        total = reg_loss + cal_loss + self.lambda_rank * rank_loss  # scalar

        return {
            "total": total,
            "regression": reg_loss,
            "ranking": rank_loss,
            "calibration": cal_loss,
        }

"""Validation script for a trained UQ Estimator.

Generates score distribution plots and a text report.

Usage:
    python scripts/validate_uq.py --checkpoint checkpoints/uq/best.pt \
        --feature_dir ./data/features --label_file ./data/labels/uq_labels.pt
    python scripts/validate_uq.py --checkpoint checkpoints/uq/best.pt --mock
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from uq_estimator.dataset import UQFeatureDataset
from uq_estimator.losses import CombinedUQLoss
from uq_estimator.model import UQEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def try_spearman(pred: list[float], target: list[float]) -> float | None:
    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(pred, target)
        return float(rho)
    except ImportError:
        return None


def compute_separation(
    preds: list[float], labels: list[float],
    low: float = 0.4, high: float = 0.6,
) -> float:
    low_p = [p for p, l in zip(preds, labels) if l < low]
    high_p = [p for p, l in zip(preds, labels) if l > high]
    if not low_p or not high_p:
        return float("nan")
    return float(np.mean(high_p) - np.mean(low_p))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_score_distribution(
    preds: list[float], labels: list[float], out_path: Path,
) -> None:
    """Two-panel figure: histogram + grouped box plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram of all scores
    ax1.hist(preds, bins=20, range=(0, 1), edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Predicted Score")
    ax1.set_ylabel("Count")
    ax1.set_title("Score Distribution")

    # Right: box plot by label group
    low = [p for p, l in zip(preds, labels) if l < 0.4]
    mid = [p for p, l in zip(preds, labels) if 0.4 <= l <= 0.6]
    high = [p for p, l in zip(preds, labels) if l > 0.6]

    box_data = [d for d in [low, mid, high] if d]
    box_labels = []
    if low:
        box_labels.append(f"Low (<0.4)\nn={len(low)}")
    if mid:
        box_labels.append(f"Mid (0.4-0.6)\nn={len(mid)}")
    if high:
        box_labels.append(f"High (>0.6)\nn={len(high)}")

    if box_data:
        ax2.boxplot(box_data, labels=box_labels)
    ax2.set_ylabel("Predicted Score")
    ax2.set_title("Score by Label Group")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def plot_score_vs_label(
    preds: list[float], labels: list[float], out_path: Path,
) -> None:
    """Scatter plot of predicted vs true with y=x reference."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(labels, preds, alpha=0.5, s=15)
    ax.plot([0, 1], [0, 1], "r--", label="y = x")
    ax.set_xlabel("True Label")
    ax.set_ylabel("Predicted Score")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")

    rho = try_spearman(preds, labels)
    rho_str = f"{rho:.3f}" if rho is not None else "N/A"
    ax.set_title(f"Predicted vs True  (Spearman = {rho_str})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(
    preds: list[float],
    labels: list[float],
    val_loss: float,
    n_params: int,
    n_samples: int,
    out_path: Path,
) -> str:
    """Write validation_report.txt and return its content."""
    arr = np.array(preds)
    rho = try_spearman(preds, labels)
    sep = compute_separation(preds, labels)

    lines = [
        "=" * 50,
        "UQ Estimator Validation Report",
        "=" * 50,
        f"Model parameters : {n_params:,}",
        f"Dataset samples  : {n_samples}",
        f"Val loss         : {val_loss:.4f}",
        f"Spearman rho     : {rho:.4f}" if rho is not None else "Spearman rho     : N/A (scipy missing)",
        f"Separation       : {sep:.4f}" if not np.isnan(sep) else "Separation       : N/A (insufficient groups)",
        "",
        "Predicted score statistics:",
        f"  Mean : {arr.mean():.4f}",
        f"  Std  : {arr.std():.4f}",
        f"  Min  : {arr.min():.4f}",
        f"  Max  : {arr.max():.4f}",
        "=" * 50,
    ]
    text = "\n".join(lines)
    out_path.write_text(text)
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Validate trained UQ Estimator.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--feature_dir", type=str, default="")
    parser.add_argument("--label_file", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./reports/uq_validation")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_cfg = cfg["model"]

    model = UQEstimator(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model ({n_params:,} params) from {args.checkpoint}")

    # ── Dataset ───────────────────────────────────────────────────────────
    feature_dir = args.feature_dir or cfg["data"]["feature_dir"]
    label_file = args.label_file or cfg["data"]["label_file"]

    ds = UQFeatureDataset(
        feature_dir=feature_dir,
        label_file=label_file,
        split="val",
        val_ratio=cfg["data"]["val_ratio"],
        mock=args.mock,
        mock_size=64,
        n_views=model_cfg["n_views"],
        n_patches=model_cfg["n_patches"],
        d_patch=model_cfg["d_patch"],
        stat_cache_file=cfg["data"].get("stat_cache_file", ""),
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    print(f"Validation samples: {len(ds)}")

    # ── Inference ─────────────────────────────────────────────────────────
    criterion = CombinedUQLoss()
    all_preds: list[float] = []
    all_labels: list[float] = []
    loss_accum = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            tokens = batch["patch_tokens"].to(device)    # [B, N_views, N_patches, D]
            stats = batch["stat_features"].to(device)     # [B, 5]
            labels = batch["label"].to(device)             # [B, 1]

            out = model(tokens, stats)
            loss_dict = criterion(out.score, labels)

            loss_accum += loss_dict["total"].item()
            n_batches += 1

            all_preds.extend(out.score.squeeze(-1).cpu().tolist())
            all_labels.extend(labels.squeeze(-1).cpu().tolist())

    val_loss = loss_accum / max(n_batches, 1)

    # ── Output ────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_score_distribution(all_preds, all_labels, out_dir / "score_distribution.png")
    plot_score_vs_label(all_preds, all_labels, out_dir / "score_vs_label.png")
    report = write_report(
        all_preds, all_labels, val_loss, n_params, len(ds),
        out_dir / "validation_report.txt",
    )
    print(f"\n{report}")
    print(f"\nOutputs saved to {out_dir}")


if __name__ == "__main__":
    main()

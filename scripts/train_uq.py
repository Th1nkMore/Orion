"""Training script for UQ Estimator.

Usage:
    python scripts/train_uq.py --config configs/uq_train.yaml
    python scripts/train_uq.py --mock --smoke
    python scripts/train_uq.py --config configs/uq_train.yaml --resume checkpoints/uq/best.pt
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from uq_estimator.dataset import UQFeatureDataset, FastTensorLoader
from uq_estimator.losses import CombinedUQLoss
from uq_estimator.model import UQEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def warmup_lambda(epoch: int, warmup_epochs: int) -> float:
    """Linear warmup from 0.1 to 1.0 over warmup_epochs, then 1.0."""
    if epoch < warmup_epochs:
        return 0.1 + 0.9 * epoch / max(warmup_epochs, 1)
    return 1.0


def compute_separation(
    pred_scores: list[float],
    labels: list[float],
    low_thresh: float = 0.4,
    high_thresh: float = 0.6,
) -> float:
    """Mean predicted score gap between high-UQ and low-UQ samples."""
    low_preds = [p for p, l in zip(pred_scores, labels) if l < low_thresh]
    high_preds = [p for p, l in zip(pred_scores, labels) if l > high_thresh]
    if not low_preds or not high_preds:
        return float("nan")
    return float(np.mean(high_preds) - np.mean(low_preds))


def try_spearman(pred: list[float], target: list[float]) -> float | None:
    """Compute Spearman correlation, return None if scipy unavailable."""
    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(pred, target)
        return float(rho)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedUQLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run one training epoch. Returns mean losses."""
    model.train()
    accum = {"total": 0.0, "regression": 0.0, "ranking": 0.0, "calibration": 0.0}
    n_batches = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        patch_tokens = batch["patch_tokens"].to(device)   # [B, N_views, N_patches, D]
        stat_features = batch["stat_features"].to(device)  # [B, 5]
        labels = batch["label"].to(device)                  # [B, 1]

        out = model(patch_tokens, stat_features)
        loss_dict = criterion(out.score, labels)

        optimizer.zero_grad()
        loss_dict["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in accum:
            accum[k] += loss_dict[k].item()
        n_batches += 1

    if n_batches == 0:
        return accum
    return {k: v / n_batches for k, v in accum.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedUQLoss,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[dict[str, float], list[float], list[float]]:
    """Run validation. Returns (mean_losses, all_pred_scores, all_labels)."""
    model.eval()
    accum = {"total": 0.0, "regression": 0.0, "ranking": 0.0, "calibration": 0.0}
    n_batches = 0
    all_preds: list[float] = []
    all_labels: list[float] = []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        patch_tokens = batch["patch_tokens"].to(device)   # [B, N_views, N_patches, D]
        stat_features = batch["stat_features"].to(device)  # [B, 5]
        labels = batch["label"].to(device)                  # [B, 1]

        out = model(patch_tokens, stat_features)
        loss_dict = criterion(out.score, labels)

        for k in accum:
            accum[k] += loss_dict[k].item()
        n_batches += 1

        all_preds.extend(out.score.squeeze(-1).cpu().tolist())
        all_labels.extend(labels.squeeze(-1).cpu().tolist())

    if n_batches == 0:
        return accum, all_preds, all_labels
    return {k: v / n_batches for k, v in accum.items()}, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train UQ Estimator.")
    parser.add_argument("--config", type=str, default="configs/uq_train.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Override feature_dir")
    parser.add_argument("--label_file", type=str, default=None, help="Override label_file")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    parser.add_argument("--smoke", action="store_true", help="2 epochs, 3 batches each")
    args = parser.parse_args()

    # ── 1. Config & seed ──────────────────────────────────────────────────
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    log_cfg = cfg["logging"]
    loss_cfg = cfg.get("loss", {})

    set_seed(train_cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    epochs = 2 if args.smoke else train_cfg["epochs"]
    max_batches = 3 if args.smoke else None
    save_interval = log_cfg["save_interval"]

    # ── 2. Data ───────────────────────────────────────────────────────────
    feature_dir = args.data_dir or data_cfg["feature_dir"]
    label_file = args.label_file or data_cfg["label_file"]
    val_ratio = data_cfg["val_ratio"]

    stat_cache_file = data_cfg.get("stat_cache_file", "")
    preload = data_cfg.get("preload", False) and not args.mock
    preload_workers = data_cfg.get("preload_workers", 16)

    n_patches_subsample = data_cfg.get("n_patches_subsample", 0)
    ds_kwargs = dict(
        feature_dir=feature_dir,
        label_file=label_file,
        val_ratio=val_ratio,
        mock=args.mock,
        mock_size=64,
        n_views=model_cfg["n_views"],
        n_patches=model_cfg["n_patches"],
        d_patch=model_cfg["d_patch"],
        stat_cache_file=stat_cache_file,
        preload=preload,
        preload_workers=preload_workers,
        n_patches_subsample=n_patches_subsample,
    )
    train_ds = UQFeatureDataset(split="train", **ds_kwargs)
    val_ds = UQFeatureDataset(split="val", **ds_kwargs)

    use_cuda = device.type == "cuda"
    batch_size = train_cfg["batch_size"] if not args.smoke else 4

    if preload and not args.mock:
        train_loader = FastTensorLoader(train_ds, batch_size=batch_size, shuffle=True, device=device)
        val_loader = FastTensorLoader(val_ds, batch_size=batch_size, shuffle=False, device=device)
    else:
        effective_workers = 0 if args.mock else 4
        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=effective_workers,
            pin_memory=use_cuda,
            prefetch_factor=2 if effective_workers > 0 else None,
        )
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # ── 3. Model ──────────────────────────────────────────────────────────
    model = UQEstimator(model_cfg).to(device)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # ── 4. Optimiser & scheduler ──────────────────────────────────────────
    lr = train_cfg["lr"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg["weight_decay"],
    )
    warmup_epochs = train_cfg["warmup_epochs"]
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: warmup_lambda(ep, warmup_epochs)
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    start_epoch = 1
    lambda_cal = loss_cfg.get("lambda_cal", 0.1) if loss_cfg.get("use_calibration", True) else 0.0
    lambda_rank = loss_cfg.get("lambda_rank", 0.5) if loss_cfg.get("use_ranking", True) else 0.0
    criterion = CombinedUQLoss(lambda_cal=lambda_cal, lambda_rank=lambda_rank)
    ablation_info = []
    if not loss_cfg.get("use_ranking", True):
        ablation_info.append("ranking_loss=OFF")
    if not loss_cfg.get("use_calibration", True):
        ablation_info.append("calibration_reg=OFF")
    if not model_cfg.get("use_stat_features", True):
        ablation_info.append("stat_features=OFF")
    if not model_cfg.get("use_transformer_decoder", True):
        ablation_info.append("transformer_decoder=OFF (mean pool)")
    if ablation_info:
        print(f"[Ablation] {', '.join(ablation_info)}")

    # ── Resume ────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_epoch = ckpt.get("best_epoch", 0)
        # Restore scheduler states so LR curve continues correctly
        if "warmup_sched_state" in ckpt:
            warmup_sched.load_state_dict(ckpt["warmup_sched_state"])
        if "cosine_sched_state" in ckpt:
            cosine_sched.load_state_dict(ckpt["cosine_sched_state"])
        print(f"Resumed from epoch {ckpt['epoch']} (starting at {start_epoch}), best_val_loss={best_val_loss:.4f}")

    # ── 5. Training loop ──────────────────────────────────────────────────
    save_dir = Path(log_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    last_spearman = None
    spearman_warned = False

    for epoch in range(start_epoch, epochs + 1):  # always train up to epoch `epochs`
        # — train —
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, max_batches
        )

        # — scheduler step —
        if epoch <= warmup_epochs:
            warmup_sched.step()
        else:
            cosine_sched.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # — validate —
        val_losses, val_preds, val_labels = validate(
            model, val_loader, criterion, device, max_batches
        )

        # — metrics —
        spearman = try_spearman(val_preds, val_labels)
        if spearman is None and not spearman_warned:
            print("[WARN] scipy not installed — skipping Spearman correlation")
            spearman_warned = True
        last_spearman = spearman

        separation = compute_separation(val_preds, val_labels)

        spearman_str = f"{spearman:.2f}" if spearman is not None else "N/A"
        sep_str = f"{separation:.2f}" if not math.isnan(separation) else "N/A"

        print(
            f"Epoch {epoch:02d}/{start_epoch + epochs - 1:02d} | "
            f"train_loss: {train_losses['total']:.4f} | "
            f"val_loss: {val_losses['total']:.4f} | "
            f"spearman: {spearman_str} | "
            f"separation: {sep_str} | "
            f"lr: {current_lr:.2e}"
        )

        # — save checkpoint —
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "warmup_sched_state": warmup_sched.state_dict(),
            "cosine_sched_state": cosine_sched.state_dict(),
            "val_loss": val_losses["total"],
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "config": cfg,
        }

        if epoch % save_interval == 0 or args.smoke:
            ckpt_path = save_dir / f"checkpoint_epoch{epoch:05d}.pt"
            torch.save(ckpt_data, str(ckpt_path))

        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_epoch = epoch
            torch.save(ckpt_data, str(save_dir / "best.pt"))

    # ── 6. Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Training complete.")
    print(f"Best val_loss   : {best_val_loss:.4f} (epoch {best_epoch})")
    if last_spearman is not None:
        print(f"Final spearman  : {last_spearman:.4f}")
    print(f"Checkpoints     : {save_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

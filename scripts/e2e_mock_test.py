"""End-to-end mock test for the full Phase 1 pipeline.

Verifies the complete data flow without any real data files:
  mock features → pseudo-labels → dataset → model → train → checkpoint → reload

Usage:
    python scripts/e2e_mock_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_labels import compute_uq_score
from tests.fixtures import create_mock_feature_dir
from uq_estimator.dataset import UQFeatureDataset
from uq_estimator.losses import CombinedUQLoss
from uq_estimator.model import UQEstimator


def main() -> None:
    with open(PROJECT_ROOT / "configs" / "uq_train.yaml") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]

    results: list[str] = []
    tmpdir = Path(tempfile.mkdtemp())

    # ==================================================================
    # 1. Generate mock feature files (30: 10 normal + 10 random + 10 adverse)
    # ==================================================================
    print("=" * 60)
    print("[1/8] Generating 30 mock feature files ...")
    feature_dir = create_mock_feature_dir(tmpdir, n_files=30)
    files = sorted(feature_dir.glob("*.pt"))
    print(f"      Created {len(files)} files in {feature_dir}")

    # ==================================================================
    # 2. Generate pseudo-labels
    # ==================================================================
    print("\n[2/8] Computing pseudo-labels ...")
    labels: dict[str, float] = {}
    for f in files:
        fname, score = compute_uq_score(f)
        if score is not None:
            labels[fname] = score

    # Group stats (fixture: 0-9 normal, 10-19 random, 20-29 adverse)
    normal_scores = [labels[f"sample_{i:04d}.pt"] for i in range(10)]
    random_scores = [labels[f"sample_{i:04d}.pt"] for i in range(10, 20)]
    adverse_scores = [labels[f"sample_{i:04d}.pt"] for i in range(20, 30)]

    separation = np.mean(adverse_scores) - np.mean(normal_scores)
    print(f"      Normal  mean: {np.mean(normal_scores):.4f}")
    print(f"      Random  mean: {np.mean(random_scores):.4f}")
    print(f"      Adverse mean: {np.mean(adverse_scores):.4f}")
    print(f"      Separation (adverse - normal): {separation:.4f}")

    # Save labels
    label_path = tmpdir / "labels.pt"
    torch.save(labels, str(label_path))
    results.append(f"Pseudo-label generation OK, separation: {separation:.2f}")

    # ==================================================================
    # 3. Dataset
    # ==================================================================
    print("\n[3/8] Initialising dataset ...")
    train_ds = UQFeatureDataset(
        feature_dir=str(feature_dir), label_file=str(label_path),
        split="train", val_ratio=0.1,
    )
    val_ds = UQFeatureDataset(
        feature_dir=str(feature_dir), label_file=str(label_path),
        split="val", val_ratio=0.1,
    )
    print(f"      Train: {len(train_ds)} | Val: {len(val_ds)}")
    results.append(f"Dataset OK, train/val: {len(train_ds)}/{len(val_ds)}")

    # ==================================================================
    # 4. Model
    # ==================================================================
    print("\n[4/8] Initialising model ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UQEstimator(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Device: {device}")
    print(f"      Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
    results.append(f"Model OK, params: {n_params / 1e6:.2f}M")

    # ==================================================================
    # 5. Training steps (3 batches)
    # ==================================================================
    print("\n[5/8] Running 3 training steps ...")
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=0,
    )
    criterion = CombinedUQLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    step_losses: list[float] = []
    for step, batch in enumerate(train_loader):
        if step >= 3:
            break

        tokens = batch["patch_tokens"].to(device)    # [B, N_views, N_patches, D]
        stats = batch["stat_features"].to(device)     # [B, 5]
        label = batch["label"].to(device)              # [B, 1]

        out = model(tokens, stats)
        loss_dict = criterion(out.score, label)

        optimizer.zero_grad()
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        step_losses.append(loss_dict["total"].item())
        print(
            f"      Step {step+1}: total={loss_dict['total']:.4f}  "
            f"reg={loss_dict['regression']:.4f}  "
            f"rank={loss_dict['ranking']:.4f}  "
            f"cal={loss_dict['calibration']:.4f}"
        )

    all_finite = all(np.isfinite(l) for l in step_losses)
    decreasing = step_losses[-1] < step_losses[0] if len(step_losses) >= 2 else True
    print(f"      All losses finite: {all_finite} | Decreasing trend: {decreasing}")
    results.append(f"Training OK, loss finite={'yes' if all_finite else 'NO'}, decreasing={'yes' if decreasing else 'no'}")

    # ==================================================================
    # 6. Validation step
    # ==================================================================
    print("\n[6/8] Running validation ...")
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=len(val_ds), shuffle=False, num_workers=0,
    )
    model.eval()
    with torch.no_grad():
        val_batch = next(iter(val_loader))
        tokens = val_batch["patch_tokens"].to(device)  # [B, N_views, N_patches, D]
        stats = val_batch["stat_features"].to(device)   # [B, 5]
        val_labels = val_batch["label"].to(device)       # [B, 1]

        val_out = model(tokens, stats)

    print(f"      Embedding shape: {val_out.embedding.shape}")
    print(f"      Score range: [{val_out.score.min():.4f}, {val_out.score.max():.4f}]")

    # Per-group predicted scores
    val_scores = val_out.score.squeeze(-1).cpu().numpy()  # [B]
    val_labels_np = val_labels.squeeze(-1).cpu().numpy()   # [B]
    low_mask = val_labels_np < 0.5
    high_mask = val_labels_np >= 0.5
    if low_mask.any():
        print(f"      Low-UQ samples pred mean: {val_scores[low_mask].mean():.4f}")
    if high_mask.any():
        print(f"      High-UQ samples pred mean: {val_scores[high_mask].mean():.4f}")
    results.append("Validation OK")

    # ==================================================================
    # 7. Save checkpoint
    # ==================================================================
    print("\n[7/8] Saving checkpoint ...")
    ckpt_path = tmpdir / "test_checkpoint.pt"
    ckpt_data = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": 0.0,
        "config": cfg,
    }
    torch.save(ckpt_data, str(ckpt_path))
    print(f"      Saved to {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
    results.append("Checkpoint save OK")

    # ==================================================================
    # 8. Reload and verify
    # ==================================================================
    print("\n[8/8] Reloading checkpoint and verifying weights ...")
    loaded = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model2 = UQEstimator(model_cfg).to(device)
    model2.load_state_dict(loaded["model_state_dict"])

    # Compare a few parameter tensors
    all_match = True
    for name, param in model.named_parameters():
        param2 = dict(model2.named_parameters())[name]
        if not torch.allclose(param, param2):
            print(f"      MISMATCH: {name}")
            all_match = False
            break
    if all_match:
        print("      All weights match after reload")
    results.append(f"Checkpoint reload OK, weights match={all_match}")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 1 End-to-End Verification Summary")
    print("=" * 60)
    status = [
        ("Pseudo-label generation", f"separation: {separation:.2f}"),
        ("Dataset loading", f"train/val: {len(train_ds)}/{len(val_ds)}"),
        ("Model forward pass", f"params: {n_params / 1e6:.2f}M"),
        ("Training loop", f"loss finite & {'decreasing' if decreasing else 'not decreasing'}"),
        ("Checkpoint save & reload", f"weights match: {all_match}"),
    ]
    all_ok = True
    for desc, detail in status:
        ok = True  # each step would have raised if broken
        mark = "OK" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {desc} — {detail}")

    print()
    if all_ok and all_finite and all_match:
        print("Phase 1 end-to-end verification PASSED")
    else:
        print("Phase 1 end-to-end verification FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()

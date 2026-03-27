"""Generate uncertainty pseudo-labels from pre-extracted feature files.

Usage:
    python scripts/generate_labels.py --feature_dir ./data/features --dry_run
    python scripts/generate_labels.py --feature_dir ./data/features --output_file ./data/labels/uq_labels.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Indicator weights
# ---------------------------------------------------------------------------
W_GRADIENT = 0.3
W_ENTROPY = 0.3
W_CONSISTENCY = 0.4


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------
def compute_uq_score(feature_path: Path) -> tuple[str, Optional[float], str]:
    """Compute a single uncertainty pseudo-label for one feature file.

    Args:
        feature_path: path to a .pt file with keys 'tokens' and optionally 'image'.

    Returns:
        (filename, uq_score, scene_type) where score is in [0, 1], or None on failure.
    """
    fname = feature_path.name
    try:
        data = torch.load(str(feature_path), map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"[WARN] failed to load {fname}: {e}")
        return fname, None, "unknown"

    tokens = data["tokens"]  # [N_views, N_patches, D]
    has_image = "image" in data
    scene_type = data.get("scene_type", "unknown")

    # ---- 1. gradient_score (weight 0.3) -----------------------------------
    if has_image:
        image = data["image"]  # [N_views, 3, H, W]
        gradient_score = _compute_gradient_score(image)
    else:
        gradient_score = 0.5  # neutral fallback

    # ---- 2. entropy_score (weight 0.3) ------------------------------------
    entropy_score = _compute_entropy_score(tokens)

    # ---- 3. consistency_score (weight 0.4) --------------------------------
    consistency_score = _compute_consistency_score(tokens)

    uq_score = (
        W_GRADIENT * gradient_score
        + W_ENTROPY * entropy_score
        + W_CONSISTENCY * consistency_score
    )
    uq_score = float(torch.clamp(torch.tensor(uq_score), 0.0, 1.0).item())

    # ---- 4. scene_type calibration ----------------------------------------
    if scene_type == "normal":
        uq_score = float(torch.tensor(uq_score).clip(0.0, 0.45).item())
    elif scene_type == "adverse":
        uq_score = float(torch.tensor(uq_score).clip(0.55, 1.0).item())
    # unknown: keep original value

    return fname, uq_score, scene_type


# ---------------------------------------------------------------------------
# Sub-indicators
# ---------------------------------------------------------------------------
def _compute_gradient_score(image: torch.Tensor) -> float:
    """High gradient → clear image → low uncertainty → return 1 - normalised.

    Args:
        image: [N_views, 3, H, W]

    Returns:
        Scalar in [0, 1].
    """
    gray = image.mean(dim=1, keepdim=True)  # [N_views, 1, H, W]

    # Sobel kernels
    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=gray.dtype
    ).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    sobel_y = sobel_x.transpose(-1, -2)  # [1, 1, 3, 3]

    # Apply per-view
    N_views = gray.shape[0]
    gx = F.conv2d(gray.reshape(N_views, 1, *gray.shape[2:]), sobel_x, padding=1)  # [N_views, 1, H, W]
    gy = F.conv2d(gray.reshape(N_views, 1, *gray.shape[2:]), sobel_y, padding=1)  # [N_views, 1, H, W]
    mag = (gx.pow(2) + gy.pow(2)).sqrt()  # [N_views, 1, H, W]

    grad_mean = mag.mean().item()  # scalar
    normalised = min(max(grad_mean, 0.0), 10.0) / 10.0
    return 1.0 - normalised  # high gradient → low score


def _compute_entropy_score(tokens: torch.Tensor) -> float:
    """High entropy → dispersed activations → high uncertainty.

    Args:
        tokens: [N_views, N_patches, D]

    Returns:
        Scalar in [0, 1].
    """
    D = tokens.shape[-1]
    p = F.softmax(tokens, dim=-1)  # [N_views, N_patches, D]
    entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)  # [N_views, N_patches]
    max_entropy = torch.log(torch.tensor(float(D)))  # scalar
    normalised = (entropy.mean() / max_entropy).item()
    return float(min(max(normalised, 0.0), 1.0))


def _compute_consistency_score(tokens: torch.Tensor) -> float:
    """High cross-view similarity → consistent → low uncertainty → return 1 - sim.

    Args:
        tokens: [N_views, N_patches, D]

    Returns:
        Scalar in [0, 1].
    """
    view_feats = tokens.mean(dim=1)  # [N_views, D]
    view_feats = F.normalize(view_feats, dim=-1)  # [N_views, D]
    sim = view_feats @ view_feats.T  # [N_views, N_views]

    N_views = sim.shape[0]
    mask = torch.triu(torch.ones(N_views, N_views, dtype=torch.bool), diagonal=1)
    if mask.sum() == 0:
        return 0.5
    mean_sim = sim[mask].mean().item()
    # cosine sim is in [-1, 1]; map to [0, 1] then invert
    normalised = (mean_sim + 1.0) / 2.0
    return 1.0 - normalised  # high similarity → low score


# ---------------------------------------------------------------------------
# ASCII histogram
# ---------------------------------------------------------------------------
def _ascii_histogram(values: list[float], bins: int = 10, width: int = 40) -> str:
    """Return an ASCII histogram string."""
    import numpy as np

    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    max_count = max(counts) if max(counts) > 0 else 1
    lines = []
    for i, count in enumerate(counts):
        bar_len = int(count / max_count * width)
        bar = "#" * bar_len
        lines.append(f"  [{edges[i]:.1f}, {edges[i+1]:.1f}) | {bar} {count}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batched GPU scoring
# ---------------------------------------------------------------------------
def _compute_scores_batch_gpu(
    tokens_batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched entropy and consistency scores on GPU.

    Args:
        tokens_batch: [B, N_views, N_patches, D] fp16 on CUDA

    Returns:
        entropy_scores:     [B] float32
        consistency_scores: [B] float32
    """
    B, N_views, N_patches, D = tokens_batch.shape
    t = tokens_batch.float()  # work in fp32

    # Entropy score: [B]
    p = F.softmax(t, dim=-1)                              # [B, N_views, N_patches, D]
    entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)     # [B, N_views, N_patches]
    max_entropy = torch.log(torch.tensor(float(D), device=t.device))
    entropy_scores = (entropy.mean(dim=(-1, -2)) / max_entropy).clamp(0.0, 1.0)  # [B]

    # Consistency score: [B]
    view_feats = t.mean(dim=2)                            # [B, N_views, D]
    view_feats = F.normalize(view_feats, dim=-1)
    sim = torch.bmm(view_feats, view_feats.transpose(1, 2))  # [B, N_views, N_views]
    mask = torch.triu(
        torch.ones(N_views, N_views, dtype=torch.bool, device=t.device), diagonal=1
    )
    sim_vals = sim[:, mask]                               # [B, N_pairs]
    mean_sim = sim_vals.mean(dim=-1)                      # [B]
    consistency_scores = (1.0 - (mean_sim + 1.0) / 2.0).clamp(0.0, 1.0)  # [B]

    return entropy_scores, consistency_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UQ pseudo-labels.")
    parser.add_argument("--feature_dir", type=str, required=True, help="Directory of .pt feature files")
    parser.add_argument("--output_file", type=str, default="./data/labels/uq_labels.pt", help="Output label file")
    parser.add_argument("--n_workers", type=int, default=4, help="Parallel workers (unused with --device cuda)")
    parser.add_argument("--dry_run", action="store_true", help="Process first 10 files, print stats, don't save")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for GPU processing")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

    pt_files = sorted(feature_dir.glob("*.pt"))
    if len(pt_files) == 0:
        print("No .pt files found. Exiting.")
        return

    if args.dry_run:
        pt_files = pt_files[:10]
        print(f"[dry_run] Processing first {len(pt_files)} files.")

    labels: dict[str, dict] = {}
    failed: list[str] = []

    if args.device == "cuda":
        # ── GPU batched path ──────────────────────────────────────────────
        # Pass 1: compute raw scores for all files
        print(f"Using GPU batched processing (batch_size={args.batch_size})")
        raw_records: list[tuple[str, float, str]] = []   # (fname, raw_score, scene_type)

        for batch_start in tqdm(range(0, len(pt_files), args.batch_size),
                                desc="Computing UQ scores (GPU)"):
            batch_files = pt_files[batch_start: batch_start + args.batch_size]
            tokens_list, fnames, scene_types_b = [], [], []

            for fp in batch_files:
                try:
                    d = torch.load(str(fp), map_location="cpu", weights_only=True)
                    tokens_list.append(d["tokens"])          # [N_views, N_patches, D]
                    fnames.append(fp.name)
                    scene_types_b.append(d.get("scene_type", "unknown"))
                except Exception as e:
                    print(f"[WARN] failed to load {fp.name}: {e}")
                    failed.append(fp.name)

            if not tokens_list:
                continue

            tokens_batch = torch.stack(tokens_list).cuda()  # [B, N_views, N_patches, D]
            with torch.no_grad():
                entropy_scores, consistency_scores = _compute_scores_batch_gpu(tokens_batch)

            uq_scores = (
                W_GRADIENT * 0.5                        # no images → neutral
                + W_ENTROPY * entropy_scores.cpu()
                + W_CONSISTENCY * consistency_scores.cpu()
            ).clamp(0.0, 1.0)                           # [B]

            for fname, st, score in zip(fnames, scene_types_b, uq_scores.tolist()):
                raw_records.append((fname, score, st))

        # Pass 2: per-class min-max normalisation → target range
        # normal  → [0.00, 0.45]   adverse → [0.55, 1.00]   unknown → keep raw
        # This preserves within-class variance rather than hard-clipping everyone
        # to the same boundary value.
        import numpy as np
        for class_name, lo, hi in [("normal", 0.0, 0.45), ("adverse", 0.55, 1.0)]:
            class_scores = [s for _, s, st in raw_records if st == class_name]
            if not class_scores:
                continue
            c_min, c_max = min(class_scores), max(class_scores)
            span = c_max - c_min if c_max > c_min else 1.0
            for i, (fname, raw, st) in enumerate(raw_records):
                if st != class_name:
                    continue
                normed = (raw - c_min) / span          # [0, 1] within class
                calibrated = lo + normed * (hi - lo)   # map to target range
                raw_records[i] = (fname, float(calibrated), st)

        for fname, score, st in raw_records:
            labels[fname] = {"score": score, "scene_type": st}

    else:
        # ── CPU multiprocessing fallback ──────────────────────────────────
        import multiprocessing as mp
        with mp.Pool(processes=args.n_workers) as pool:
            results = list(tqdm(
                pool.imap(compute_uq_score, pt_files),
                total=len(pt_files),
                desc="Computing UQ scores (CPU)",
            ))
        for fname, score, scene_type in results:
            if score is None:
                failed.append(fname)
            else:
                labels[fname] = {"score": score, "scene_type": scene_type}

    # Statistics
    import numpy as np

    scores = [v["score"] for v in labels.values()]
    scene_types = [v["scene_type"] for v in labels.values()]

    arr = np.array(scores)
    print(f"\n{'='*50}")
    print(f"Total processed : {len(scores)}")
    print(f"Failed          : {len(failed)}")
    print(f"Mean            : {arr.mean():.4f}")
    print(f"Std             : {arr.std():.4f}")
    print(f"Min             : {arr.min():.4f}")
    print(f"Max             : {arr.max():.4f}")

    # Per-scene-type statistics
    for st in ["normal", "adverse", "unknown"]:
        indices = [i for i, s in enumerate(scene_types) if s == st]
        if indices:
            st_scores = [scores[i] for i in indices]
            st_arr = np.array(st_scores)
            print(f"\n  [{st}] n={len(st_scores):4d}  mean={st_arr.mean():.4f}  std={st_arr.std():.4f}")

    print(f"\nScore distribution:")
    print(_ascii_histogram(scores))
    print(f"{'='*50}")

    if failed:
        print(f"\nFailed files ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")

    if args.dry_run:
        print("\n[dry_run] Exiting without saving.")
        return

    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(labels, str(output_path))
    print(f"\nSaved {len(labels)} labels to {output_path}")


if __name__ == "__main__":
    main()

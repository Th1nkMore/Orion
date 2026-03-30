"""Generate uncertainty pseudo-labels from pre-extracted feature files.

Usage (fast path using stat_cache):
    python scripts/generate_labels.py --feature_dir ./data/features --stat_cache ./data/stat_cache.pt --output_file ./data/labels/uq_labels.pt

Usage (legacy path):
    python scripts/generate_labels.py --feature_dir ./data/features --output_file ./data/labels/uq_labels.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Indicator weights (v2: based on Cohen's d analysis of stat features)
#   max_mean  d=1.07 → strongest discriminator (lower in adverse)
#   cosim     d=0.53 → moderate discriminator  (lower in adverse)
#   entropy   d=0.06 → weak, kept for diversity
# ---------------------------------------------------------------------------
W_MAX_MEAN = 0.50
W_COSIM = 0.35
W_ENTROPY = 0.15

# Calibration target ranges with wider gap for better AUROC
NORMAL_RANGE = (0.03, 0.38)
ADVERSE_RANGE = (0.62, 0.97)
UNKNOWN_RANGE = (0.30, 0.70)


# ---------------------------------------------------------------------------
# Core scoring (v2) — stat-cache based, unified for CPU and GPU
# ---------------------------------------------------------------------------
def compute_raw_score_from_stats(stat: np.ndarray) -> float:
    """Compute raw uncertainty score from 5-dim stat features.

    stat: [5] = [var, entropy, cosim, abs_mean, max_mean]
    Higher return value = higher uncertainty.
    """
    _var, entropy, cosim, _abs_mean, max_mean = stat

    # max_mean: lower → higher uncertainty. Normalise to [0, 1] via empirical range.
    # Observed range: ~13.2 to ~15.8 across dataset.
    max_mean_score = 1.0 - np.clip((max_mean - 13.0) / (16.0 - 13.0), 0.0, 1.0)

    # cosim: lower → higher uncertainty. Already in ~[0.5, 1.0].
    cosim_score = 1.0 - np.clip((cosim - 0.5) / (1.0 - 0.5), 0.0, 1.0)

    # entropy: already in [0, 1], higher = more uncertain.
    entropy_score = np.clip(entropy, 0.0, 1.0)

    raw = (W_MAX_MEAN * max_mean_score
           + W_COSIM * cosim_score
           + W_ENTROPY * entropy_score)
    return float(np.clip(raw, 0.0, 1.0))


def calibrate_scores(
    raw_records: list[tuple[str, float, str]],
) -> list[tuple[str, float, str]]:
    """Percentile-based calibration per scene_type class.

    More robust than min-max: uses p2/p98 to clip outliers, then linearly
    maps each class to its target range with a wide normal/adverse gap.
    """
    calibrated = list(raw_records)

    for class_name, (lo, hi) in [
        ("normal", NORMAL_RANGE),
        ("adverse", ADVERSE_RANGE),
        ("unknown", UNKNOWN_RANGE),
    ]:
        indices = [i for i, (_, _, st) in enumerate(calibrated) if st == class_name]
        if not indices:
            continue
        class_scores = np.array([calibrated[i][1] for i in indices])

        p2, p98 = np.percentile(class_scores, [2, 98])
        span = p98 - p2 if p98 > p2 else 1.0
        for idx in indices:
            fname, raw, st = calibrated[idx]
            normed = np.clip((raw - p2) / span, 0.0, 1.0)
            calibrated[idx] = (fname, float(lo + normed * (hi - lo)), st)

    return calibrated


# ---------------------------------------------------------------------------
# Legacy scoring function (CPU single-file, kept for backward compat)
# ---------------------------------------------------------------------------
def compute_uq_score(feature_path: Path) -> tuple[str, Optional[float], str]:
    """Compute a single uncertainty pseudo-label for one feature file.

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
    scene_type = data.get("scene_type", "unknown")

    entropy_score = _compute_entropy_score(tokens)
    consistency_score = _compute_consistency_score(tokens)
    max_mean_score = _compute_max_mean_score(tokens)

    uq_score = (
        W_MAX_MEAN * max_mean_score
        + W_COSIM * consistency_score
        + W_ENTROPY * entropy_score
    )
    uq_score = float(np.clip(uq_score, 0.0, 1.0))
    return fname, uq_score, scene_type


# ---------------------------------------------------------------------------
# Sub-indicators
# ---------------------------------------------------------------------------
def _compute_gradient_score(image: torch.Tensor) -> float:
    """High gradient → clear image → low uncertainty → return 1 - normalised."""
    gray = image.mean(dim=1, keepdim=True)
    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=gray.dtype
    ).unsqueeze(0).unsqueeze(0)
    sobel_y = sobel_x.transpose(-1, -2)
    N_views = gray.shape[0]
    gx = F.conv2d(gray.reshape(N_views, 1, *gray.shape[2:]), sobel_x, padding=1)
    gy = F.conv2d(gray.reshape(N_views, 1, *gray.shape[2:]), sobel_y, padding=1)
    mag = (gx.pow(2) + gy.pow(2)).sqrt()
    grad_mean = mag.mean().item()
    normalised = min(max(grad_mean, 0.0), 10.0) / 10.0
    return 1.0 - normalised


def _compute_entropy_score(tokens: torch.Tensor) -> float:
    """High entropy → dispersed activations → high uncertainty."""
    D = tokens.shape[-1]
    p = F.softmax(tokens, dim=-1)
    entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(D)))
    normalised = (entropy.mean() / max_entropy).item()
    return float(min(max(normalised, 0.0), 1.0))


def _compute_consistency_score(tokens: torch.Tensor) -> float:
    """High cross-view similarity → consistent → low uncertainty → return 1 - sim."""
    view_feats = tokens.mean(dim=1)
    view_feats = F.normalize(view_feats, dim=-1)
    sim = view_feats @ view_feats.T
    N_views = sim.shape[0]
    mask = torch.triu(torch.ones(N_views, N_views, dtype=torch.bool), diagonal=1)
    if mask.sum() == 0:
        return 0.5
    mean_sim = sim[mask].mean().item()
    normalised = (mean_sim + 1.0) / 2.0
    return 1.0 - normalised


def _compute_max_mean_score(tokens: torch.Tensor) -> float:
    """Lower max activation → degraded features → higher uncertainty."""
    max_mean = tokens.amax(dim=-1).mean().item()
    normalised = min(max(max_mean - 13.0, 0.0), 3.0) / 3.0
    return 1.0 - normalised


# ---------------------------------------------------------------------------
# ASCII histogram
# ---------------------------------------------------------------------------
def _ascii_histogram(values: list[float], bins: int = 10, width: int = 40) -> str:
    """Return an ASCII histogram string."""
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    max_count = max(counts) if max(counts) > 0 else 1
    lines = []
    for i, count in enumerate(counts):
        bar_len = int(count / max_count * width)
        bar = "#" * bar_len
        lines.append(f"  [{edges[i]:.1f}, {edges[i+1]:.1f}) | {bar} {count}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batched GPU scoring (v2)
# ---------------------------------------------------------------------------
def _compute_scores_batch_gpu(
    tokens_batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched scoring on GPU.

    Args:
        tokens_batch: [B, N_views, N_patches, D] fp16 on CUDA

    Returns:
        entropy_scores:     [B] float32
        consistency_scores: [B] float32
        max_mean_scores:    [B] float32
    """
    B, N_views, N_patches, D = tokens_batch.shape
    t = tokens_batch.float()

    # Entropy: [B]
    p = F.softmax(t, dim=-1)
    entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(D), device=t.device))
    entropy_scores = (entropy.mean(dim=(-1, -2)) / max_entropy).clamp(0.0, 1.0)

    # Consistency (1 - normalised cosim): [B]
    view_feats = t.mean(dim=2)
    view_feats = F.normalize(view_feats, dim=-1)
    sim = torch.bmm(view_feats, view_feats.transpose(1, 2))
    mask = torch.triu(
        torch.ones(N_views, N_views, dtype=torch.bool, device=t.device), diagonal=1
    )
    sim_vals = sim[:, mask]
    mean_sim = sim_vals.mean(dim=-1)
    consistency_scores = (1.0 - (mean_sim + 1.0) / 2.0).clamp(0.0, 1.0)

    # Max-mean (inverted): [B]
    max_mean = t.amax(dim=-1).mean(dim=(-1, -2))  # [B]
    max_mean_scores = (1.0 - ((max_mean - 13.0) / 3.0).clamp(0.0, 1.0))

    return entropy_scores, consistency_scores, max_mean_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UQ pseudo-labels.")
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory of .pt feature files")
    parser.add_argument("--output_file", type=str,
                        default="./data/labels/uq_labels.pt",
                        help="Output label file")
    parser.add_argument("--stat_cache", type=str, default="",
                        help="Path to stat_cache.pt for fast label generation")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="Parallel workers (CPU fallback only)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print stats, don't save")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for GPU processing")
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

    # Build fname → scene_type mapping from feature files (first pass, lightweight)
    fname_to_scene: dict[str, str] = {}
    stat_cache: dict[str, torch.Tensor] = {}
    if args.stat_cache and Path(args.stat_cache).is_file():
        stat_cache = torch.load(args.stat_cache, weights_only=True, map_location="cpu")
        print(f"Loaded stat_cache: {len(stat_cache)} entries")

    # ── Fast path: stat_cache available ───────────────────────────────────
    if stat_cache:
        print("Using stat_cache fast path (v2 scoring)")
        raw_records: list[tuple[str, float, str]] = []

        for fp in tqdm(pt_files, desc="Loading scene_types"):
            fname = fp.name
            if fname not in stat_cache:
                failed.append(fname)
                continue
            try:
                d = torch.load(str(fp), map_location="cpu", weights_only=True)
                scene_type = d.get("scene_type", "unknown")
            except Exception:
                scene_type = "unknown"

            stat = stat_cache[fname].float().numpy()  # [5]
            raw = compute_raw_score_from_stats(stat)
            raw_records.append((fname, raw, scene_type))

        calibrated = calibrate_scores(raw_records)
        for fname, score, st in calibrated:
            labels[fname] = {"score": score, "scene_type": st}

    elif args.device == "cuda":
        # ── GPU batched path ──────────────────────────────────────────────
        print(f"Using GPU batched processing (batch_size={args.batch_size})")
        raw_records = []

        for batch_start in tqdm(range(0, len(pt_files), args.batch_size),
                                desc="Computing UQ scores (GPU)"):
            batch_files = pt_files[batch_start: batch_start + args.batch_size]
            tokens_list, fnames, scene_types_b = [], [], []

            for fp in batch_files:
                try:
                    d = torch.load(str(fp), map_location="cpu", weights_only=True)
                    tokens_list.append(d["tokens"])
                    fnames.append(fp.name)
                    scene_types_b.append(d.get("scene_type", "unknown"))
                except Exception as e:
                    print(f"[WARN] failed to load {fp.name}: {e}")
                    failed.append(fp.name)

            if not tokens_list:
                continue

            tokens_batch = torch.stack(tokens_list).cuda()
            with torch.no_grad():
                ent, cons, maxm = _compute_scores_batch_gpu(tokens_batch)

            uq_scores = (
                W_MAX_MEAN * maxm.cpu()
                + W_COSIM * cons.cpu()
                + W_ENTROPY * ent.cpu()
            ).clamp(0.0, 1.0)

            for fname, st, score in zip(fnames, scene_types_b, uq_scores.tolist()):
                raw_records.append((fname, score, st))

        calibrated = calibrate_scores(raw_records)
        for fname, score, st in calibrated:
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
        raw_records = []
        for fname, score, scene_type in results:
            if score is None:
                failed.append(fname)
            else:
                raw_records.append((fname, score, scene_type))
        calibrated = calibrate_scores(raw_records)
        for fname, score, st in calibrated:
            labels[fname] = {"score": score, "scene_type": st}

    # Statistics
    scores = [v["score"] for v in labels.values()]
    scene_types_list = [v["scene_type"] for v in labels.values()]

    arr = np.array(scores)
    print(f"\n{'='*50}")
    print(f"Total processed : {len(scores)}")
    print(f"Failed          : {len(failed)}")
    print(f"Mean            : {arr.mean():.4f}")
    print(f"Std             : {arr.std():.4f}")
    print(f"Min             : {arr.min():.4f}")
    print(f"Max             : {arr.max():.4f}")

    for st in ["normal", "adverse", "unknown"]:
        indices = [i for i, s in enumerate(scene_types_list) if s == st]
        if indices:
            st_scores = [scores[i] for i in indices]
            st_arr = np.array(st_scores)
            print(f"\n  [{st}] n={len(st_scores):4d}  mean={st_arr.mean():.4f}  "
                  f"std={st_arr.std():.4f}  min={st_arr.min():.4f}  max={st_arr.max():.4f}")

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

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(labels, str(output_path))
    print(f"\nSaved {len(labels)} labels to {output_path}")


if __name__ == "__main__":
    main()

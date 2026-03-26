"""Generate uncertainty pseudo-labels from pre-extracted feature files.

Usage:
    python scripts/generate_labels.py --feature_dir ./data/features --dry_run
    python scripts/generate_labels.py --feature_dir ./data/features --output_file ./data/labels/uq_labels.pt
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
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
def compute_uq_score(feature_path: Path) -> tuple[str, Optional[float]]:
    """Compute a single uncertainty pseudo-label for one feature file.

    Args:
        feature_path: path to a .pt file with keys 'tokens' and optionally 'image'.

    Returns:
        (filename, uq_score) where score is in [0, 1], or None on failure.
    """
    fname = feature_path.name
    try:
        data = torch.load(str(feature_path), map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"[WARN] failed to load {fname}: {e}")
        return fname, None

    tokens = data["tokens"]  # [N_views, N_patches, D]
    has_image = "image" in data

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
    return fname, uq_score


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
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UQ pseudo-labels.")
    parser.add_argument("--feature_dir", type=str, required=True, help="Directory of .pt feature files")
    parser.add_argument("--output_file", type=str, default="./data/labels/uq_labels.pt", help="Output label file")
    parser.add_argument("--n_workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--dry_run", action="store_true", help="Process first 10 files, print stats, don't save")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
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

    # Parallel processing
    labels: dict[str, float] = {}
    failed: list[str] = []

    with mp.Pool(processes=args.n_workers) as pool:
        results = list(tqdm(
            pool.imap(compute_uq_score, pt_files),
            total=len(pt_files),
            desc="Computing UQ scores",
        ))

    for fname, score in results:
        if score is None:
            failed.append(fname)
        else:
            labels[fname] = score

    # Statistics
    scores = list(labels.values())
    import numpy as np

    arr = np.array(scores)
    print(f"\n{'='*50}")
    print(f"Total processed : {len(scores)}")
    print(f"Failed          : {len(failed)}")
    print(f"Mean            : {arr.mean():.4f}")
    print(f"Std             : {arr.std():.4f}")
    print(f"Min             : {arr.min():.4f}")
    print(f"Max             : {arr.max():.4f}")
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

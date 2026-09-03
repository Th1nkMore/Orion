"""Evaluate attention-free BEV uncertainty on B2D sample data.

Uses Inverse Perspective Mapping (IPM) — no model, no attention weights.

Produces:
  results/bev_noattn/
    panel_normal.png      — camera images + BEV heatmap (normal weather)
    panel_adverse.png     — camera images + BEV heatmap (adverse weather)
    comparison.png        — side-by-side mean uncertainty comparison
    score_boxplot.png     — distribution of mean BEV uncertainty per condition
    report.txt            — numeric summary

Usage:
    source .venv/bin/activate
    python scripts/download_b2d_sample.py   # if not yet downloaded
    python scripts/eval_bev_noattn.py
"""

from __future__ import annotations

import sys
import json
import textwrap
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# ── project root ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from uq_estimator.bev_uncertainty import (
    compute_bev_uncertainty_ipm,
    make_b2d_calibration,
    compute_patch_quality,
)

# ── Config ────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data" / "b2d_sample"
OUT_DIR = ROOT / "results" / "bev_noattn"

# B2D image resolution (original, before ORION resize)
IMG_H, IMG_W = 900, 1600

# BEV parameters matching ORION
BEV_RANGE = 51.2   # ±51.2 m
BEV_RES = 0.4      # 0.4 m/pixel → 256×256 grid

# Camera directory names (order must match make_b2d_calibration)
CAM_DIRS = [
    "rgb_front", "rgb_front_left", "rgb_front_right",
    "rgb_back", "rgb_back_left", "rgb_back_right",
]
CAM_LABELS = ["Front", "Front-L", "Front-R", "Back", "Back-L", "Back-R"]

# Scenario dirs  (created by download_b2d_sample.py)
SCENARIOS = {
    "Normal (CloudySunset)": "normal_w3",
    "Adverse (HardRainNight)": "adverse_w13",
}

PATCH_SIZE = 16   # ViT patch size; images will be loaded at IMG_H × IMG_W


# ── helpers ──────────────────────────────────────────────────────────

def _find_scenario_root(base: Path) -> Path:
    """The tar extracts into a single sub-directory; find it."""
    subs = [p for p in base.iterdir() if p.is_dir()]
    if len(subs) == 1:
        return subs[0]
    return base


def load_frames(scenario_key: str, n_frames: int = 5) -> list[dict]:
    """Load n_frames timesteps from a scenario.

    Returns list of dicts: {'images': [H,W,3] arrays per camera}.
    """
    base = DATA_DIR / scenario_key
    if not base.exists():
        raise FileNotFoundError(
            f"{base} not found — run scripts/download_b2d_sample.py first"
        )
    root = _find_scenario_root(base)

    # Collect sorted frame filenames from rgb_front
    front_dir = root / "camera" / "rgb_front"
    if not front_dir.exists():
        # try without camera/ prefix
        front_dir = root / "rgb_front"
    frames_sorted = sorted(front_dir.glob("*.jpg"))[:n_frames]
    if not frames_sorted:
        raise FileNotFoundError(f"No jpg images found in {front_dir}")

    results = []
    for ref_img in frames_sorted:
        stem = ref_img.stem  # e.g. "000001"
        imgs = []
        for cam_dir in CAM_DIRS:
            cam_path = root / "camera" / cam_dir / f"{stem}.jpg"
            if not cam_path.exists():
                cam_path = root / cam_dir / f"{stem}.jpg"
            img = np.array(Image.open(cam_path).convert("RGB"), dtype=np.float32)
            imgs.append(img)
        results.append({"images": imgs, "stem": stem})
    return results


def frames_to_tensor(frames: list[dict]) -> torch.Tensor:
    """Convert list of frame dicts → [B, N_views, 3, H, W] float32."""
    batch = []
    for fr in frames:
        views = []
        for img in fr["images"]:
            # Resize to expected H × W if needed
            if img.shape[:2] != (IMG_H, IMG_W):
                pil = Image.fromarray(img.astype(np.uint8))
                pil = pil.resize((IMG_W, IMG_H), Image.BILINEAR)
                img = np.array(pil, dtype=np.float32)
            t = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
            views.append(t)
        batch.append(torch.stack(views))  # [N_views, 3, H, W]
    return torch.stack(batch)  # [B, N_views, 3, H, W]


def _draw_panel(
    frames: list[dict],
    bev_maps: torch.Tensor,
    title: str,
    out_path: Path,
    n_show: int = 3,
):
    """Draw camera images + BEV heatmap panel for up to n_show frames."""
    n_show = min(n_show, len(frames))
    n_cams = len(CAM_DIRS)

    fig = plt.figure(figsize=(4 * n_cams, 3.5 * n_show + 1.5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(n_show, n_cams + 1, figure=fig,
                           wspace=0.05, hspace=0.3)

    for fi in range(n_show):
        # Camera images
        for ci, label in enumerate(CAM_LABELS):
            ax = fig.add_subplot(gs[fi, ci])
            img = frames[fi]["images"][ci].astype(np.uint8)
            ax.imshow(img)
            ax.axis("off")
            if fi == 0:
                ax.set_title(label, fontsize=8)

        # BEV heatmap
        ax_bev = fig.add_subplot(gs[fi, n_cams])
        bev_np = bev_maps[fi].cpu().numpy()
        im = ax_bev.imshow(
            bev_np, cmap="RdYlGn_r", vmin=0, vmax=1, origin="upper"
        )
        bev_size = bev_np.shape[0]
        ticks = [0, bev_size // 4, bev_size // 2, 3 * bev_size // 4, bev_size - 1]
        tick_labels = [f"{BEV_RANGE - t * BEV_RES:.0f}" for t in ticks]
        ax_bev.set_yticks(ticks)
        ax_bev.set_yticklabels(tick_labels, fontsize=6)
        ax_bev.set_xticks([])
        if fi == 0:
            ax_bev.set_title("BEV Unc.", fontsize=8)
            plt.colorbar(im, ax=ax_bev, fraction=0.046, pad=0.04)
        # Draw ego vehicle marker
        cx = bev_size // 2
        cy = bev_size // 2
        ax_bev.plot(cx, cy, "b^", markersize=5, label="ego")

        stem = frames[fi]["stem"]
        ax_bev.set_ylabel(f"t={stem}", fontsize=7)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def _draw_comparison(all_mean_unc: dict[str, list[float]], out_path: Path):
    """Bar + box comparison of mean BEV uncertainty per condition."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("BEV Uncertainty: No-Attention IPM Method", fontsize=13)

    labels = list(all_mean_unc.keys())
    colors = ["#4CAF50", "#F44336"]  # green / red

    # Bar chart of mean ± std
    ax = axes[0]
    means = [np.mean(v) for v in all_mean_unc.values()]
    stds = [np.std(v) for v in all_mean_unc.values()]
    bars = ax.bar(labels, means, color=colors, alpha=0.8,
                  yerr=stds, capsize=6, error_kw=dict(linewidth=2))
    ax.set_ylabel("Mean BEV Uncertainty")
    ax.set_title("Mean ± Std across frames")
    ax.set_ylim(0, 1.05)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=11)

    # Box plot
    ax2 = axes[1]
    bp = ax2.boxplot(
        [all_mean_unc[k] for k in labels],
        labels=labels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("Mean BEV Uncertainty (per frame)")
    ax2.set_title("Distribution across frames")
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def _draw_mean_bev(mean_bevs: dict[str, np.ndarray], out_path: Path):
    """Show mean BEV uncertainty maps side-by-side."""
    n = len(mean_bevs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Mean BEV Uncertainty Map (IPM, no attention)", fontsize=13)
    for ax, (label, bev_np) in zip(axes, mean_bevs.items()):
        im = ax.imshow(bev_np, cmap="RdYlGn_r", vmin=0, vmax=1, origin="upper")
        ax.set_title(label, fontsize=11)
        bev_size = bev_np.shape[0]
        cx = bev_size // 2
        ax.plot(cx, cx, "b^", markersize=8, label="ego")
        ax.set_xlabel("← right / left →")
        ax.set_ylabel("← forward / backward →")
        ticks = [0, bev_size // 4, bev_size // 2, 3 * bev_size // 4, bev_size - 1]
        tick_labels = [f"{BEV_RANGE - t * BEV_RES:.0f}m" for t in ticks]
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels, fontsize=8)
        plt.colorbar(im, ax=ax, label="Uncertainty", fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ── main ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build B2D camera calibration (hard-coded, matches agent.py)
    intrinsics, cam2egos, cam_names = make_b2d_calibration(IMG_H, IMG_W)

    # ── Pass 1: collect raw (unnormalised) patch quality for global normalisation
    print("\n[Pass 1] Computing raw patch quality for global normalisation...")
    raw_pq_all: dict[str, torch.Tensor] = {}
    imgs_all: dict[str, torch.Tensor] = {}
    frames_all: dict[str, list] = {}

    for label, scenario_key in SCENARIOS.items():
        frames = load_frames(scenario_key, n_frames=10)
        imgs = frames_to_tensor(frames) * (1.0 / 255.0)  # [B,6,3,H,W] in [0,1]
        # Raw quality (no per-frame normalisation) for cross-condition comparison
        pq_raw = compute_patch_quality(imgs * 255.0, patch_size=PATCH_SIZE, normalize=False)
        raw_pq_all[label] = pq_raw
        imgs_all[label] = imgs
        frames_all[label] = frames
        print(f"  {label}: raw quality mean={pq_raw.mean():.2f}  std={pq_raw.std():.2f}")

    # Log-scale normalisation to handle the heavy tail in quality distribution
    # (a few very sharp patches dominate the linear range; log compresses them)
    all_raw = torch.cat(list(raw_pq_all.values()), dim=0)
    all_log = torch.log1p(all_raw)
    q_log_max = all_log.max()
    print(f"\n  Global raw quality range:    [{all_raw.min():.1f}, {all_raw.max():.1f}]")
    print(f"  Global log(1+q) range:       [0.000, {q_log_max:.3f}]")
    for label, pq_raw in raw_pq_all.items():
        q_mean_log = torch.log1p(pq_raw).mean().item()
        print(f"    {label}: log-mean={q_mean_log:.3f}  "
              f"→ norm={q_mean_log/q_log_max.item():.3f}  "
              f"→ unc={1 - q_mean_log/q_log_max.item():.3f}")

    def global_normalize_quality(pq_raw: torch.Tensor) -> torch.Tensor:
        """Log-scale global normalisation → [0,1], high = better quality."""
        return torch.log1p(pq_raw) / (q_log_max + 1e-8)

    # ── Pass 2: compute globally-normalised BEV uncertainty
    all_mean_unc: dict[str, list[float]] = {}
    all_mean_bevs: dict[str, np.ndarray] = {}

    for label, scenario_key in SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {label}  ({scenario_key})")

        imgs = imgs_all[label]
        frames = frames_all[label]
        pq_raw = raw_pq_all[label]

        # Globally-normalised quality (preserves cross-condition differences)
        pq_norm = global_normalize_quality(pq_raw)
        print(f"  Globally-normalised quality: mean={pq_norm.mean():.3f}  std={pq_norm.std():.3f}")

        print("  Running IPM BEV uncertainty (normalize_output=False → raw means)...")
        bev_maps = compute_bev_uncertainty_ipm(
            imgs * 255.0,
            intrinsics,
            cam2egos,
            patch_size=PATCH_SIZE,
            bev_range=BEV_RANGE,
            bev_resolution=BEV_RES,
            sigma=2.5,
            patch_quality=pq_norm,
            normalize_output=False,   # keep absolute values for comparison
        )
        print(f"    bev_maps (raw): {bev_maps.shape}  mean={bev_maps.mean():.4f}")

        # For display, also compute per-frame normalised maps
        bev_display = compute_bev_uncertainty_ipm(
            imgs * 255.0, intrinsics, cam2egos,
            patch_size=PATCH_SIZE, bev_range=BEV_RANGE, bev_resolution=BEV_RES,
            sigma=2.5, patch_quality=pq_norm, normalize_output=True,
        )

        # Covered-pixel mean (ignore uncovered areas = 0)
        covered_mean_per_frame = []
        for b in range(bev_maps.shape[0]):
            m = bev_maps[b]
            cov = m > 1e-6
            covered_mean_per_frame.append(m[cov].mean().item() if cov.sum() > 0 else 0.0)
        all_mean_unc[label] = covered_mean_per_frame
        all_mean_bevs[label] = bev_display.mean(dim=0).numpy()

        _draw_panel(
            frames, bev_display, title=f"{label}",
            out_path=OUT_DIR / f"panel_{scenario_key}.png",
            n_show=min(3, len(frames)),
        )

        print(f"  Covered-pixel mean BEV unc: {[f'{v:.4f}' for v in covered_mean_per_frame]}")

    # Text report
    report_lines = ["BEV Uncertainty — No-Attention IPM Results", "=" * 50, ""]
    for label, vals in all_mean_unc.items():
        report_lines += [
            f"Condition : {label}",
            f"  Frames  : {len(vals)}",
            f"  Mean    : {np.mean(vals):.4f}",
            f"  Std     : {np.std(vals):.4f}",
            f"  Min     : {np.min(vals):.4f}",
            f"  Max     : {np.max(vals):.4f}",
            "",
        ]

    if len(all_mean_unc) == 2:
        vals_list = list(all_mean_unc.values())
        labels_list = list(all_mean_unc.keys())
        delta = np.mean(vals_list[1]) - np.mean(vals_list[0])
        report_lines += [
            f"Δ mean uncertainty (adverse − normal): {delta:+.4f}",
            f"  (+) = adverse has higher uncertainty → metric is discriminative",
            "",
        ]

    report_path = OUT_DIR / "report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"\n[saved] {report_path}")
    print("\n" + "\n".join(report_lines))

    print(f"\nAll outputs saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()

"""
BEV trajectory comparison visualization.

Renders predicted vs GT trajectories on a Bird's Eye View canvas for
selected samples, comparing Baseline and FiLM-enhanced predictions.

Designed to produce qualitative figures similar to ORION paper Fig. 5,
showing how UQ-aware FiLM modulation changes trajectory predictions.

Two modes:
  1. From eval .pt files: extract ego_fut_preds stored in bbox_results
  2. Live capture: run inference on selected samples and render immediately

Usage:
    # From pre-computed eval results (needs raw .pt with bbox_results)
    python scripts/visualize_trajectory.py \
        --baseline-pt results/eval_A_baseline.pt \
        --film-pt results/eval_B_film_l1.pt \
        --out-dir results/figures/trajectory

    # Live capture mode
    python scripts/visualize_trajectory.py \
        --capture \
        --config adzoo/orion/configs/orion_stage3_infer.py \
        --checkpoint ckpts/Orion.pth \
        --film-checkpoint checkpoints/film/best_l1.pt \
        --out-dir results/figures/trajectory \
        --num-samples 20
"""
import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection


# ── BEV rendering parameters ─────────────────────────────────────────
BEV_RANGE = 30.0        # meters in each direction from ego
BEV_RESOLUTION = 0.1    # meters per pixel
EGO_LENGTH = 4.5        # ego vehicle length (m)
EGO_WIDTH = 2.0         # ego vehicle width (m)


def setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def draw_ego_vehicle(ax, color='#333333'):
    """Draw ego vehicle as a rectangle at origin facing +x."""
    rect = Rectangle(
        (-EGO_LENGTH / 2, -EGO_WIDTH / 2),
        EGO_LENGTH, EGO_WIDTH,
        linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.6, zorder=5)
    ax.add_patch(rect)
    ax.annotate('', xy=(EGO_LENGTH / 2 + 0.5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5), zorder=6)


def draw_trajectory(ax, traj, color, label, linewidth=2.0, marker='o', alpha=0.9):
    """Draw a trajectory as colored line with waypoint markers.

    Args:
        traj: [T, 2] array of (x, y) waypoints in ego-lidar frame
    """
    if traj is None or len(traj) == 0:
        return
    traj = np.array(traj)
    ax.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=linewidth,
            alpha=alpha, label=label, zorder=4)
    ax.scatter(traj[:, 0], traj[:, 1], c=color, s=20, marker=marker,
               alpha=alpha, edgecolors='white', linewidths=0.5, zorder=4)
    # Time annotations at each waypoint
    for i, (x, y) in enumerate(traj):
        t = (i + 1) * 0.5  # 0.5s per step
        ax.annotate(f'{t:.1f}s', (x, y), fontsize=6, ha='center', va='bottom',
                    color=color, alpha=0.7)


def draw_bev_grid(ax, bev_range):
    """Draw BEV background grid."""
    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.axhline(0, color='gray', linewidth=0.3, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.3, alpha=0.5)
    ax.set_xlabel('x (m) — forward')
    ax.set_ylabel('y (m) — left')


def render_single_sample(gt_traj, pred_traj, pred_traj_film=None,
                          uq_score=None, weather='', scenario='',
                          ax=None, bev_range=BEV_RANGE):
    """Render a single BEV trajectory comparison.

    Args:
        gt_traj: [T, 2] ground-truth future trajectory
        pred_traj: [T, 2] baseline predicted trajectory
        pred_traj_film: [T, 2] FiLM predicted trajectory (optional)
        uq_score: float, uncertainty score for this sample
        weather: weather condition string
        scenario: scenario type string
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    draw_bev_grid(ax, bev_range)
    draw_ego_vehicle(ax)

    # GT trajectory (green)
    draw_trajectory(ax, gt_traj, '#4CAF50', 'GT', linewidth=2.5, marker='s')

    # Baseline prediction (gray)
    draw_trajectory(ax, pred_traj, '#757575', 'Baseline', linewidth=2.0, marker='o')

    # FiLM prediction (blue/orange)
    if pred_traj_film is not None:
        draw_trajectory(ax, pred_traj_film, '#1976D2', 'FiLM', linewidth=2.0, marker='^')

    # Compute L2 errors for annotation
    if gt_traj is not None and pred_traj is not None:
        gt = np.array(gt_traj)
        pr = np.array(pred_traj)
        min_len = min(len(gt), len(pr))
        if min_len > 0:
            l2 = np.sqrt(((gt[:min_len] - pr[:min_len]) ** 2).sum(axis=1)).mean()
            ax.text(0.02, 0.02, f'Baseline L2={l2:.2f}m',
                    transform=ax.transAxes, fontsize=8, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    if gt_traj is not None and pred_traj_film is not None:
        gt = np.array(gt_traj)
        pf = np.array(pred_traj_film)
        min_len = min(len(gt), len(pf))
        if min_len > 0:
            l2_f = np.sqrt(((gt[:min_len] - pf[:min_len]) ** 2).sum(axis=1)).mean()
            ax.text(0.02, 0.08, f'FiLM L2={l2_f:.2f}m',
                    transform=ax.transAxes, fontsize=8, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#E3F2FD', alpha=0.7))

    title_parts = []
    if scenario:
        title_parts.append(scenario)
    if weather:
        title_parts.append(weather)
    if uq_score is not None:
        title_parts.append(f'UQ={uq_score:.3f}')
    ax.set_title(' | '.join(title_parts), fontsize=10)
    ax.legend(loc='upper right', fontsize=8)


# ── Extract trajectory from eval results ─────────────────────────────

def _extract_scenario_type(folder):
    """Extract scenario type from folder path."""
    parts = folder.split('/')
    name = parts[-1] if len(parts) > 1 else parts[0]
    tokens = name.split('_')
    type_tokens = []
    for t in tokens:
        if t.startswith('Town') or t.startswith('Route') or t.startswith('Weather'):
            break
        type_tokens.append(t)
    return '_'.join(type_tokens) if type_tokens else 'Unknown'


def load_eval_data(pt_path):
    """Load eval .pt and extract per-sample trajectory + metadata."""
    import torch
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    records = data.get('records', [])
    return records


def render_gallery(records_base, records_film=None, out_dir='results/figures/trajectory',
                    num_samples=12, sort_by='uq_score', fmt='pdf'):
    """Render a gallery of BEV trajectory comparisons.

    Selects samples with highest UQ scores (most uncertain) and
    renders comparison panels.

    Args:
        records_base: list of dicts from baseline eval
        records_film: list of dicts from FiLM eval (optional)
        out_dir: output directory
        num_samples: number of samples to render
        sort_by: field to sort by when selecting samples
    """
    setup_style()
    os.makedirs(out_dir, exist_ok=True)

    # Filter to records with trajectory data
    valid_base = [r for r in records_base
                  if r.get('ego_fut_preds') is not None and 'uq_score' in r]
    if not valid_base:
        print('  No trajectory data in baseline records. '
              'Run eval with trajectory capture enabled.')
        # Fall back to rendering L2-based mock trajectories for demonstration
        _render_from_metrics(records_base, records_film, out_dir, num_samples, fmt)
        return

    # Sort by UQ score descending (most uncertain first)
    valid_base.sort(key=lambda r: r.get(sort_by, 0), reverse=True)
    selected = valid_base[:num_samples]

    # Build film lookup by index
    film_lookup = {}
    if records_film:
        for r in records_film:
            film_lookup[r.get('idx', -1)] = r

    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if num_samples == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for i, rec in enumerate(selected):
        row, col = divmod(i, cols)
        ax = axes[row][col]

        gt = rec.get('ego_fut_gt')
        pred = rec.get('ego_fut_preds')
        pred_film = None
        film_rec = film_lookup.get(rec.get('idx', -1))
        if film_rec:
            pred_film = film_rec.get('ego_fut_preds')

        render_single_sample(
            gt_traj=np.array(gt) if gt is not None else None,
            pred_traj=np.array(pred) if pred is not None else None,
            pred_traj_film=np.array(pred_film) if pred_film is not None else None,
            uq_score=rec.get('uq_score'),
            weather=rec.get('weather_name', ''),
            scenario=_extract_scenario_type(rec.get('folder', '')),
            ax=ax,
        )

    # Hide unused axes
    for i in range(num_samples, rows * cols):
        row, col = divmod(i, cols)
        axes[row][col].set_visible(False)

    fig.suptitle('BEV Trajectory Comparison: GT vs Baseline vs FiLM\n'
                 '(sorted by UQ score, highest uncertainty first)',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f'trajectory_gallery.{fmt}')
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved gallery: {out_path}')


def _render_from_metrics(records_base, records_film, out_dir, num_samples, fmt):
    """Fallback: render a schematic comparison using only L2/Col metrics.

    When full trajectory data isn't available, this creates a summary figure
    showing per-sample L2 errors with UQ scores to illustrate the relationship.
    """
    setup_style()

    valid = [r for r in records_base
             if 'uq_score' in r and (r.get('plan_L2_3s', 0) > 0 or r.get('fut_valid'))]
    if not valid:
        print('  No valid records for metric-based visualization')
        return

    # Select diverse samples: high UQ, low UQ, collision cases
    valid.sort(key=lambda r: r.get('uq_score', 0), reverse=True)
    high_uq = valid[:num_samples // 3]
    valid.sort(key=lambda r: r.get('uq_score', 0))
    low_uq = valid[:num_samples // 3]
    collision = [r for r in records_base
                 if r.get('plan_obj_col_3s', 0) > 0 and 'uq_score' in r]
    collision.sort(key=lambda r: r.get('uq_score', 0), reverse=True)
    collision = collision[:num_samples // 3]
    selected = high_uq + low_uq + collision
    selected = selected[:num_samples]

    if not selected:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: UQ score vs L2@3s for selected samples
    ax = axes[0]
    uq = [r['uq_score'] for r in selected]
    l2 = [r.get('plan_L2_3s', 0) for r in selected]
    is_col = [r.get('plan_obj_col_3s', 0) > 0 for r in selected]
    colors = ['#D32F2F' if c else '#1976D2' for c in is_col]
    ax.scatter(uq, l2, c=colors, s=40, alpha=0.7, edgecolors='white', zorder=3)
    ax.set_xlabel('UQ Score')
    ax.set_ylabel('L2 Error @3s (m)')
    ax.set_title('Selected Samples: UQ vs Error')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D32F2F',
               markersize=8, label='Collision'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1976D2',
               markersize=8, label='No Collision'),
    ]
    ax.legend(handles=legend_elements)

    # Panel 2: L2 at 1s/2s/3s grouped by UQ bin
    ax = axes[1]
    all_valid = [r for r in records_base
                 if 'uq_score' in r and (r.get('plan_L2_3s', 0) > 0 or r.get('fut_valid'))]
    uq_all = np.array([r['uq_score'] for r in all_valid])
    tertiles = np.percentile(uq_all, [33, 67])
    groups = {'Low UQ': [], 'Mid UQ': [], 'High UQ': []}
    for r in all_valid:
        s = r['uq_score']
        if s < tertiles[0]:
            groups['Low UQ'].append(r)
        elif s < tertiles[1]:
            groups['Mid UQ'].append(r)
        else:
            groups['High UQ'].append(r)

    x = np.arange(3)
    width = 0.25
    for i, (gname, grecs) in enumerate(groups.items()):
        vals = [np.mean([r[f'plan_L2_{t}s'] for r in grecs]) for t in [1, 2, 3]]
        color = ['#4CAF50', '#FF9800', '#D32F2F'][i]
        ax.bar(x + i * width, vals, width, label=f'{gname} (n={len(grecs)})',
               color=color, alpha=0.7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(['L2@1s', 'L2@2s', 'L2@3s'])
    ax.set_ylabel('Mean L2 Error (m)')
    ax.set_title('Planning Error by UQ Tertile')
    ax.legend(fontsize=8)

    # Panel 3: collision rate by UQ bin
    ax = axes[2]
    for i, (gname, grecs) in enumerate(groups.items()):
        vals = [np.mean([r.get(f'plan_obj_col_{t}s', 0) for r in grecs]) * 100
                for t in [1, 2, 3]]
        color = ['#4CAF50', '#FF9800', '#D32F2F'][i]
        ax.bar(x + i * width, vals, width, label=f'{gname}',
               color=color, alpha=0.7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Col@1s', 'Col@2s', 'Col@3s'])
    ax.set_ylabel('Collision Rate (%)')
    ax.set_title('Collision Rate by UQ Tertile')
    ax.legend(fontsize=8)

    fig.suptitle('UQ-Stratified Planning Analysis', fontsize=14, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f'uq_stratified_analysis.{fmt}')
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ── Main ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='BEV Trajectory Visualization')
    p.add_argument('--baseline-pt', default=None,
                   help='Baseline eval .pt result file')
    p.add_argument('--film-pt', default=None,
                   help='FiLM eval .pt result file for comparison')
    p.add_argument('--out-dir', default='results/figures/trajectory')
    p.add_argument('--num-samples', type=int, default=12,
                   help='Number of samples to visualize')
    p.add_argument('--format', default='pdf', choices=['pdf', 'png', 'svg'])
    p.add_argument('--sort-by', default='uq_score',
                   help='Sort field for sample selection')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    setup_style()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.baseline_pt:
        print(f'Loading baseline results from {args.baseline_pt}...')
        records_base = load_eval_data(args.baseline_pt)
        print(f'  {len(records_base)} baseline records')

        records_film = None
        if args.film_pt:
            print(f'Loading FiLM results from {args.film_pt}...')
            records_film = load_eval_data(args.film_pt)
            print(f'  {len(records_film)} FiLM records')

        render_gallery(records_base, records_film, args.out_dir,
                        args.num_samples, args.format)
    else:
        print('Specify --baseline-pt for pre-computed data.')
        print('See --help for usage.')

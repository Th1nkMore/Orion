"""
Offline BEV-only GIF renderer — no ORION environment required.

Reads the cached trajectory_data.pt and renders pure BEV trajectory
comparison plots (GT vs FiLM) per scenario, assembled into animated GIFs.

Note: the 'baseline' records in the cache were generated with a buggy
disable_film() (gamma ≠ 1) and are NOT a valid baseline; they are shown
with a dashed style and marked as "Baseline*" to indicate this.

Usage:
    python scripts/render_bev_gifs.py \
        --cache results/gifs/trajectory_data.pt \
        --out-dir results/gifs/bev_only \
        --fps 4
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle
import imageio.v2 as imageio


# ── Style constants ────────────────────────────────────────────────────

COLOR_GT   = '#43A047'  # green
COLOR_BASE = '#EF5350'  # red  (buggy, shown dashed)
COLOR_FILM = '#1E88E5'  # blue

BG_COLOR   = '#1a1a2e'
EGO_LENGTH = 4.5
EGO_WIDTH  = 2.0

WEATHER_NAMES = {
    0: 'ClearNoon',        1: 'ClearSunset',
    2: 'CloudyNoon',       3: 'CloudySunset',
    4: 'WetNoon',          5: 'WetSunset',
    6: 'MidRainyNoon',     7: 'MidRainSunset',
    8: 'HardRainNoon',     9: 'HardRainSunset',
    10: 'SoftRainNoon',    11: 'SoftRainSunset',
    12: 'ClearNight',      13: 'ClearNight2',
    14: 'FoggyNoon',       15: 'FoggyNight',
    20: 'HardRain2',       21: 'HardRain3',
    22: 'HardRainNight',
}

ADVERSE_IDS = {6, 7, 8, 9, 10, 11, 14, 15, 20, 21, 22}


def parse_weather_id(folder: str) -> int:
    import re
    m = re.search(r'Weather(\d+)', folder)
    return int(m.group(1)) if m else -1


# ── BEV frame renderer ─────────────────────────────────────────────────

def render_bev_frame(gt_traj, base_traj, film_traj,
                     uq_score, frame_idx, scenario_name, weather_name,
                     l2_base, l2_film, col_base, col_film,
                     fig_size=(8, 6), dpi=100):
    """Render a single BEV comparison frame; returns RGB numpy array."""

    all_pts = []
    for t in [gt_traj, base_traj, film_traj]:
        if t is not None and len(t) > 0:
            all_pts.append(np.array(t))
    if all_pts:
        combined = np.concatenate(all_pts, axis=0)
        max_abs = max(np.abs(combined).max(), 1.0)
    else:
        max_abs = 5.0
    pad = max(1.5, max_abs * 0.3)
    rng = max_abs + pad

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-rng, rng)
    ax.set_ylim(-rng * 0.2, rng * 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.12, linewidth=0.5, color='white')
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)
    ax.set_xlabel('lateral (m)', fontsize=9, color='white')
    ax.set_ylabel('forward (m)', fontsize=9, color='white')

    # Ego vehicle box
    rect = Rectangle((-EGO_WIDTH / 2, -EGO_LENGTH / 2), EGO_WIDTH, EGO_LENGTH,
                      linewidth=2, edgecolor='white', facecolor='#444',
                      alpha=0.8, zorder=5)
    ax.add_patch(rect)
    ax.annotate('', xy=(0, EGO_LENGTH / 2 + 0.4), xytext=(0, EGO_LENGTH / 2),
                arrowprops=dict(arrowstyle='->', color='white', lw=2), zorder=6)

    def _plot(traj, color, label, ls='-', marker='o', lw=2.5, ms=50):
        if traj is None or len(traj) == 0:
            return
        t = np.array(traj)
        ax.plot(t[:, 0], t[:, 1], ls, color=color, linewidth=lw,
                alpha=0.95, label=label, zorder=4)
        ax.scatter(t[:, 0], t[:, 1], c=color, s=ms, marker=marker,
                   alpha=0.95, edgecolors='white', linewidths=0.8, zorder=4)
        for i, (x, y) in enumerate(t):
            if i % 2 == 1 or i == len(t) - 1:
                ax.annotate(f'{(i+1)*0.5:.1f}s', (x, y),
                            fontsize=7, color=color, ha='center', va='bottom',
                            textcoords='offset points', xytext=(0, 6))

    _plot(gt_traj,   COLOR_GT,   'GT',          ls='-',  marker='s', lw=3.0, ms=65)
    _plot(base_traj, COLOR_BASE, 'Baseline*',   ls='--', marker='o', lw=2.0, ms=45)
    _plot(film_traj, COLOR_FILM, 'Ours (FiLM)', ls='-',  marker='^', lw=3.0, ms=55)

    leg = ax.legend(loc='upper left', fontsize=9, framealpha=0.85,
                    facecolor=BG_COLOR, edgecolor='white', labelcolor='white')

    # Title block
    is_adv = parse_weather_id(scenario_name) in ADVERSE_IDS
    adv_tag = ' [ADVERSE]' if is_adv else ' [normal]'
    title_str = f'{scenario_name}\n{weather_name}{adv_tag}  |  Frame {frame_idx}'
    ax.set_title(title_str, fontsize=9, color='white', pad=8)

    # Metrics text box (upper right)
    metric_lines = []
    if uq_score is not None:
        metric_lines.append(f'UQ score: {uq_score:.3f}')
    if l2_film > 0:
        metric_lines.append(f'Ours L2@3s: {l2_film:.2f} m')
    if l2_base > 0:
        metric_lines.append(f'Baseline* L2@3s: {l2_base:.2f} m')
    if col_film > 0:
        metric_lines.append('⚠ Ours: COLLISION')
    if col_base > 0:
        metric_lines.append('⚠ Baseline*: COLLISION')

    if metric_lines:
        metrics_str = '\n'.join(metric_lines)
        ax.text(0.98, 0.98, metrics_str,
                transform=ax.transAxes,
                fontsize=8, color='white', va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d0d1a',
                          edgecolor='white', alpha=0.8))

    # Note about buggy baseline
    ax.text(0.5, 0.01,
            '* Baseline generated with buggy disable_film — not a valid baseline',
            transform=ax.transAxes, fontsize=6.5, color='#aaa',
            ha='center', va='bottom')

    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)[:, :, :3].copy()
    plt.close(fig)
    return img


# ── GIF assembly ───────────────────────────────────────────────────────

def generate_bev_gif(scenario_name, sdata, out_dir, fps):
    folder = sdata.get('folder', '')
    weather_id = sdata.get('weather_id', parse_weather_id(folder))
    weather_name = WEATHER_NAMES.get(weather_id, f'Weather{weather_id}')

    base_records = sdata.get('baseline', [])
    film_records = sdata.get('film', [])

    if not base_records and not film_records:
        print(f'  SKIP {scenario_name}: no records')
        return None

    # Use whichever pass has data; GT comes from baseline records
    if base_records:
        primary = base_records
        film_lookup = {r['frame_idx']: r for r in film_records}
    else:
        primary = film_records
        film_lookup = {}

    frames = []
    for rec in primary:
        fidx = rec['frame_idx']
        film_rec = film_lookup.get(fidx, {})

        gt_traj   = rec.get('ego_fut_gt')
        base_traj = rec.get('ego_fut_preds')       # buggy baseline
        film_traj = film_rec.get('ego_fut_preds')  # real FiLM output

        uq_score = film_rec.get('uq_score') or rec.get('uq_score')

        img = render_bev_frame(
            gt_traj=gt_traj,
            base_traj=base_traj,
            film_traj=film_traj,
            uq_score=uq_score,
            frame_idx=fidx,
            scenario_name=scenario_name,
            weather_name=weather_name,
            l2_base=rec.get('plan_L2_3s', 0.0),
            l2_film=film_rec.get('plan_L2_3s', 0.0),
            col_base=rec.get('plan_obj_col_3s', 0.0),
            col_film=film_rec.get('plan_obj_col_3s', 0.0),
        )
        frames.append(img)

    if not frames:
        print(f'  SKIP {scenario_name}: no frames rendered')
        return None

    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, f'{scenario_name}.gif')
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    size_mb = os.path.getsize(gif_path) / 1024 / 1024
    print(f'  {gif_path}  ({len(frames)} frames, {size_mb:.1f} MB)')
    return gif_path


# ── Main ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Render BEV-only GIFs from cached trajectory data')
    p.add_argument('--cache', default='results/gifs/trajectory_data.pt')
    p.add_argument('--out-dir', default='results/gifs/bev_only')
    p.add_argument('--fps', type=int, default=4)
    p.add_argument('--scenarios', nargs='*', default=None,
                   help='limit to specific scenario names (default: all)')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.cache):
        print(f'Cache not found: {args.cache}')
        return

    print(f'Loading cache: {args.cache}')
    all_data = torch.load(args.cache, map_location='cpu', weights_only=False)
    print(f'  {len(all_data)} scenarios in cache')

    scenarios = list(all_data.keys())
    if args.scenarios:
        scenarios = [s for s in scenarios if any(t in s for t in args.scenarios)]
        print(f'  Filtered to {len(scenarios)} scenarios')

    gif_paths = []
    for i, name in enumerate(scenarios, 1):
        print(f'\n[{i}/{len(scenarios)}] {name}')
        path = generate_bev_gif(name, all_data[name], args.out_dir, args.fps)
        if path:
            gif_paths.append(path)

    print(f'\nDone — {len(gif_paths)} GIFs saved to {args.out_dir}/')


if __name__ == '__main__':
    main()

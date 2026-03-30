"""
Generate trajectory comparison GIFs for selected scenarios.

For each scenario, runs ORION inference twice (baseline + FiLM) and captures
per-frame predicted trajectories alongside GT. Then renders composite frames
(front camera + BEV trajectory plot) and assembles into animated GIFs.

Usage:
    python scripts/generate_trajectory_gifs.py \
        adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth \
        --film-checkpoint checkpoints/film/best_l1l2_col_v3.pt \
        --scenarios ControlLoss_Town04_Route170_Weather14 \
                    ConstructionObstacle_Town10HD_Route74_Weather22 \
                    YieldToEmergencyVehicle_Town04_Route166_Weather10 \
        --out-dir results/gifs \
        --frame-step 5
"""
import argparse
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from mmcv.utils import set_random_seed, Config, load_checkpoint, ProgressBar
from mmcv.models import build_model
from mmcv.datasets import build_dataset
from mmcv.parallel.collate import collate as mm_collate

from eval_openloop import (
    UQScoreCapture, custom_wrap_fp16_model,
    parse_weather_id, is_adverse, WEATHER_NAMES,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import imageio.v2 as imageio
import cv2


# ── Args ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Generate trajectory comparison GIFs')
    p.add_argument('config', help='ORION config file')
    p.add_argument('checkpoint', help='ORION checkpoint file')
    p.add_argument('--film-checkpoint', default='checkpoints/film/best_l1l2_col_v3.pt')
    p.add_argument('--ann-file', default='data/infos/b2d_infos_val.pkl')
    p.add_argument('--data-root', default='data/bench2drive')
    p.add_argument('--scenarios', nargs='+', default=[
        'ControlLoss_Town04_Route170_Weather14',
        'ConstructionObstacle_Town10HD_Route74_Weather22',
        'YieldToEmergencyVehicle_Town04_Route166_Weather10',
    ])
    p.add_argument('--frame-step', type=int, default=5)
    p.add_argument('--out-dir', default='results/gifs')
    p.add_argument('--fps', type=int, default=4)
    p.add_argument('--cache', default='results/gifs/trajectory_data.pt',
                   help='cache trajectory data for offline re-rendering')
    p.add_argument('--render-only', action='store_true',
                   help='skip inference, use cached data')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


# ── Scenario helpers ──────────────────────────────────────────────────

def group_scenarios(data_infos):
    scenarios = defaultdict(list)
    for idx, info in enumerate(data_infos):
        scenarios[info['folder']].append((info['frame_idx'], idx))
    for folder in scenarios:
        scenarios[folder].sort(key=lambda x: x[0])
    return dict(scenarios)


def match_scenario(folder, target_names):
    """Check if folder matches any target scenario name."""
    leaf = folder.split('/')[-1]
    for t in target_names:
        if t in leaf:
            return t
    return None


def collate_single(dataset, idx):
    raw_data = dataset[idx]
    batch = mm_collate([raw_data], samples_per_gpu=1)
    for key, val in batch.items():
        if key != 'img_metas' and torch.is_tensor(val[0]):
            val[0] = val[0].cuda()
        if key == 'input_ids':
            for i in range(len(val[0])):
                for k in range(len(val[0][i])):
                    val[0][i][k] = val[0][i][k].cuda()
    return batch


# ── Build model ───────────────────────────────────────────────────────

def build_orion_model(cfg, checkpoint_path, film_checkpoint=None):
    """Build ORION model with UQ + optional FiLM weights."""
    cfg.model.pts_bbox_head.transformer.use_uncertainty = True
    cfg.model.pts_bbox_head.uq_checkpoint = 'checkpoints/uq/best.pt'

    if film_checkpoint and os.path.exists(film_checkpoint):
        ckpt_peek = torch.load(film_checkpoint, map_location='cpu', weights_only=False)
        if 'film_gamma_l2_weight' in ckpt_peek:
            cfg.model.use_uncertainty_l2 = True

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        custom_wrap_fp16_model(model)

    ckpt = load_checkpoint(model, checkpoint_path, map_location='cpu')

    # Reload UQ weights
    uq_path = cfg.model.pts_bbox_head.get('uq_checkpoint', '')
    if uq_path and os.path.exists(uq_path):
        uq_ckpt = torch.load(uq_path, map_location='cpu', weights_only=False)
        model.pts_bbox_head.uq_estimator.load_state_dict(
            uq_ckpt['model_state_dict'], strict=False)
        print(f'[UQ] Reloaded UQEstimator from {uq_path}')

    # Load FiLM weights
    if film_checkpoint and os.path.exists(film_checkpoint):
        film_ckpt = torch.load(film_checkpoint, map_location='cpu', weights_only=False)
        transformer = model.pts_bbox_head.transformer
        if hasattr(transformer, 'film_gamma') and 'film_gamma_weight' in film_ckpt:
            transformer.film_gamma.weight.data = film_ckpt['film_gamma_weight']
            transformer.film_gamma.bias.data = film_ckpt['film_gamma_bias']
            transformer.film_beta.weight.data = film_ckpt['film_beta_weight']
            transformer.film_beta.bias.data = film_ckpt['film_beta_bias']
            print(f'[UQ] Loaded FiLM L1 weights')
        if hasattr(model, 'film_gamma_l2') and 'film_gamma_l2_weight' in film_ckpt:
            model.film_gamma_l2.weight.data = film_ckpt['film_gamma_l2_weight']
            model.film_gamma_l2.bias.data = film_ckpt['film_gamma_l2_bias']
            model.film_beta_l2.weight.data = film_ckpt['film_beta_l2_weight']
            model.film_beta_l2.bias.data = film_ckpt['film_beta_l2_bias']
            print(f'[UQ] Loaded FiLM L2 weights')

    if 'CLASSES' in ckpt.get('meta', {}):
        model.CLASSES = ckpt['meta']['CLASSES']

    model.cuda()

    from torch.nn import DataParallel
    model = DataParallel(model, device_ids=[0])
    model.eval()
    return model


def disable_film(model):
    """Reset FiLM layers to identity (gamma=1, beta=0) for baseline inference."""
    m = model.module if hasattr(model, 'module') else model
    transformer = m.pts_bbox_head.transformer
    if hasattr(transformer, 'film_gamma'):
        torch.nn.init.ones_(transformer.film_gamma.weight)
        torch.nn.init.zeros_(transformer.film_gamma.bias)
        torch.nn.init.zeros_(transformer.film_beta.weight)
        torch.nn.init.zeros_(transformer.film_beta.bias)
        print('[Baseline] FiLM L1 reset to identity')
    if hasattr(m, 'film_gamma_l2'):
        torch.nn.init.ones_(m.film_gamma_l2.weight)
        torch.nn.init.zeros_(m.film_gamma_l2.bias)
        torch.nn.init.zeros_(m.film_beta_l2.weight)
        torch.nn.init.zeros_(m.film_beta_l2.bias)
        print('[Baseline] FiLM L2 reset to identity')


def reload_film(model, film_checkpoint):
    """Re-load FiLM weights for FiLM inference pass."""
    m = model.module if hasattr(model, 'module') else model
    film_ckpt = torch.load(film_checkpoint, map_location='cpu', weights_only=False)
    transformer = m.pts_bbox_head.transformer
    if hasattr(transformer, 'film_gamma') and 'film_gamma_weight' in film_ckpt:
        transformer.film_gamma.weight.data = film_ckpt['film_gamma_weight']
        transformer.film_gamma.bias.data = film_ckpt['film_gamma_bias']
        transformer.film_beta.weight.data = film_ckpt['film_beta_weight']
        transformer.film_beta.bias.data = film_ckpt['film_beta_bias']
        print('[FiLM] Reloaded FiLM L1 weights')
    if hasattr(m, 'film_gamma_l2') and 'film_gamma_l2_weight' in film_ckpt:
        m.film_gamma_l2.weight.data = film_ckpt['film_gamma_l2_weight']
        m.film_gamma_l2.bias.data = film_ckpt['film_gamma_l2_bias']
        m.film_beta_l2.weight.data = film_ckpt['film_beta_l2_weight']
        m.film_beta_l2.bias.data = film_ckpt['film_beta_l2_bias']
        print('[FiLM] Reloaded FiLM L2 weights')


# ── Inference loop ────────────────────────────────────────────────────

def run_scenario_inference(model, dataset, frame_indices, data_infos,
                           frame_step, uq_capture):
    """Run inference on sampled frames, return per-frame trajectory data."""
    sampled = frame_indices[::frame_step]
    records = []

    for frame_idx, ds_idx in sampled:
        try:
            batch = collate_single(dataset, ds_idx)
            with torch.no_grad():
                result = model(batch, return_loss=False)
        except Exception as e:
            print(f'  [WARN] frame {frame_idx}: {e}')
            continue

        if isinstance(result, list) and len(result) > 0:
            r = result[0]
        elif isinstance(result, dict):
            r = result
        else:
            continue

        pts = r.get('pts_bbox', r)
        if not isinstance(pts, dict) or 'ego_fut_preds' not in pts:
            continue

        ego_fut_preds = pts['ego_fut_preds']
        if isinstance(ego_fut_preds, torch.Tensor):
            ego_fut_preds = ego_fut_preds.cpu().numpy()
        ego_fut_preds = np.array(ego_fut_preds, dtype=np.float32)

        # GT future trajectory from dataset's computed ego_fut_trajs
        ego_fut_gt = None
        if 'ego_fut_trajs' in batch:
            gt_raw = batch['ego_fut_trajs']
            if isinstance(gt_raw, list):
                gt_raw = gt_raw[0]
            if isinstance(gt_raw, torch.Tensor):
                gt_raw = gt_raw.cpu().numpy().squeeze()
            ego_fut_gt = np.cumsum(gt_raw, axis=-2).astype(np.float32)
            if ego_fut_gt.ndim > 2:
                ego_fut_gt = ego_fut_gt[0]
            ego_fut_gt = ego_fut_gt[:, :2] if ego_fut_gt.shape[-1] > 2 else ego_fut_gt

        # UQ score for this frame
        uq_score = None
        if uq_capture.scores:
            uq_score = uq_capture.scores[-1].squeeze().item()

        # Planning metrics
        metric = r.get('metric_results', {})

        records.append({
            'frame_idx': frame_idx,
            'ds_idx': ds_idx,
            'ego_fut_preds': ego_fut_preds[:6, :2],  # [6, 2]
            'ego_fut_gt': ego_fut_gt[:6] if ego_fut_gt is not None else None,
            'uq_score': uq_score,
            'plan_L2_3s': float(metric.get('plan_L2_3s', 0)),
            'plan_obj_col_3s': float(metric.get('plan_obj_col_3s', 0)),
        })

    return records


# ── Rendering ─────────────────────────────────────────────────────────

EGO_LENGTH = 4.5
EGO_WIDTH = 2.0

# Trajectory colors (consistent across camera and BEV)
COLOR_GT = '#43A047'
COLOR_BASE = '#EF5350'
COLOR_FILM = '#1E88E5'


def _auto_bev_range(gt_traj, base_traj, film_traj, pad=0.4):
    """Compute tight BEV range to make all trajectories clearly visible."""
    pts = []
    for t in [gt_traj, base_traj, film_traj]:
        if t is not None and len(t) > 0:
            pts.append(np.array(t))
    if not pts:
        return 3.0
    all_pts = np.concatenate(pts, axis=0)
    max_abs = max(np.abs(all_pts).max(), 0.3)
    return max_abs + pad


def _draw_cam_trajectories(cam_img, gt_traj, base_traj, film_traj):
    """Draw trajectory direction indicators directly on the camera image.

    Uses a simple perspective mapping: lidar (x_lat, y_fwd) ->
    camera pixel with vanishing point at image center-top.
    """
    h, w = cam_img.shape[:2]
    overlay = cam_img.copy()

    vanish_x, vanish_y = w // 2, int(h * 0.42)
    ego_x, ego_y = w // 2, h - 10

    def _lidar_to_pixel(x_lat, y_fwd, scale=200):
        """Approximate perspective: map lidar (lat, fwd) to pixel coords."""
        t = min(y_fwd * scale / max(h, 1), 0.85)
        t = max(t, 0.0)
        px = int(ego_x + (vanish_x - ego_x) * t - x_lat * scale * (1 - t * 0.6))
        py = int(ego_y + (vanish_y - ego_y) * t)
        return px, py

    trajs = [
        (gt_traj, COLOR_GT, 'GT'),
        (base_traj, COLOR_BASE, 'Baseline'),
        (film_traj, COLOR_FILM, 'Ours'),
    ]

    for traj, hex_color, label in trajs:
        if traj is None or len(traj) == 0:
            continue
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        bgr = (b, g, r)

        pts_px = []
        for x_lat, y_fwd in traj:
            px, py = _lidar_to_pixel(x_lat, y_fwd)
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            pts_px.append((px, py))

        for i in range(len(pts_px) - 1):
            cv2.line(overlay, pts_px[i], pts_px[i + 1], bgr, 4, cv2.LINE_AA)
        for i, (px, py) in enumerate(pts_px):
            cv2.circle(overlay, (px, py), 6, bgr, -1, cv2.LINE_AA)
            cv2.circle(overlay, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)

        if pts_px:
            cv2.putText(overlay, label, (pts_px[-1][0] + 8, pts_px[-1][1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2, cv2.LINE_AA)
            cv2.putText(overlay, label, (pts_px[-1][0] + 8, pts_px[-1][1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    result = cv2.addWeighted(overlay, 0.7, cam_img, 0.3, 0)
    return result


def render_composite_frame(cam_img, gt_traj, base_traj, film_traj,
                           uq_score, frame_idx, scenario_name, weather,
                           l2_base, l2_film, col_base, col_film):
    """Render composite: camera with trajectory overlay (top) + BEV inset (bottom-right).

    Returns:
        np.ndarray [H, W, 3] uint8 RGB image
    """
    cam_with_traj = _draw_cam_trajectories(cam_img.copy(), gt_traj, base_traj, film_traj)

    # --- Render BEV as a matplotlib figure → image ---
    bev_range = _auto_bev_range(gt_traj, base_traj, film_traj, pad=0.5)
    fig_bev, ax_bev = plt.subplots(1, 1, figsize=(4.0, 4.0), dpi=100)
    fig_bev.patch.set_facecolor('#1a1a2e')
    ax_bev.set_facecolor('#1a1a2e')

    ax_bev.set_xlim(-bev_range, bev_range)
    ax_bev.set_ylim(-bev_range * 0.3, bev_range * 1.5)
    ax_bev.set_aspect('equal')
    ax_bev.grid(True, alpha=0.15, linewidth=0.5, color='white')

    # Ego vehicle
    rect = Rectangle((-EGO_WIDTH / 2, -EGO_LENGTH / 2), EGO_WIDTH, EGO_LENGTH,
                      linewidth=2, edgecolor='white', facecolor='#555', alpha=0.7, zorder=5)
    ax_bev.add_patch(rect)
    ax_bev.annotate('', xy=(0, EGO_LENGTH / 2 + 0.15), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='white', lw=2), zorder=6)

    def _plot_bev(ax, traj, color, label, marker='o', lw=3.5, ms=60, zorder=4):
        if traj is None or len(traj) == 0:
            return
        t = np.array(traj)
        ax.plot(t[:, 0], t[:, 1], '-', color=color, linewidth=lw,
                alpha=0.95, label=label, zorder=zorder)
        ax.scatter(t[:, 0], t[:, 1], c=color, s=ms, marker=marker,
                   alpha=0.95, edgecolors='white', linewidths=1.0, zorder=zorder)
        for i, (x, y) in enumerate(t):
            if i % 2 == 1 or i == len(t) - 1:
                ax.annotate(f'{(i+1)*0.5:.1f}s', (x, y),
                            fontsize=7, fontweight='bold',
                            ha='center', va='bottom', color=color,
                            textcoords='offset points', xytext=(0, 6))

    _plot_bev(ax_bev, gt_traj, COLOR_GT, 'GT', marker='s', lw=3.5, ms=70, zorder=3)
    _plot_bev(ax_bev, base_traj, COLOR_BASE, 'Baseline', marker='o', lw=3.0, ms=55)
    _plot_bev(ax_bev, film_traj, COLOR_FILM, 'Ours (FiLM)', marker='^', lw=3.0, ms=55)

    ax_bev.legend(loc='upper left', fontsize=8, framealpha=0.85,
                  facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax_bev.tick_params(colors='white', labelsize=7)
    for spine in ax_bev.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)
    ax_bev.set_xlabel('lateral (m)', fontsize=8, color='white')
    ax_bev.set_ylabel('forward (m)', fontsize=8, color='white')

    fig_bev.canvas.draw()
    buf = np.frombuffer(fig_bev.canvas.buffer_rgba(), dtype=np.uint8)
    bw, bh = fig_bev.canvas.get_width_height()
    bev_img = buf.reshape(bh, bw, 4)[:, :, :3].copy()
    plt.close(fig_bev)

    # --- Compose final image ---
    cam_h, cam_w = cam_with_traj.shape[:2]
    out_w = cam_w
    bev_inset_h = int(cam_h * 0.55)
    bev_inset_w = bev_inset_h
    bev_resized = cv2.resize(bev_img, (bev_inset_w, bev_inset_h), interpolation=cv2.INTER_AREA)

    # Place BEV inset in bottom-right corner with border
    pad = 6
    y0 = cam_h - bev_inset_h - pad
    x0 = cam_w - bev_inset_w - pad
    cam_with_traj[y0:y0 + bev_inset_h, x0:x0 + bev_inset_w] = bev_resized
    cv2.rectangle(cam_with_traj, (x0 - 1, y0 - 1),
                  (x0 + bev_inset_w, y0 + bev_inset_h), (255, 255, 255), 2)

    # Info overlay (top-left)
    info_lines = [scenario_name]
    info_lines.append(f'Weather: {weather}  |  Frame: {frame_idx}')
    if uq_score is not None:
        info_lines.append(f'UQ Score: {uq_score:.3f}')

    y_text = 28
    for line in info_lines:
        cv2.putText(cam_with_traj, line, (12, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(cam_with_traj, line, (12, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y_text += 24

    # Metrics overlay (top-right, above BEV inset)
    metric_lines = []
    if l2_base > 0:
        metric_lines.append(f'Baseline L2: {l2_base:.2f}m')
    if l2_film > 0:
        metric_lines.append(f'Ours L2: {l2_film:.2f}m')
    if col_base > 0:
        metric_lines.append('Baseline: COLLISION')
    if col_film > 0:
        metric_lines.append('Ours: COLLISION')

    y_metric = 28
    for line in metric_lines:
        tw = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
        cv2.putText(cam_with_traj, line, (cam_w - tw - 12, y_metric),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(cam_with_traj, line, (cam_w - tw - 12, y_metric),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y_metric += 22

    return cam_with_traj


def load_camera_image(data_root, folder, frame_idx, target_size=(960, 540)):
    """Load front camera image for a frame."""
    img_path = os.path.join(data_root, folder, 'camera', 'rgb_front', f'{frame_idx:05d}.jpg')
    if not os.path.exists(img_path):
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    img = Image.open(img_path)
    img = img.resize(target_size, Image.LANCZOS)
    return np.array(img)


# ── GIF assembly ──────────────────────────────────────────────────────

def generate_gif(scenario_name, base_records, film_records, data_root,
                 folder, weather_name, out_dir, fps):
    """Generate a single GIF for one scenario."""

    # Build film lookup by frame_idx
    film_lookup = {r['frame_idx']: r for r in film_records}

    frames = []
    for rec in base_records:
        fidx = rec['frame_idx']
        film_rec = film_lookup.get(fidx, {})

        cam_img = load_camera_image(data_root, folder, fidx)

        gt_traj = rec.get('ego_fut_gt')
        base_traj = rec.get('ego_fut_preds')
        film_traj = film_rec.get('ego_fut_preds')

        uq_score = film_rec.get('uq_score', rec.get('uq_score'))

        composite = render_composite_frame(
            cam_img=cam_img,
            gt_traj=gt_traj,
            base_traj=base_traj,
            film_traj=film_traj,
            uq_score=uq_score,
            frame_idx=fidx,
            scenario_name=scenario_name,
            weather=weather_name,
            l2_base=rec.get('plan_L2_3s', 0),
            l2_film=film_rec.get('plan_L2_3s', 0),
            col_base=rec.get('plan_obj_col_3s', 0),
            col_film=film_rec.get('plan_obj_col_3s', 0),
        )
        frames.append(composite)

    if not frames:
        print(f'  [WARN] No frames rendered for {scenario_name}')
        return None

    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, f'{scenario_name}.gif')
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    file_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
    print(f'  Saved: {gif_path} ({len(frames)} frames, {file_size_mb:.1f} MB)')
    return gif_path


# ── Main ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_random_seed(args.seed, deterministic=True)
    os.makedirs(args.out_dir, exist_ok=True)

    all_data = {}

    if args.render_only and os.path.exists(args.cache):
        print(f'Loading cached trajectory data from {args.cache}...')
        all_data = torch.load(args.cache, map_location='cpu', weights_only=False)
    else:
        # ── Build model and dataset ──
        cfg = Config.fromfile(args.config)
        cfg.model.pts_bbox_head.transformer.use_uncertainty = True
        cfg.model.pts_bbox_head.uq_checkpoint = 'checkpoints/uq/best.pt'
        if args.ann_file:
            cfg.data.test.ann_file = args.ann_file

        dataset = build_dataset(cfg.data.test)
        data_infos = dataset.data_infos
        print(f'Dataset: {len(dataset)} samples')

        scenarios = group_scenarios(data_infos)

        # Find matching scenario folders
        target_folders = {}
        for folder in scenarios:
            name = match_scenario(folder, args.scenarios)
            if name:
                target_folders[name] = folder
                print(f'  Matched: {name} -> {folder} ({len(scenarios[folder])} frames)')

        if not target_folders:
            print('ERROR: No matching scenarios found!')
            return

        total_sampled = sum(
            len(scenarios[f][::args.frame_step])
            for f in target_folders.values())
        print(f'Total frames to process per pass: {total_sampled}')
        print(f'Estimated time: ~{total_sampled * 2 * 2 / 60:.0f} min (2 passes)')

        # ── Build model ──
        print('\nBuilding ORION model...')
        model = build_orion_model(cfg, args.checkpoint, args.film_checkpoint)

        uq_capture = UQScoreCapture()
        uq_capture.register(model)

        # ── Pass 1: Baseline (FiLM disabled) ──
        print('\n═══ Pass 1: Baseline inference (FiLM → identity) ═══')
        disable_film(model)

        for scenario_name, folder in target_folders.items():
            print(f'\n  Processing: {scenario_name}')
            frame_indices = scenarios[folder]
            uq_capture.scores.clear()
            torch.cuda.empty_cache()

            t0 = time.time()
            records = run_scenario_inference(
                model, dataset, frame_indices, data_infos,
                args.frame_step, uq_capture)
            elapsed = time.time() - t0
            print(f'    {len(records)} frames captured in {elapsed:.1f}s '
                  f'({elapsed/max(len(records),1):.2f}s/frame)')

            if scenario_name not in all_data:
                all_data[scenario_name] = {
                    'folder': folder,
                    'weather_id': parse_weather_id(folder),
                }
            all_data[scenario_name]['baseline'] = records

        # ── Pass 2: FiLM ──
        print('\n═══ Pass 2: FiLM inference ═══')
        reload_film(model, args.film_checkpoint)

        for scenario_name, folder in target_folders.items():
            print(f'\n  Processing: {scenario_name}')
            frame_indices = scenarios[folder]
            uq_capture.scores.clear()
            torch.cuda.empty_cache()

            t0 = time.time()
            records = run_scenario_inference(
                model, dataset, frame_indices, data_infos,
                args.frame_step, uq_capture)
            elapsed = time.time() - t0
            print(f'    {len(records)} frames captured in {elapsed:.1f}s '
                  f'({elapsed/max(len(records),1):.2f}s/frame)')

            all_data[scenario_name]['film'] = records

        # ── Cache trajectory data ──
        cache_path = args.cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(all_data, cache_path)
        print(f'\nCached trajectory data: {cache_path}')

    # ── Render GIFs ──
    print('\n═══ Rendering GIFs ═══')
    gif_paths = []
    for scenario_name, sdata in all_data.items():
        folder = sdata['folder']
        weather_id = sdata['weather_id']
        weather_name = WEATHER_NAMES.get(weather_id, f'Weather{weather_id}')
        base_records = sdata.get('baseline', [])
        film_records = sdata.get('film', [])

        if not base_records:
            print(f'  SKIP {scenario_name}: no baseline data')
            continue

        print(f'\n  Rendering {scenario_name} ({len(base_records)} frames)...')
        gif_path = generate_gif(
            scenario_name, base_records, film_records,
            args.data_root, folder, weather_name,
            args.out_dir, args.fps)
        if gif_path:
            gif_paths.append(gif_path)

    print(f'\nDone! Generated {len(gif_paths)} GIFs:')
    for p in gif_paths:
        print(f'  {p}')


if __name__ == '__main__':
    main()

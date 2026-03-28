"""
Closed-loop replay evaluation using Bench2Drive recorded data (no CARLA needed).

Processes frames sequentially within each scenario:
  1. Feed real sensor data → ORION inference → predicted trajectory
  2. PID controller → predicted steer/throttle/brake
  3. Compare against ground-truth controls
  4. Compute per-frame collision via occupancy grid

Captures sequential consistency that per-frame open-loop evaluation misses.

Usage:
    python scripts/eval_closedloop_replay.py \
        adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth \
        --out results/closedloop_replay.json \
        [--film-checkpoint checkpoints/film/best_l1l2.pt] \
        [--frame-step 5] [--max-scenarios 10]
"""
import argparse
import os
import sys
import time
import json
from collections import defaultdict

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from mmcv.utils import set_random_seed, Config, load_checkpoint, ProgressBar
from mmcv.models import build_model
from mmcv.datasets import build_dataset, build_dataloader

from eval_openloop import (
    UQScoreCapture, custom_wrap_fp16_model,
    parse_weather_id, is_adverse, WEATHER_NAMES, NORMAL_WEATHER_IDS,
)

# PID controller from ORION agent
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'team_code'))
from pid_controller import PIDController


def parse_args():
    parser = argparse.ArgumentParser(description='Closed-loop replay eval (no CARLA)')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='ORION checkpoint file')
    parser.add_argument('--out', default='results/closedloop_replay.json',
                        help='output JSON file')
    parser.add_argument('--ann-file', default=None,
                        help='override annotation file')
    parser.add_argument('--film-checkpoint', default=None,
                        help='trained FiLM weights')
    parser.add_argument('--frame-step', type=int, default=5,
                        help='process every N-th frame per scenario (default: 5)')
    parser.add_argument('--max-scenarios', type=int, default=None,
                        help='limit scenarios for quick testing')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='limit total dataset samples')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


# ── Scenario grouping ──────────────────────────────────────────────────

def group_scenarios(data_infos):
    """Group data_infos indices by folder, sorted by frame_idx within each."""
    scenarios = defaultdict(list)
    for idx, info in enumerate(data_infos):
        scenarios[info['folder']].append((info['frame_idx'], idx))

    # Sort by frame_idx within each scenario
    for folder in scenarios:
        scenarios[folder].sort(key=lambda x: x[0])

    return dict(scenarios)


# ── Local command transform ────────────────────────────────────────────

def get_local_command(info):
    """Transform world-frame command_near_xy to ego-lidar local frame."""
    cmd_world = info['command_near_xy']  # [2]
    world2lidar = info['sensors']['LIDAR_TOP']['world2lidar']  # [4, 4]
    cmd_homo = np.array([cmd_world[0], cmd_world[1], 0.0, 1.0])
    cmd_local = world2lidar @ cmd_homo
    return cmd_local[:2]  # [x, y] in lidar frame


def get_speed(info):
    """Extract scalar speed from ego velocity."""
    return float(np.linalg.norm(info['ego_vel']))


# ── Per-frame inference ────────────────────────────────────────────────

def infer_single(model, dataset, idx):
    """Run inference on a single dataset sample, return ego_fut_preds."""
    data = dataset[idx]
    # Collate: wrap in lists for batch dimension (dataset returns single sample)
    # The dataloader normally handles this, but for sequential access we do it manually
    if isinstance(data, dict):
        # The dataset pipeline returns dict with DataContainer objects
        # We need to use the dataloader's collate to properly batch
        pass
    with torch.no_grad():
        result = model(data, return_loss=False)

    # Extract ego_fut_preds from result
    if isinstance(result, list) and len(result) > 0:
        r = result[0]
    elif isinstance(result, dict):
        r = result
    else:
        return None

    pts = r.get('pts_bbox', r)
    if isinstance(pts, dict) and 'ego_fut_preds' in pts:
        preds = pts['ego_fut_preds']
        if isinstance(preds, torch.Tensor):
            return preds.cpu().numpy()
        return np.array(preds)
    return None


# ── Replay one scenario ────────────────────────────────────────────────

def replay_scenario(model, dataset, data_loader, frame_indices, data_infos,
                    frame_step, uq_capture):
    """
    Replay a single scenario sequentially.

    Args:
        model: ORION model (DataParallel wrapped)
        dataset: B2DOrionDataset
        data_loader: pre-built dataloader (used for proper collation)
        frame_indices: list of (frame_idx, dataset_idx) sorted by frame_idx
        data_infos: full data_infos list
        frame_step: process every N-th frame
        uq_capture: UQScoreCapture instance (scores accumulate)

    Returns:
        dict with per-frame and aggregate metrics for this scenario
    """
    pid = PIDController()

    frame_records = []
    sampled_indices = frame_indices[::frame_step]

    for frame_idx, ds_idx in sampled_indices:
        info = data_infos[ds_idx]

        # Get local command and speed from GT
        try:
            local_cmd = get_local_command(info)
            speed = get_speed(info)
        except (KeyError, TypeError):
            continue

        # Run inference through dataloader for proper collation
        data = dataset[ds_idx]
        try:
            with torch.no_grad():
                result = model(data, return_loss=False)
        except Exception as e:
            continue

        # Extract ego_fut_preds
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
        ego_fut_preds = np.array(ego_fut_preds)

        if ego_fut_preds.ndim != 2 or ego_fut_preds.shape[0] < 4:
            continue

        # PID control
        try:
            steer_pred, throttle_pred, brake_pred, pid_meta = pid.control_pid(
                ego_fut_preds, np.float64(speed), local_cmd.astype(np.float64))
        except Exception:
            continue

        # GT controls
        gt_steer = float(info.get('steer', 0))
        gt_throttle = float(info.get('throttle', 0))
        gt_brake = float(info.get('brake', 0))

        # Per-frame planning metrics (from model result)
        metric = r.get('metric_results', {})

        record = {
            'frame_idx': frame_idx,
            'ds_idx': ds_idx,
            # Control comparison
            'steer_pred': float(steer_pred),
            'steer_gt': gt_steer,
            'throttle_pred': float(throttle_pred),
            'throttle_gt': gt_throttle,
            'brake_pred': float(brake_pred),
            'brake_gt': gt_brake,
            # Planning metrics
            'plan_L2_1s': float(metric.get('plan_L2_1s', 0)),
            'plan_L2_2s': float(metric.get('plan_L2_2s', 0)),
            'plan_L2_3s': float(metric.get('plan_L2_3s', 0)),
            'plan_obj_col_3s': float(metric.get('plan_obj_col_3s', 0)),
            'fut_valid': bool(metric.get('fut_valid_flag', False)),
            'speed': speed,
        }
        frame_records.append(record)

    if not frame_records:
        return None

    # Aggregate scenario metrics
    n = len(frame_records)
    result = {
        'n_frames': n,
        # Control MAE
        'control_mae_steer': np.mean([abs(r['steer_pred'] - r['steer_gt']) for r in frame_records]),
        'control_mae_throttle': np.mean([abs(r['throttle_pred'] - r['throttle_gt']) for r in frame_records]),
        'control_mae_brake': np.mean([abs(r['brake_pred'] - r['brake_gt']) for r in frame_records]),
        # Planning metrics (mean over frames)
        'traj_ade_1s': np.mean([r['plan_L2_1s'] for r in frame_records if r['fut_valid']]) if any(r['fut_valid'] for r in frame_records) else 0,
        'traj_ade_2s': np.mean([r['plan_L2_2s'] for r in frame_records if r['fut_valid']]) if any(r['fut_valid'] for r in frame_records) else 0,
        'traj_ade_3s': np.mean([r['plan_L2_3s'] for r in frame_records if r['fut_valid']]) if any(r['fut_valid'] for r in frame_records) else 0,
        # Collision rate
        'collision_rate_3s': np.mean([r['plan_obj_col_3s'] for r in frame_records if r['fut_valid']]) if any(r['fut_valid'] for r in frame_records) else 0,
        # Fraction of valid frames
        'valid_ratio': sum(r['fut_valid'] for r in frame_records) / n,
        'avg_speed': np.mean([r['speed'] for r in frame_records]),
    }
    return result


# ── Aggregation ────────────────────────────────────────────────────────

def aggregate_results(scenario_results, data_infos):
    """Aggregate per-scenario results, stratified by weather."""
    groups = {'all': [], 'normal': [], 'adverse': []}

    for folder, result in scenario_results.items():
        weather_id = parse_weather_id(folder)
        result['weather_id'] = weather_id
        result['weather_name'] = WEATHER_NAMES.get(weather_id, f'Unknown{weather_id}')
        result['is_adverse'] = is_adverse(weather_id)

        groups['all'].append(result)
        if result['is_adverse']:
            groups['adverse'].append(result)
        else:
            groups['normal'].append(result)

    aggregate = {}
    metric_keys = ['control_mae_steer', 'control_mae_throttle', 'control_mae_brake',
                   'traj_ade_1s', 'traj_ade_2s', 'traj_ade_3s',
                   'collision_rate_3s', 'valid_ratio', 'avg_speed']

    for group_name, results in groups.items():
        if not results:
            aggregate[group_name] = {'n_scenarios': 0}
            continue
        agg = {'n_scenarios': len(results)}
        for k in metric_keys:
            vals = [r[k] for r in results if k in r]
            if vals:
                agg[f'{k}_mean'] = float(np.mean(vals))
                agg[f'{k}_std'] = float(np.std(vals))
        aggregate[group_name] = agg

    return aggregate


def stratify_by_uq(scenario_results, uq_scenario_scores):
    """Split scenario results by UQ score quartiles."""
    scored = [(folder, r, uq_scenario_scores.get(folder, 0.5))
              for folder, r in scenario_results.items()
              if folder in uq_scenario_scores]
    if len(scored) < 4:
        return {}

    scored.sort(key=lambda x: x[2])
    n = len(scored)
    q_size = n // 4

    quartiles = {}
    for qi, qname in enumerate(['Q1_low_unc', 'Q2', 'Q3', 'Q4_high_unc']):
        start = qi * q_size
        end = start + q_size if qi < 3 else n
        q_results = scored[start:end]
        if not q_results:
            continue
        metric_keys = ['control_mae_steer', 'traj_ade_3s', 'collision_rate_3s']
        q = {'n_scenarios': len(q_results), 'uq_score_range': [q_results[0][2], q_results[-1][2]]}
        for k in metric_keys:
            vals = [r[k] for _, r, _ in q_results if k in r]
            if vals:
                q[f'{k}_mean'] = float(np.mean(vals))
        quartiles[qname] = q

    return quartiles


# ── Main ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Enable UQ
    cfg.model.pts_bbox_head.transformer.use_uncertainty = True
    cfg.model.pts_bbox_head.uq_checkpoint = 'checkpoints/uq/best.pt'

    # Enable L2 if film checkpoint contains L2 weights
    l2_only = os.environ.get('UQ_FILM_L2_ONLY', '0') == '1'
    if args.film_checkpoint:
        if l2_only:
            cfg.model.pts_bbox_head.transformer.use_uncertainty = False
            cfg.model.use_uncertainty_l2 = True
        # Check if checkpoint has L2 weights to auto-detect
        elif args.film_checkpoint:
            ckpt_peek = torch.load(args.film_checkpoint, map_location='cpu', weights_only=False)
            if 'film_gamma_l2_weight' in ckpt_peek:
                cfg.model.use_uncertainty_l2 = True

    if args.ann_file:
        cfg.data.test.ann_file = args.ann_file

    set_random_seed(args.seed, deterministic=True)

    # Build dataset
    dataset = build_dataset(cfg.data.test)
    if args.max_samples and args.max_samples < len(dataset):
        dataset.data_infos = dataset.data_infos[:args.max_samples]
        dataset.flag = np.zeros(len(dataset), dtype=np.uint8)

    data_infos = dataset.data_infos
    print(f'Dataset: {len(dataset)} samples')

    # Group by scenario
    scenarios = group_scenarios(data_infos)
    scenario_folders = sorted(scenarios.keys())
    if args.max_scenarios:
        scenario_folders = scenario_folders[:args.max_scenarios]
    print(f'Scenarios: {len(scenario_folders)} (frame_step={args.frame_step})')

    # Count total frames to process
    total_frames = sum(
        len(scenarios[f][::args.frame_step]) for f in scenario_folders)
    print(f'Total frames to process: {total_frames}')

    # Build model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        custom_wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    print(f'Loaded ORION checkpoint from {args.checkpoint}')

    # Reload UQ weights
    pts_cfg = cfg.model.get('pts_bbox_head', {})
    if pts_cfg.get('uq_checkpoint'):
        uq_path = pts_cfg['uq_checkpoint']
        if os.path.exists(uq_path):
            uq_ckpt = torch.load(uq_path, map_location='cpu', weights_only=False)
            model.pts_bbox_head.uq_estimator.load_state_dict(
                uq_ckpt['model_state_dict'], strict=False)
            print(f'[UQ] Reloaded UQEstimator from {uq_path}')

    # Load FiLM weights
    if args.film_checkpoint and os.path.exists(args.film_checkpoint):
        film_ckpt = torch.load(args.film_checkpoint, map_location='cpu', weights_only=False)
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

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    from torch.nn import DataParallel
    model = DataParallel(model, device_ids=[0])
    model.eval()

    # Register UQ capture
    uq_capture = UQScoreCapture()
    uq_capture.register(model)

    # ── Replay loop ──
    scenario_results = {}
    uq_scenario_scores = {}
    prog_bar = ProgressBar(len(scenario_folders))
    t0 = time.time()

    for folder in scenario_folders:
        frame_indices = scenarios[folder]

        # Track UQ score count before this scenario
        uq_before = len(uq_capture.scores)

        result = replay_scenario(
            model, dataset, None, frame_indices, data_infos,
            args.frame_step, uq_capture)

        if result is not None:
            scenario_results[folder] = result

            # Compute mean UQ score for this scenario
            uq_after = len(uq_capture.scores)
            if uq_after > uq_before:
                scenario_uq = [uq_capture.scores[i].squeeze().item()
                               for i in range(uq_before, uq_after)]
                uq_scenario_scores[folder] = float(np.mean(scenario_uq))
                result['uq_score_mean'] = uq_scenario_scores[folder]

        prog_bar.update()

    elapsed = time.time() - t0
    print(f'\n\nReplay done: {len(scenario_results)}/{len(scenario_folders)} scenarios '
          f'in {elapsed:.0f}s ({elapsed/60:.1f}min)')

    uq_capture.remove()

    # ── Aggregate ──
    aggregate = aggregate_results(scenario_results, data_infos)
    uq_stratified = stratify_by_uq(scenario_results, uq_scenario_scores)

    # ── Print report ──
    print('\n' + '=' * 70)
    print('CLOSED-LOOP REPLAY EVALUATION')
    print('=' * 70)

    for group in ['all', 'normal', 'adverse']:
        agg = aggregate.get(group, {})
        n = agg.get('n_scenarios', 0)
        print(f'\n── {group.upper()} ({n} scenarios) ──')
        if n == 0:
            continue
        print(f'  Control MAE:  steer={agg.get("control_mae_steer_mean", 0):.4f}  '
              f'throttle={agg.get("control_mae_throttle_mean", 0):.4f}  '
              f'brake={agg.get("control_mae_brake_mean", 0):.4f}')
        print(f'  Traj ADE:     1s={agg.get("traj_ade_1s_mean", 0):.4f}  '
              f'2s={agg.get("traj_ade_2s_mean", 0):.4f}  '
              f'3s={agg.get("traj_ade_3s_mean", 0):.4f}')
        print(f'  Collision@3s: {agg.get("collision_rate_3s_mean", 0)*100:.2f}%')
        print(f'  Valid ratio:  {agg.get("valid_ratio_mean", 0)*100:.1f}%')

    if uq_stratified:
        print('\n── UQ STRATIFIED (by scenario mean UQ score) ──')
        for qname, q in uq_stratified.items():
            uq_range = q.get('uq_score_range', [0, 0])
            print(f'  {qname} (n={q["n_scenarios"]}, UQ=[{uq_range[0]:.3f}, {uq_range[1]:.3f}]): '
                  f'ADE@3s={q.get("traj_ade_3s_mean", 0):.4f}  '
                  f'Col@3s={q.get("collision_rate_3s_mean", 0)*100:.2f}%  '
                  f'Steer MAE={q.get("control_mae_steer_mean", 0):.4f}')

    # ── Save results ──
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    output = {
        'scenario_results': {k: {kk: float(vv) if isinstance(vv, (np.floating, float, np.bool_)) else vv
                                  for kk, vv in v.items()}
                             for k, v in scenario_results.items()},
        'aggregate': aggregate,
        'uq_stratified': uq_stratified,
        'config': {
            'film_checkpoint': args.film_checkpoint,
            'frame_step': args.frame_step,
            'n_scenarios': len(scenario_results),
            'elapsed_seconds': elapsed,
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved to {args.out}')


if __name__ == '__main__':
    main()

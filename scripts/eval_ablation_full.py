"""
Full ablation evaluation with FiLM hot-swap.

Loads model ONCE, then hot-swaps FiLM weights to evaluate 4 groups:
  A = Baseline (no FiLM)
  B = FiLM L1 only (QT-Former)
  C = FiLM L2 only (VAE)
  D = FiLM L1 + L2

This avoids reloading the 7.5B-param model for each group (~95min saved per swap).

Usage:
    python scripts/eval_ablation_full.py \
        adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth \
        --film-l1 checkpoints/film/best_l1.pt \
        --film-l2 checkpoints/film/best_l2.pt \
        --film-l1l2 checkpoints/film/best_l1l2.pt \
        --out-dir results/ablation \
        [--max-samples 500]
"""
import argparse
import os
import sys
import time
import json
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

# Add project root and scripts dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmcv.utils import set_random_seed, Config, load_checkpoint, ProgressBar
from mmcv.models import build_model
from mmcv.datasets import build_dataset, build_dataloader

# Import reusable functions from eval_openloop
from eval_openloop import (
    run_inference, collect_per_sample_metrics, compute_aggregate_stats,
    compute_uq_correlation, UQScoreCapture, custom_wrap_fp16_model,
    WEATHER_NAMES, NORMAL_WEATHER_IDS, parse_weather_id, is_adverse,
)

# ── Ablation groups ─────────────────────────────────────────────────────
ABLATION_GROUPS = OrderedDict([
    ('A', {'label': 'Baseline', 'l1': False, 'l2': False, 'ckpt_key': None}),
    ('B', {'label': 'L1 only',  'l1': True,  'l2': False, 'ckpt_key': 'l1'}),
    ('C', {'label': 'L2 only',  'l1': False, 'l2': True,  'ckpt_key': 'l2'}),
    ('D', {'label': 'L1+L2',    'l1': True,  'l2': True,  'ckpt_key': 'l1l2'}),
])


def parse_args():
    parser = argparse.ArgumentParser(description='Full ablation eval with FiLM hot-swap')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='ORION checkpoint file')
    parser.add_argument('--film-l1', default='checkpoints/film/best_l1.pt',
                        help='FiLM L1 checkpoint (Group B)')
    parser.add_argument('--film-l2', default='checkpoints/film/best_l2.pt',
                        help='FiLM L2 checkpoint (Group C)')
    parser.add_argument('--film-l1l2', default='checkpoints/film/best_l1l2.pt',
                        help='FiLM L1+L2 checkpoint (Group D)')
    parser.add_argument('--out-dir', default='results/ablation',
                        help='output directory')
    parser.add_argument('--ann-file', default=None,
                        help='override annotation file')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='limit samples for quick testing')
    parser.add_argument('--groups', default='A,B,C,D',
                        help='comma-separated groups to evaluate (default: A,B,C,D)')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def reset_film_to_identity(model):
    """Reset all FiLM params to identity init (gamma*x + beta = x when gamma=1, beta=0)."""
    transformer = model.pts_bbox_head.transformer
    if hasattr(transformer, 'film_gamma'):
        nn.init.zeros_(transformer.film_gamma.weight)
        nn.init.ones_(transformer.film_gamma.bias)
        nn.init.zeros_(transformer.film_beta.weight)
        nn.init.zeros_(transformer.film_beta.bias)
    if hasattr(model, 'film_gamma_l2'):
        nn.init.zeros_(model.film_gamma_l2.weight)
        nn.init.ones_(model.film_gamma_l2.bias)
        nn.init.zeros_(model.film_beta_l2.weight)
        nn.init.zeros_(model.film_beta_l2.bias)


def load_film_weights(model, film_ckpt):
    """Load FiLM weights from checkpoint dict in-place."""
    transformer = model.pts_bbox_head.transformer
    # FiLM L1
    if hasattr(transformer, 'film_gamma') and 'film_gamma_weight' in film_ckpt:
        transformer.film_gamma.weight.data.copy_(film_ckpt['film_gamma_weight'])
        transformer.film_gamma.bias.data.copy_(film_ckpt['film_gamma_bias'])
        transformer.film_beta.weight.data.copy_(film_ckpt['film_beta_weight'])
        transformer.film_beta.bias.data.copy_(film_ckpt['film_beta_bias'])
    # FiLM L2
    if hasattr(model, 'film_gamma_l2') and 'film_gamma_l2_weight' in film_ckpt:
        model.film_gamma_l2.weight.data.copy_(film_ckpt['film_gamma_l2_weight'])
        model.film_gamma_l2.bias.data.copy_(film_ckpt['film_gamma_l2_bias'])
        model.film_beta_l2.weight.data.copy_(film_ckpt['film_beta_l2_weight'])
        model.film_beta_l2.bias.data.copy_(film_ckpt['film_beta_l2_bias'])


def configure_film_group(model, group_id, film_checkpoints):
    """Configure model for a specific ablation group by toggling flags and swapping weights."""
    cfg = ABLATION_GROUPS[group_id]
    transformer = model.pts_bbox_head.transformer

    # Reset all FiLM to identity first
    reset_film_to_identity(model)

    # Toggle L1 (QT-Former FiLM)
    transformer.use_uncertainty = cfg['l1']

    # Toggle L2 (VAE FiLM)
    if hasattr(model, 'use_uncertainty_l2'):
        model.use_uncertainty_l2 = cfg['l2']

    # Load weights if this group has a checkpoint
    ckpt_key = cfg['ckpt_key']
    if ckpt_key and ckpt_key in film_checkpoints:
        load_film_weights(model, film_checkpoints[ckpt_key])

    print(f'  Configured: L1={cfg["l1"]}, L2={cfg["l2"]}, '
          f'ckpt={"identity" if not ckpt_key else ckpt_key}')


def print_comparison_table(all_results):
    """Print side-by-side comparison table."""
    print('\n' + '=' * 90)
    print('ABLATION COMPARISON')
    print('=' * 90)

    header = f'{"Group":20s} | {"Split":8s} | {"L2@1s":7s} | {"L2@2s":7s} | {"L2@3s":7s} | {"Col@1s":7s} | {"Col@2s":7s} | {"Col@3s":7s}'
    print(header)
    print('-' * len(header))

    metrics = ['plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s',
               'plan_obj_col_1s', 'plan_obj_col_2s', 'plan_obj_col_3s']

    for gid, gdata in all_results.items():
        stats = gdata['stats']
        label = f'{gid} ({ABLATION_GROUPS[gid]["label"]})'
        for split in ['all', 'normal', 'adverse']:
            s = stats.get(split, {})
            vals = []
            for m in metrics:
                if 'col' in m:
                    vals.append(f'{s.get(f"{m}_rate", 0)*100:6.2f}%')
                else:
                    vals.append(f'{s.get(f"{m}_mean", 0):7.4f}')
            print(f'{label:20s} | {split:8s} | {" | ".join(vals)}')
        print('-' * len(header))

    # Relative improvement vs baseline
    if 'A' in all_results:
        base = all_results['A']['stats']
        print('\nRelative change vs Baseline (negative = better for L2/Col):')
        for gid in all_results:
            if gid == 'A':
                continue
            s = all_results[gid]['stats']
            for split in ['all', 'adverse']:
                for m in ['plan_L2_3s_mean', 'plan_obj_col_3s_rate']:
                    b = base.get(split, {}).get(m, 0)
                    c = s.get(split, {}).get(m, 0)
                    if b > 0:
                        pct = (c - b) / b * 100
                        short_m = m.replace('plan_', '').replace('_mean', '').replace('_rate', '')
                        print(f'  {gid} ({ABLATION_GROUPS[gid]["label"]}) {split} {short_m}: {pct:+.1f}%')


def main():
    args = parse_args()
    groups_to_run = [g.strip() for g in args.groups.split(',')]
    for g in groups_to_run:
        if g not in ABLATION_GROUPS:
            print(f'Unknown group: {g}. Valid: {list(ABLATION_GROUPS.keys())}')
            return

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Build config with ALL FiLM layers allocated ──
    cfg = Config.fromfile(args.config)
    # Enable both L1 and L2 so layers are created (we toggle at runtime)
    cfg.model.pts_bbox_head.transformer.use_uncertainty = True
    cfg.model.use_uncertainty_l2 = True
    cfg.model.pts_bbox_head.uq_checkpoint = 'checkpoints/density_uq/best.pt'

    if args.ann_file:
        cfg.data.test.ann_file = args.ann_file

    set_random_seed(args.seed, deterministic=True)

    # ── Build dataset and dataloader (once) ──
    dataset = build_dataset(cfg.data.test)
    if args.max_samples and args.max_samples < len(dataset):
        dataset.data_infos = dataset.data_infos[:args.max_samples]
        dataset.flag = np.zeros(len(dataset), dtype=np.uint8)
        print(f'Truncated dataset to {len(dataset)} samples')

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    print(f'Dataset: {len(dataset)} samples')

    # ── Build model and load checkpoint (once) ──
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        custom_wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    print(f'Loaded ORION checkpoint from {args.checkpoint}')

    # Reload UQ checkpoint
    pts_cfg = cfg.model.get('pts_bbox_head', {})
    if pts_cfg.get('uq_checkpoint'):
        uq_path = pts_cfg['uq_checkpoint']
        if os.path.exists(uq_path):
            uq_ckpt = torch.load(uq_path, map_location='cpu', weights_only=False)
            from uq_estimator.density import get_uq_state_dict
            model.pts_bbox_head.uq_estimator.load_state_dict(
                get_uq_state_dict(uq_ckpt), strict=False)
            print(f'[UQ] Reloaded UQEstimator from {uq_path}')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # ── Load FiLM checkpoints into memory ──
    film_checkpoints = {}
    ckpt_map = {'l1': args.film_l1, 'l2': args.film_l2, 'l1l2': args.film_l1l2}
    for key, path in ckpt_map.items():
        if path and os.path.exists(path):
            film_checkpoints[key] = torch.load(path, map_location='cpu', weights_only=False)
            print(f'[FiLM] Loaded {key} checkpoint from {path}')
        else:
            print(f'[FiLM] WARNING: {key} checkpoint not found at {path}')

    # ── Move to GPU (once) ──
    from torch.nn import DataParallel
    model = DataParallel(model, device_ids=[0])
    model_inner = model.module  # unwrap for FiLM access

    # ── Run evaluation for each group ──
    all_results = {}
    total_t0 = time.time()

    for gid in groups_to_run:
        gcfg = ABLATION_GROUPS[gid]
        print(f'\n{"="*60}')
        print(f'Group {gid}: {gcfg["label"]}')
        print(f'{"="*60}')

        configure_film_group(model_inner, gid, film_checkpoints)

        # Register UQ capture hook
        uq_capture = UQScoreCapture()
        uq_capture.register(model)

        # Run inference
        t0 = time.time()
        bbox_results = run_inference(model, data_loader)
        elapsed = time.time() - t0
        print(f'\nInference: {elapsed:.1f}s ({elapsed/max(len(dataset),1):.2f}s/sample)')

        uq_capture.remove()

        # Collect metrics
        uq_scores = uq_capture.get_scores()
        print(f'Captured {len(uq_scores)} UQ scores for {len(bbox_results)} samples')
        records = collect_per_sample_metrics(bbox_results, dataset.data_infos, uq_scores)
        stats = compute_aggregate_stats(records)
        corr = compute_uq_correlation(records)

        # Quick summary
        s = stats.get('all', {})
        print(f'  L2@3s={s.get("plan_L2_3s_mean", 0):.4f}  '
              f'Col@3s={s.get("plan_obj_col_3s_rate", 0)*100:.2f}%')

        # Save per-group results
        group_out = os.path.join(args.out_dir, f'eval_{gid}_{gcfg["label"].replace(" ", "_")}.pt')
        torch.save({'records': records, 'stats': stats, 'correlation': corr,
                     'group': gid, 'label': gcfg['label'],
                     'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, group_out)
        print(f'  Saved: {group_out}')

        all_results[gid] = {'records': records, 'stats': stats, 'correlation': corr}

    total_elapsed = time.time() - total_t0
    print(f'\nTotal eval time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)')

    # ── Print comparison table ──
    print_comparison_table(all_results)

    # ── Save comparison JSON ──
    json_out = os.path.join(args.out_dir, 'ablation_comparison.json')
    json_data = {}
    for gid, gdata in all_results.items():
        json_data[f'{gid} ({ABLATION_GROUPS[gid]["label"]})'] = {
            k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else int(vv)
                 for kk, vv in v.items()}
            for k, v in gdata['stats'].items()
        }
    with open(json_out, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f'\nComparison saved to {json_out}')


if __name__ == '__main__':
    main()

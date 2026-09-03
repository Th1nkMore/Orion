#!/usr/bin/env python3
"""Recompute frame + clip open-loop stats from an existing eval_openloop *.pt file.

No mmcv / ORION inference required.

Usage:
    python scripts/aggregate_openloop_results.py results/eval_openloop_full.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from uq_estimator.openloop_aggregate import build_summary_payload, run_aggregation


def _print_stats_block(title, stats, count_key='n_valid'):
    print(f'\n── {title} ──')
    for group in ['all', 'normal', 'adverse']:
        s = stats.get(group, {})
        n_count = s.get(count_key, s.get('n_total', 0))
        n_frames = s.get('n_frames', '')
        suffix = f', {n_frames} frames' if n_frames else ''
        print(f'  [{group.upper()}] {n_count} {count_key.replace("n_", "")}s{suffix}')
        if s.get(count_key, s.get('n_valid', 0)) == 0:
            print('    (no valid samples)')
            continue
        print(f'    L2 (m):  1s={s["plan_L2_1s_mean"]:.4f}  '
              f'2s={s["plan_L2_2s_mean"]:.4f}  3s={s["plan_L2_3s_mean"]:.4f}')
        if 'avg_l2_2s' in s:
            print(f'    Avg. L2 (paper @2s): {s["avg_l2_2s"]:.4f}')
        print(f'    Col rate: 1s={s["plan_obj_col_1s_rate"]:.4f}  '
              f'2s={s["plan_obj_col_2s_rate"]:.4f}  3s={s["plan_obj_col_3s_rate"]:.4f}')
        if 'uq_score_mean' in s:
            print(f'    UQ: mean={s["uq_score_mean"]:.4f}  median={s["uq_score_median"]:.4f}')


def _frame_uq_correlation(records):
    import numpy as np
    valid = [r for r in records if r.get('fut_valid') and 'uq_score' in r]
    if len(valid) < 10:
        return {}
    uq = np.array([r['uq_score'] for r in valid])
    l2_3s = np.array([r['plan_L2_3s'] for r in valid])
    col_3s = np.array([r['plan_obj_col_3s'] for r in valid])
    corr = {}
    try:
        from scipy.stats import spearmanr, pearsonr
        sp_l2, sp_l2_p = spearmanr(uq, l2_3s)
        sp_col, sp_col_p = spearmanr(uq, col_3s)
        pe_l2, pe_l2_p = pearsonr(uq, l2_3s)
        corr['spearman_uq_vs_L2_3s'] = {'rho': float(sp_l2), 'p': float(sp_l2_p)}
        corr['spearman_uq_vs_col_3s'] = {'rho': float(sp_col), 'p': float(sp_col_p)}
        corr['pearson_uq_vs_L2_3s'] = {'r': float(pe_l2), 'p': float(pe_l2_p)}
    except ImportError:
        if float(np.std(uq)) > 0 and float(np.std(l2_3s)) > 0:
            corr['pearson_uq_vs_L2_3s'] = float(np.corrcoef(uq, l2_3s)[0, 1])
    try:
        from sklearn.metrics import roc_auc_score
        labels = np.array([1 if r['is_adverse'] else 0 for r in records if 'uq_score' in r])
        scores = np.array([r['uq_score'] for r in records if 'uq_score' in r])
        if len(set(labels.tolist())) == 2:
            corr['auroc_adverse'] = float(roc_auc_score(labels, scores))
    except ImportError:
        pass
    return corr


def main():
    parser = argparse.ArgumentParser(description='Re-aggregate open-loop eval results')
    parser.add_argument('pt_path', help='eval_openloop *.pt with records[]')
    parser.add_argument('--no-write', action='store_true', help='print only, do not overwrite')
    args = parser.parse_args()

    print(f'Loading {args.pt_path}')
    data = torch.load(args.pt_path, map_location='cpu', weights_only=False)
    records = data['records']
    print(f'  {len(records)} frame records')

    stats_frame, stats_clip, clip_records, corr, clip_corr = run_aggregation(
        records, _frame_uq_correlation,
    )

    print('\n' + '=' * 70)
    print('Open-Loop Aggregation (frame micro + clip macro)')
    print('=' * 70)
    _print_stats_block('FRAME-LEVEL (micro)', stats_frame, count_key='n_valid')
    _print_stats_block('CLIP-LEVEL (macro, paper Avg. L2)', stats_clip, count_key='n_clips')
    if corr.get('auroc_adverse') is not None:
        print(f'\n  Frame AUROC(UQ→adverse): {corr["auroc_adverse"]:.4f}')
    if clip_corr.get('auroc_adverse') is not None:
        print(f'  Clip  AUROC(UQ→adverse): {clip_corr["auroc_adverse"]:.4f}')
    print('=' * 70)

    summary = build_summary_payload(stats_frame, stats_clip, clip_records, corr, clip_corr)
    print(f'\nPaper Avg. L2 (clip macro, all): {summary["avg_l2_2s_clip_macro"]:.4f} m')
    print(f'Frame L2@2s (micro, all):        {summary["stats_frame"]["all"]["plan_L2_2s_mean"]:.4f} m')

    if args.no_write:
        return

    data['stats'] = stats_frame
    data['stats_frame'] = stats_frame
    data['stats_clip'] = stats_clip
    data['clip_records'] = clip_records
    data['correlation'] = corr
    data['correlation_clip'] = clip_corr
    data['n_clips'] = len(clip_records)
    torch.save(data, args.pt_path)

    json_out = args.pt_path.replace('.pt', '_summary.json')
    with open(json_out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nUpdated {args.pt_path}')
    print(f'Summary  {json_out}')


if __name__ == '__main__':
    main()

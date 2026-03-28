"""
Open-loop evaluation script for ORION with UQ score analysis.

Runs ORION inference on the validation set, collects per-sample planning
metrics (L2 error, collision rate) and UQ scores, then reports aggregate
statistics split by normal vs adverse weather conditions.

Usage:
    python scripts/eval_openloop.py \
        adzoo/orion/configs/orion_stage3_infer.py \
        ckpts/Orion.pth \
        --out results/eval_openloop.pt
"""
import argparse
import os
import sys
import time
import json
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmcv.utils import (
    get_dist_info, set_random_seed, Config, load_checkpoint, ProgressBar
)
from mmcv.models import build_model
from mmcv.datasets import build_dataset, build_dataloader

# ── Weather classification ──────────────────────────────────────────────
# CARLA weather IDs: 0-3 are clear/cloudy daytime (normal)
# Everything else is adverse (rain, wet, fog, night)
NORMAL_WEATHER_IDS = {0, 1, 2, 3}

WEATHER_NAMES = {
    0: 'ClearNoon', 1: 'ClearSunset', 2: 'CloudyNoon', 3: 'CloudySunset',
    5: 'WetNoon', 6: 'WetSunset', 7: 'MidRainyNoon', 8: 'MidRainSunset',
    9: 'WetCloudyNoon', 10: 'WetCloudySunset', 11: 'HardRainNoon',
    12: 'HardRainSunset', 13: 'SoftRainNoon', 14: 'SoftRainSunset',
    15: 'ClearNight', 18: 'CloudyNight', 19: 'WetNight',
    20: 'WetCloudyNight', 21: 'MidRainyNight', 22: 'HardRainNight',
    23: 'SoftRainNight', 25: 'FoggyNoon', 26: 'FoggySunset',
}


def parse_weather_id(folder: str) -> int:
    """Extract weather ID from folder name like 'v1/Scene_Town_Route_WeatherN'."""
    parts = folder.split('_')
    for p in reversed(parts):
        if p.startswith('Weather'):
            return int(p.replace('Weather', ''))
    raise ValueError(f'Cannot parse weather from folder: {folder}')


def is_adverse(weather_id: int) -> bool:
    return weather_id not in NORMAL_WEATHER_IDS


# ── UQ Score Hook ───────────────────────────────────────────────────────
class UQScoreCapture:
    """Forward hook to capture UQ scores from UQEstimator during inference."""

    def __init__(self):
        self.scores = []
        self._handle = None

    def hook_fn(self, module, input, output):
        # output is UQOutput dataclass with .score attribute  # [B, 1]
        self.scores.append(output.score.detach().cpu())

    def register(self, model):
        """Register hook on UQEstimator if it exists."""
        head = model
        # Handle DataParallel wrapper
        if hasattr(model, 'module'):
            head = model.module
        # Navigate to pts_bbox_head.uq_estimator
        if hasattr(head, 'pts_bbox_head') and hasattr(head.pts_bbox_head, 'uq_estimator'):
            self._handle = head.pts_bbox_head.uq_estimator.register_forward_hook(self.hook_fn)
            print('[UQ] Registered score capture hook on UQEstimator')
            return True
        print('[UQ] WARNING: UQEstimator not found, scores will not be captured')
        return False

    def remove(self):
        if self._handle:
            self._handle.remove()

    def get_scores(self) -> list:
        """Return list of scalar UQ scores."""
        return [s.squeeze().item() for s in self.scores]


# ── Custom FP16 wrapper (from test.py) ──────────────────────────────────
custom_fp16 = dict(map_head=False, pts_bbox_head=False)

def custom_wrap_fp16_model(model):
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True
    for module_name, v in custom_fp16.items():
        model._modules[module_name].fp16_enabled = v


# ── Main ────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='ORION open-loop eval with UQ analysis')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='ORION checkpoint file')
    parser.add_argument('--out', default='results/eval_openloop.pt',
                        help='output file for per-sample results')
    parser.add_argument('--ann-file', default=None,
                        help='override annotation file (default: from config)')
    parser.add_argument('--film-checkpoint', default=None,
                        help='trained FiLM weights (enables FiLM in transformer)')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def run_inference(model, data_loader):
    """Run single-GPU inference, return list of bbox results."""
    model.eval()
    bbox_results = []
    prog_bar = ProgressBar(len(data_loader.dataset))

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(data, return_loss=False)

        if isinstance(result, dict):
            if 'bbox_results' in result:
                bbox_results.extend(result['bbox_results'])
                batch_size = len(result['bbox_results'])
            else:
                batch_size = 1
        else:
            bbox_results.extend(result)
            batch_size = len(result)

        for _ in range(batch_size):
            prog_bar.update()

    return bbox_results


def collect_per_sample_metrics(bbox_results, data_infos, uq_scores):
    """Collect per-sample planning metrics and metadata."""
    records = []
    for i, (result, info) in enumerate(zip(bbox_results, data_infos)):
        metric = result.get('metric_results', {})
        weather_id = parse_weather_id(info['folder'])

        record = {
            'idx': i,
            'folder': info['folder'],
            'frame_idx': info['frame_idx'],
            'weather_id': weather_id,
            'weather_name': WEATHER_NAMES.get(weather_id, f'Unknown{weather_id}'),
            'is_adverse': is_adverse(weather_id),
            'fut_valid': bool(metric.get('fut_valid_flag', False)),
            # Planning metrics (L2 error in meters)
            'plan_L2_1s': float(metric.get('plan_L2_1s', 0)),
            'plan_L2_2s': float(metric.get('plan_L2_2s', 0)),
            'plan_L2_3s': float(metric.get('plan_L2_3s', 0)),
            # Collision metrics
            'plan_obj_col_1s': float(metric.get('plan_obj_col_1s', 0)),
            'plan_obj_col_2s': float(metric.get('plan_obj_col_2s', 0)),
            'plan_obj_col_3s': float(metric.get('plan_obj_col_3s', 0)),
            'plan_obj_box_col_1s': float(metric.get('plan_obj_box_col_1s', 0)),
            'plan_obj_box_col_2s': float(metric.get('plan_obj_box_col_2s', 0)),
            'plan_obj_box_col_3s': float(metric.get('plan_obj_box_col_3s', 0)),
        }
        # Attach UQ score if available
        if i < len(uq_scores):
            record['uq_score'] = uq_scores[i]

        records.append(record)
    return records


def compute_aggregate_stats(records):
    """Compute aggregate statistics grouped by normal/adverse."""
    groups = {'all': records, 'normal': [], 'adverse': []}
    for r in records:
        if r['is_adverse']:
            groups['adverse'].append(r)
        else:
            groups['normal'].append(r)

    stats = {}
    for group_name, group_records in groups.items():
        # Include all samples with non-zero L2 (model always outputs 6-step traj)
        valid = [r for r in group_records if r['plan_L2_3s'] > 0 or r['fut_valid']]
        n_total = len(group_records)
        n_valid = len(valid)

        if not valid:
            stats[group_name] = {'n_total': n_total, 'n_valid': 0}
            continue

        s = {'n_total': n_total, 'n_valid': n_valid}

        # Planning L2 (mean over valid samples)
        for t in ['1s', '2s', '3s']:
            vals = [r[f'plan_L2_{t}'] for r in valid]
            s[f'plan_L2_{t}_mean'] = np.mean(vals)
            s[f'plan_L2_{t}_std'] = np.std(vals)

        # Collision rate (fraction of valid samples with collision)
        for t in ['1s', '2s', '3s']:
            col_vals = [r[f'plan_obj_col_{t}'] for r in valid]
            box_col_vals = [r[f'plan_obj_box_col_{t}'] for r in valid]
            s[f'plan_obj_col_{t}_rate'] = np.mean(col_vals)
            s[f'plan_obj_box_col_{t}_rate'] = np.mean(box_col_vals)

        # UQ scores
        uq_vals = [r['uq_score'] for r in group_records if 'uq_score' in r]
        if uq_vals:
            s['uq_score_mean'] = np.mean(uq_vals)
            s['uq_score_std'] = np.std(uq_vals)
            s['uq_score_median'] = np.median(uq_vals)

        stats[group_name] = s

    return stats


def compute_uq_correlation(records):
    """Compute correlation between UQ scores and planning errors."""
    valid = [r for r in records if r['fut_valid'] and 'uq_score' in r]
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
        corr['spearman_uq_vs_L2_3s'] = {'rho': sp_l2, 'p': sp_l2_p}
        corr['spearman_uq_vs_col_3s'] = {'rho': sp_col, 'p': sp_col_p}
        corr['pearson_uq_vs_L2_3s'] = {'r': pe_l2, 'p': pe_l2_p}
    except ImportError:
        # Fallback: numpy-only Pearson
        if np.std(uq) > 0 and np.std(l2_3s) > 0:
            corr['pearson_uq_vs_L2_3s'] = float(np.corrcoef(uq, l2_3s)[0, 1])

    # AUROC: can UQ score distinguish normal vs adverse?
    try:
        from sklearn.metrics import roc_auc_score
        labels = np.array([1 if r['is_adverse'] else 0 for r in records if 'uq_score' in r])
        scores = np.array([r['uq_score'] for r in records if 'uq_score' in r])
        if len(np.unique(labels)) == 2:
            corr['auroc_adverse'] = float(roc_auc_score(labels, scores))
    except ImportError:
        pass

    return corr


def print_report(stats, corr):
    """Print formatted evaluation report."""
    print('\n' + '=' * 70)
    print('ORION Open-Loop Evaluation with UQ Analysis')
    print('=' * 70)

    for group in ['all', 'normal', 'adverse']:
        s = stats.get(group, {})
        n_total = s.get('n_total', 0)
        n_valid = s.get('n_valid', 0)
        print(f'\n── {group.upper()} ({n_total} samples, {n_valid} valid) ──')

        if n_valid == 0:
            print('  No valid samples')
            continue

        print(f'  Planning L2 (m):  1s={s["plan_L2_1s_mean"]:.4f}  '
              f'2s={s["plan_L2_2s_mean"]:.4f}  3s={s["plan_L2_3s_mean"]:.4f}')
        print(f'  Collision rate:   1s={s["plan_obj_col_1s_rate"]:.4f}  '
              f'2s={s["plan_obj_col_2s_rate"]:.4f}  3s={s["plan_obj_col_3s_rate"]:.4f}')
        print(f'  Box collision:    1s={s["plan_obj_box_col_1s_rate"]:.4f}  '
              f'2s={s["plan_obj_box_col_2s_rate"]:.4f}  3s={s["plan_obj_box_col_3s_rate"]:.4f}')

        if 'uq_score_mean' in s:
            print(f'  UQ score:         mean={s["uq_score_mean"]:.4f}  '
                  f'std={s["uq_score_std"]:.4f}  median={s["uq_score_median"]:.4f}')

    if corr:
        print(f'\n── UQ CORRELATION ──')
        if 'spearman_uq_vs_L2_3s' in corr:
            c = corr['spearman_uq_vs_L2_3s']
            print(f'  Spearman(UQ, L2@3s):   rho={c["rho"]:.4f}  p={c["p"]:.2e}')
        if 'spearman_uq_vs_col_3s' in corr:
            c = corr['spearman_uq_vs_col_3s']
            print(f'  Spearman(UQ, col@3s):  rho={c["rho"]:.4f}  p={c["p"]:.2e}')
        if 'pearson_uq_vs_L2_3s' in corr:
            c = corr['pearson_uq_vs_L2_3s']
            if isinstance(c, dict):
                print(f'  Pearson(UQ, L2@3s):    r={c["r"]:.4f}  p={c["p"]:.2e}')
            else:
                print(f'  Pearson(UQ, L2@3s):    r={c:.4f}')
        if 'auroc_adverse' in corr:
            print(f'  AUROC(UQ → adverse):   {corr["auroc_adverse"]:.4f}')

    print('\n' + '=' * 70)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Enable FiLM in transformer if film checkpoint provided
    l2_only = os.environ.get('UQ_FILM_L2_ONLY', '0') == '1'
    if args.film_checkpoint:
        if l2_only:
            cfg.model.pts_bbox_head.transformer.use_uncertainty = False
            cfg.model.use_uncertainty_l2 = True
            print('[UQ] Eval mode: FiLM L2 only')
        else:
            cfg.model.pts_bbox_head.transformer.use_uncertainty = True

    # Override annotation file if specified
    if args.ann_file:
        cfg.data.test.ann_file = args.ann_file
        cfg.ann_file_test = args.ann_file

    set_random_seed(args.seed, deterministic=True)

    # Build dataset and dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    print(f'Dataset: {len(dataset)} samples')

    # Build model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        custom_wrap_fp16_model(model)

    # Load ORION checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    print(f'Loaded ORION checkpoint from {args.checkpoint}')

    # Reload UQ checkpoint (same logic as test.py)
    pts_cfg = cfg.model.get('pts_bbox_head', {})
    if pts_cfg.get('use_uncertainty') and pts_cfg.get('uq_checkpoint'):
        uq_ckpt_path = pts_cfg['uq_checkpoint']
        if os.path.exists(uq_ckpt_path):
            uq_ckpt = torch.load(uq_ckpt_path, map_location='cpu', weights_only=False)
            model.pts_bbox_head.uq_estimator.load_state_dict(
                uq_ckpt['model_state_dict'], strict=False)
            print(f'[UQ] Reloaded UQEstimator from {uq_ckpt_path}')

    # Load trained FiLM weights if provided
    if args.film_checkpoint and os.path.exists(args.film_checkpoint):
        film_ckpt = torch.load(args.film_checkpoint, map_location='cpu', weights_only=False)
        # FiLM L1 (QT-Former)
        transformer = model.pts_bbox_head.transformer
        if hasattr(transformer, 'film_gamma') and 'film_gamma_weight' in film_ckpt:
            transformer.film_gamma.weight.data = film_ckpt['film_gamma_weight']
            transformer.film_gamma.bias.data = film_ckpt['film_gamma_bias']
            transformer.film_beta.weight.data = film_ckpt['film_beta_weight']
            transformer.film_beta.bias.data = film_ckpt['film_beta_bias']
            print(f'[UQ] Loaded FiLM L1 weights from {args.film_checkpoint}')
        # FiLM L2 (VAE)
        if hasattr(model, 'film_gamma_l2') and 'film_gamma_l2_weight' in film_ckpt:
            model.film_gamma_l2.weight.data = film_ckpt['film_gamma_l2_weight']
            model.film_gamma_l2.bias.data = film_ckpt['film_gamma_l2_bias']
            model.film_beta_l2.weight.data = film_ckpt['film_beta_l2_weight']
            model.film_beta_l2.bias.data = film_ckpt['film_beta_l2_bias']
            print(f'[UQ] Loaded FiLM L2 weights from {args.film_checkpoint}')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # Wrap in DataParallel for single GPU
    from torch.nn import DataParallel
    model = DataParallel(model, device_ids=[0])

    # Register UQ score capture hook
    uq_capture = UQScoreCapture()
    uq_capture.register(model)

    # Run inference
    print(f'\nRunning inference on {len(dataset)} samples...')
    t0 = time.time()
    bbox_results = run_inference(model, data_loader)
    elapsed = time.time() - t0
    print(f'\nInference done in {elapsed:.1f}s ({elapsed/len(dataset):.2f}s/sample)')

    uq_capture.remove()

    # Collect per-sample metrics
    uq_scores = uq_capture.get_scores()
    print(f'Captured {len(uq_scores)} UQ scores for {len(bbox_results)} samples')

    records = collect_per_sample_metrics(bbox_results, dataset.data_infos, uq_scores)

    # Compute statistics
    stats = compute_aggregate_stats(records)
    corr = compute_uq_correlation(records)

    # Print report
    print_report(stats, corr)

    # Save results
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_data = {
        'records': records,
        'stats': stats,
        'correlation': corr,
        'config': str(args.config),
        'checkpoint': args.checkpoint,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(records),
    }
    torch.save(save_data, args.out)
    print(f'\nResults saved to {args.out}')

    # Also save a human-readable JSON summary
    json_out = args.out.replace('.pt', '_summary.json')
    json_data = {
        'stats': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                       for kk, vv in v.items()} for k, v in stats.items()},
        'correlation': {},
    }
    # Flatten correlation for JSON
    for k, v in corr.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                json_data['correlation'][f'{k}_{kk}'] = float(vv)
        else:
            json_data['correlation'][k] = float(v)

    with open(json_out, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f'Summary saved to {json_out}')


if __name__ == '__main__':
    main()

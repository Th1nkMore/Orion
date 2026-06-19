"""
Open-loop evaluation script for ORION with UQ score analysis.

Runs ORION inference on the validation set, collects per-sample planning
metrics (L2 error, collision rate) and UQ scores, then reports aggregate
statistics split by normal vs adverse weather conditions.

Aggregation modes (both written to summary JSON):
  - frame (micro): mean over all frames — used for UQ / weather stratification
  - clip  (macro): mean per route folder, then mean over clips — aligns with
    Bench2Drive / ORION paper Table 1 (Avg. L2 @ 2s on 50 val clips)

Usage:
    python scripts/eval_openloop.py \
        adzoo/orion/configs/orion_stage3_infer.py \
        ckpts/Orion.pth \
        --out results/eval_openloop.pt

    # Re-aggregate an existing result without re-running inference:
    python scripts/eval_openloop.py --aggregate-only results/eval_openloop.pt
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

from uq_estimator.openloop_aggregate import (
    build_summary_payload,
    run_aggregation as _run_aggregation_core,
)

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
    parser.add_argument('config', nargs='?', default=None, help='config file path')
    parser.add_argument('checkpoint', nargs='?', default=None, help='ORION checkpoint file')
    parser.add_argument('--out', default='results/eval_openloop.pt',
                        help='output file for per-sample results')
    parser.add_argument('--ann-file', default=None,
                        help='override annotation file (default: from config)')
    parser.add_argument('--film-checkpoint', default=None,
                        help='trained FiLM weights (enables FiLM in transformer)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='limit number of samples for quick eval')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--aggregate-only', default=None, metavar='PT_PATH',
        help='recompute frame/clip stats from an existing .pt (no inference)',
    )
    args = parser.parse_args()
    if args.aggregate_only:
        return args
    if not args.config or not args.checkpoint:
        parser.error('config and checkpoint are required unless --aggregate-only is set')
    return args


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


def _print_stats_block(title, stats, count_key='n_valid'):
    print(f'\n── {title} ──')
    for group in ['all', 'normal', 'adverse']:
        s = stats.get(group, {})
        n_count = s.get(count_key, s.get('n_total', 0))
        n_extra = s.get('n_frames', '')
        suffix = f', {n_extra} frames' if n_extra else ''
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


def print_report(stats_frame, stats_clip, corr, clip_corr=None):
    """Print formatted evaluation report."""
    print('\n' + '=' * 70)
    print('ORION Open-Loop Evaluation with UQ Analysis')
    print('=' * 70)

    _print_stats_block('FRAME-LEVEL (micro, all frames)', stats_frame, count_key='n_valid')
    _print_stats_block(
        'CLIP-LEVEL (macro, paper Avg. L2 — mean per route then over clips)',
        stats_clip,
        count_key='n_clips',
    )

    if corr:
        print(f'\n── UQ CORRELATION (frame-level) ──')
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

    if clip_corr and 'auroc_adverse' in clip_corr:
        print(f'\n── UQ CORRELATION (clip-level) ──')
        print(f'  AUROC(UQ → adverse):   {clip_corr["auroc_adverse"]:.4f}')

    print('\n' + '=' * 70)


def run_aggregation(records):
    """Compute all aggregates from per-frame records."""
    return _run_aggregation_core(records, compute_uq_correlation)


def aggregate_only_main(pt_path):
    """Recompute stats from an existing eval .pt without inference."""
    import runpy
    agg_script = os.path.join(os.path.dirname(__file__), 'aggregate_openloop_results.py')
    old_argv = sys.argv
    try:
        sys.argv = [agg_script, pt_path]
        runpy.run_path(agg_script, run_name='__main__')
    finally:
        sys.argv = old_argv


def main():
    args = parse_args()
    if args.aggregate_only:
        aggregate_only_main(args.aggregate_only)
        return

    from mmcv.utils import (
        set_random_seed, Config, load_checkpoint, ProgressBar,
    )
    from mmcv.models import build_model
    from mmcv.datasets import build_dataset, build_dataloader

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
    if args.max_samples and args.max_samples < len(dataset):
        import numpy as np
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
            from uq_estimator.density import get_uq_state_dict
            model.pts_bbox_head.uq_estimator.load_state_dict(
                get_uq_state_dict(uq_ckpt), strict=False)
            print(f'[UQ] Reloaded UQEstimator from {uq_ckpt_path}')

    uq_token_path = os.environ.get(
        'UQ_TOKEN_CHECKPOINT', cfg.model.get('uq_token_checkpoint', ''))
    if cfg.model.get('use_uq_token') and uq_token_path:
        if not os.path.exists(uq_token_path):
            raise FileNotFoundError(f'UQ token checkpoint not found: {uq_token_path}')
        from uq_estimator.training import load_uq_token_weights
        loaded = load_uq_token_weights(model, uq_token_path)
        print(f'[UQ Token] Loaded {loaded} adaptation tensors from {uq_token_path}')

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

    stats_frame, stats_clip, clip_records, corr, clip_corr = run_aggregation(records)

    print_report(stats_frame, stats_clip, corr, clip_corr)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    save_data = {
        'records': records,
        'clip_records': clip_records,
        'stats': stats_frame,
        'stats_frame': stats_frame,
        'stats_clip': stats_clip,
        'correlation': corr,
        'correlation_clip': clip_corr,
        'config': str(args.config),
        'checkpoint': args.checkpoint,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(records),
        'n_clips': len(clip_records),
    }
    torch.save(save_data, args.out)
    print(f'\nResults saved to {args.out} ({len(records)} frames, {len(clip_records)} clips)')

    scores_npz = args.out.replace('.pt', '_scores.npz')
    if scores_npz != args.out:
        from uq_estimator.roc_plot import export_openloop_scores

        export_openloop_scores(records, scores_npz, source=args.out)
        print(f'Score cache saved to {scores_npz}')

    json_out = args.out.replace('.pt', '_summary.json')
    summary = build_summary_payload(stats_frame, stats_clip, clip_records, corr, clip_corr)
    with open(json_out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary saved to {json_out}')


if __name__ == '__main__':
    main()

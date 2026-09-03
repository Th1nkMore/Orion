"""Frame- and clip-level open-loop metric aggregation (no mmcv dependency)."""
from collections import defaultdict

import numpy as np


def _valid_records(group_records):
    return [r for r in group_records if r['plan_L2_3s'] > 0 or r['fut_valid']]


def _summarize_metric_group(valid, n_total, uq_source=None):
    n_valid = len(valid)
    if not valid:
        return {'n_total': n_total, 'n_valid': 0}

    s = {'n_total': n_total, 'n_valid': n_valid}
    for t in ['1s', '2s', '3s']:
        vals = [r[f'plan_L2_{t}'] for r in valid]
        s[f'plan_L2_{t}_mean'] = float(np.mean(vals))
        s[f'plan_L2_{t}_std'] = float(np.std(vals))

    for t in ['1s', '2s', '3s']:
        col_vals = [r[f'plan_obj_col_{t}'] for r in valid]
        box_col_vals = [r[f'plan_obj_box_col_{t}'] for r in valid]
        s[f'plan_obj_col_{t}_rate'] = float(np.mean(col_vals))
        s[f'plan_obj_box_col_{t}_rate'] = float(np.mean(box_col_vals))

    if uq_source:
        uq_vals = [r['uq_score'] for r in uq_source if 'uq_score' in r]
        if uq_vals:
            s['uq_score_mean'] = float(np.mean(uq_vals))
            s['uq_score_std'] = float(np.std(uq_vals))
            s['uq_score_median'] = float(np.median(uq_vals))

    return s


def _split_weather_groups(items, adverse_key='is_adverse'):
    groups = {'all': items, 'normal': [], 'adverse': []}
    for r in items:
        if r[adverse_key]:
            groups['adverse'].append(r)
        else:
            groups['normal'].append(r)
    return groups


def compute_aggregate_stats(records):
    """Frame-level (micro) statistics grouped by normal/adverse."""
    stats = {}
    for group_name, group_records in _split_weather_groups(records).items():
        valid = _valid_records(group_records)
        stats[group_name] = _summarize_metric_group(
            valid, len(group_records), uq_source=group_records,
        )
    return stats


def build_clip_records(records):
    """Per-route (clip) means — one dict per folder in b2d_infos_val."""
    by_folder = defaultdict(list)
    for r in records:
        by_folder[r['folder']].append(r)

    clip_records = []
    for folder, frame_recs in sorted(by_folder.items()):
        valid = _valid_records(frame_recs)
        if not valid:
            continue
        ref = frame_recs[0]
        clip = {
            'folder': folder,
            'weather_id': ref['weather_id'],
            'weather_name': ref['weather_name'],
            'is_adverse': ref['is_adverse'],
            'n_frames': len(frame_recs),
            'n_valid_frames': len(valid),
        }
        for t in ['1s', '2s', '3s']:
            clip[f'plan_L2_{t}'] = float(np.mean([r[f'plan_L2_{t}'] for r in valid]))
            clip[f'plan_obj_col_{t}'] = float(np.mean([r[f'plan_obj_col_{t}'] for r in valid]))
            clip[f'plan_obj_box_col_{t}'] = float(
                np.mean([r[f'plan_obj_box_col_{t}'] for r in valid])
            )
        uq_vals = [r['uq_score'] for r in frame_recs if 'uq_score' in r]
        if uq_vals:
            clip['uq_score'] = float(np.mean(uq_vals))
        clip_records.append(clip)
    return clip_records


def compute_clip_aggregate_stats(clip_records):
    """Clip-level (macro) statistics — matches B2D paper Avg. L2 reporting."""
    stats = {}
    for group_name, group_clips in _split_weather_groups(clip_records).items():
        n_total = len(group_clips)
        if not group_clips:
            stats[group_name] = {'n_clips': 0, 'n_valid': 0}
            continue
        pseudo = []
        for c in group_clips:
            pseudo.append({
                'plan_L2_1s': c['plan_L2_1s'],
                'plan_L2_2s': c['plan_L2_2s'],
                'plan_L2_3s': c['plan_L2_3s'],
                'plan_obj_col_1s': c['plan_obj_col_1s'],
                'plan_obj_col_2s': c['plan_obj_col_2s'],
                'plan_obj_col_3s': c['plan_obj_col_3s'],
                'plan_obj_box_col_1s': c['plan_obj_box_col_1s'],
                'plan_obj_box_col_2s': c['plan_obj_box_col_2s'],
                'plan_obj_box_col_3s': c['plan_obj_box_col_3s'],
                'fut_valid': True,
                'uq_score': c.get('uq_score'),
            })
        s = _summarize_metric_group(pseudo, n_total, uq_source=pseudo)
        s['n_clips'] = n_total
        s['n_frames'] = int(sum(c['n_frames'] for c in group_clips))
        s['avg_l2_2s'] = s['plan_L2_2s_mean']
        stats[group_name] = s
    return stats


def compute_clip_uq_correlation(clip_records):
    labeled = [c for c in clip_records if 'uq_score' in c]
    if len(labeled) < 2:
        return {}
    try:
        from sklearn.metrics import roc_auc_score
        labels = np.array([1 if c['is_adverse'] else 0 for c in labeled])
        scores = np.array([c['uq_score'] for c in labeled])
        if len(np.unique(labels)) == 2:
            return {'auroc_adverse': float(roc_auc_score(labels, scores))}
    except ImportError:
        pass
    return {}


def run_aggregation(records, compute_uq_correlation_fn):
    """Compute frame + clip aggregates; UQ frame correlation injected by caller."""
    clip_records = build_clip_records(records)
    stats_frame = compute_aggregate_stats(records)
    stats_clip = compute_clip_aggregate_stats(clip_records)
    corr = compute_uq_correlation_fn(records)
    clip_corr = compute_clip_uq_correlation(clip_records)
    return stats_frame, stats_clip, clip_records, corr, clip_corr


def floatify_stats(stats):
    out = {}
    for k, v in stats.items():
        out[k] = {
            kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
            for kk, vv in v.items()
        }
    return out


def flatten_correlation(corr):
    flat = {}
    for k, v in corr.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f'{k}_{kk}'] = float(vv)
        else:
            flat[k] = float(v)
    return flat


def build_summary_payload(stats_frame, stats_clip, clip_records, corr, clip_corr):
    all_clip = stats_clip.get('all', {})
    return {
        'n_samples': stats_frame.get('all', {}).get('n_total', 0),
        'n_clips': all_clip.get('n_clips', len(clip_records)),
        'avg_l2_2s_clip_macro': all_clip.get('avg_l2_2s'),
        'stats': floatify_stats(stats_frame),
        'stats_frame': floatify_stats(stats_frame),
        'stats_clip': floatify_stats(stats_clip),
        'correlation': flatten_correlation(corr),
        'correlation_clip': flatten_correlation(clip_corr),
    }

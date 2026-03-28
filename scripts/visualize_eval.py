"""
Paper-ready visualization from open-loop evaluation results.

Generates figures for paper Section 3.1 (UQ Score Analysis):
  1. UQ score distribution: normal vs adverse histogram
  2. AUROC curve: UQ score as adverse detector
  3. UQ score vs planning L2 error scatter
  4. Per-weather-type box plot of UQ scores
  5. Planning metrics comparison bar chart (normal vs adverse)

Usage:
    python scripts/visualize_eval.py \
        --input results/eval_openloop_full.pt \
        --out-dir results/figures
"""
import argparse
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Weather metadata ────────────────────────────────────────────────────
WEATHER_NAMES = {
    0: 'ClearNoon', 1: 'ClearSunset', 2: 'CloudyNoon', 3: 'CloudySunset',
    5: 'WetNoon', 6: 'WetSunset', 7: 'MidRainyNoon', 8: 'MidRainSunset',
    9: 'WetCloudyNoon', 10: 'WetCloudySunset', 11: 'HardRainNoon',
    12: 'HardRainSunset', 13: 'SoftRainNoon', 14: 'SoftRainSunset',
    15: 'ClearNight', 18: 'CloudyNight', 19: 'WetNight',
    20: 'WetCloudyNight', 21: 'MidRainyNight', 22: 'HardRainNight',
    23: 'SoftRainNight', 25: 'FoggyNoon', 26: 'FoggySunset',
}

# Group adverse conditions by type for finer-grained analysis
WEATHER_GROUPS = {
    'Clear': [0, 1],
    'Cloudy': [2, 3],
    'Wet': [5, 6, 9, 10],
    'Rain': [7, 8, 11, 12, 13, 14],
    'Night': [15, 18],
    'Night+Wet/Rain': [19, 20, 21, 22, 23],
    'Fog': [25, 26],
}


def parse_args():
    p = argparse.ArgumentParser(description='Visualize eval results')
    p.add_argument('--input', required=True, help='eval_openloop .pt result file')
    p.add_argument('--input-film', default=None,
                   help='optional second result file (FiLM L1) for comparison')
    p.add_argument('--out-dir', default='results/figures')
    p.add_argument('--format', default='pdf', choices=['pdf', 'png', 'svg'])
    return p.parse_args()


def load_records(path):
    data = torch.load(path, weights_only=False)
    return data['records']


def setup_matplotlib():
    """Configure matplotlib for paper-quality figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })
    return plt


# ── Figure 1: UQ Score Distribution ────────────────────────────────────
def plot_score_distribution(records, plt, out_path):
    """Normal vs adverse UQ score histograms with KDE overlay."""
    normal = [r['uq_score'] for r in records if not r['is_adverse'] and 'uq_score' in r]
    adverse = [r['uq_score'] for r in records if r['is_adverse'] and 'uq_score' in r]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    bins = np.linspace(0, 1, 50)
    ax.hist(normal, bins=bins, alpha=0.6, color='#2196F3', label=f'Normal (n={len(normal)})',
            density=True, edgecolor='white', linewidth=0.5)
    ax.hist(adverse, bins=bins, alpha=0.6, color='#F44336', label=f'Adverse (n={len(adverse)})',
            density=True, edgecolor='white', linewidth=0.5)

    # Add means as vertical lines
    if normal:
        ax.axvline(np.mean(normal), color='#1565C0', linestyle='--', linewidth=1.5,
                    label=f'Normal mean={np.mean(normal):.3f}')
    if adverse:
        ax.axvline(np.mean(adverse), color='#C62828', linestyle='--', linewidth=1.5,
                    label=f'Adverse mean={np.mean(adverse):.3f}')

    ax.set_xlabel('UQ Score')
    ax.set_ylabel('Density')
    ax.set_title('Uncertainty Score Distribution: Normal vs Adverse')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1)

    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved: {out_path}')

    # Print separation stats
    if normal and adverse:
        sep = np.mean(adverse) - np.mean(normal)
        print(f'  Separation: adverse_mean - normal_mean = {sep:.4f}')


# ── Figure 2: AUROC Curve ──────────────────────────────────────────────
def plot_auroc(records, plt, out_path):
    """ROC curve for UQ score as adverse weather detector."""
    scores = [r['uq_score'] for r in records if 'uq_score' in r]
    labels = [1 if r['is_adverse'] else 0 for r in records if 'uq_score' in r]

    if len(set(labels)) < 2:
        print('  Skipping AUROC: only one class present')
        return

    try:
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        print('  Skipping AUROC: sklearn not available')
        return

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(fpr, tpr, color='#E65100', linewidth=2,
            label=f'UQ Score (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC: UQ Score as Adverse Detector')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved: {out_path} (AUROC={roc_auc:.4f})')


# ── Figure 3: UQ vs Planning Error Scatter ─────────────────────────────
def plot_uq_vs_l2(records, plt, out_path):
    """Scatter plot of UQ score vs trajectory L2 error at 3s."""
    valid = [r for r in records if (r['plan_L2_3s'] > 0 or r['fut_valid']) and 'uq_score' in r]
    if not valid:
        print('  Skipping UQ vs L2: no valid records')
        return

    uq = np.array([r['uq_score'] for r in valid])
    l2 = np.array([r['plan_L2_3s'] for r in valid])
    is_adv = np.array([r['is_adverse'] for r in valid])

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Plot normal and adverse with different colors
    ax.scatter(uq[~is_adv], l2[~is_adv], alpha=0.15, s=8, c='#2196F3',
               label='Normal', rasterized=True)
    ax.scatter(uq[is_adv], l2[is_adv], alpha=0.15, s=8, c='#F44336',
               label='Adverse', rasterized=True)

    # Add trend line
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(uq, l2)
        z = np.polyfit(uq, l2, 1)
        trend_x = np.linspace(uq.min(), uq.max(), 100)
        ax.plot(trend_x, np.polyval(z, trend_x), 'k-', linewidth=1.5, alpha=0.7,
                label=f'Trend (Spearman ρ={rho:.3f})')
    except ImportError:
        pass

    ax.set_xlabel('UQ Score')
    ax.set_ylabel('Planning L2 Error @3s (m)')
    ax.set_title('Uncertainty Score vs Planning Error')
    ax.legend()

    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ── Figure 4: Per-Weather Box Plot ─────────────────────────────────────
def plot_weather_boxplot(records, plt, out_path):
    """Box plot of UQ scores grouped by weather type."""
    # Group by weather group
    group_scores = {}
    for group_name, weather_ids in WEATHER_GROUPS.items():
        scores = [r['uq_score'] for r in records
                  if r['weather_id'] in weather_ids and 'uq_score' in r]
        if scores:
            group_scores[group_name] = scores

    if not group_scores:
        print('  Skipping weather boxplot: no data')
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

    labels = list(group_scores.keys())
    data = [group_scores[l] for l in labels]
    n_samples = [len(d) for d in data]
    x_labels = [f'{l}\n(n={n})' for l, n in zip(labels, n_samples)]

    # Color by category
    colors = []
    for l in labels:
        if l in ('Clear', 'Cloudy'):
            colors.append('#2196F3')  # blue for normal
        else:
            colors.append('#F44336')  # red for adverse

    bp = ax.boxplot(data, labels=x_labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('UQ Score')
    ax.set_title('Uncertainty Score by Weather Condition')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ── Figure 5: Planning Metrics Bar Chart ───────────────────────────────
def plot_planning_bars(records, plt, out_path):
    """Bar chart comparing planning metrics: normal vs adverse."""
    groups = {'Normal': [], 'Adverse': []}
    for r in records:
        key = 'Adverse' if r['is_adverse'] else 'Normal'
        if r['plan_L2_3s'] > 0 or r['fut_valid']:
            groups[key].append(r)

    if not groups['Normal'] or not groups['Adverse']:
        print('  Skipping planning bars: missing group data')
        return

    metrics = {
        'L2@1s': 'plan_L2_1s',
        'L2@2s': 'plan_L2_2s',
        'L2@3s': 'plan_L2_3s',
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: L2 errors
    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.35
    normal_vals = [np.mean([r[k] for r in groups['Normal']]) for k in metrics.values()]
    adverse_vals = [np.mean([r[k] for r in groups['Adverse']]) for k in metrics.values()]
    normal_stds = [np.std([r[k] for r in groups['Normal']]) for k in metrics.values()]
    adverse_stds = [np.std([r[k] for r in groups['Adverse']]) for k in metrics.values()]

    ax.bar(x - width/2, normal_vals, width, yerr=normal_stds, capsize=3,
           label=f'Normal (n={len(groups["Normal"])})', color='#2196F3', alpha=0.7)
    ax.bar(x + width/2, adverse_vals, width, yerr=adverse_stds, capsize=3,
           label=f'Adverse (n={len(groups["Adverse"])})', color='#F44336', alpha=0.7)
    ax.set_ylabel('L2 Error (m)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.set_title('Planning L2 Error')
    ax.legend()

    # Right: Collision rates
    ax = axes[1]
    col_metrics = {
        'Col@1s': 'plan_obj_col_1s',
        'Col@2s': 'plan_obj_col_2s',
        'Col@3s': 'plan_obj_col_3s',
    }
    x = np.arange(len(col_metrics))
    normal_cols = [np.mean([r[k] for r in groups['Normal']]) for k in col_metrics.values()]
    adverse_cols = [np.mean([r[k] for r in groups['Adverse']]) for k in col_metrics.values()]

    ax.bar(x - width/2, normal_cols, width, capsize=3,
           label='Normal', color='#2196F3', alpha=0.7)
    ax.bar(x + width/2, adverse_cols, width, capsize=3,
           label='Adverse', color='#F44336', alpha=0.7)
    ax.set_ylabel('Collision Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(col_metrics.keys())
    ax.set_title('Object Collision Rate')
    ax.legend()

    fig.suptitle('Planning Metrics: Normal vs Adverse', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ── Figure 6: Comparison (baseline vs FiLM L1) ────────────────────────
def plot_comparison(records_base, records_film, plt, out_path):
    """Side-by-side comparison of baseline vs FiLM L1 results."""
    def get_group_means(records):
        groups = {'Normal': [], 'Adverse': []}
        for r in records:
            key = 'Adverse' if r['is_adverse'] else 'Normal'
            if r['plan_L2_3s'] > 0 or r['fut_valid']:
                groups[key].append(r)
        means = {}
        for g, recs in groups.items():
            if recs:
                means[g] = {
                    'L2@3s': np.mean([r['plan_L2_3s'] for r in recs]),
                    'Col@3s': np.mean([r['plan_obj_col_3s'] for r in recs]),
                    'UQ': np.mean([r['uq_score'] for r in recs if 'uq_score' in r]),
                }
        return means

    base_means = get_group_means(records_base)
    film_means = get_group_means(records_film)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for idx, scene in enumerate(['Normal', 'Adverse']):
        ax = axes[idx]
        if scene not in base_means or scene not in film_means:
            ax.text(0.5, 0.5, f'No {scene} data', ha='center', va='center')
            continue

        metrics = ['L2@3s', 'Col@3s']
        base_vals = [base_means[scene][m] for m in metrics]
        film_vals = [film_means[scene][m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width/2, base_vals, width, label='Baseline', color='#9E9E9E', alpha=0.8)
        ax.bar(x + width/2, film_vals, width, label='FiLM L1', color='#4CAF50', alpha=0.8)

        # Show improvement percentage
        for i, (b, f) in enumerate(zip(base_vals, film_vals)):
            if b > 0:
                pct = (f - b) / b * 100
                color = '#4CAF50' if pct < 0 else '#F44336'
                ax.annotate(f'{pct:+.1f}%', xy=(i + width/2, f),
                            ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_title(f'{scene} Scenes')
        ax.legend()

    fig.suptitle('Baseline vs FiLM L1: Planning Metrics', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ── Text Summary ───────────────────────────────────────────────────────
def write_summary(records, out_path):
    """Write a text summary of key statistics."""
    normal = [r for r in records if not r['is_adverse']]
    adverse = [r for r in records if r['is_adverse']]

    with open(out_path, 'w') as f:
        f.write('UQ-ORION Open-Loop Evaluation Summary\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'Total samples: {len(records)}\n')
        f.write(f'Normal: {len(normal)}, Adverse: {len(adverse)}\n\n')

        for name, group in [('Normal', normal), ('Adverse', adverse), ('All', records)]:
            valid = [r for r in group if r['plan_L2_3s'] > 0 or r['fut_valid']]
            uq = [r['uq_score'] for r in group if 'uq_score' in r]
            f.write(f'--- {name} ({len(group)} total, {len(valid)} with metrics) ---\n')
            if uq:
                f.write(f'  UQ score: mean={np.mean(uq):.4f}, std={np.std(uq):.4f}, '
                        f'median={np.median(uq):.4f}\n')
            if valid:
                for t in ['1s', '2s', '3s']:
                    l2 = [r[f'plan_L2_{t}'] for r in valid]
                    col = [r[f'plan_obj_col_{t}'] for r in valid]
                    f.write(f'  L2@{t}: {np.mean(l2):.4f} +/- {np.std(l2):.4f}\n')
                    f.write(f'  Col@{t}: {np.mean(col):.4f}\n')
            f.write('\n')

        # Correlation
        valid_all = [r for r in records if (r['plan_L2_3s'] > 0 or r['fut_valid']) and 'uq_score' in r]
        if len(valid_all) > 10:
            uq = np.array([r['uq_score'] for r in valid_all])
            l2 = np.array([r['plan_L2_3s'] for r in valid_all])
            try:
                from scipy.stats import spearmanr, pearsonr
                sp, sp_p = spearmanr(uq, l2)
                pe, pe_p = pearsonr(uq, l2)
                f.write(f'Spearman(UQ, L2@3s): rho={sp:.4f}, p={sp_p:.2e}\n')
                f.write(f'Pearson(UQ, L2@3s):  r={pe:.4f}, p={pe_p:.2e}\n')
            except ImportError:
                pass

            try:
                from sklearn.metrics import roc_auc_score
                labels = np.array([1 if r['is_adverse'] else 0 for r in records if 'uq_score' in r])
                scores = np.array([r['uq_score'] for r in records if 'uq_score' in r])
                if len(np.unique(labels)) == 2:
                    f.write(f'AUROC(UQ -> adverse): {roc_auc_score(labels, scores):.4f}\n')
            except ImportError:
                pass

    print(f'  Saved: {out_path}')


# ── Main ───────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    plt = setup_matplotlib()
    os.makedirs(args.out_dir, exist_ok=True)
    ext = args.format

    print(f'Loading results from {args.input}...')
    records = load_records(args.input)
    print(f'  {len(records)} samples loaded')

    n_uq = sum(1 for r in records if 'uq_score' in r)
    print(f'  {n_uq} with UQ scores')

    print('\nGenerating figures:')
    plot_score_distribution(records, plt, os.path.join(args.out_dir, f'fig1_score_dist.{ext}'))
    plot_auroc(records, plt, os.path.join(args.out_dir, f'fig2_auroc.{ext}'))
    plot_uq_vs_l2(records, plt, os.path.join(args.out_dir, f'fig3_uq_vs_l2.{ext}'))
    plot_weather_boxplot(records, plt, os.path.join(args.out_dir, f'fig4_weather_boxplot.{ext}'))
    plot_planning_bars(records, plt, os.path.join(args.out_dir, f'fig5_planning_bars.{ext}'))

    # Comparison figure if second result provided
    if args.input_film:
        print(f'\nLoading FiLM results from {args.input_film}...')
        records_film = load_records(args.input_film)
        plot_comparison(records, records_film, plt,
                        os.path.join(args.out_dir, f'fig6_comparison.{ext}'))

    # Text summary
    write_summary(records, os.path.join(args.out_dir, 'summary.txt'))
    print('\nDone.')


if __name__ == '__main__':
    main()

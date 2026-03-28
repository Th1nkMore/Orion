"""
Attention Map Visualization: QT-Former cross-attention before/after FiLM.

Captures cross-attention weights from PETRTransformerDecoderLayer and visualizes
how FiLM modulation changes the attention distribution. Under high uncertainty,
attention should concentrate more on nearby/safe regions.

Two modes:
  1. Single-run: visualize attention maps for a set of samples
  2. Comparison: run same samples with FiLM ON vs OFF, show side-by-side

Usage:
    # Single run (from eval results with attention captured)
    python scripts/visualize_attention.py \
        --input results/eval_openloop_full.pt \
        --output-dir results/figures/attention

    # Comparison: baseline vs FiLM (needs two eval runs with --capture-attention)
    python scripts/visualize_attention.py \
        --input results/eval_openloop_full.pt \
        --input-film results/eval_openloop_film_l1.pt \
        --output-dir results/figures/attention

    # Standalone capture mode (hook-based, runs inference on selected samples)
    python scripts/visualize_attention.py \
        --capture \
        --config adzoo/orion/configs/orion_stage3_infer.py \
        --checkpoint ckpts/Orion.pth \
        --ann-file data/infos/b2d_infos_val.pkl \
        --num-samples 20 \
        --output-dir results/figures/attention
"""
import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Attention Hook ──────────────────────────────────────────────────────

class AttentionCapture:
    """Registers forward hooks on QT-Former cross-attention layers to capture
    attention weights during inference.

    Hook target: PETRTransformerDecoderLayer.transformer_layers[2]
    which is a MultiHeadAttentionwDropout wrapping nn.MultiheadAttention.
    The inner nn.MultiheadAttention returns (output, attn_weights).
    """

    def __init__(self, model):
        self.attn_weights = {}  # {layer_idx: [B, num_query, num_key]}
        self._hooks = []
        self._register(model)

    def _register(self, model):
        """Find and hook cross-attention modules in QT-Former decoder."""
        transformer = model.pts_bbox_head.transformer
        decoder = transformer.query_decoder

        for layer_idx, layer in enumerate(decoder._layers):
            # Index 2 = cross-attention (MultiHeadAttentionwDropout)
            cross_attn = layer.transformer_layers[2]

            # Hook the inner nn.MultiheadAttention which returns attn_weights
            inner_attn = cross_attn.attn
            hook = inner_attn.register_forward_hook(
                self._make_hook(layer_idx))
            self._hooks.append(hook)

        print(f'[Attention] Registered hooks on {len(self._hooks)} decoder layers')

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # nn.MultiheadAttention returns (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_w = output[1]  # [B, num_query, num_key] or None
                if attn_w is not None:
                    self.attn_weights[layer_idx] = attn_w.detach().cpu()
        return hook_fn

    def get_and_clear(self):
        """Return captured attention and reset."""
        result = dict(self.attn_weights)
        self.attn_weights.clear()
        return result

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── Visualization Functions ─────────────────────────────────────────────

def setup_style():
    """Paper-quality matplotlib style."""
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


def plot_attention_heatmap(attn_weights, layer_idx, title, ax):
    """Plot single attention heatmap.

    Args:
        attn_weights: [num_query, num_key] attention matrix
        layer_idx: decoder layer index
        title: plot title
        ax: matplotlib axis
    """
    im = ax.imshow(attn_weights, aspect='auto', cmap='viridis',
                   interpolation='nearest')
    ax.set_xlabel('Key (memory tokens)')
    ax.set_ylabel('Query (detection queries)')
    ax.set_title(title, fontsize=10)
    return im


def plot_attention_entropy(attn_weights_dict, ax, label='', color='steelblue'):
    """Plot per-query attention entropy across layers.

    Higher entropy = more dispersed attention.
    Lower entropy = more focused attention (expected under high UQ).

    Args:
        attn_weights_dict: {layer_idx: [num_query, num_key]}
    """
    layers = sorted(attn_weights_dict.keys())
    entropies = []
    for l in layers:
        w = attn_weights_dict[l]  # [num_query, num_key]
        # Clamp to avoid log(0)
        w = np.clip(w, 1e-10, 1.0)
        ent = -np.sum(w * np.log(w), axis=-1)  # [num_query]
        entropies.append(ent.mean())

    ax.plot(layers, entropies, 'o-', color=color, label=label, linewidth=1.5,
            markersize=4)
    ax.set_xlabel('Decoder Layer')
    ax.set_ylabel('Mean Attention Entropy')
    ax.set_title('Cross-Attention Entropy per Layer')


def plot_attention_spatial(attn_weights, num_views=6, patches_per_view=1600,
                           hw=(40, 40), ax=None, title=''):
    """Visualize spatial attention pattern for one view.

    Reshapes key dimension back to spatial grid and plots mean attention
    across queries.

    Args:
        attn_weights: [num_query, num_key] where num_key = num_views * patches_per_view
        num_views: number of camera views
        patches_per_view: patches per view (40*40=1600)
        hw: (H, W) spatial dimensions of patches
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # Mean attention across queries: [num_key]
    mean_attn = attn_weights.mean(axis=0)

    # Split by views and take first view (front camera, index 0)
    total_patches = num_views * patches_per_view
    if mean_attn.shape[0] >= total_patches:
        # Key includes temporal memory, take only current frame
        mean_attn = mean_attn[:total_patches]

    if mean_attn.shape[0] == total_patches:
        # Reshape to per-view spatial maps
        view_maps = mean_attn.reshape(num_views, hw[0], hw[1])
    elif mean_attn.shape[0] == patches_per_view:
        view_maps = mean_attn.reshape(1, hw[0], hw[1])
    else:
        # Fallback: just show as 1D
        ax.bar(range(len(mean_attn)), mean_attn, width=1.0, color='steelblue')
        ax.set_title(title)
        return

    # Show front camera (view 0)
    im = ax.imshow(view_maps[0], cmap='hot', interpolation='bilinear')
    ax.set_title(title if title else 'Front Camera Attention')
    ax.set_xlabel('Patch x')
    ax.set_ylabel('Patch y')
    plt.colorbar(im, ax=ax, shrink=0.8)


def visualize_single_sample(attn_dict, sample_info, output_dir, prefix=''):
    """Generate all attention visualizations for a single sample.

    Args:
        attn_dict: {layer_idx: [num_query, num_key]} attention weights
        sample_info: dict with 'uq_score', 'weather_id', 'scene_token' etc.
        output_dir: directory to save figures
        prefix: filename prefix
    """
    setup_style()
    os.makedirs(output_dir, exist_ok=True)

    uq_score = sample_info.get('uq_score', 0.0)
    weather_id = sample_info.get('weather_id', -1)

    layers = sorted(attn_dict.keys())
    n_layers = len(layers)

    # ── Figure 1: Heatmaps per layer ──
    fig, axes = plt.subplots(1, min(n_layers, 6), figsize=(3 * min(n_layers, 6), 3))
    if n_layers == 1:
        axes = [axes]
    for i, l in enumerate(layers[:6]):
        w = attn_dict[l].numpy() if isinstance(attn_dict[l], torch.Tensor) else attn_dict[l]
        if w.ndim == 3:
            w = w[0]  # take first batch
        plot_attention_heatmap(w, l, f'Layer {l}', axes[i])
    fig.suptitle(f'Cross-Attention | UQ={uq_score:.3f} | Weather={weather_id}',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{prefix}heatmaps.png'))
    plt.close(fig)

    # ── Figure 2: Spatial attention (last layer, front camera) ──
    last_layer = layers[-1]
    w = attn_dict[last_layer].numpy() if isinstance(attn_dict[last_layer], torch.Tensor) else attn_dict[last_layer]
    if w.ndim == 3:
        w = w[0]
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    plot_attention_spatial(w, ax=ax,
                          title=f'Spatial Attention (Last Layer) | UQ={uq_score:.3f}')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{prefix}spatial.png'))
    plt.close(fig)

    # ── Figure 3: Entropy per layer ──
    attn_np = {}
    for l in layers:
        w = attn_dict[l].numpy() if isinstance(attn_dict[l], torch.Tensor) else attn_dict[l]
        if w.ndim == 3:
            w = w[0]
        attn_np[l] = w

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    plot_attention_entropy(attn_np, ax, label='attention')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{prefix}entropy.png'))
    plt.close(fig)


def visualize_comparison(attn_baseline, attn_film, sample_info, output_dir, prefix=''):
    """Side-by-side comparison of attention with and without FiLM.

    Args:
        attn_baseline: {layer_idx: [num_query, num_key]} without FiLM
        attn_film: {layer_idx: [num_query, num_key]} with FiLM
        sample_info: dict with metadata
        output_dir: save directory
    """
    setup_style()
    os.makedirs(output_dir, exist_ok=True)

    uq_score = sample_info.get('uq_score', 0.0)
    weather_id = sample_info.get('weather_id', -1)

    layers = sorted(set(attn_baseline.keys()) & set(attn_film.keys()))
    last_layer = layers[-1]

    # ── Figure 1: Spatial attention comparison (front camera) ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    def to_np(d, l):
        w = d[l].numpy() if isinstance(d[l], torch.Tensor) else d[l]
        return w[0] if w.ndim == 3 else w

    plot_attention_spatial(to_np(attn_baseline, last_layer), ax=axes[0],
                          title='Baseline (no FiLM)')
    plot_attention_spatial(to_np(attn_film, last_layer), ax=axes[1],
                          title='With FiLM')
    fig.suptitle(f'Spatial Attention Comparison | UQ={uq_score:.3f} | Weather={weather_id}',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{prefix}spatial_comparison.png'))
    plt.close(fig)

    # ── Figure 2: Entropy comparison across layers ──
    base_np = {l: to_np(attn_baseline, l) for l in layers}
    film_np = {l: to_np(attn_film, l) for l in layers}

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    plot_attention_entropy(base_np, ax, label='Baseline', color='steelblue')
    plot_attention_entropy(film_np, ax, label='With FiLM', color='orangered')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{prefix}entropy_comparison.png'))
    plt.close(fig)

    # ── Figure 3: Attention difference heatmap (last layer) ──
    w_base = to_np(attn_baseline, last_layer)
    w_film = to_np(attn_film, last_layer)
    diff = w_film - w_base

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    plot_attention_heatmap(w_base, last_layer, 'Baseline', axes[0])
    plot_attention_heatmap(w_film, last_layer, 'With FiLM', axes[1])
    im = axes[2].imshow(diff, aspect='auto', cmap='RdBu_r', interpolation='nearest',
                        vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[2].set_title('Difference (FiLM - Baseline)')
    axes[2].set_xlabel('Key')
    axes[2].set_ylabel('Query')
    plt.colorbar(im, ax=axes[2], shrink=0.8)
    fig.suptitle(f'Attention Maps (Layer {last_layer}) | UQ={uq_score:.3f}', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{prefix}diff_heatmap.png'))
    plt.close(fig)


def visualize_aggregate_entropy(all_entropies, output_dir):
    """Aggregate entropy analysis across multiple samples.

    Shows how FiLM changes entropy distribution for normal vs adverse weather.

    Args:
        all_entropies: list of dicts with 'uq_score', 'weather', 'entropy_baseline',
                       'entropy_film' (optional)
    """
    setup_style()
    os.makedirs(output_dir, exist_ok=True)

    scores = np.array([e['uq_score'] for e in all_entropies])
    entropies = np.array([e['entropy'] for e in all_entropies])
    is_adverse = np.array([e.get('is_adverse', False) for e in all_entropies])

    # ── Figure: UQ score vs attention entropy ──
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    normal_mask = ~is_adverse
    ax.scatter(scores[normal_mask], entropies[normal_mask], alpha=0.4, s=10,
               c='steelblue', label='Normal')
    ax.scatter(scores[is_adverse], entropies[is_adverse], alpha=0.4, s=10,
               c='orangered', label='Adverse')
    ax.set_xlabel('UQ Score')
    ax.set_ylabel('Mean Cross-Attention Entropy')
    ax.set_title('Uncertainty Score vs Attention Entropy')
    ax.legend()

    # Add correlation
    from scipy import stats
    rho, p = stats.spearmanr(scores, entropies)
    ax.text(0.02, 0.98, f'Spearman ρ={rho:.3f} (p={p:.1e})',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'uq_vs_entropy.png'))
    plt.close(fig)


# ── Standalone Capture Mode ─────────────────────────────────────────────

def capture_attention_standalone(args):
    """Run inference on selected samples and capture attention maps.

    This is for when you want to visualize attention without a full eval run.
    """
    from mmcv.utils import Config, ProgressBar
    from mmcv.models import build_model
    from mmcv.datasets import build_dataset, build_dataloader
    from mmcv.runner import load_checkpoint

    # Build model
    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)

    # Reload UQ weights
    uq_ckpt_path = cfg.model.get('pts_bbox_head', {}).get('uq_checkpoint', 'checkpoints/uq/best.pt')
    if os.path.exists(uq_ckpt_path):
        uq_ckpt = torch.load(uq_ckpt_path, map_location='cpu', weights_only=False)
        model.pts_bbox_head.uq_estimator.load_state_dict(
            uq_ckpt['model_state_dict'], strict=False)
        print(f'[UQ] Loaded UQEstimator from {uq_ckpt_path}')

    # Load FiLM weights if provided
    if args.film_checkpoint and os.path.exists(args.film_checkpoint):
        film_ckpt = torch.load(args.film_checkpoint, map_location='cpu', weights_only=False)
        transformer = model.pts_bbox_head.transformer
        if hasattr(transformer, 'film_gamma') and 'film_gamma_weight' in film_ckpt:
            transformer.film_gamma.weight.data = film_ckpt['film_gamma_weight']
            transformer.film_gamma.bias.data = film_ckpt['film_gamma_bias']
            transformer.film_beta.weight.data = film_ckpt['film_beta_weight']
            transformer.film_beta.bias.data = film_ckpt['film_beta_bias']
            # Enable FiLM
            transformer.use_uncertainty = True
            print(f'[UQ] Loaded FiLM L1 weights, enabled FiLM')

    model.cuda().eval()

    # Register attention hooks
    attn_capture = AttentionCapture(model)

    # Also capture UQ scores
    from scripts.eval_openloop import UQScoreCapture
    uq_capture = UQScoreCapture(model)

    # Build dataset
    if args.ann_file:
        cfg.data.test.ann_file = args.ann_file
    dataset = build_dataset(cfg.data.test)
    dataloader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=1, shuffle=False, dist=False)

    # Weather classification
    NORMAL_WEATHER = {0, 1, 2, 3}

    # Run inference on selected samples
    num_samples = min(args.num_samples, len(dataset))
    results = []
    prog = ProgressBar(num_samples)

    for i, data in enumerate(dataloader):
        if i >= num_samples:
            break

        with torch.no_grad():
            model.simple_test(data['img_metas'][0], **{
                k: v for k, v in data.items() if k != 'img_metas'
            })

        # Capture results
        attn = attn_capture.get_and_clear()
        uq_score = uq_capture.get_and_clear()

        weather_id = data['img_metas'][0][0].get('weather', -1)
        sample_token = data['img_metas'][0][0].get('scene_token', f'sample_{i}')

        sample_info = {
            'uq_score': uq_score if uq_score is not None else 0.0,
            'weather_id': weather_id,
            'is_adverse': weather_id not in NORMAL_WEATHER,
            'sample_token': sample_token,
            'index': i,
        }

        results.append({
            'attn': attn,
            'info': sample_info,
        })

        prog.update()

    attn_capture.remove()

    # Save raw data
    torch.save(results, os.path.join(args.output_dir, 'attention_data.pt'))
    print(f'\nSaved attention data for {len(results)} samples')

    # Generate visualizations
    print('Generating visualizations...')
    for r in results:
        idx = r['info']['index']
        uq = r['info']['uq_score']
        prefix = f'sample_{idx:04d}_uq{uq:.3f}_'
        visualize_single_sample(r['attn'], r['info'], args.output_dir, prefix)

    # Aggregate entropy analysis
    all_entropies = []
    for r in results:
        layers = sorted(r['attn'].keys())
        if not layers:
            continue
        # Mean entropy across all layers
        ent_list = []
        for l in layers:
            w = r['attn'][l].numpy() if isinstance(r['attn'][l], torch.Tensor) else r['attn'][l]
            if w.ndim == 3:
                w = w[0]
            w = np.clip(w, 1e-10, 1.0)
            ent = -np.sum(w * np.log(w), axis=-1).mean()
            ent_list.append(ent)
        all_entropies.append({
            'uq_score': r['info']['uq_score'],
            'entropy': np.mean(ent_list),
            'is_adverse': r['info']['is_adverse'],
        })

    if all_entropies:
        visualize_aggregate_entropy(all_entropies, args.output_dir)

    print(f'Done! Figures saved to {args.output_dir}')


# ── From Pre-computed Eval Results ──────────────────────────────────────

def visualize_from_eval(args):
    """Generate attention visualizations from pre-captured eval data."""
    data = torch.load(args.input, map_location='cpu', weights_only=False)

    if isinstance(data, list) and len(data) > 0 and 'attn' in data[0]:
        # Attention data from capture mode
        print(f'Loaded {len(data)} samples with attention data')
        for r in data:
            idx = r['info']['index']
            uq = r['info']['uq_score']
            prefix = f'sample_{idx:04d}_uq{uq:.3f}_'
            visualize_single_sample(r['attn'], r['info'], args.output_dir, prefix)

        # Aggregate
        all_entropies = []
        for r in data:
            layers = sorted(r['attn'].keys())
            if not layers:
                continue
            ent_list = []
            for l in layers:
                w = r['attn'][l].numpy() if isinstance(r['attn'][l], torch.Tensor) else r['attn'][l]
                if w.ndim == 3:
                    w = w[0]
                w = np.clip(w, 1e-10, 1.0)
                ent = -np.sum(w * np.log(w), axis=-1).mean()
                ent_list.append(ent)
            all_entropies.append({
                'uq_score': r['info']['uq_score'],
                'entropy': np.mean(ent_list),
                'is_adverse': r['info']['is_adverse'],
            })
        if all_entropies:
            visualize_aggregate_entropy(all_entropies, args.output_dir)
    else:
        print('Input file does not contain attention data.')
        print('Use --capture mode to generate attention data first.')
        return

    print(f'Done! Figures saved to {args.output_dir}')


# ── Main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='QT-Former Attention Map Visualization')

    # Mode selection
    p.add_argument('--capture', action='store_true',
                   help='Standalone capture mode: run inference and capture attention')
    p.add_argument('--input', default=None,
                   help='Pre-computed attention data file (.pt)')
    p.add_argument('--input-film', default=None,
                   help='Attention data with FiLM for comparison')

    # Capture mode args
    p.add_argument('--config', default='adzoo/orion/configs/orion_stage3_infer.py')
    p.add_argument('--checkpoint', default='ckpts/Orion.pth')
    p.add_argument('--film-checkpoint', default=None,
                   help='FiLM checkpoint for capture mode')
    p.add_argument('--ann-file', default=None)
    p.add_argument('--num-samples', type=int, default=20,
                   help='Number of samples to capture (capture mode)')

    # Output
    p.add_argument('--output-dir', default='results/figures/attention')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.capture:
        capture_attention_standalone(args)
    elif args.input:
        visualize_from_eval(args)
    else:
        print('Specify --capture for live capture or --input for pre-computed data.')
        print('See --help for usage.')

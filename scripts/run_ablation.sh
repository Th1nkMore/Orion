#!/bin/bash
# =============================================================================
# UQ-ORION Stage 4: Ablation Study (A/B/C/D)
#
# Four groups:
#   A = Baseline ORION (no UQ)
#   B = ORION + UQ + FiLM L1 only
#   C = ORION + UQ + FiLM L2 only
#   D = ORION + UQ + FiLM L1 + L2 (full model)
#
# Prerequisites:
#   - Stage 2a eval complete (results/eval_openloop_full.pt = Group A)
#   - FiLM L1 trained (checkpoints/film/best_l1.pt)
#   - FiLM L2 trained (checkpoints/film/best_l2.pt)
#   - FiLM L1+L2 trained (checkpoints/film/best_l1l2.pt)
#
# Usage:
#   bash scripts/run_ablation.sh [--train] [--eval] [--viz] [--all]
#
#   --train  Run FiLM training for all groups (B, C, D)
#   --eval   Run open-loop eval for all groups
#   --viz    Generate comparison visualizations
#   --all    Run everything in sequence
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Config ──────────────────────────────────────────────────────────────
CONFIG="adzoo/orion/configs/orion_stage3_infer.py"
CHECKPOINT="ckpts/Orion.pth"
ANN_FILE="data/infos/b2d_infos_val.pkl"
RESULTS_DIR="results/ablation"

# FiLM checkpoints (output of training)
FILM_L1_CKPT="checkpoints/film/best_l1.pt"
FILM_L2_CKPT="checkpoints/film/best_l2.pt"
FILM_L1L2_CKPT="checkpoints/film/best_l1l2.pt"

# Training hyperparams
EPOCHS=3
LR="1e-3"
GRAD_ACCUM=4

DO_TRAIN=false
DO_EVAL=false
DO_VIZ=false

for arg in "$@"; do
    case $arg in
        --train) DO_TRAIN=true ;;
        --eval)  DO_EVAL=true ;;
        --viz)   DO_VIZ=true ;;
        --all)   DO_TRAIN=true; DO_EVAL=true; DO_VIZ=true ;;
        *)       echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

if ! $DO_TRAIN && ! $DO_EVAL && ! $DO_VIZ; then
    echo "Usage: bash scripts/run_ablation.sh [--train] [--eval] [--viz] [--all]"
    exit 0
fi

mkdir -p "$RESULTS_DIR" checkpoints/film

# ── Helper ──────────────────────────────────────────────────────────────
log() { echo ""; echo "========== $1 =========="; echo ""; }
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# =============================================================================
# TRAINING
# =============================================================================
if $DO_TRAIN; then

    # ── Group B: FiLM L1 only ──
    log "Training Group B: FiLM L1 only ($(timestamp))"
    # Config: use_uncertainty=True (head), use_uncertainty=True (transformer), use_uncertainty_l2=False
    # train_film.py trains film_gamma/film_beta in transformer
    python scripts/train_film.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --ann-file "$ANN_FILE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --grad-accum "$GRAD_ACCUM" \
        --out "$FILM_L1_CKPT" \
        2>&1 | tee "$RESULTS_DIR/train_l1.log"
    echo "Group B training done: $FILM_L1_CKPT"

    # ── Group C: FiLM L2 only ──
    # Need a modified config that enables L2 but disables L1 in transformer
    log "Training Group C: FiLM L2 only ($(timestamp))"
    # Use env var to signal L2-only mode
    USE_FILM_L2_ONLY=1 python scripts/train_film.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --ann-file "$ANN_FILE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --grad-accum "$GRAD_ACCUM" \
        --out "$FILM_L2_CKPT" \
        2>&1 | tee "$RESULTS_DIR/train_l2.log"
    echo "Group C training done: $FILM_L2_CKPT"

    # ── Group D: FiLM L1 + L2 ──
    log "Training Group D: FiLM L1 + L2 ($(timestamp))"
    USE_FILM_L1L2=1 python scripts/train_film.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --ann-file "$ANN_FILE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --grad-accum "$GRAD_ACCUM" \
        --out "$FILM_L1L2_CKPT" \
        2>&1 | tee "$RESULTS_DIR/train_l1l2.log"
    echo "Group D training done: $FILM_L1L2_CKPT"

fi

# =============================================================================
# EVALUATION
# =============================================================================
if $DO_EVAL; then

    # ── Group A: Baseline ──
    # If Stage 2a result already exists, skip
    if [ -f "results/eval_openloop_full.pt" ]; then
        log "Group A: Using existing baseline eval (results/eval_openloop_full.pt)"
        cp results/eval_openloop_full.pt "$RESULTS_DIR/eval_A_baseline.pt"
        cp results/eval_openloop_full_summary.json "$RESULTS_DIR/eval_A_baseline_summary.json" 2>/dev/null || true
    else
        log "Group A: Baseline eval ($(timestamp))"
        python scripts/eval_openloop.py \
            "$CONFIG" "$CHECKPOINT" \
            --ann-file "$ANN_FILE" \
            --out "$RESULTS_DIR/eval_A_baseline.pt" \
            2>&1 | tee "$RESULTS_DIR/eval_A.log"
    fi

    # ── Group B: L1 only ──
    log "Group B: Eval with FiLM L1 ($(timestamp))"
    if [ -f "$FILM_L1_CKPT" ]; then
        python scripts/eval_openloop.py \
            "$CONFIG" "$CHECKPOINT" \
            --ann-file "$ANN_FILE" \
            --film-checkpoint "$FILM_L1_CKPT" \
            --out "$RESULTS_DIR/eval_B_l1only.pt" \
            2>&1 | tee "$RESULTS_DIR/eval_B.log"
    else
        echo "SKIP: $FILM_L1_CKPT not found. Run --train first."
    fi

    # ── Group C: L2 only ──
    log "Group C: Eval with FiLM L2 ($(timestamp))"
    if [ -f "$FILM_L2_CKPT" ]; then
        UQ_FILM_L2_ONLY=1 python scripts/eval_openloop.py \
            "$CONFIG" "$CHECKPOINT" \
            --ann-file "$ANN_FILE" \
            --film-checkpoint "$FILM_L2_CKPT" \
            --out "$RESULTS_DIR/eval_C_l2only.pt" \
            2>&1 | tee "$RESULTS_DIR/eval_C.log"
    else
        echo "SKIP: $FILM_L2_CKPT not found. Run --train first."
    fi

    # ── Group D: L1 + L2 ──
    log "Group D: Eval with FiLM L1+L2 ($(timestamp))"
    if [ -f "$FILM_L1L2_CKPT" ]; then
        python scripts/eval_openloop.py \
            "$CONFIG" "$CHECKPOINT" \
            --ann-file "$ANN_FILE" \
            --film-checkpoint "$FILM_L1L2_CKPT" \
            --out "$RESULTS_DIR/eval_D_l1l2.pt" \
            2>&1 | tee "$RESULTS_DIR/eval_D.log"
    else
        echo "SKIP: $FILM_L1L2_CKPT not found. Run --train first."
    fi

fi

# =============================================================================
# VISUALIZATION
# =============================================================================
if $DO_VIZ; then

    log "Generating ablation comparison figures ($(timestamp))"
    mkdir -p "$RESULTS_DIR/figures"

    python -c "
import torch, json, os, sys
import numpy as np

results_dir = '$RESULTS_DIR'
groups = {
    'A (Baseline)': 'eval_A_baseline.pt',
    'B (L1 only)': 'eval_B_l1only.pt',
    'C (L2 only)': 'eval_C_l2only.pt',
    'D (L1+L2)': 'eval_D_l1l2.pt',
}

NORMAL_WEATHER = {0, 1, 2, 3}

# Load available results
data = {}
for label, fname in groups.items():
    path = os.path.join(results_dir, fname)
    if os.path.exists(path):
        records = torch.load(path, weights_only=False)
        data[label] = records
        print(f'Loaded {label}: {len(records)} samples')
    else:
        print(f'SKIP {label}: {path} not found')

if len(data) < 2:
    print('Need at least 2 groups for comparison. Exiting.')
    sys.exit(0)

# ── Compute metrics per group ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'figure.dpi': 300})

metrics_names = ['plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s',
                 'plan_obj_col_1s', 'plan_obj_col_2s', 'plan_obj_col_3s']

table = {}
for label, records in data.items():
    normal = [r for r in records if r.get('weather_id', -1) in NORMAL_WEATHER and (r.get('plan_L2_3s', 0) > 0 or r.get('fut_valid', False))]
    adverse = [r for r in records if r.get('weather_id', -1) not in NORMAL_WEATHER and (r.get('plan_L2_3s', 0) > 0 or r.get('fut_valid', False))]
    all_valid = normal + adverse

    row = {'n_normal': len(normal), 'n_adverse': len(adverse)}
    for split_name, split_data in [('all', all_valid), ('normal', normal), ('adverse', adverse)]:
        for m in metrics_names:
            vals = [r[m] for r in split_data if m in r]
            row[f'{split_name}_{m}'] = np.mean(vals) if vals else float('nan')
    table[label] = row

# Print results table
print('\n' + '='*80)
print('ABLATION RESULTS')
print('='*80)
header = f\"{'Group':20s} | {'Split':8s} | {'L2@1s':7s} | {'L2@2s':7s} | {'L2@3s':7s} | {'Col@1s':7s} | {'Col@2s':7s} | {'Col@3s':7s}\"
print(header)
print('-'*len(header))
for label, row in table.items():
    for split in ['all', 'normal', 'adverse']:
        vals = [f\"{row.get(f'{split}_{m}', float('nan')):7.4f}\" for m in metrics_names]
        print(f\"{label:20s} | {split:8s} | {' | '.join(vals)}\")
    print('-'*len(header))

# Save to JSON
json_out = {}
for label, row in table.items():
    json_out[label] = {k: float(v) if not isinstance(v, (int, float)) else v for k, v in row.items()}
with open(os.path.join(results_dir, 'ablation_results.json'), 'w') as f:
    json.dump(json_out, f, indent=2)
print(f'\nSaved: {results_dir}/ablation_results.json')

# ── Bar chart: L2 error by group (normal vs adverse) ──
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, horizon in enumerate(['1s', '2s', '3s']):
    m = f'plan_L2_{horizon}'
    labels_list = list(table.keys())
    normal_vals = [table[l].get(f'normal_{m}', 0) for l in labels_list]
    adverse_vals = [table[l].get(f'adverse_{m}', 0) for l in labels_list]

    x = np.arange(len(labels_list))
    w = 0.35
    axes[i].bar(x - w/2, normal_vals, w, label='Normal', color='steelblue', alpha=0.8)
    axes[i].bar(x + w/2, adverse_vals, w, label='Adverse', color='orangered', alpha=0.8)
    axes[i].set_ylabel('L2 Error (m)')
    axes[i].set_title(f'Planning L2 @ {horizon}')
    axes[i].set_xticks(x)
    axes[i].set_xticklabels([l.split('(')[0].strip() for l in labels_list], fontsize=8)
    axes[i].legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(results_dir, 'figures', 'ablation_l2_bars.png'))
plt.close(fig)
print(f'Saved: {results_dir}/figures/ablation_l2_bars.png')

# ── Bar chart: Collision rate by group ──
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, horizon in enumerate(['1s', '2s', '3s']):
    m = f'plan_obj_col_{horizon}'
    labels_list = list(table.keys())
    normal_vals = [table[l].get(f'normal_{m}', 0) * 100 for l in labels_list]
    adverse_vals = [table[l].get(f'adverse_{m}', 0) * 100 for l in labels_list]

    x = np.arange(len(labels_list))
    w = 0.35
    axes[i].bar(x - w/2, normal_vals, w, label='Normal', color='steelblue', alpha=0.8)
    axes[i].bar(x + w/2, adverse_vals, w, label='Adverse', color='orangered', alpha=0.8)
    axes[i].set_ylabel('Collision Rate (%)')
    axes[i].set_title(f'Collision @ {horizon}')
    axes[i].set_xticks(x)
    axes[i].set_xticklabels([l.split('(')[0].strip() for l in labels_list], fontsize=8)
    axes[i].legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(results_dir, 'figures', 'ablation_collision_bars.png'))
plt.close(fig)
print(f'Saved: {results_dir}/figures/ablation_collision_bars.png')

# ── Improvement table (relative to baseline) ──
if 'A (Baseline)' in table:
    base = table['A (Baseline)']
    print('\nRelative improvement vs Baseline (negative = better):')
    for label, row in table.items():
        if label == 'A (Baseline)':
            continue
        for split in ['adverse']:
            for m in ['plan_L2_3s', 'plan_obj_col_3s']:
                base_val = base.get(f'{split}_{m}', 0)
                cur_val = row.get(f'{split}_{m}', 0)
                if base_val > 0:
                    pct = (cur_val - base_val) / base_val * 100
                    print(f'  {label} {split} {m}: {pct:+.1f}%')
"

fi

log "Ablation complete ($(timestamp))"

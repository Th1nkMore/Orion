# UQ-ORION Session Handover

> Last updated: 2026-03-28

## Current Status

**Stage 2a: Open-Loop Eval** — running in background (~6h total, ~1.8s/sample).

```bash
# Running command:
python scripts/eval_openloop.py \
    adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth \
    --ann-file data/infos/b2d_infos_val.pkl \
    --out results/eval_openloop_full.pt
```

- 12806 samples, FiLM OFF (transformer `use_uncertainty=False`)
- UQ scores captured via forward hook
- Output: `results/eval_openloop_full.pt` + `_summary.json`

## Completed Stages

| Stage | Status | Key Output |
|-------|--------|------------|
| 0: Feature Extraction | ✅ | `data/features/` (235GB, 12806 samples) |
| 1a: Pseudo-labels | ✅ | `data/labels/uq_labels.pt` (scene_type calibrated) |
| 1b: UQEstimator Training | ✅ | `checkpoints/uq/best.pt` (Spearman ρ=0.96, epoch 15) |
| 2a: Open-Loop Eval | 🔄 | Running (~6h) |
| 2b: FiLM L1 Fine-tune | ⏳ | Script ready: `scripts/train_film.py` |

## After Eval Completes

```bash
# 1. Check results
python -c "
import torch, json
data = torch.load('results/eval_openloop_full.pt', weights_only=False)
with open('results/eval_openloop_full_summary.json') as f:
    print(json.dumps(json.load(f), indent=2))
"

# 2. Train FiLM L1 (~2h on val set)
python scripts/train_film.py \
    --config adzoo/orion/configs/orion_stage3_infer.py \
    --checkpoint ckpts/Orion.pth \
    --epochs 3 --lr 1e-3 \
    --out checkpoints/film/best.pt

# 3. Eval with trained FiLM
python scripts/eval_openloop.py \
    adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth \
    --ann-file data/infos/b2d_infos_val.pkl \
    --film-checkpoint checkpoints/film/best.pt \
    --out results/eval_openloop_film_l1.pt
```

## Key Technical Discoveries

### FiLM L1 Training Gap
- UQEstimator is trained (outputs meaningful scores/embeddings)
- FiLM layers (`film_gamma`, `film_beta`) in `PETRTemporalTransformer` are **NOT** in the UQ checkpoint — they need separate training
- Config has `use_uncertainty=False` in transformer → FiLM disabled by default
- Identity initialization added: gamma bias=1, beta bias=0, all weights=0

### Gradient Flow for FiLM Training
- `generate()` in LLM is NOT differentiable → can't backprop through it
- Solution: use teacher-forcing (`lm_head.forward()` with labels) which IS differentiable
- Gradient path: `trajectory_loss → VAE → ego_feature → LLM(teacher forcing) → vlm_memory → QT-Former(+FiLM) → FiLM layers`
- Implemented in `scripts/train_film.py:forward_film_training()`

### Planning Metrics
- `plan_results` in dataset JSON export is empty **by design** (not a bug)
- Actual planning metrics computed in `orion.py:compute_planner_metric_stp3()` at inference time
- Metrics stored in `result['metric_results']`: `plan_L2_1s/2s/3s`, `plan_obj_col_1s/2s/3s`
- `fut_valid_flag=False` doesn't mean invalid — metrics still computed when trajectory has 6 steps

### Weather Split
- Normal: Weather 0–3 (ClearNoon/Sunset, CloudyNoon/Sunset) → 2709 val samples
- Adverse: everything else (rain/fog/night/wet) → 10097 val samples

## ORION Files Modified

All changes searchable via `[UQ]` tag. See README.md for full details.

| File | Lines Added | Purpose |
|------|-------------|---------|
| `adzoo/orion/configs/orion_stage3_infer.py` | +3 | UQ config flags |
| `adzoo/orion/test.py` | +26 | UQ + FiLM checkpoint reload |
| `mmcv/models/dense_heads/orion_head.py` | +23 | UQEstimator in forward pass |
| `mmcv/models/utils/petr_transformers.py` | +16 | FiLM modulation + identity init |

## Data & Checkpoints

| Path | Size | Description |
|------|------|-------------|
| `ckpts/Orion.pth` | 36GB | ORION main checkpoint |
| `ckpts/pretrain_qformer/` | ~14GB | Qwen LLM weights |
| `checkpoints/uq/best.pt` | 25MB | Trained UQEstimator |
| `data/features/` | 235GB | Pre-extracted EVAViT tokens |
| `data/labels/uq_labels.pt` | 1.3MB | Pseudo-labels |
| `data/bench2drive/v1/` | 407GB | Bench2Drive raw dataset |
| `data/infos/b2d_infos_val.pkl` | 141MB | Val annotations (12806 samples) |
| `data/infos/b2d_infos_train.pkl` | 22MB | Train annotations (234848 samples) |

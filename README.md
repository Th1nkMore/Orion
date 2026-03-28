# UQ-ORION

Uncertainty-aware extension for [ORION](https://github.com/xiaomi-mlab/Orion) end-to-end autonomous driving. Improves safety in adverse weather (rain, fog, night) through lightweight uncertainty quantification — zero backbone fine-tuning, <5M new parameters.

## Architecture

```
Vision Encoder (EVAViT, frozen)
    ↓ patch_tokens [B, 6, 1600, 1024]
    ├──────────────────────────────────────────┐
    │                                          ↓
    │                                  UQEstimator (2.24M, trained)
    │                                          ↓
    │                             uncertainty_embedding [B, 256]
    │                             uncertainty_score     [B, 1]
    │                                          │
QT-Former (frozen) ←── FiLM L1 ──────────────┤   gamma * query + beta
    ↓ vlm_memory                               │
LLM / Qwen (frozen)                            │
    ↓ ego_feature                              │
VAE (frozen) ←── FiLM L2 ─────────────────────┘
    ↓
trajectory
```

**FiLM L1** modulates QT-Former detection queries: `query = γ(u) · query + β(u)` where `u` is the uncertainty embedding. Identity-initialized (γ=1, β=0) and fine-tuned with trajectory loss.

**FiLM L2** modulates ego_feature (current_states) before VAE: `current_states = γ₂(u) · current_states + β₂(u)`. Same identity init. Biases the trajectory distribution toward conservative mode under high uncertainty. ~2.11M params (Linear 256→4096 × 2).

## Project Structure

```
uq-orion/
├── uq_estimator/                  # UQ extension module (all new code)
│   ├── model.py                   # UQEstimator: 2.24M params
│   ├── losses.py                  # Regression + ranking + calibration losses
│   └── dataset.py                 # UQFeatureDataset + compute_stat_features
├── scripts/
│   ├── extract_orion_features.py  # Stage 0: extract patch tokens from ORION
│   ├── generate_labels.py         # Stage 1a: compute uncertainty pseudo-labels
│   ├── train_uq.py               # Stage 1b: train UQEstimator
│   ├── validate_uq.py            # Stage 1c: validation report + plots
│   ├── eval_openloop.py           # Stage 2: open-loop eval with UQ analysis
│   ├── train_film.py             # Stage 2: FiLM L1 fine-tuning
│   └── e2e_mock_test.py          # Pipeline smoke test (no data needed)
├── configs/
│   └── uq_train.yaml             # UQEstimator model/training config
├── tests/                         # Pytest tests with mock data
├── adzoo/                         # ORION original code (4 files modified, see below)
├── mmcv/                          # ORION dependency (2 files modified, see below)
├── checkpoints/uq/best.pt        # Trained UQEstimator weights
├── PLAN.md                        # Detailed project plan with ablation design
└── CLAUDE.md                      # Development conventions
```

## ORION File Modifications

All modifications to ORION code are marked with `[UQ]` comments and can be found by searching for that tag. Total: **~128 lines** across 5 files.

### `adzoo/orion/configs/orion_stage3_infer.py` (+4 lines)

Added UQ config entries:
```python
use_uncertainty=True,                          # pts_bbox_head: enable UQEstimator
uq_checkpoint='checkpoints/uq/best.pt',       # pts_bbox_head: trained weights path
use_uncertainty=False,                         # transformer: FiLM L1 disabled until trained
use_uncertainty_l2=False,                      # model: FiLM L2 disabled until trained
```

### `adzoo/orion/test.py` (+32 lines)

Three checkpoint reloading blocks after the main ORION checkpoint loads:
1. **UQEstimator reload** (lines 187–199): The 36GB ORION checkpoint doesn't contain UQ weights; after `load_checkpoint()`, we reload the UQEstimator state dict from `checkpoints/uq/best.pt`.
2. **FiLM L1 weight reload** (lines 201–211): If `UQ_FILM_CHECKPOINT` env var is set, load trained FiLM L1 gamma/beta weights into the transformer.
3. **FiLM L2 weight reload** (lines 213–218): Same checkpoint, loads FiLM L2 gamma/beta weights into the detector model.

### `mmcv/models/dense_heads/orion_head.py` (+25 lines)

- **Constructor** (lines 298–312): When `use_uncertainty=True`, instantiates `UQEstimator` from `uq_estimator/model.py`, loads config from `configs/uq_train.yaml`, optionally loads checkpoint.
- **Forward** (lines 759–770): Reshapes image features `[B, N_views, C, H, W]` → patch tokens `[B, N_views, H*W, C]`, computes 5-dim stat features, runs UQEstimator → `uncertainty_emb [B, 256]`.
- **Transformer call** (line 798): Passes `uncertainty_emb` to the PETRTemporalTransformer.
- **Return** (line 978): Returns 3-element tuple `(outs, vlm_memory, uncertainty_emb)` for FiLM L2 use.

### `mmcv/models/utils/petr_transformers.py` (+16 lines)

- **Constructor** (lines 271–279): Creates `film_gamma` and `film_beta` Linear(256→256) layers when `use_uncertainty=True`. Identity-initialized: `gamma.bias=1, beta.bias=0, weights=0`.
- **Forward** (lines 307–311): Applies FiLM L1: `query = gamma(emb) * query + beta(emb)` to detection queries before the decoder, gated by `use_uncertainty` flag.

### `mmcv/models/detectors/orion.py` (+35 lines)

- **Constructor**: When `use_uncertainty_l2=True`, creates `film_gamma_l2` and `film_beta_l2` Linear(256→4096) with identity init (~2.11M params).
- **Training path** (`forward_pts_train`): Applies FiLM L2 to `current_states` before VAE: `current_states = γ₂(u) · current_states + β₂(u)`.
- **Inference path** (`simple_test_pts`): Same FiLM L2 modulation.
- Both paths unpack 3-element tuple from head: `(outs, det_query, uncertainty_emb)`.

## Pipeline Stages

### Stage 0: Feature Extraction ✅
```bash
python scripts/extract_orion_features.py \
    --checkpoint ckpts/Orion.pth \
    --output_dir data/features \
    --ann_file data/infos/b2d_infos_val.pkl
```
Extracts EVAViT patch tokens per sample → `data/features/*.pt` (~235GB for 12806 samples).

### Stage 1: UQEstimator Training ✅
```bash
# 1a. Generate pseudo-labels (3-component: gradient + entropy + cross-view consistency)
python scripts/generate_labels.py \
    --feature_dir data/features --output_file data/labels/uq_labels.pt

# 1b. Train (50 epochs, stops early, ~2h on A100)
python scripts/train_uq.py --config configs/uq_train.yaml

# 1c. Validate
python scripts/validate_uq.py --checkpoint checkpoints/uq/best.pt \
    --feature_dir data/features --label_file data/labels/uq_labels.pt
```
Result: `checkpoints/uq/best.pt` — Spearman ρ=0.96 at epoch 15.

### Stage 2a: Open-Loop Eval + UQ Score Analysis 🔄
```bash
python scripts/eval_openloop.py \
    adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth \
    --ann-file data/infos/b2d_infos_val.pkl \
    --out results/eval_openloop_full.pt
```
Runs ORION inference on full val set (12806 samples, ~6h), captures per-sample:
- Planning metrics: L2 error @1s/2s/3s, collision rate
- UQ scores via forward hook on UQEstimator
- Splits by weather: normal (Weather 0–3) vs adverse (rain/fog/night/wet)

Output: `results/eval_openloop_full.pt` (per-sample records) + `_summary.json`.

### Stage 2b: FiLM L1 Fine-tuning (next)
```bash
python scripts/train_film.py \
    --config adzoo/orion/configs/orion_stage3_infer.py \
    --checkpoint ckpts/Orion.pth \
    --epochs 3 --lr 1e-3 \
    --out checkpoints/film/best.pt
```
Freezes all ORION params (~1B), trains only FiLM gamma/beta (~131K params) using teacher-forcing through LLM for gradient flow. Uses trajectory planning loss.

### Stage 2b: L1 Eval (after FiLM training)
```bash
python scripts/eval_openloop.py \
    adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth \
    --ann-file data/infos/b2d_infos_val.pkl \
    --film-checkpoint checkpoints/film/best.pt \
    --out results/eval_openloop_film_l1.pt
```

## Weather Classification

CARLA Weather IDs used for normal/adverse split:

| Weather ID | Name | Category |
|-----------|------|----------|
| 0 | ClearNoon | normal |
| 1 | ClearSunset | normal |
| 2 | CloudyNoon | normal |
| 3 | CloudySunset | normal |
| 5–14 | Wet/Rain variants | adverse |
| 15, 18–23 | Night variants | adverse |
| 25–26 | Foggy variants | adverse |

Val set: 2709 normal / 10097 adverse samples.

## Environment

```bash
# Python 3.11+, CUDA, 1x A100 80GB
# Dependencies managed separately from ORION:
pip install -r requirements_uq.txt   # UQ-specific deps
pip install -r requirements.txt       # ORION deps (do not modify)
```

Key: ORION model (Orion.pth, 36GB) + Qwen LLM (14GB) → ~40GB VRAM at inference.

## Reproduction Checklist

1. Ensure `ckpts/Orion.pth` and `ckpts/pretrain_qformer/` exist
2. Ensure `data/bench2drive/v1/` has 1001 scenario folders
3. Ensure `data/infos/b2d_infos_val.pkl` and `b2d_infos_train.pkl` exist
4. Run `pytest tests/ -v` to verify UQ module integrity
5. Follow pipeline stages above in order

## Citation

```bibtex
@inproceedings{fu2025orion,
  title={ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation},
  author={Haoyu Fu and others},
  booktitle={ICCV},
  year={2025}
}
```

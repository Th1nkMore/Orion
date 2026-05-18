# UQ-ORION

Uncertainty-aware extension for [ORION](https://github.com/xiaomi-mlab/Orion) end-to-end autonomous driving. Improves safety in adverse weather (rain, fog, night) through lightweight uncertainty quantification — zero backbone fine-tuning, <5M new parameters.

## Documentation map

| Document | What it is |
|----------|------------|
| **[REPORT.md](./REPORT.md)** | Full project report: pseudo-label v1→v3, UQ/FiLM results, closed-loop metrics, collision-aware training, known bugs (init_weights, LayerNorm), Score-Gated FiLM rationale. |
| **[docs/plan_v2.md](./docs/plan_v2.md)** | Research plan (IPM BEV, ablations, risks). |
| **[docs_learning/](./docs_learning/)** | In-repo learning notes (architecture walkthroughs; not necessarily git-tracked). |
| **[CLAUDE.md](./CLAUDE.md)** | Dev conventions: tensor shapes, tests, commit rules, environment (`uv` / `.venv`). |

---

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
QT-Former (frozen) ←── FiLM L1 ──────────────┤   score-gated γ, β on queries
    ↓ vlm_memory                               │
LLM / Qwen (frozen)                            │
    ↓ ego_feature                              │
VAE (frozen) ←── FiLM L2 ─────────────────────┘   score-gated γ, β on current_states
    ↓
trajectory
```

### What each part does (plain language)

- **Vision Encoder (EVAViT)** — Turns multi-camera images into a grid of patch tokens per view. It stays **frozen**; we never backprop into it for UQ/FiLM training.
- **UQEstimator** — A small side network that reads those patch tokens (plus 5 hand-crafted statistics). It outputs a **scalar score** in (0,1) (“how uncertain / degraded is perception?”) and a **256-D embedding** used only for conditioning.
- **Pseudo-labels (Stage 1a)** — There is no human “uncertainty ground truth.” Labels are **derived from the same tokens** (e.g. max activation, cross-view consistency, entropy) and **weather-aware calibration** so training aligns with normal vs adverse splits used in eval (see `REPORT.md` v3).
- **QT-Former** — Maps image tokens to BEV-style representations for detection / VLM. **Frozen.** **FiLM L1** slightly rescales/shifts its queries using the UQ embedding; **score gating** makes modulation vanish when `uncertainty_score → 0`.
- **LLM (Qwen)** — Language-side reasoning; **frozen** during FiLM training; teacher forcing is used so gradients reach FiLM.
- **VAE (trajectory head)** — Samples/filters trajectory modes. **Frozen.** **FiLM L2** modulates the ego planning feature before the VAE, again **gated by UQ score** so clear-weather frames stay near baseline.

**FiLM L1 (QT-Former)** — With uncertainty embedding `u` and score `s`, the implementation uses raw linear maps `γ_raw(u), β_raw(u)` then applies **Score-Gated FiLM**: `γ = 1 + s·(γ_raw - 1)`, `β = s·β_raw`, then `query = γ ⊙ query + β` (broadcast over queries). If `s` is omitted, it falls back to ungated modulation.

**FiLM L2 (before VAE)** — Same gating idea on `current_states`: scale branch uses `1 + s·(γ_raw - 1)`; shift branch uses `s·β_raw`. **~2.11M params** (Linear 256→4096 × 2 for γ and β).

---

## Project Structure

```
uq-orion/
├── uq_estimator/
│   ├── model.py              # UQEstimator + UQOutput
│   ├── losses.py             # MSE + calibration + pairwise ranking
│   ├── dataset.py            # UQFeatureDataset, stat features, FastTensorLoader
│   └── bev_uncertainty.py    # Patch quality, IPM BEV maps, trajectory cost helpers
├── scripts/
│   ├── extract_orion_features.py   # Stage 0: patch tokens → data/features/
│   ├── generate_labels.py        # Stage 1a: pseudo-labels (+ stat_cache, scene_type_map)
│   ├── train_uq.py               # Stage 1b: train UQEstimator
│   ├── validate_uq.py            # Stage 1c: metrics + report
│   ├── eval_openloop.py          # Stage 2a: open-loop + UQ hooks
│   ├── train_film.py             # Stage 2b: FiLM only (env: L1 / L2 / L1+L2)
│   ├── eval_ablation_full.py     # Hot-swap FiLM ablations
│   ├── eval_closedloop_replay.py # Bench2Drive replay metrics
│   ├── visualize_eval.py         # Figures / summaries
│   ├── generate_trajectory_gifs.py
│   └── e2e_mock_test.py          # Smoke test (no dataset)
├── configs/
│   ├── uq_train.yaml             # Default UQ training + ablation yaml variants
│   └── uq_ablation_*.yaml
├── checkpoints/
│   ├── uq/best.pt                # Trained UQEstimator (keep in sync with labels)
│   └── film/*.pt                 # e.g. best_l1l2_col_v3.pt (collision-aware FiLM)
├── tests/                        # pytest (mock data)
├── adzoo/ , mmcv/                # ORION + deps (`[UQ]` patches)
├── REPORT.md                     # Detailed results & narrative
└── CLAUDE.md                     # Contributor / CI conventions
```

---

## ORION File Modifications

All modifications are marked with `[UQ]` — search: `grep -r "\[UQ]" adzoo/ mmcv/`.

### `adzoo/orion/configs/orion_stage3_infer.py`

`use_uncertainty`, `uq_checkpoint`, transformer `use_uncertainty`, `use_uncertainty_l2` flags.

### `adzoo/orion/test.py`

Reload **UQEstimator** from `uq_checkpoint` (not in the 36GB ORION ckpt); optional **FiLM** weights via `UQ_FILM_CHECKPOINT`.

### `mmcv/models/dense_heads/orion_head.py`

Builds `UQEstimator` from `configs/uq_train.yaml`, reshapes backbone features to patch tokens, computes 5-D stats, returns `(outs, vlm_memory, uncertainty_emb, uncertainty_score)` for downstream FiLM / BEV helpers.

### `mmcv/models/utils/petr_transformers.py`

FiLM L1 layers + **Score-Gated** forward using `uncertainty_emb` and optional `uncertainty_score`; `init_weights` skips FiLM weights so identity init is preserved.

### `mmcv/models/detectors/orion.py`

FiLM L2 layers + **Score-Gated** modulation on the VAE path; consumes 4-tuple from head.

---

## Pipeline stages (current scripts & defaults)

### Stage 0: Feature extraction

```bash
python scripts/extract_orion_features.py \
    --checkpoint ckpts/Orion.pth \
    --output_dir data/features \
    --ann_file data/infos/b2d_infos_val.pkl \
    --batch_size 8 --num_workers 1
```

**Output:** one `.pt` per sample with patch tokens `[6, 1600, 1024]` (order of ~235GB for the full val list — scale depends on how many infos you pass).

---

### Stage 1: UQEstimator (labels → train → validate)

#### 1a. Pseudo-labels (v3-style, weather-aligned)

Labels are a weighted mix of **max-mean**, **cross-view cosine similarity**, and **token entropy**, then **percentile calibration** per `scene_type`. For labels consistent with open-loop **weather ID** splits (normal = Weather 0–3), pass a weather map (see `REPORT.md` / `data/weather_scene_type_map.pt` if you generated it):

```bash
# Optional: precompute 5-D stats once (speeds up label generation)
# python -c "..."  # or your cache script → data/stat_cache.pt

python scripts/generate_labels.py \
    --feature_dir data/features \
    --output_file data/labels/uq_labels.pt \
    --stat_cache data/stat_cache.pt \
    --scene_type_map data/weather_scene_type_map.pt
```

`--stat_cache` and `--scene_type_map` are optional but recommended for speed and eval alignment.

#### 1b. Train UQEstimator (matches `configs/uq_train.yaml`)

Default settings in repo:

| Setting | Typical value (see YAML) |
|---------|----------------------------|
| Epochs | `training.epochs` (default **20**) |
| Optimizer | AdamW (`lr`, `weight_decay`) |
| LR schedule | Linear **warmup** (`warmup_epochs`) then **cosine** to `lr * 0.01` |
| Batch | `training.batch_size` (default **64**); use **≥4** so batch std in calibration loss is defined |
| Loss | `MSE + λ_cal·calibration + λ_rank·ranking` (`loss` section; ablations flip `use_ranking` / `use_calibration`) |
| Data | `n_patches_subsample` (e.g. **256**) subsamples patches per view in RAM; stats can still come from full-res cache |
| Speed | Set `data.preload: true` to use **FastTensorLoader** (pinned memory + bulk index) |

```bash
python scripts/train_uq.py --config configs/uq_train.yaml
# Resume:
python scripts/train_uq.py --config configs/uq_train.yaml --resume checkpoints/uq/checkpoint_epochXXXXX.pt
```

**Checkpointing:** `best.pt` = lowest **validation total loss**; periodic `checkpoint_epoch*.pt`. **No early stopping** in script — runs all `epochs`.

**Smoke / CI:** `python scripts/train_uq.py --mock --smoke`

#### 1c. Validate

```bash
python scripts/validate_uq.py \
    --checkpoint checkpoints/uq/best.pt \
    --feature_dir data/features \
    --label_file data/labels/uq_labels.pt \
    --output_dir reports/uq_validation
```

**Reference metrics:** After v3 label + eval alignment, **AUROC ≈ 0.954** (UQ vs weather-based adverse); see `REPORT.md` for Spearman / separation history.

---

### Stage 2a: Open-loop eval + UQ logging

```bash
python scripts/eval_openloop.py \
    adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth \
    --ann-file data/infos/b2d_infos_val.pkl \
    --out results/eval_openloop_full.pt
```

Per sample: planning L2 / collision metrics, **UQ score** (hook), weather split. Add `--film-checkpoint path.pt` when evaluating trained FiLM.

---

### Stage 2b: FiLM training

`train_film.py` **freezes** the ORION stack and trains only FiLM parameters. Mode is selected with **environment variables** (used by `run_ablation.sh`):

| Mode | Env |
|------|-----|
| FiLM L1 only (default) | *(none)* |
| FiLM L2 only | `USE_FILM_L2_ONLY=1` |
| FiLM L1 + L2 | `USE_FILM_L1L2=1` |

```bash
# L1 only (default)
python scripts/train_film.py \
    --config adzoo/orion/configs/orion_stage3_infer.py \
    --checkpoint ckpts/Orion.pth \
    --epochs 3 --lr 1e-3 \
    --out checkpoints/film/best.pt

# L1+L2 with collision-aware term (example hyperparams from REPORT)
USE_FILM_L1L2=1 python scripts/train_film.py \
    --max-samples 3000 --epochs 5 \
    --lr 1e-3 --lambda-col 0.5 --col-margin 4.0 \
    --out checkpoints/film/best_l1l2_col_v3.pt
```

**Notes:** `--grad-accum` default 4; UQ checkpoint path is set inside the script (`checkpoints/uq/best.pt`). Collision loss uses **GT agents** and optional **detached UQ score** as weight. See `REPORT.md` §9 for shapes and caveats.

---

### Stage 2b (eval with FiLM)

```bash
python scripts/eval_openloop.py \
    adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth \
    --ann-file data/infos/b2d_infos_val.pkl \
    --film-checkpoint checkpoints/film/best_l1l2_col_v3.pt \
    --out results/eval_openloop_film.pt
```

At inference, load FiLM via `UQ_FILM_CHECKPOINT` in `test.py` as documented in ORION patches.

---

## Weather classification

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

---

## Environment

```bash
# Recommended: project venv (see CLAUDE.md)
# source .venv/bin/activate
uv pip install -r requirements_uq.txt   # UQ stack
uv pip install -r requirements.txt       # ORION base (do not modify casually)
```

Python **3.11+**, CUDA. Full ORION inference: **~40GB VRAM** (36GB ORION + Qwen, order-of-magnitude).

---

## Reproduction checklist

1. `ckpts/Orion.pth` and `ckpts/pretrain_qformer/` (or equivalent LLM path in config)
2. `data/bench2drive/v1/` and `data/infos/b2d_infos_val.pkl` (and train pkl if needed)
3. `pytest tests/ -v`
4. Run pipeline stages in order: features → labels → `train_uq` → (optional) `train_film` → eval scripts
5. For paper-grade numbers and caveats (FiLM vs baseline, init_weights bug, Normal ADE), read **`REPORT.md`**

---

## Citation

```bibtex
@inproceedings{fu2025orion,
  title={ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation},
  author={Haoyu Fu and others},
  booktitle={ICCV},
  year={2025}
}
```

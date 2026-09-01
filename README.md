# UQ-ORION

Research extension for [ORION](https://github.com/xiaomi-mlab/Orion) that studies whether spatial observation uncertainty can be grounded by the VLM and used by the planning stack. Closed-loop safety improvement is the target claim, not an established result.

> **Current-state authority (2026-09-01):** [docs/CURRENT_STATE.md](./docs/CURRENT_STATE.md) records the active architecture, exact completed gates, failed gates, open blockers, and next executable milestone. Machine-readable experiment authority remains in `configs/scenario_factory/` and hash-bound result records in `results/scenario_factory/`.
>
> The original FiLM, scalar Density-UQ, explicit scalar-token, and hard-governor paths are retained for historical comparison only. They are not the current mainline.

## Documentation map

| Document | What it is |
|----------|------------|
| **[docs/CURRENT_STATE.md](./docs/CURRENT_STATE.md)** | Canonical current status and execution order. Read this first. |
| **[docs/spatial_uq_two_stage_v2.md](./docs/spatial_uq_two_stage_v2.md)** | Active Stage-1/Stage-2 responsibility contract. |
| **[configs/scenario_factory/protocol_v1.json](./configs/scenario_factory/protocol_v1.json)** | Machine-readable architecture and scenario-factory contract. |
| **[REPORT.md](./REPORT.md)** | Full project report: pseudo-label v1→v3, UQ/FiLM results, closed-loop metrics, collision-aware training, known bugs (init_weights, LayerNorm), Score-Gated FiLM rationale. |
| **[docs/plan_v2.md](./docs/plan_v2.md)** | Research plan (IPM BEV, ablations, risks). |
| **[docs_learning/](./docs_learning/)** | In-repo learning notes (architecture walkthroughs; not necessarily git-tracked). |
| **[CLAUDE.md](./CLAUDE.md)** | Dev conventions: tensor shapes, tests, commit rules, environment (`uv` / `.venv`). |

---

## Current architecture

```text
multi-view observations + temporal context
                    |
                    v
 frozen Stage-1 observation-UQ adapter
 U: view × position × component × time
                    |
                    v
 frozen task-agnostic U-tokenizer
                    |
                    v
 ORION visual evidence + navigation + ego state
                    |
                    v
 Stage-2L ORION/VLM: task relevance R + structured QA
                    |
                    v
 Stage-2P planner: risk-aware trajectory response
```

- Stage 1 may estimate only observation evidence loss. It must not receive route, actor, TTC, collision, corruption-family, or action labels.
- Stage 2L owns task relevance and interpretable uncertainty semantics.
- Stage 2P owns trajectory response and remains locked until Stage 2L passes held-out semantic and false-conservatism gates.
- Legacy Density UQ, scalar UQ tokens, FiLM, and scalar speed governors remain disabled in the current mainline.

The largest immediate gap is **semantic identifiability and held-out generalization of task relevance R**, not generic six-view balance. A CPU audit rebuilt all 80 geometry targets and their 10x10 trainer tensors at zero error, ruling out stale lineage or a deterministic camera/projection mismatch. A second frozen inventory then decomposed the union target: among 21 strict clean-off non-held-out routes, the route corridor contributes 1,103 front frame-view positives but only 15/11 front-left/front-right and none behind; actor support contributes 179 front, 90 front-left, 29 front-right, 263 back, 128 back-left and zero back-right. Thus the front-dominant loss mixes route binding with a differently distributed conflict-actor problem. Until those components are supervised and evaluated separately, the language bridge cannot receive a valid held-out U-to-task signal. See [docs/CURRENT_STATE.md](./docs/CURRENT_STATE.md).

The original v11 run (Job `1120666`) remains recorded as a 10x10 consumer-grid control-contract failure. Its separately versioned v11.1 replacement fixes that defect exactly; Job `1120954` then validly localized the failure to frozen R (`0.8667` train versus `0.55` dev) before language optimization. The calibrated v12 within-group view objective passes its CPU integrity checks but cannot separate route-corridor and conflict-actor learning inside one union-R loss. The next permitted work is therefore a CPU-only factorized-R target/interface preflight with an explicit unsupported-view boundary—not more GPU epochs and not a custom rear-right route. Language, Stage2-P, learned-U closed loop and benchmark work remain unauthorized.

The coverage-repair audit has now exhausted the cheap existing-data path. Route196 contributes five geometry-valid train-only frames and raises front-right independent-event coverage, but not back-right. A scan of every aligned raw frame in all 14 accepted train event identities found zero `CAM_BACK_RIGHT` relevance support. The preregistered fallback therefore froze Route167 (`YieldToEmergencyVehicle`) for one outcome-blind `clean_off` collection under a non-held-out `train_coverage_repair` split. Its first Job `1121242` was an invalid relative-path launch. The fixed Job `1121244` loaded Town03, ORION and real control, then failed when CARLA `world.tick` stalled for 301 seconds. A CPU scan of its 15 complete six-view/meta frames found `CAM_BACK=2` but `CAM_BACK_RIGHT=0`, so Route167 is retired as the missing-view fallback and will not be retried. The partial frames are diagnostic only; no R/U/language/planning optimization is unlocked.

The subsequent full raw-asset component inventory changes the next action. All 21 eligible route identities still have zero back-right union or actor support, so back-right is now an explicit unsupported claim region rather than a universal release gate. Front-right actor grounding exists in only Routes157/177/192/194; notably, Route202's nine front-right union positives are all route-only. Because 951 of 1,130 front union positives are route-only while every back/back-left positive is actor-grounded, a single union-R metric hides the actual supervision imbalance. A new CARLA collection and the R-only GPU smoke remain locked until route and actor relevance are factorized and audited on CPU.

That factorized-R CPU preflight now passes on all 80 frozen groups with exact route/actor/union reconstruction and finite gradients. Under the prospective `>=2` train-event and `>=1` dev-event rule, route relevance is identifiable only on front, while actor relevance is identifiable on front, front-left, front-right, back and back-left; back-right stays outside the claim. The frozen single-union R reaches dev recall `0.6744` on route/front but `0 / 0 / 0.0089` on actor front-left/front-right/back-left, supporting one bounded factorized-R-only engineering smoke as the next experiment. Front-right remains fragile at only two train events and one dev event. No GPU run, language training, Stage2-P or closed-loop method is unlocked by the CPU result alone.

The separate v12.1 trainer preflight now also passes on the exact 17-event/80-group assets: all component targets load, the duplicated route/actor warm start is bitwise identical to the old single-R probability, and neither ambiguous union convention enters the loss. Exactly one hash-attested 40-step R-only run was submitted as Slurm Job `1121553`; at the recorded `2026-09-01T15:28:11+08:00` scheduler snapshot it was `PENDING (Priority)`. This is a submission state, not a model result, and it does not unlock U, language, Stage2-P or closed loop.

---

## Historical baseline architecture (superseded)

The remainder of this README documents the original UQEstimator/FiLM baseline and is kept to make earlier experiments reproducible. Do not use it as current execution guidance.

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

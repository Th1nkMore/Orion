# UQ-ORION Session Handover

## 1. Background Process Status

**Feature Extraction Process:**
- PID: 145925 (main worker)
- Progress: 204 / 12806 samples (~1.6%)
- Speed: ~2 samples/sec
- Estimated completion: ~106 minutes from start (~07:00 UTC, so ~08:50 UTC)
- Command running:
  ```bash
  PYTHONPATH=. PYTHONUNBUFFERED=1 python3 scripts/extract_orion_features.py \
      --checkpoint /workspace/uq-orion/ckpts/Orion.pth \
      --output_dir /workspace/uq-orion/data/features \
      --ann_file /workspace/uq-orion/data/infos/b2d_infos_val.pkl \
      --batch_size 8 \
      --num_workers 1 \
      2>&1 | tee /workspace/extract_log_v2.txt
  ```
- File size: ~20MB per sample (fp16 tokens only, no image)
- Total disk usage: ~250GB for all 12806 samples

**After extraction completes, run:**

```bash
# STEP 2: Generate pseudo-labels
python scripts/generate_labels.py \
    --feature_dir /workspace/uq-orion/data/features \
    --output_file /workspace/uq-orion/data/labels/uq_labels.pt \
    --n_workers 4

# STEP 3: Train UQ Estimator
python scripts/train_uq.py \
    --config configs/uq_train.yaml \
    --data_dir /workspace/uq-orion/data/features \
    --label_file /workspace/uq-orion/data/labels/uq_labels.pt \
    2>&1 | tee /workspace/train_uq_log.txt
```

## 2. Completed Work (Commits)

```
f4dffab [UQ] perf: remove image storage, use fp16 tokens, fix stat features
c9c9eea [UQ] feat: add scene_type calibration to pseudo-label generation
5ad9d09 [UQ] feat: integrate UQEstimator into QT-Former via FiLM modulation
1459370 [UQ] fix: np.bool deprecation (numpy 1.24+), use bool instead
0304ca5 docs: add ORION reproduction report
ef5df97 chore: add cloud environment snapshot (PyTorch 2.10, CUDA 13.0)
537a4b2 [UQ] config: set absolute paths for vast.ai environment
```

## 3. Current File Status

```
M adzoo/orion/configs/orion_stage3_infer.py  # use_uncertainty=False (can set True after training)
M configs/uq_train.yaml                      # d_patch=1024, n_patches=1600
M mmcv/models/dense_heads/orion_head.py      # UQ integration
M mmcv/models/utils/petr_transformers.py   # FiLM modulation in PETRTemporalTransformer
M scripts/generate_labels.py                 # scene_type calibration
M uq_estimator/dataset.py                   # stat features from tokens only
M tests/fixtures.py                         # scene_type in mock data
M tests/test_generate_labels.py              # updated assertions
```

## 4. Remaining Steps

### STEP 2: Generate Pseudo-labels
```bash
cd /workspace/uq-orion
python scripts/generate_labels.py \
    --feature_dir /workspace/uq-orion/data/features \
    --output_file /workspace/uq-orion/data/labels/uq_labels.pt \
    --n_workers 4
```
- scene_type calibration: normal→[0,0.45], adverse→[0.55,1.0]
- Label format: `{filename: {'score': float, 'scene_type': str}}`

### STEP 3: Train UQ Estimator
```bash
cd /workspace/uq-orion
python scripts/train_uq.py \
    --config configs/uq_train.yaml \
    --data_dir /workspace/uq-orion/data/features \
    --label_file /workspace/uq-orion/data/labels/uq_labels.pt \
    2>&1 | tee /workspace/train_uq_log.txt
```
- Checkpoint saved to: `checkpoints/uq/best.pt`
- Report every 5 epochs: spearman correlation + separation metrics

### STEP 4: Validate
```bash
python scripts/validate_uq.py \
    --checkpoint /workspace/uq-orion/checkpoints/uq/best.pt \
    --feature_dir /workspace/uq-orion/data/features \
    --label_file /workspace/uq-orion/data/labels/uq_labels.pt \
    --output_dir /workspace/uq-orion/reports/uq_validation
```
- Acceptance criteria:
  - normal场景均值 < 0.45
  - adverse场景均值 > 0.55
  - Spearman相关系数 > 0.5

### STEP 5: Integrate into Inference
Set in `adzoo/orion/configs/orion_stage3_infer.py`:
```python
pts_bbox_head.use_uncertainty = True  # instead of False
```
Then run inference to see uncertainty scores per sample.

## 5. Known Issues & Notes

| Item | Detail |
|------|--------|
| **transformers version** | 5.3.0 (original was 4.35.0) |
| **fut_valid_flag bug** | Known bug in `b2d_orion_dataset.py:873` - `metric_dict=None` when all `fut_valid_flag=False`; does NOT affect UQ work |
| **EVAViT dimensions** | `d_patch=1024`, `n_patches=1600` (40×40 from 640×640 image with stride 16) |
| **Token storage** | fp16 - must convert `.float()` before computation in `dataset.py` |
| **UQ model** | Located in `uq_estimator/model.py` - 2.24M params |
| **FiLM integration** | `PETRTemporalTransformer` + `OrionTransformerDecoder` both support `use_uncertainty=True` with `uncertainty_emb=None` fallback |

## 6. Config Files

- `configs/uq_train.yaml`: Model dims (d_patch=1024, n_patches=1600, d_stat=5, d_out=256)
- `adzoo/orion/configs/orion_stage3_infer.py`: Orion inference config; set `use_uncertainty=True` in `pts_bbox_head` to enable UQ
- No changes needed to `petr_transformers.py` or `orion_head.py` to switch between modes

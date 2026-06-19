# Research and Execution Log

Use one entry per meaningful implementation change or experimental run.

Required run fields:

```text
Date/time:
Git commit:
Machine:
Command:
Configuration:
Input checkpoint:
Output path:
Split:
Result:
Failure/notes:
Next action:
```

---

## 2026-06-20: Density UQ Stage Completed

Git commit:

```text
45a4c4b Add density-based EVAViT uncertainty estimation
```

Machine:

```text
AutoDL RTX 4090
root@connect.weste.seetacloud.com:39408
```

Inputs:

```text
12,806 EVAViT feature files
tokens [6, 1600, 1024], float16
50 Bench2Drive validation routes
```

Completed:

- generated 12,806 descriptors of dimension 12,288;
- fitted normal-only standardization, PCA, and Ledoit-Wolf covariance;
- screened PCA dimensions 8, 16, 32, 64, 128, and 256;
- compared shrinkage Mahalanobis, diagonal distance, and kNN;
- selected 16-component shrinkage Mahalanobis;
- integrated runtime model into ORION;
- verified FP16-wrapped real-feature forward;
- verified complete ORION construction.

Final result:

```text
AUROC: 0.79898
AUPRC: 0.95116
Route-bootstrap AUROC 95% CI: 0.67547-0.91509
Active embedding dimension: 16
Compatibility output dimension: 256
```

Artifacts:

```text
checkpoints/density_uq/best.pt
reports/density_uq/metrics.json
reports/density_uq/weather_metrics.csv
reports/density_uq/score_distribution.png
reports/density_uq/embedding_projection.png
```

Notes:

- The 256-component model was rejected at AUROC 0.627.
- Weather 25 remains difficult in the held-out routes and has score statistics
  close to normal conditions.
- The current result is based on only 50 routes, so final claims require a
  larger train/validation protocol.

Next action:

Implement the single-token UQ projector and inject its output into the LLM
visual token sequence.

---

## 2026-06-20: Primary Conditioning Strategy Changed

Decision:

Make LLM uncertainty tokens the primary method. Reclassify FiLM L1 and L2 as
comparison methods.

Reason:

L2 bypasses LLM decision formation, while L1 only changes visual features
indirectly. Explicit tokens make the model input and the research claim
consistent.

Implementation status:

Not started.

---

## 2026-06-20: UQ Token Implementation and Backward Smoke Test

Git commit:

```text
working tree after 45a4c4b
```

Machine:

```text
AutoDL RTX 4090
root@connect.weste.seetacloud.com:39408
```

Implemented:

- `UQOutput.active_embedding` for the 16-D density direction;
- `UQTokenProjector`, configured as `17 -> 512 -> 4096`;
- one learned-null, score-gated continuous token;
- shared token append path for teacher forcing, `inference_ego`, and generation;
- dedicated route-filtered `train_uq_token.py`;
- hard freeze and gradient audits for projector + LLM LoRA only.

Verification:

```text
Unit tests: 15 passed
Visual sequence: [1, 513, 4096] -> [1, 514, 4096]
Projector parameters: 2,115,584
LoRA parameters: 16,777,216
Matched train-route frames: 8,316
```

Smoke command:

```bash
python scripts/train_uq_token.py \
  --config adzoo/orion/configs/orion_stage3_infer.py \
  --checkpoint ckpts/Orion.pth \
  --density-checkpoint checkpoints/density_uq/best.pt \
  --ann-file data/infos/b2d_infos_val.pkl \
  --split train \
  --epochs 1 \
  --max-samples 100 \
  --max-steps 1 \
  --workers 0 \
  --grad-accum 1 \
  --lambda-col 0 \
  --out /root/autodl-tmp/orion_assets/checkpoints/uq_token/smoke.pt
```

Result:

```text
Gradient audit: passed
Loss: 0.138143
Checkpoint size: 217 MiB, including optimizer state
Checkpoint tensors: 7 projector + 256 LoRA
Post-update token norm at score 1: approximately 0.137
Post-update learned-null norm at score 0: approximately 0.0064
```

Notes:

- The first 8-frame smoke subset had no valid future trajectory masks and was
  correctly skipped.
- Expanding to 100 frames reached a valid sample and completed one backward
  step.
- The learned null token is trainable and is not a strict zero token after an
  update. Low-UQ consistency remains to be implemented.
- A fresh full ORION model loaded all 263 adaptation tensors successfully.
- `test.py` and open-loop evaluation now accept `UQ_TOKEN_CHECKPOINT`.

Next action:

Add checkpoint loading for evaluation, implement low-UQ consistency, and run a
short multi-step overfit test before full training.

---

## 2026-06-20: Consistency Loss and Multi-Step Diagnostics

Implemented:

- no-gradient no-token LLM reference forward;
- `(1 - score)`-weighted waypoint-feature MSE;
- configurable planning, VAE, VLM, collision, and consistency weights;
- per-step structured metrics stored inside the adaptation checkpoint.

Ten-step stability run:

```text
Valid optimization steps: 10
Mean total loss with lambda_vlm=0.01: 0.102490
Final observed token norm: 2.671621
OOM/NaN: none
```

This run used different samples and is a stability test, not a strict
single-example overfit test.

Loss-balance diagnostic with `lambda_vlm=0.001`:

| Step | Plan | Weighted VLM | Raw consistency | Weighted consistency | Total | Token norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00282 | 0.01031 | 0.00374 | 0.00019 | 0.01332 | 0.000 |
| 2 | 0.01005 | 0.00976 | 0.00176 | 0.00009 | 0.01990 | 0.555 |
| 3 | 0.00331 | 0.01008 | 0.00106 | 0.00005 | 0.01344 | 0.491 |

Checkpoint:

```text
/root/autodl-tmp/orion_assets/checkpoints/uq_token/diagnostic3_balanced.pt
size: approximately 217 MiB
history records: 3
```

Interpretation:

- consistency remains a light regularizer at the current weight;
- the reduced VLM weight prevents language loss from overwhelming planning;
- projector and LoRA gradients remain finite;
- formal training should not start before calibration-route evaluation and
  early stopping are implemented.

Next action:

Implement calibration-route loss evaluation and epoch-level early stopping,
then run a limited training pilot before committing to the full 8,316-frame
training schedule.

---

## 2026-06-20: Full Training Paused for Grounding Validation

Issue:

The continuous token has no pretrained semantic meaning. LoRA provides a way
for the LLM to learn its functional meaning, but planning improvement would not
prove that uncertainty itself is encoded or used.

Decision:

- pause full planning training;
- add a grounding head that predicts the fixed density score from the waypoint
  hidden state;
- require correct UQ to outperform no-token, zero-token, and shuffled-token
  controls;
- add fixed-image counterfactual token interventions.

Current status:

```text
Token injection: implemented
Projector + LoRA training: implemented
Low-UQ consistency: implemented
Multi-step stability: verified
Grounding head: not implemented
Full training: not started
```

Next action:

Implement the grounding head and small-route grounding pilot before any full
planning run.

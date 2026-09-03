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

## 2026-06-21: Reliability Language and Planning Sensitivity

Completed:

- balanced R2d reliability alignment;
- 100-frame correct/shuffled calibration evaluation;
- none/zero controls;
- two minimal free risk-synthesis pilots;
- fixed-noise reliability-history planning sensitivity pilot.

R2d result:

```text
correct parse / accuracy: 0.97 / 0.90
shuffled parse / accuracy: 0.99 / 0.96
intervention response: 0.936
none and zero controls: 0/20 parseable
```

Planning pilot:

```text
valid frames: 10
correct-text ADE: 0.2362 m
shuffled-text ADE: 0.2367 m
correct-vs-shuffled hidden L2: 0.4301
correct-vs-shuffled trajectory displacement: 0.00283 m
```

Conclusion:

The LLM reads and verbalizes the continuous UQ token, but the frozen planning
path does not naturally convert reliability meaning into behavior. Free-form
risk synthesis also failed at this data scale. The next stage requires
planning supervision rather than additional prompt-only tuning.

Artifacts:

```text
reports/risk_qa/r2d_results.md
reports/risk_qa/r2d_summary.png
reports/risk_qa/risk_planning_pilot.md
```

---

## 2026-06-21: Paired Corruption Planning Pilot

Implemented deterministic blur, exposure reduction, and camera-dropout
corruptions, a Density UQ response audit, paired planning supervision, and
corrupted-view intervention evaluation.

The UQ audit selected one-camera dropout:

```text
mean UQ increase: +0.0895
increase rate: 10/10
```

Blur and exposure reduction were rejected because the current Density UQ
estimator did not respond consistently.

The 50-frame paired pilot failed:

```text
corrupted ADE: none 0.1910, shuffled 0.2147, correct 0.2447
clean ADE: none 0.0808, correct 0.1031
```

The next pilot adds correct-versus-shuffled ranking relative to the clean
trajectory reference. Scaling the original consistency-only configuration is
blocked.

Counterfactual ranking was subsequently active on 14/20 pilot frames, but it
also failed:

```text
corrupted ADE:
  none     0.1964
  shuffled 0.3054
  correct  0.3159
```

Decision:

The explicit-token planning route has now failed two controlled paired
architecture pilots. Stop scaling it. Preserve R2d as the semantic grounding
result and move behavioral adaptation to a pre-LLM uncertainty adapter.

---

## 2026-06-21: Pre-LLM Vision Adapter Pilot

Implemented an identity-initialized low-rank residual adapter on the 513
visual queries before the LLM. The final 100-step pilot freezes LoRA updates
and trains only the adapter.

```text
Route1115 first 50 corrupted:
  none 0.1991, shuffled 0.1583, correct 0.1520 ADE

Route504 first 50 corrupted:
  none 0.7284, shuffled 0.6144, correct 0.5547 ADE

Route504 clean:
  none 0.5039, correct 0.4650 ADE
```

The harder later section of Route1115 does not improve:

```text
first 100 frames:
  none 2.7355, shuffled 2.7740, correct 2.7726 ADE
```

Retain the adapter as the behavioral candidate, but require route-balanced
training and evaluation before making a general claim.

Artifact:

```text
reports/uq_token/vision_adapter_pilot.md
```

---

## 2026-06-23: Route-balanced Adapter Evaluation and Visualization

Git commit:

```text
pending
```

Machine:

```text
AutoDL RTX 4090
root@connect.weste.seetacloud.com:39408
```

Commands:

```text
python scripts/train_uq_token.py ... \
  --conditioning vision_adapter \
  --eval-only --eval-planning \
  --eval-route-balanced --eval-route-samples 50 --eval-route-limit 10 \
  --eval-corruption --corruption camera_dropout --corruption-severity 1

python scripts/eval_uq_adapter_stratified.py ... \
  --eval-route-samples 50 --eval-route-limit 10 \
  --eval-corruption --corruption camera_dropout --corruption-severity 1 \
  --bins median

python scripts/render_uq_adapter_gifs.py ... \
  --routes VehicleTurningRoute_Town15_Route504_Weather10 \
           BlockedIntersection_Town03_Route135_Weather5 \
           YieldToEmergencyVehicle_Town04_Route166_Weather10
```

Configuration:

```text
checkpoint: /root/autodl-tmp/orion_assets/checkpoints/uq_vision_adapter/pilot100.pt
conditioning: pre-LLM vision adapter
split: calibration
candidate routes: 10
candidate frames per route: 50
valid planning frames: 350
corruption: one-camera dropout, severity 1
```

Route-balanced result:

```text
none     ADE 1.4429, FDE 2.5210
zero     ADE 1.4429, FDE 2.5210
shuffled ADE 1.2286, FDE 2.1969
correct  ADE 1.1641, FDE 2.0955

correct vs none ADE:     +19.3%
correct vs shuffled ADE: +5.3%
```

Per-route notes:

```text
correct improves over none on 9/10 routes
correct improves over shuffled on 8/10 routes
largest gains: VanillaSignalizedTurnEncounterRedLight, OppositeVehicleRunningRedLight,
HighwayExit, VehicleTurningRoute, BlockedIntersection
weak/failure routes: YieldToEmergencyVehicle, SignalizedJunctionLeftTurnEnterFlow
```

High/low UQ stratification:

```text
high-UQ: count 192
  none 1.673, shuffled 1.526, correct 1.502 ADE
  correct vs none +10.2%, correct vs shuffled +1.6%

low-UQ: count 158
  none 1.262, shuffled 1.089, correct 1.052 ADE
  correct vs none +16.7%, correct vs shuffled +3.4%
```

Conclusion:

The adapter has a route-balanced average gain, and correct UQ is better than
shuffled on average. However, the high-UQ half does not show larger relative
gain than the low-UQ half. The current result supports "UQ-conditioned visual
evidence can improve average planning under corruption", but it does not yet
support "higher scalar UQ implies larger planning benefit".

Clean safety check:

```text
clean none     ADE 0.8884, FDE 1.3707
clean shuffled ADE 0.9008, FDE 1.4265
clean correct  ADE 0.8971, FDE 1.4248

correct vs none:
  ADE degradation: 1.0%
  FDE degradation: 3.9%
```

Clean ADE passes the 3% degradation gate. Clean FDE is slightly above the
original 3% gate, so the next training run needs a stronger clean preservation
term or identity regularization.

Artifacts:

```text
reports/uq_token/assets/route_balanced_eval/pilot100_corrupted_route10x50.json
reports/uq_token/assets/route_balanced_eval/pilot100_corrupted_route10x50_median.json
reports/uq_token/assets/route_balanced_eval/pilot100_clean_route10x50_median.json
reports/uq_token/assets/route_balanced_eval/route_balanced_ade_by_route.png
reports/uq_token/assets/route_balanced_eval/route_balanced_correct_improvement.png
reports/uq_token/assets/route_balanced_eval/uq_stratified_ade.png
reports/uq_token/assets/uq_adapter_gifs/
scripts/eval_uq_adapter_stratified.py
scripts/render_uq_adapter_gifs.py
scripts/summarize_uq_adapter_results.py
```

Next action:

Run clean route-balanced safety evaluation, then train a larger route-balanced
adapter with score+active-embedding conditioning or stronger score calibration.

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

---

## 2026-06-20: Grounding Implementation and Pilot Sequence

Git commit:

```text
working tree after 43f88ed
```

Machine:

```text
AutoDL RTX 4090
root@connect.weste.seetacloud.com:39408
```

Implemented:

- score grounding head on the 4096-D waypoint representation;
- detached SmoothL1 density-score target;
- deterministic correct, zero, shuffled, and no-token controls;
- calibration-route MAE, Pearson, and Spearman;
- explicit `score_basis` in the UQ token projector;
- same-image correct/shuffled counterfactual grounding;
- score-only and score+direction input modes.

Initial independent 30-step runs, 50 calibration frames:

| Mode | MAE | Pearson | Spearman |
| --- | ---: | ---: | ---: |
| none | 0.1287 | 0.3773 | 0.3649 |
| zero | 0.1220 | 0.3424 | 0.3766 |
| shuffled | 0.1305 | 0.4103 | 0.4163 |
| correct | 0.1227 | 0.4203 | 0.3663 |

Interpretation:

Independent training did not isolate token use. The LLM and grounding head
could infer uncertainty from the image.

Score+direction counterfactual run, 100 steps:

```text
correct:  MAE 0.2698, Pearson 0.8699, Spearman 0.7556
shuffled: MAE 0.3004, Pearson 0.1697, Spearman 0.2414
none:     MAE 0.0588, Pearson 0.6875, Spearman 0.7212
```

This established intervention sensitivity but showed poor absolute calibration.

Score-only counterfactual command:

```bash
python scripts/train_uq_token.py \
  --config adzoo/orion/configs/orion_stage3_infer.py \
  --checkpoint ckpts/Orion.pth \
  --density-checkpoint checkpoints/density_uq/best.pt \
  --descriptor-cache data/density_uq/descriptors.pt \
  --ann-file data/infos/b2d_infos_val.pkl \
  --split train \
  --grounding-only \
  --counterfactual-grounding \
  --token-input score_only \
  --uq-mode correct \
  --epochs 1 \
  --max-samples 750 \
  --max-steps 100 \
  --eval-max-samples 50 \
  --workers 2 \
  --grad-accum 1 \
  --lambda-vlm 0 \
  --lambda-ground 1 \
  --lambda-consistency 0 \
  --seed 42 \
  --intervention-eval-after-train \
  --out /root/autodl-tmp/orion_assets/checkpoints/uq_token/grounding_counterfactual_score_only100.pt
```

Result:

| Intervention | MAE | Pearson | Spearman |
| --- | ---: | ---: | ---: |
| none | 0.1158 | 0.4135 | 0.4479 |
| zero | 0.1318 | 0.4048 | 0.4734 |
| shuffled | 0.2742 | -0.0055 | 0.1052 |
| correct | 0.1501 | 0.7531 | 0.6414 |

Artifact:

```text
/root/autodl-tmp/orion_assets/checkpoints/uq_token/grounding_counterfactual_score_only100.pt
size: 217 MiB
```

Conclusion:

The correct-to-shuffled correlation collapse is evidence that the LLM
representation uses the injected score. Absolute calibration is not yet good
enough: correct-token MAE remains worse than the no-token visual shortcut.

Next action:

Run a 300-step score-only counterfactual pilot and evaluate 200 calibration
frames. Do not start full planning adaptation unless both causal and MAE gates
pass.

---

## 2026-06-20: Score-Only Grounding Gate Passed

Machine:

```text
AutoDL RTX 4090
root@connect.weste.seetacloud.com:39408
```

Configuration:

```text
Training candidates: 1,500
Effective optimization steps: 300
Calibration frames: 200
Token input: score only
Objective: same-image correct/shuffled counterfactual grounding
lambda_vlm: 0
lambda_ground: 1
lambda_consistency: 0
```

Result:

| Intervention | MAE | Pearson | Spearman |
| --- | ---: | ---: | ---: |
| none | 0.1087 | -0.6553 | -0.5898 |
| zero | 0.1129 | -0.6810 | -0.6208 |
| shuffled | 0.2940 | -0.0517 | -0.0372 |
| correct | **0.0406** | **0.8611** | **0.7253** |

Training summary:

```text
Mean grounding loss: 0.013720
Final reported token norm: 13.830158
OOM/NaN: none
```

Artifact:

```text
/root/autodl-tmp/orion_assets/checkpoints/uq_token/grounding_counterfactual_score_only300.pt
size: approximately 217 MiB
```

Interpretation:

- correct UQ is substantially better than no-token and zero-token controls;
- shuffled UQ destroys correlation and triples-to-septuples MAE;
- the waypoint representation therefore retains and uses the supplied
  sample-level score;
- the 100-step calibration failure was primarily insufficient optimization,
  not evidence against the score-token method.

Next action:

Run a limited planning diagnostic, then verify whether the grounding-only
weights preserve base-ORION trajectory behavior.

---

## 2026-06-20: Grounding-Only Initializer Rejected

Stage 2D 50-step diagnostic:

```text
Initialization: grounding_counterfactual_score_only300.pt
Loss: plan + 0.1 VAE + 0.001 VLM + 0.1 grounding + 0.05 consistency
Effective steps: 50
Collision loss: disabled
```

Grounding intervention on 50 calibration frames:

| Mode | MAE | Pearson | Spearman |
| --- | ---: | ---: | ---: |
| none | 0.0806 | 0.4716 | 0.6342 |
| zero | 0.0983 | 0.4820 | 0.6018 |
| shuffled | 0.1483 | 0.1417 | 0.2320 |
| correct | **0.0449** | **0.6631** | **0.6284** |

Fixed-noise trajectory metrics on 35 valid calibration frames:

| Checkpoint / mode | ADE | FDE |
| --- | ---: | ---: |
| Base ORION / none | **0.0803** | **0.0745** |
| Stage 2D pilot / none | 0.0854 | 0.0923 |
| Stage 2D pilot / shuffled | 0.0904 | 0.0890 |
| Stage 2D pilot / correct | 0.0894 | 0.0856 |

The Stage 2D pilot retained score grounding but remained worse than base ORION.

Root-cause evaluation of the grounding-only checkpoint:

| Mode | ADE | FDE | Grounding MAE | Grounding Pearson |
| --- | ---: | ---: | ---: | ---: |
| none | 0.0815 | 0.0804 | 0.2505 | 0.5187 |
| correct | 1.8923 | 2.2267 | 0.0303 | 0.9522 |

Conclusion:

The grounding-only model learned a highly decodable score representation that
was incompatible with the unchanged trajectory decoder. Planning training
could mostly recover trajectory quality in 50 steps, but the grounded
checkpoint is not a defensible or stable initializer.

Next action:

Start from base ORION and jointly optimize correct-token planning,
correct-token grounding, same-image shuffled-token grounding, and low-UQ
consistency. Use lower LoRA/projector learning rates and gradient accumulation.

---

## 2026-06-20: Joint Grounding and Planning Pilot, 100 Samples

Configuration:

```text
Initialization: base ORION
Effective samples: 100
Optimizer updates: 25
Gradient accumulation: 4
Projector LR: 5e-5
LoRA LR: 5e-6
lambda_plan: 1
lambda_vlm: 0.001
lambda_ground: 1
lambda_consistency: 0.1
Counterfactual grounding: enabled
```

Grounding on 50 calibration frames:

| Mode | MAE | Pearson | Spearman |
| --- | ---: | ---: | ---: |
| none | 0.0404 | 0.7403 | 0.7640 |
| zero | 0.0432 | 0.7320 | 0.7726 |
| shuffled | 0.1750 | 0.3020 | 0.4403 |
| correct | 0.1041 | **0.8682** | **0.8096** |

Fixed-noise trajectory metrics on 35 valid frames:

| Mode | ADE | FDE |
| --- | ---: | ---: |
| Base ORION / none | **0.0803** | **0.0745** |
| Joint checkpoint / none | 0.0803 | 0.0804 |
| Joint checkpoint / zero | 0.0923 | 0.0930 |
| Joint checkpoint / shuffled | 0.1375 | 0.1708 |
| Joint checkpoint / correct | 0.0941 | 0.1070 |

Interpretation:

- the correct token carries recoverable score information;
- shuffled UQ strongly changes and worsens the trajectory, proving the token
  affects planning;
- correct UQ is still worse than base ORION, so the planning gate fails;
- grounding frequently dominates the weighted per-step objective;
- 25 optimizer updates are insufficient to learn a beneficial behavioral use.

Next action:

Resume to 400 effective samples with `lambda_ground=0.2`. Keep the same
counterfactual objective and lower learning rates. If correct-UQ ADE/FDE remain
worse than baseline, stop scaling and redesign behavioral supervision.

---

## 2026-06-20: Joint 400 Pilot Failed Planning Gate

Continuation:

```text
Resume: joint_pilot100.pt
Effective samples: 400 total
Gradient accumulation: 4
lambda_ground: 0.2
Other learning rates and losses unchanged
```

Grounding:

| Mode | MAE | Pearson | Spearman |
| --- | ---: | ---: | ---: |
| none | 0.0461 | 0.6942 | 0.7135 |
| zero | 0.0545 | 0.6704 | 0.7018 |
| shuffled | 0.2018 | 0.2128 | 0.3114 |
| correct | 0.1157 | **0.8833** | **0.8228** |

Fixed-noise trajectory metrics on 35 valid frames:

| Mode | ADE | FDE |
| --- | ---: | ---: |
| Base ORION / none | **0.0803** | **0.0745** |
| Joint 400 / none | 0.0818 | 0.0839 |
| Joint 400 / zero | 0.0973 | 0.1002 |
| Joint 400 / shuffled | 0.1874 | 0.2665 |
| Joint 400 / correct | 0.1046 | 0.1162 |

Artifacts:

```text
/root/autodl-tmp/orion_assets/checkpoints/uq_token/joint_pilot100.pt
/root/autodl-tmp/orion_assets/checkpoints/uq_token/joint_pilot400.pt
```

Conclusion:

- the UQ token is read and causally affects both grounding and trajectory;
- the current waypoint-state grounding objective causes harmful planning use;
- more samples and a lower grounding weight did not reverse the degradation;
- full training is stopped.

Next action:

Add a dedicated LLM `<uq_state>` readout token. Ground score from that token,
not from the waypoint hidden state. Preserve the current counterfactual controls
and evaluate whether the waypoint trajectory uses UQ beneficially without
being forced to reconstruct it.

---

## 2026-06-20: Explicit Reliability QA Pre-Experiment

Implemented:

- structured and natural Risk QA target generation;
- Bench2Drive category normalization and nearest critical-object selection;
- correct/zero/shuffled/no-token generation interventions;
- counterfactual teacher-forced Risk QA alignment;
- level-only reliability QA for multi-round decomposition.

R0 target audit, 50 calibration frames:

```text
Parse success: 100%
Critical-object coverage: 100%
Mean target length: 162.8 tokens
Maximum target length: 167 tokens
```

R1 with the previous planning adaptation:

```text
All modes generated the original ORION scene/planning response.
Structured Risk QA parse rate: 0%.
```

R2 structured and R2b combined natural-language targets both failed free
generation after 300 effective frames. Training/generation token prefixes were
verified identical. The failure was attributed to simultaneously retraining
reliability semantics and detailed scene description.

R2c separated the existing critical-object QA from a new reliability-only
round:

```text
Target: Visual reliability is LEVEL.
Levels: very low, low, moderate, high, very high
Training frames: 200
Optimizer updates: 50
```

Fifty-frame result:

```text
Correct parse rate: 1.00
Correct level accuracy: 0.88
Correct ordinal MAE: 0.12

Shuffled parse rate: 0.86
Shuffled level accuracy: 0.68
Shuffled ordinal MAE: 0.21
Shuffled level Spearman: 0.953

Different-target intervention frames: 33
Generated level changed: 31
Intervention response: 93.9%
```

Artifacts:

```text
/root/autodl-tmp/orion_assets/checkpoints/risk_qa/r2c_level200.pt
/root/autodl-tmp/orion_assets/reports/risk_qa/r2c_eval50.json
```

Conclusion:

The language model can explicitly verbalize the continuous UQ token as a
calibrated reliability level. The remaining error is concentrated in the
very-high-reliability tail and should be addressed by balanced level sampling,
not by changing the token interface.

---

## 2026-06-21: Balanced Reliability QA Passed

Machine:

```text
AutoDL RTX 4090
root@connect.weste.seetacloud.com:39408
```

Training:

```text
Initialization: r2c_level200.pt
Training frames: 300
Frames per level: 60
Effective steps: 300
Gradient accumulation: 4
Projector LR: 3e-5
LoRA LR: 3e-6
```

Balanced 100-frame route-disjoint calibration evaluation:

| Intervention | Parse rate | Accuracy | Ordinal MAE | Spearman |
| --- | ---: | ---: | ---: | ---: |
| correct | 0.97 | 0.90 | 0.072 | 0.981 |
| shuffled | 0.99 | 0.96 | 0.030 | 0.985 |

Counterfactual intervention:

```text
Different target levels: 78
Generated output changed: 73
Response rate: 93.6%
```

Controls:

```text
No UQ token parse rate: 0/20
Zero UQ token parse rate: 0/20
```

Artifacts:

```text
/root/autodl-tmp/orion_assets/checkpoints/risk_qa/r2d_balanced300.pt
/root/autodl-tmp/orion_assets/reports/risk_qa/r2d_eval100.json
/root/autodl-tmp/orion_assets/reports/risk_qa/r2d_controls20.json
reports/risk_qa/r2d_summary.png
reports/risk_qa/r2d_results.md
```

Conclusion:

The explicit reliability-language stage passes. For the midterm report, the
claim should remain limited to uncertainty-aware scene reliability
understanding. Multi-round risk synthesis and planning effects remain future
work.

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

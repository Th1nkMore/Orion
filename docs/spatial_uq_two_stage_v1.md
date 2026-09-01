# Spatial uncertainty and behavior adaptation: v1 research contract

> Superseded on 2026-08-28 by `docs/spatial_uq_two_stage_v2.md`. This file is
> retained as design history; its fixed path-risk aggregator is no longer the
> paper mainline.

> Status: active design contract, 2026-08-26
> Scope: spatial uncertainty extraction, oracle-conditioned behavior adaptation,
> and closed-loop validation. Diffusion decoding is intentionally out of scope.

## 1. Claim boundary

The project separates three quantities that must not be used as synonyms:

1. **Perception uncertainty** estimates where the current visual/perception
   prediction is likely to be wrong.
2. **Path risk** combines perception failure probability with occupancy, the
   candidate route corridor, and time to collision.
3. **Behavior response** changes speed or trajectory only when path risk
   warrants intervention.

The existing Density-UQ score is retained as a **global anomaly/OOD baseline**.
It is not a target for the new spatial head and is not called semantically
correct uncertainty.

The current route-146 controlled-stop result establishes an exploratory oracle
mechanism only. It does not validate learned uncertainty or LLM understanding.

## 2. Literature basis

The first-stage design follows these primary sources:

- Kendall and Gal separate input-dependent aleatoric uncertainty from model
  uncertainty and train heteroscedastic predictions with proper likelihoods:
  [NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html).
- Deep ensembles provide the offline epistemic teacher and variance
  decomposition baseline:
  [NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html).
- MC dropout is retained as a cheaper epistemic baseline, not assumed to be a
  faithful posterior for the frozen VLM:
  [ICML 2016](https://proceedings.mlr.press/v48/gal16.html).
- Calibration uses a held-out calibration split and proper scoring rules, not
  score spread as a surrogate:
  [ICML 2017](https://proceedings.mlr.press/v70/guo17a.html).
- Spatial anomaly evaluation follows the AUPRC/FPR discipline of Fishyscapes,
  while explicitly keeping anomaly detection separate from path risk:
  [ICCVW 2019](https://openaccess.thecvf.com/content_ICCVW_2019/html/ADW/Blum_Fishyscapes_A_Benchmark_for_Safe_Semantic_Segmentation_in_Autonomous_Driving_ICCVW_2019_paper.html).
- Object-level class and localization uncertainty are evaluated separately as
  recommended by probabilistic object detection:
  [WACV 2020](https://arxiv.org/abs/1811.10800).
- Clean/adverse correspondence and explicit uncertain regions follow the data
  principle used by ACDC:
  [ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.html).
- Corruption families and severities are informed by the 3D common-corruption
  benchmark; camera dropout remains a diagnostic rather than the main signal:
  [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Dong_Benchmarking_Robustness_of_3D_Object_Detection_to_Common_Corruptions_CVPR_2023_paper.html).
- Selective intervention is assessed with risk-coverage behavior and an
  explicit cost for unnecessary intervention:
  [ICML 2019](https://proceedings.mlr.press/v97/geifman19a.html).

## 3. Frozen public interface

The v1 interface uses dynamic spatial dimensions; no model code may assume a
fixed 40 x 40 patch grid.

```text
observed_patch_tokens       [B, V, P, D]
patch_error_mean            [B, V, P]
patch_aleatoric_variance    [B, V, P]
patch_epistemic_variance    [B, V, P]
bev_failure_probability     [B, H_bev, W_bev]
bev_valid_mask              [B, H_bev, W_bev]
path_risk                   [B, M]
```

The spatial UQ head produces all fields except `path_risk`. A separate,
auditable path-risk aggregator produces `path_risk`; route information is not
an input to the spatial UQ head.

At inference time the behavior adapter may consume:

```text
planning_feature            [B, D_plan]
path_risk                   [B, M]
selected_path_risk          [B, 1]
```

Its first implementation predicts bounded route/speed modulation and a stop
probability. It is identity-initialized so clean behavior is unchanged before
training.

## 4. First-stage targets

Every training record is tied to the same CARLA/Bench2Drive state under a clean
observation `x` and a corrupted observation `x_tilde`.

### 4.1 Primary: actual perception error

Where available, targets come from frozen ORION predictions compared with
privileged ground truth:

- object miss/classification/localization error;
- BEV cell occupancy or object-presence error;
- map/lane error;
- later, CARLA semantic/depth error when those sensors are recorded.

The main BEV target is a calibrated probability that perception is wrong in a
cell, not a label that a corruption was applied.

### 4.2 Primary fallback: corruption-attributable representation error

For the first executable pilot, a spatial target can be obtained without new
annotations:

```text
e_attr = max(0, error(x_tilde, y) - error(x, y))
```

When task-space error is unavailable, patch cosine discrepancy between frozen
clean and corrupted visual features is used as an explicitly named
**representation-error proxy**. It is not presented as full semantic UQ.

### 4.3 Auxiliary only: corruption mask

The known corruption mask receives a low-weight localization loss. It cannot be
the primary target, because predicting where an augmentation was drawn does not
show that perception failed there.

### 4.4 Epistemic teacher

Three independently initialized lightweight spatial heads form the initial
head-level ensemble. Their predictive disagreement is distilled into the
single-pass student. Because the visual backbone is shared and frozen, the
result is reported as **head-level epistemic uncertainty**, not full-model
Bayesian uncertainty.

## 5. Loss contract

The complete v1 design target is:

```text
L_uq = L_error_NLL
     + lambda_fail * L_failure_Brier
     + lambda_epi  * L_ensemble_distill
     + lambda_rank * L_error_ranking
     + lambda_cf   * L_counterfactual_equivariance
     + lambda_time * L_temporal_response
     + lambda_mask * L_corruption_mask_aux
     + lambda_clean * L_clean_preservation
```

The executable trainer currently implements error NLL, failure Brier,
ensemble distillation, measured-error ranking, corruption-mask auxiliary, and
clean false-positive terms.  Counterfactual-equivariance and temporal-response
losses remain gated on records carrying explicit region transforms and event
boundaries; configuration files list them as planned rather than silently
claiming they are active.

Requirements:

- heteroscedastic NLL includes the log-variance penalty;
- log variance is bounded for numerical stability;
- ranking is applied only when measured perception/representation error also
  increases with severity;
- temporal smoothing is disabled across known event boundaries so it does not
  reward delayed onset or recovery;
- clean preservation covers both false UQ activation and unchanged ORION output.

The old `1 / std(prediction)` spread regularizer is not considered calibration
and is not used by the spatial-UQ mainline.

## 6. Path-risk aggregation

UQ remains route-independent. For candidate route `k`, the fixed aggregator
uses:

```text
cell_risk = P_fail * P_occupied * route_corridor_k * TTC_weight
R_k = top-q mean or CVaR(cell_risk)
```

Global mean pooling is prohibited as the main aggregator because a small
pedestrian region would be diluted by the road background. The implementation
must expose all component maps for visualization and ablation.

For a spatially translated local corruption, local UQ should translate with the
corruption. `R_k` should change only when that high-UQ region overlaps the
candidate route or hazard. This on-path/off-path test is a required causal
diagnostic.

## 7. Parallel workstreams

### A. Spatial UQ extraction

1. Add deterministic local blur, glare, occlusion, darkness and diagnostic
   dropout with explicit masks and severity.
2. Generate clean/corrupt pairs online around the frozen image backbone.
3. Train three small spatial heads and distil a single-pass student.
4. Add BEV failure targets from paired frozen-ORION errors.
5. Calibrate on route/Town-disjoint data.

### B. Oracle-conditioned behavior adapter

1. Reuse the fixed `path_risk` interface with oracle maps.
2. Build behavior targets from oracle intervention traces and privileged safe
   trajectories/controls.
3. Train an identity-initialized adapter with hazard and no-hazard examples.
4. Penalize collision proxies, lack of progress, unnecessary intervention,
   discomfort, and delayed recovery.

This workstream does not wait for learned UQ.

### C. Closed-loop and evidence tooling

1. Keep route 146 as a development route, not the only paper route.
2. Add compatible held-out hazard/no-hazard and on-path/off-path cases.
3. Automatically render front input, raw front, BEV, UQ, route-corridor overlap,
   speed, TTC, braking and event timing.
4. Compare equal or matched intervention budgets, not only an arbitrary
   constant score.

## 8. Stage gates

### Gate 1: spatial-UQ offline validity

Required evidence:

- error-mask AUPRC as the primary spatial metric, with AUROC/FPR95 secondary;
- NLL and Brier score plus reliability diagram;
- AURC/risk-coverage;
- severity monotonicity;
- onset and recovery latency;
- clean false-positive duration;
- distance/object/weather/Town stratification;
- local translation equivariance;
- seen-corruption training and unseen-corruption evaluation;
- comparison with brightness, black-pixel ratio, blur metric and Density UQ.

Corruption-mask IoU is diagnostic only and cannot pass this gate alone.

### Gate 2: oracle-adapter validity

On held-out scenarios, oracle-conditioned adapter must approach the verified
oracle controller while retaining route completion and avoiding unnecessary
stops in no-hazard/off-path conditions.

### Gate 3: learned-UQ causal matrix

Only after Gates 1 and 2 pass, compare:

- response off;
- simple corruption detector;
- matched-intervention constant;
- temporally shuffled UQ;
- spatially shuffled UQ;
- aligned learned UQ;
- oracle UQ.

The learned system supports the central claim only if it beats the simple
detector, matched constant, and both shuffled controls and begins to approach
oracle performance.

## 9. Resource contract

- Shared data, checkpoints and raw traces live under
  `/public/share/lidachuan/orion_assets`.
- Closed-loop ORION jobs request one A800 and about 192 GB host memory.
- CPU preprocessing and visualization use CPU partitions where possible.
- Before Gate 1, no large multi-route/multi-seed learned-UQ matrix is submitted.
- One training job and one closed-loop job may run concurrently on different
  nodes; data-generation jobs must not overwrite paired manifests.

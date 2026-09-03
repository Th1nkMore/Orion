# Stage-1 actual perception-failure target design

> Date: 2026-08-26  
> Status: standalone v2 record/training/evaluation contract implemented; no cluster job submitted  
> Scope: a real Stage-1 target for the existing spatial UQ head. Diffusion and
> behavior-adapter training are out of scope here.

## 1. Decision

The minimum defensible Stage-1 target is:

> **Frozen-ORION object/occupancy error against privileged Bench2Drive ground
> truth, attributed to visible camera patches.**

This is preferable to supervising the head with a corruption mask or a
clean/corrupt feature residual. The magnitude of the label is an actual error
of ORION's frozen perception outputs. The mapping from an object/BEV error back
to image patches is still an **attribution proxy** and must be described as
such; it does not prove that a particular pixel caused the error.

For the first implementation, use safety-critical dynamic objects and affected
traffic lights. Do not include map/lane error until its query-to-patch
attribution is validated. Keep paired EVAViT cosine residual as an explicitly
named representation-error fallback, not as semantic uncertainty.

The intended separation is:

```text
observed RGB -> frozen ORION perception -> actual task error versus GT
                                              |
                                              v
                                     view/patch attribution
                                              |
                                              v
                                      Stage-1 UQ target

route corridor + predicted UQ -> path risk        (separate module)
path risk -> conservative behavior                 (Stage 2)
```

## 2. What the repository and installed data actually provide

### 2.1 Current ORION data path

`mmcv/datasets/b2d_orion_dataset.py` exposes the following usable supervision:

- current 3D boxes, class names, actor IDs and velocities;
- six calibrated cameras through `lidar2img`, `cam_intrinsic`, and
  `lidar2cam`;
- ego future displacements and valid masks for six future steps;
- per-agent future displacements, future-valid masks, future yaw, dimensions,
  class and current state in `gt_attr_labels`;
- route command, ego state and map polylines/traffic-control geometry.

The configured `sample_interval=5` and dataset rate of 10 Hz make the six
future labels 0.5 s apart, covering 3 s. `PlanningMetric` in
`mmcv/models/dense_heads/planning_head_plugin/metric_stp3.py` already converts
the boxes and future-agent labels into vehicle/pedestrian BEV occupancy with a
200 x 200 grid at 0.5 m resolution. That rasterizer can be reused as the GT
side of the target, after its coordinate conventions are unit-tested.

The Stage-1/2/3 training pipelines currently consume RGB and object/map labels;
they do **not** load dense semantic, instance or depth targets. Therefore a new
offline target exporter is required, but new manual annotation is not.

### 2.2 Current frozen-model outputs

`mmcv/models/dense_heads/orion_head.py` already produces the quantities needed
to score perception failure:

```text
all_cls_scores          [decoder, B, query, class]
all_bbox_preds          [decoder, B, query, box_code]
all_traj_preds          [decoder, B, query, mode, time, 2]
all_traj_cls_scores     [decoder, B, query, mode]
all_traffic_states      [decoder, B, query, state/affect logits]
```

The head also exposes the Hungarian assigner used during training and decoded
box/motion results through `get_motion_bboxes`. The map head provides lane
classification and polyline predictions, but there is no native dense semantic
segmentation or occupancy head. Any predicted occupancy is consequently a
deterministic rasterization of object and motion predictions, not a native
ORION output; this provenance must be recorded.

### 2.3 Installed Bench2Drive assets, read-only audit

The current server asset root contains:

```text
/public/share/lidachuan/orion_assets/data/infos/b2d_infos_val.pkl
/public/share/lidachuan/orion_assets/data/infos/b2d_map_infos.pkl
/public/share/lidachuan/orion_assets/data/bench2drive/v1/
```

The installed validation info contains 12,806 frames from 50 route folders in
11 towns. A first record has these keys:

```text
brake, command_far, command_far_xy, command_near, command_near_xy,
ego_accel, ego_rotation_rate, ego_size, ego_translation, ego_vel, ego_yaw,
folder, frame_idx, gt_boxes, gt_ids, gt_names, npc2world, num_points,
sensors, steer, throttle, town_name, world2ego
```

The raw 50-route subset contains aligned 1600 x 900 files for every camera:

- six lossy JPEG RGB views;
- six 8-bit semantic-label PNG views;
- six instance-label RGBA PNG views;
- six 8-bit depth PNG views;
- top-down RGB, raw annotations, lidar, radar and expert assessment.

A deterministic sample of 201 annotation frames checked all 4,824 expected
RGB/semantic/instance/depth view files with no missing file. Bench2Drive's own
`docs/anno.md` states that semantic/depth/instance sensors are privileged
training information and share the RGB camera poses. Its collection code saves
only the red channel for semantic labels and saves converted depth as PNG. In
the installed files the depth is 8-bit, so it is suitable for coarse visibility
checks in the safety range but should not be treated as high-precision metric
depth without an explicit decoding/quantization audit.

Important availability limitation: no `b2d_infos_train.pkl` was found under
the current personal or shared asset trees, and the installed raw subset has
50 route directories. The 50-route validation subset is sufficient for an
exploratory, route-disjoint target pilot. It is not acceptable to train on
these routes and then claim an untouched official validation result. Formal
training requires acquiring the train info/routes or reserving a permanently
untouched benchmark split.

### 2.4 Image/patch alignment detail

ORION does not feed the raw 1600 x 900 frame directly to EVAViT. The pipeline
applies `ResizeCropFlipRotImage`, updates camera intrinsics and `lidar2img`,
resizes again, normalizes, and produces a 40 x 40 feature map per view. All
semantic/depth masks, corruption masks and projected boxes must undergo the
same recorded image transform before pooling to `[V, 1600]`. Projecting with
raw intrinsics and then indexing processed patches would create a systematic,
plausible-looking label misalignment.

## 3. What counts as an actual failure target

A target is an **actual task error** only if it compares a frozen model output
with privileged task ground truth. A clean/corrupt difference can make that
error corruption-attributable, but the clean prediction itself is not ground
truth.

The recommended records carry both quantities:

```text
E_obs(v,p)   = ORION task error on the observation shown to the UQ head
E_clean(v,p) = ORION task error on the paired clean observation
DeltaE(v,p)  = relu(E_obs(v,p) - E_clean(v,p))
```

- `E_obs` is the primary target. It can be nonzero on clean images because
  ORION can be wrong in clean conditions.
- `E_clean` is the clean paired target, not a forced all-zero map.
- `DeltaE` is used for corruption-attribution analysis and severity ranking.
  It must not replace `E_obs` everywhere, because that would label clean model
  failures as certain.

The known corruption mask remains a low-weight localization auxiliary only.

## 4. Candidate comparison

| Candidate | Is the error real? | Spatial quality | Cost | Main failure mode | Decision |
|---|---:|---:|---:|---|---|
| ORION semantic detection/state error against B2D boxes | Yes, for ORION object perception | Object support can be projected to views | Medium/high: frozen ORION inference | Sparse positives; forced Hungarian matches and score thresholds can distort misses | Retain as a component and interpretation channel |
| BEV occupancy disagreement rasterized from ORION boxes/motion versus GT | Yes, after declaring that predicted occupancy is derived | Strong in BEV and directly safety-relevant | Medium/high | View-patch back-attribution is approximate; thresholds/rasterizer matter | **Primary v1 task-error component** |
| Clean-teacher/corrupt-student feature residual | No; it is a representation proxy | Dense and exactly aligned | Low once features exist | Penalizes harmless invariances and rewards recovering appearance rather than recognizing risk | Fallback/baseline only |
| Planner counterfactual sensitivity | Sensitivity is real, but not correctness unless measured as GT-loss increase | Region-level if interventions are local | Very high if one intervention per patch/region | Entangles Stage 1 with planner/route, becomes circular with Stage 2, and may reward ADE imitation | Audit/importance signal only, not the Stage-1 target |

Two additional distinctions matter:

1. Comparing an external semantic teacher with CARLA semantic masks is a real
   error of that teacher, but not necessarily an error of ORION. It is useful as
   an auxiliary benchmark, not the primary ORION claim.
2. A projected object mask tells where the failed object was visible. It does
   not prove those exact pixels caused the failure. The error magnitude is
   actual; its patch attribution is a documented proxy.

This follows the literature principle that class and localization uncertainty
should be evaluated separately in probabilistic object detection, while
aleatoric and epistemic uncertainty should not be conflated. Proper held-out
calibration and error-localization metrics are required rather than treating
raw score spread as calibration. Relevant primary sources are Kendall and Gal
([NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html)),
Deep Ensembles
([NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)),
probabilistic object detection
([WACV 2020](https://arxiv.org/abs/1811.10800)), calibration
([ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)), and spatial
failure/OOD evaluation in Fishyscapes
([ICCVW 2019](https://openaccess.thecvf.com/content_ICCVW_2019/html/ADW/Blum_Fishyscapes_A_Benchmark_for_Safe_Semantic_Segmentation_in_Autonomous_Driving_ICCVW_2019_paper.html)).

## 5. Preferred minimum target: `orion_projected_object_failure/v1`

### 5.1 Freeze the target model

Record, for every exported target set:

- base ORION checkpoint SHA-256;
- exact inference config and git revision;
- class mapping, decoder layer and decoder/NMS thresholds;
- BEV bounds/resolution and future horizon;
- camera order and all image transforms;
- target schema version.

Run the model in `eval()` with gradients disabled. Generate perception outputs
without invoking the LLM or trajectory decoder when possible: the target needs
EVAViT, positional encoding, the object/motion head, and optionally the map
head, but not language generation. This should reduce compute, but it must be
smoked against the normal ORION path before relying on the saving; the only
currently proven full-model resource envelope is about 192 GB host memory plus
one A800.

Because `OrionHead` uses temporal memory, process each route in order and reset
memory only at route boundaries. Clean and corrupt passes must start from the
same route state and deterministic seed. Two acceptable implementations are:

- replay the same recorded route twice, once clean and once with the
  preregistered corruption schedule; or
- batch clean/corrupt branches together if a two-branch inference smoke fits
  GPU memory.

Never compare a temporally warm clean branch against a reset corrupt branch.
Record whether internal histories are clean, corrupt, or shared at every event
frame.

### 5.2 Build interpretable per-object task errors

Use the final perception decoder output and retain individual components rather
than hiding them immediately in one number.

For each GT object `j`, decode predictions and make a class-aware match using
thresholds frozen on calibration routes. The outer 4 m center-distance level
already present in the repository's detection evaluation is a reasonable
miss boundary; report sensitivity at 2 m as well. A raw Hungarian assignment
alone is insufficient because it forcibly assigns a query even when the
decoded detection should count as a miss.

Define bounded components:

```text
e_miss_j = 1 if no valid class-aware match, else 0
e_cls_j  = 1 - p(predicted correct class | matched query)
e_loc_j  = max(clamp(center_distance / d_max, 0, 1), 1 - BEV_IoU)
e_occ_j  = 1 - mean_t soft_IoU(predicted actor occupancy_t,
                               GT actor occupancy_t)
e_tl_j   = 1 - p(correct traffic-light state)  # only when state is valid
```

Use the motion mode selected by ORION's own mode score for `e_occ`; additionally
store oracle-best-mode error as a diagnostic, never as the primary target. A
miss receives occupancy error 1 on its GT footprint. An unmatched predicted
object contributes a false-positive error on its predicted footprint weighted
by its calibrated detection probability.

Do not start v1 by mixing these components with arbitrary learned weights.
Store them separately and form the compatibility scalar as a soft union:

```text
e_obj_j = 1 - product_k (1 - e_kj),  k in valid components
```

This remains in `[0,1]`, means "at least one safety-relevant perception
component failed," and leaves each component auditable. Stratified results
must still report miss, class, localization, motion-occupancy and traffic-state
errors separately.

### 5.3 Attribute actual object error to camera patches

For every GT or unmatched predicted object, construct a fractional patch
support map `A[v,p,j]`:

1. Project the 3D box with the **post-augmentation** `lidar2img` matrix.
2. Clip it to the processed view and reject points behind the camera.
3. Refine the projected polygon with class-compatible CARLA semantic pixels.
4. Resolve overlapping objects with projected depth and use recorded depth
   only as a coarse consistency check until its 8-bit encoding is validated.
5. Area-pool the refined pixel support to the exact 40 x 40 patch grid.

The actor-ID encoding in the installed instance PNGs has not yet been verified
against `gt_ids`; do not silently assume a byte order or actor-ID formula.
Instance masks can replace the projected-box refinement only after that mapping
passes an explicit audit.

Aggregate objects with a noisy-OR so a small failed pedestrian is not diluted
by the full image:

```text
E_view[v,p] = 1 - product_j (1 - A[v,p,j] * e_obj_j)
```

Also export `valid_view_patch[v,p]`. Invalid, unobserved or geometrically
ambiguous patches must be masked out of the loss; they must not be labeled as
zero uncertainty.

For audit and future BEV-head work, export the derived occupancy error sidecar:

```text
O_gt, O_pred     [T=6, H_bev=200, W_bev=200, C]
E_bev            = abs(O_pred - O_gt)
```

The Stage-1 image head's target is `E_view`; `E_bev` is not to be flattened and
pretended to be a camera patch label.

### 5.4 Tensor contract

Implemented record contract: `spatial-uq-paired-feature/v2` with target
contract `spatial-uq-target-contract/v2`:

```text
observed_patch_features    float [V=6, P=1600, D=1024]
clean_patch_features       float [V=6, P=1600, D=1024]
error_severity_target      float [V=6, P=1600]  # continuous E_obs
failure_event_target       bool/float [V=6, P=1600]  # distinct event label
target_valid_mask          bool  [V=6, P=1600]
clean_error_severity_target float [V=6, P=1600]  # E_clean, not forced zero
clean_failure_event_target bool/float [V=6, P=1600]
clean_target_valid_mask    bool  [V=6, P=1600]
component_errors           float [V=6, P=1600, K]  # optional, named axis
component_error_names      [K]
component_error_axis       -1
corruption_mask            float [V=6, P=1600]  # auxiliary only
ensemble_teacher_variance  float [V=6, P=1600]  # optional
```

The Gaussian NLL consumes only `error_severity_target`; the Brier/calibration
path consumes only `failure_event_target`. Every loss term is restricted to
the applicable valid mask. Missing clean labels remain missing and contribute
no clean loss. Representation-proxy records derive only continuous paired
cosine severity and deliberately have no failure-event target, so no Brier or
probability-calibration claim is reported for them.

Legacy v1 proxy datasets with no `failure_target` remain readable through an
audited migration. Legacy v1 records containing `failure_target` are rejected:
their severity/event meanings cannot be recovered without an explicit
regeneration or user-specified mapping.

The canonical decoded exporter records event policy
`orion.actual-failure-event-policy/v1`. It first compares the object-level
`miss`, `class`, `localization`, `motion_occupancy`, `traffic_state`, and
unmatched-prediction `false_positive` errors with their preregistered `0.50`
component thresholds. These hard object/component events are then projected
to patches wherever the failed GT or predicted object's visible support is at
least the separate frozen
`minimum_patch_support=0.01`; patch events are ORed across objects and
components. This order prevents a missed small pedestrian with support `0.10`
from being averaged below the component threshold. The continuous severity
path remains a fractional noisy-OR and is never reused as an event probability.
Changing a component threshold, support threshold, or aggregation rule
requires a new policy version.
The `0.50` values are a preregistered pilot definition, not an empirically
proven optimum; calibration-split sensitivity for both component thresholds
and the independent `0.01` support gate must be reported, especially for edge
patches and small pedestrians.

Motion error is valid only for ORION's selected motion mode. Missing
traffic-state ground truth makes that component invalid rather than negative.
Invalid projected patches remain masked throughout. Observed and clean
severity/event/component maps are measured independently and retain separate
valid masks.

Chronological observed and clean branches may have different temporal-memory
contents. Their branch-specific history IDs are therefore preserved separately
and are not required to match. Pair construction instead requires matching
frame/config/geometry plus a shared, versioned pairing protocol and corruption
schedule ID. The lower-level GT-only compatibility bridge uses
`orion.projected-object-failure-event-policy/v1` and fails closed if unmatched
false-positive predictions have no projected support. The canonical decoded
exporter instead requires predicted-object support and includes false positives
in both severity and event maps; neither path silently drops them.

Required metadata per record includes:

```text
target_provenance: actual_frozen_orion_task_error
target_version: orion_projected_object_failure/v1
error_components: [miss, class, localization, motion_occupancy, traffic_state, false_positive]
patch_attribution: projected_visible_object_support_proxy
privileged_labels_used: [3d_boxes, future_boxes, semantic, coarse_depth]
route_id, town, frame_idx, corruption family/severity/seed/window
base_checkpoint_sha256, config_sha256, camera_order, image_transform
```

## 6. Offline generation procedure

1. Create a persistent route-disjoint manifest before target generation.
2. Iterate each route chronologically using the deterministic evaluation image
   transform; save the exact camera order and transformed calibrations.
3. Run the clean observation through frozen ORION perception and export raw
   query outputs plus decoded outputs.
4. Apply one preregistered local corruption to the same recorded state and run
   the corrupt branch. Use local blur, glare, darkening and partial occlusion;
   camera dropout remains diagnostic only.
5. Compute GT occupancy from current/future boxes with the existing planning
   rasterizer and predicted occupancy from decoded boxes/motion.
6. Compute per-component actual errors, projected patch support, `E_obs`,
   `E_clean`, and `DeltaE`.
7. Save small target/metadata records. For a pilot, cached observed features are
   acceptable; for the full set, avoid duplicating hundreds of GB per
   corruption condition and generate corrupt features online or in bounded
   shards.
8. Run all gates in Section 9 before training the UQ head.

The clean and corrupt branches use the same RGB source state, GT, route command,
camera calibration and augmentation geometry. Only the preregistered visual
corruption differs. Privileged labels are used only by the offline target
generator and are never inputs to the deployed UQ head.

## 7. Resource estimate

Empirical repository logs show roughly 1.3 frames/s for an existing full ORION
open-loop path on one GPU. Target extraction will differ, so these are planning
ranges rather than promised throughput.

### Minimum signal pilot

Suggested pilot:

- 8-12 route-disjoint folders;
- 800-1,000 states, balanced between safety-critical visible objects and
  background/no-critical-object frames;
- clean plus matched on-path/off-path local blur and glare/occlusion at one
  middle severity;
- one A800, 16 CPU cores, 220-240 GB host memory, 4-6 h wall-time request.

At full-model throughput, about 4,000-5,000 forward observations take roughly
1-1.5 GPU-hours after model load; target rasterization, route replay, storage
and safety margin justify the larger wall-time request. A perception-only
runner may be faster and use less host memory, but that must be measured by a
smoke run rather than assumed.

### Full installed 50-route subset

For 12,806 states, clean plus four corruption conditions is about 64,000
forward observations: approximately 14 GPU-hours at 1.3 frames/s before
overhead, or roughly 18-30 wall-clock hours conservatively. A 12-condition
matrix would exceed 150,000 observations and should not be launched before the
pilot gates pass.

### Storage

- one `[6,1600]` float16 patch target is about 19 KB per observation;
- one `[6,200,200]` float16 BEV error sidecar is about 480 KB per observation;
- raw/decoded object outputs and metadata are small relative to features;
- one full `[6,1600,1024]` float16 feature tensor is about 19 MB per frame.

Therefore target maps are cheap, while caching a complete corrupt feature copy
for every condition is not. Store component targets and compact query/box
metadata; shard or regenerate image features.

## 8. Leakage and shortcut risks

| Risk | Consequence | Required control |
|---|---|---|
| Training on the installed 50 validation routes | Invalid untouched-val claim | Exploratory label only, or permanently reserve held-out routes; acquire train routes for formal work |
| UQ head sees route, town, weather ID, corruption type/severity or mask | Dataset/corruption classifier masquerades as UQ | Only observed visual features enter the head; metadata is for splitting/audit |
| Corruption mask is the primary label | Learns where augmentation code painted pixels, not whether perception failed | Mask loss remains low-weight auxiliary; gate on actual task error |
| Clean branch is treated as ground truth | Clean ORION failures disappear from the label | Compare both branches independently with B2D GT; save `E_clean` |
| Raw Hungarian match is called a detection success | Every GT gets a query even for a practical miss | Decode, threshold and class-aware match; report 2 m/4 m sensitivity |
| Global object density predicts label magnitude | Head fires in crowded scenes rather than uncertain ones | Balance object count, class and distance; report no-corruption crowded controls |
| Projection uses raw instead of augmented geometry | Convincing but shifted heatmaps | Transform labels/masks with the exact pipeline and post-transform `lidar2img` |
| Unknown instance-PNG ID encoding | Wrong object-to-pixel labels | Use projected boxes + semantic/depth until ID mapping is verified |
| Privileged semantic/depth leaks into inference | Unrealistic deployed system | Use only offline target creation; assert absent from model inputs/checkpoints |
| Future GT or route corridor enters Stage-1 head | Oracle leakage and route-conditioned "uncertainty" | Future GT creates labels only; route enters path-risk aggregation later |
| Base ORION checkpoint changes | Target no longer describes the deployed model | Hash checkpoint/config; invalidate and regenerate targets on model change |
| Local corruption has an obvious low-level signature | Blur/blackness detector passes in-distribution tests | Held-out corruption family, natural adverse weather, photometric baselines, equal-area off-path controls |
| Sparse positive patches dominate sampling | Good aggregate accuracy with no hazard recall | Object/class/distance-balanced sampling and AUPRC/FPR95/AURC |
| Planner sensitivity is used as the UQ truth | Circular Stage-1/Stage-2 objective and ADE imitation | Use only as a downstream relevance audit after perception-error supervision |

## 9. Gates before Stage-1 training or closed-loop expansion

### Gate G0: data and split integrity

- A persisted manifest is route-disjoint across train, validation, calibration
  and held-out splits.
- Repetitions/weather versions of the same route do not cross splits.
- Formal claims do not train on official held-out validation routes.
- Every record resolves uniquely to `(folder, frame_idx)` and has all required
  privileged labels.

**Stop** if only a frame-random split is available.

### Gate G1: target implementation validity

- Aggregate decoded detection/motion errors from the exporter reproduce the
  repository evaluator on a fixed smoke subset within stated numerical
  tolerance.
- GT occupancy from the exporter matches `PlanningMetric` on unit fixtures.
- At least 100 frames receive manual overlay QA across all six views, including
  near/far, occlusion, pedestrian, vehicle and traffic-light cases.
- The clean/corrupt image transform, camera order and route identity agree
  exactly; invalid projections are masked, not zero-filled.

**Stop** on coordinate, camera-order or temporal-memory mismatch.

### Gate G2: the corruption induces actual failure

For each proposed corruption family:

- the route-paired bootstrap 95% confidence interval for mean on-region
  `DeltaE` is above zero;
- actual error increases monotonically with preregistered severity on average;
- an equal-area on-path corruption produces more safety-object error than the
  matched off-path corruption when a relevant object is present;
- clean and off-path controls do not show a comparable global error increase.

Do not set a universal effect-size threshold before inspecting the metric's
scale. Preregister it after the target smoke and before UQ training.

**Stop or replace the corruption** if it is visually dramatic but does not
increase frozen-ORION task error.

### Gate G3: localization is meaningful

- projected patch supports cover visible safety-critical GT objects with high
  recall on manually audited frames;
- on-region `DeltaE` exceeds matched outside-region `DeltaE`;
- results hold after stratification by camera, object class, distance, town and
  weather;
- small pedestrians are not erased by image/global averaging.

This gate validates attribution quality, not pixel-level causality.

### Gate G4: UQ learnability and calibration

On route- and corruption-family-held-out data, the learned head must:

- improve actual-error AUPRC, FPR95, Brier/NLL and AURC over brightness,
  black-pixel ratio, blur score, Density UQ and representation residual;
- preserve calibration on clean frames with real clean failures;
- respond promptly and recover after fixed corruption event windows;
- beat a model trained only on the corruption mask;
- maintain the result under natural adverse weather, not only synthetic
  corruption.

If only the corruption-mask metric improves, the Stage-1 claim has failed.

### Gate G5: downstream relevance, still separate from Stage 1

Only after G0-G4 pass:

1. use the actual target as an oracle UQ signal through the fixed path-risk and
   behavior mechanism;
2. verify oracle safety improvement without unacceptable route completion or
   traffic-rule cost;
3. compare learned UQ with oracle, off, constant and shuffled controls in the
   minimal closed-loop matrix.

Planner counterfactual sensitivity can be added here as an explanatory audit:
measure the increase in GT collision/traffic-rule loss under a localized
intervention. It must not be fed back as the sole Stage-1 truth before this
separation has been established.

## 10. Recommended first experiment

The strongest minimum experiment is not a large corruption matrix. It is a
target-validation pilot:

1. Select 8-12 route-disjoint folders with visible pedestrian/vehicle hazards
   and matched no-critical-object frames.
2. Generate clean, equal-area on-path and off-path local corruptions at one
   severity for two non-dropout families.
3. Export `E_obs`, `E_clean`, component errors, projected overlays and
   `DeltaE` for 800-1,000 states.
4. Pass G1-G3 before training any UQ head.
5. Train the small head only if actual task error is induced and localized.
6. Evaluate on held-out routes and a held-out corruption family. Proceed to an
   oracle/learned closed-loop comparison only after G4.

This breaks the apparent chicken-and-egg loop: Stage 1 learns and validates a
task-grounded perception-failure signal; Stage 2 tests whether acting on that
signal improves safety. Closed-loop outcomes are not needed to define the
first signal, but they remain necessary to show that the signal is useful for
driving.

## 11. Claim boundary

If this target passes its gates, the defensible statement is:

> The spatial head predicts where frozen ORION's object/occupancy perception
> is likely to be wrong, using privileged simulator labels only during target
> construction. A separate path-risk module determines whether the predicted
> failure overlaps a candidate route.

It is still not sufficient to say that the LLM understands uncertainty. That
claim requires the later behavior/LLM intervention experiment and the oracle,
constant and shuffled controls.

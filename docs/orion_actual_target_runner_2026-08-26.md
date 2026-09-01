# Frozen-ORION actual-target runner checkpoint

> Date: 2026-08-26
> Scope: CPU implementation and preflight only. No Slurm job, GPU, CARLA, or
> training was started.

## Outcome

The minimum Route214 runner boundary is implemented in
`uq_estimator/orion_actual_target_runner.py`, with the CPU-only entry point
`scripts/preflight_orion_actual_target_runner.py`.

It freezes the smoke to two non-interleaved replays:

- clean frames `0..63`, then a fresh reset;
- observed frames `0..63`;
- 128 perception forwards in total;
- only the 43 preregistered measurement frames may become paired records;
- every other frame is forwarded only to build chronological ORION memory.

The public execution function accepts only the repository `Orion` class and
real runtime hooks. CPU mocks cannot create a G1-looking attestation.

## Dedicated config mutation

The runner reads `adzoo/orion/configs/orion_stage3_agent.py`, deep-copies it,
and mutates only the in-memory actual-target test pipeline. It does not edit
or silently change the deployed agent config.

The dedicated pipeline sets:

- `LoadAnnotations3D.with_light_state=True`;
- `LoadAnnotations3D.with_actor_ids=True`;
- `CustomCollect3D` keys include `traffic_state`, `traffic_state_mask`, and
  `gt_actor_ids`;
- batch size is one and test mode is explicit.

The preflight verifies the local formatter, range filter, name filter, and
final box-mask code all keep actor IDs and traffic state on the same GT object
axis. Every measurement target must additionally return an actor-ID/support
eligibility audit; equal tensor lengths alone are insufficient.

## Perception-only head hook

The hook executes only:

```text
extract_feat -> prepare_location -> position_embeding -> pts_bbox_head
```

It does not invoke the language model, VAE, diffusion, or planner decode. Raw
head outputs are retained and passed to `adapt_orion_head_outputs_v1`, which
preserves source query IDs, all class probabilities, all trajectory modes and
scores, the ORION-selected motion mode, and all four traffic logits.

Real patch features are taken from position-level EVAViT features with exact
shape `[1,6,C,40,40]` and transformed to row-major `[6,1600,C]`. Any different
camera count, order, or feature grid fails closed.

## Traffic semantics

ORION traffic output is not a four-way softmax. The frozen v1 contract is:

- `all_traffic_states[..., :3]`: three independent sigmoid light-state
  logits;
- `all_traffic_states[..., 3]`: affects-ego logit retained for audit only;
- GT state label: `traffic_state[:,0]`;
- GT validity: `traffic_state_mask & bool(traffic_state[:,1])`.

The affects-ego prediction is not mixed into the v1 state error. Missing or
non-affecting state is masked invalid, never encoded as a negative target.

The raw Route214 prefix audit found 1,397 traffic-light records: 1,036 with
`(state=0, affects_ego=false)` and 361 with
`(state=2, affects_ego=false)`. Therefore the traffic component has zero valid
GT lights in this smoke. Per-frame and branch attestations separately report
`raw_loader_valid_light_count` and `affects_ego_valid_count` so this cannot be
misread as successful traffic prediction.

## GT eligibility

The main object-failure target includes only:

- car, van, truck, bicycle, pedestrian: class IDs `0,1,2,3,7`;
- traffic light class `6` only when loader-valid and affecting ego.

Traffic sign, cone, others, and non-affecting traffic lights are excluded, not
turned into negative labels. Boxes, classes, GT attributes, state/mask,
projected support, and actor IDs must be filtered with one boolean mask.

## Production geometry connections

`build_production_runtime_hooks_v1` directly fixes these repository
implementations; callers do not supply replacements:

- selected-mode occupancy:
  `selected_mode_occupancy_callback_v1`, ID
  `orion-selected-mode-derived-occupancy/v1`;
- GT occupancy: `rasterize_planningmetric_gt_v1`, ID
  `planningmetric-gt-side-effect-free-parity/v1`;
- pairwise IoU: `pairwise_bev_iou_v1`, ID
  `orion-side-effect-free-continuous-rotated-bev-iou/v1`;
- visible support: `project_boxes_to_visible_patch_support`, canonical six
  cameras, post-augmentation matrices, and exact `40x40` patches.

The z-origin policy is deliberately different across branches:

- canonical `gt_bboxes_3d.tensor`: `bottom` origin. The B2D dataset constructs
  `LiDARInstance3DBoxes(origin=(0.5,0.5,0.5))`; the base box wrapper converts
  its stored tensor to destination `(0.5,0.5,0)`;
- decoded ORION boxes at the adapter boundary: `center` origin. The head is
  trained against `gravity_center`; the repository subtracts half-height only
  later when wrapping decoded boxes.

The preflight records source hashes for this evidence. Treating both branches
as one global z-origin is forbidden. Projection overlay QA remains an
independent required gate.

## Frozen pilot thresholds

The runner rejects target bundles unless they use:

- event component thresholds `0.50` and patch support `0.01`, policy ID
  `preregistered-pilot-thresholds-0p50-support-0p01/v1`;
- decoded score `>=0.50` and center distance `<=4.0 m`, with `2.0 m`
  sensitivity analysis;
- matching policy `class-score-distance-gated-one-to-one/v1`.

These are preregistered pilot heuristics, not calibration-optimal thresholds.

## Current fail-closed status

The production GT and predicted occupancy rasterizers correctly have distinct
IDs. Exporter branch bundle v2 now persists both IDs independently and checks
grid/time shape compatibility. Runner preflight detects
`orion.actual-target-branch-bundle/v2` and passes the dual-ID gate only when a
production hook bundle declares exactly the selected-mode predicted ID and the
PlanningMetric-parity GT ID; it never forges equality.

The metadata-only CLI still reports `execution_ready=false` because it does
not install real runtime hooks or projection overlay QA, and its local default
does not run the server dataset file preflight. On the server the already
verified source plan can close the file gate, but a real branch target builder,
actor/support-axis QA, overlay QA, and record sink are still required before
the first A800 smoke.

## Verification

```bash
.venv/bin/python scripts/preflight_orion_actual_target_runner.py
.venv/bin/python -m pytest -q tests/test_orion_actual_target_runner.py
```

Current local result: 10 runner tests passed; the runner plus exporter,
rasterizer, and projection suite passes 42 tests. The dry-run reports
`execution_ready=false`, `job_submitted=false`, `gpu_used=false`,
`carla_used=false`, and `training_performed=false`.

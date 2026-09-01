# Concrete frozen-ORION branch-target builder

> Date: 2026-08-26
> Status: CPU implementation and fixture validation complete. No A800, Slurm,
> CARLA, or training work was started by this task.

`uq_estimator/orion_actual_target_builder.py` implements the concrete callable
used as `branch_target_builder` by the chronological runner. It consumes one
`DecodedORIONFrameV1`, one canonical pipeline batch, and the runner's audited
production-hook context, then returns `BuiltBranchTargetV1`.

## Construction path

For each measurement frame the builder:

1. clones canonical GT boxes, classes, PlanningMetric attributes, traffic
   state/mask, and aligned actor IDs to private CPU tensors;
2. projects every original GT box with post-augmentation matrices and explicit
   GT `bottom` z-origin;
3. applies one eligibility mask to boxes, classes, attributes, state/mask,
   projected support, and actor IDs;
4. requires eligible `gt_attr_labels[:,27]` to equal eligible
   `gt_labels_3d`, then passes `gt_attr.unsqueeze(0)` to the side-effect-free
   PlanningMetric GT rasterizer;
5. moves decoded outputs to private CPU tensors, retaining selected motion,
   four traffic logits, source queries, and mode scores;
6. projects decoded boxes with `center` z-origin;
7. computes continuous rotated pairwise BEV IoU;
8. constructs `PrivilegedGroundTruthFrameV1`, `TargetProvenanceV1`,
   `FrameChronologyV1`, and `ObservationConditionV1`;
9. calls `build_actual_target_branch` with score `>=0.50`, distance `<=4 m`,
   preregistered event thresholds, and the production geometry IDs.

Invalid traffic labels use numeric zero only as a storage placeholder with
`traffic_state_valid=false`. They never become negative supervision.

## Enforced identities

- predicted occupancy:
  `orion-selected-mode-derived-occupancy/v1`;
- GT occupancy:
  `planningmetric-gt-side-effect-free-parity/v1`;
- pairwise IoU:
  `orion-side-effect-free-continuous-rotated-bev-iou/v1`;
- support projection:
  `orion.six-view-projected-visible-support/v1`;
- eligibility:
  `safety-actors-plus-affecting-traffic-light/v1`;
- z-origin:
  GT `bottom`, decoded ORION `center`.

The exporter v2 retains the two rasterizer IDs separately. The builder checks
both after branch construction and does not alias them.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_orion_actual_target_builder.py
```

The four CPU tests cover a full branch-target fixture, clean/observed
observation provenance, invalid-traffic masking, actor/support-axis filtering,
class/attribute disagreement, context-policy drift, post-augmentation matrix
drift, distinct rasterizer IDs, and distinct GT/predicted z origins.

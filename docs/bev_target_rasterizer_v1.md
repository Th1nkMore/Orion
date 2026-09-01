# Side-effect-free BEV target rasterizer v1

This module supplies the geometry still missing between the decoded frozen
ORION head and the actual Stage-1 target bridge:

- implementation: `uq_estimator/bev_target_rasterizer.py`;
- tests: `tests/test_bev_target_rasterizer.py`;
- GT policy ID: `planningmetric-gt-side-effect-free-parity/v1`;
- predicted policy ID: `orion-selected-mode-derived-occupancy/v1`;
- continuous IoU policy ID:
  `orion-side-effect-free-continuous-rotated-bev-iou/v1`.

This is a CPU/dependency-light geometry module. It does not load ORION, MMCV,
CARLA, a checkpoint, or a CUDA extension.

## GT parity and side effects

`rasterize_planningmetric_gt_v1` follows the exact six-step convention in
`mmcv/models/dense_heads/planning_head_plugin/metric_stp3.py`:

- bounds `[-50, 50] m` in both BEV axes;
- `0.5 m` resolution and a `200 x 200` raster;
- cumulative XY step deltas and cumulative future-yaw deltas;
- the same current-yaw conversion and OpenCV polygon rounding/fill;
- vehicle classes `[0, 1, 2]` and the historical human classes `[3, 7]`.

The repository reference obtains a NumPy view of a CPU box tensor and then
edits the yaw field in place. NumPy may share that tensor storage. The v1
module clones every tensor before NumPy conversion, and the tests separately
assert pixel parity and immutability of both boxes and future features.

The return value contains per-object `[N,T,H,W]` occupancy. Union-all,
vehicle-union, and human-union grids are optional audit sidecars and are not
created unless `include_union=True`.

## Bicycle parity caveat

The repository class comment identifies class index 3 as `bicycle`, but
`PlanningMetric.category_index` includes 3 in the human/pedestrian layer. V1
preserves that behavior solely to make evaluator parity auditable. It is not a
claim that a bicycle is semantically a pedestrian. Any corrected grouping
must receive a new rasterizer version and cannot silently replace v1 targets.

## Predicted selected-mode occupancy

ORION does not expose a native actor-occupancy output. The predicted tensor is
therefore described only as deterministic, **derived occupancy**:

1. start from the decoded current box `(x,y,z,w,l,h,yaw,...)`;
2. take the trajectory mode selected by ORION's own mode scores;
3. cumulatively sum its per-step XY deltas;
4. add the decoded current center;
5. rasterize each future box with the PlanningMetric grid convention.

There is no future actor-yaw output on this decoded path. V1 freezes the
converted current yaw across the future horizon under policy
`planningmetric-convert-current-yaw-then-freeze/v1`. The immutable provenance
sets `native_orion_occupancy=False`; this output must never be described as an
ORION occupancy-head prediction or as oracle-best-mode occupancy.

The dependency-light adapter callback is:

```python
from uq_estimator.bev_target_rasterizer import (
    SELECTED_MODE_RASTERIZER_ID,
    selected_mode_occupancy_callback_v1,
)

config = ORIONDecodeAdapterConfigV1(
    # ... frozen decoder fields ...
    occupancy_rasterizer_id=SELECTED_MODE_RASTERIZER_ID,
)
decoded = adapt_orion_head_outputs_v1(
    head_outputs,
    config=config,
    occupancy_rasterizer=selected_mode_occupancy_callback_v1,
)
```

The callback uses a structural interface and intentionally does not import or
modify `decoded_actual_target_export.py`.

## Pairwise BEV IoU

`pairwise_bev_iou_v1` computes continuous rotated-rectangle intersection on
private copies and returns `[N_gt,N_pred]` on the input device. This is box
IoU, not occupancy-raster IoU. Its corner order and row-vector rotation match
`mmcv/core/bbox/box_np_ops.py::box2d_to_corner_jit`; tests compare those
repository corners, analytic overlap fixtures, and a non-axis-aligned polygon
fixture. Source boxes are checked unchanged after every operation.

## Current limits and gate

- The predicted yaw-freeze approximation must remain visible in every export
  manifest and paper description.
- GT parity is proven against CPU fixtures. The real smoke must still record
  parity tolerance against decoded production boxes before G1 is declared.
- Objects outside the PlanningMetric category groups receive zero GT occupancy
  in this parity version.
- The module provides geometry only. It does not certify decoder indices,
  temporal replay, traffic-state validity, projection support, or false-
  positive support; those remain separate actual-target gates.

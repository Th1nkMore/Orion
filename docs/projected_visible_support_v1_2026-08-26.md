# Six-view projected visible support v1

Date: 2026-08-26

Implementation:
`uq_estimator/projected_visible_support.py`

CPU tests:
`tests/test_projected_visible_support.py`

## Claim boundary

This module supplies the missing image-patch **attribution proxy** for the
Stage-1 actual target. It does not identify causal pixels and it does not prove
that a projected region caused ORION's error. Its valid claim is narrower:

> A decoded or privileged 3D object projects onto these processed-camera
> patches under the recorded post-augmentation geometry.

Consequently, `attribution_is_causal` is frozen to `false` and the provenance
string remains `projected_visible_object_support_proxy`. Object-level error is
task-grounded by decoded prediction versus privileged ground truth; mapping
that error back to camera patches remains approximate attribution.

## Input contract

`project_boxes_to_visible_patch_support` requires:

- full lidar boxes `[J,D>=7]` with fields
  `x,y,z,dx,dy,dz,yaw,...`;
- an explicit `box_z_origin` of either `center` or `bottom`;
- final, post-resize/crop/flip/rotation `lidar2img [V,4,4]` matrices;
- processed image shape `[V,2]` in height/width order;
- three separately declared orders: feature/camera order, matrix order and
  image-shape order;
- the transform identity and exact patch grid;
- an optional input patch-valid mask.

The production defaults are the frozen ORION order

```text
CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
```

and `patch_hw=(40,40)`. Any camera reorder, matrix reorder, image-shape reorder
or 40 x 40 mismatch fails closed. V1 also requires a uniform processed image
shape across the six cameras because `PatchSupportProvenanceV1` stores one
shared image shape.

The caller must audit whether the concrete ORION/MMDetection box tensor uses a
bottom or center z origin before invoking the projector. V1 deliberately has
no inferred default.

## Geometry

For every object and camera, the implementation:

1. converts the complete rotated 3D box to its eight lidar-frame corners;
2. transforms all corners with the post-augmentation projection matrix;
3. clips all 12 box edges against a positive configurable near plane;
4. projects the retained/clipped edge points and takes their 2D convex hull;
5. clips that polygon to the processed image boundary;
6. intersects the clipped polygon with each exact patch rectangle;
7. divides the intersection area by patch area.

This produces fractional support `[V,P,J]` in `[0,1]`, with row-major patch
index `p = patch_y * patch_width + patch_x`. Objects fully behind a camera or
outside its processed image receive exact zero support and
`object_visible_mask=false`; they are not fabricated as visible labels.

`valid_patch_mask [V,P]` is kept separate. Invalid patches are zeroed in the
support tensor but must be excluded from target loss rather than treated as a
negative/certain label.

## Direct actual-target connection

The returned tensors already match the decoded exporter axes:

```python
gt_projection = project_boxes_to_visible_patch_support(
    gt_boxes_lidar,
    post_aug_lidar2img,
    processed_image_hw,
    camera_order=camera_order,
    matrix_camera_order=matrix_camera_order,
    image_shape_camera_order=image_shape_camera_order,
    image_transform_id=image_transform_id,
    box_z_origin="bottom",  # only after the concrete box convention is audited
)

pred_projection = project_boxes_to_visible_patch_support(
    decoded_boxes_lidar,
    post_aug_lidar2img,
    processed_image_hw,
    camera_order=camera_order,
    matrix_camera_order=matrix_camera_order,
    image_shape_camera_order=image_shape_camera_order,
    image_transform_id=image_transform_id,
    box_z_origin="bottom",
)

gt_projected_support = gt_projection.support       # [6,1600,G]
pred_projected_support = pred_projection.support   # [6,1600,N]
target_valid_mask = gt_projection.valid_patch_mask & pred_projection.valid_patch_mask
support_provenance = gt_projection.support_provenance
```

These values can be supplied directly to
`build_actual_target_branch_from_decoded`; no transpose or inferred camera
mapping is required. This implementation intentionally does not modify
`decoded_actual_target_export.py`.

For paired clean/observed replay, the privileged GT geometry support must be
identical across the two branches under the same frame, transform and
calibration. Prediction support is branch-specific because decoded boxes may
differ. Observation-derived refinement must not silently make the shared GT
support branch-dependent.

## Optional semantic/depth refinement

The base v1 result is box-polygon support. `ProjectedSupportRefinerV1` is an
explicit optional interface for semantic and/or depth refinement. A refiner
must declare:

- a non-empty versioned `refinement_id`;
- an exact tuple of required modalities from `semantic` and `depth`;
- a non-null payload for every declared modality;
- refined fractional support, validity mask and an audit note.

Refinement may only remove support from the projected-box polygon; it cannot
expand support outside that conservative geometric attribution region.

Providing data without a refiner, configuring a refiner without all required
data, returning a malformed map, or naming unrecognized modalities fails
closed. When no refiner is configured, provenance records no refinement ID,
zero calls and `refinement_applied=false`. Configuring a refiner is also not
reported as applied if no projected visible object invoked it.

This preserves the current research boundary: the actor-ID encoding in B2D
instance images and the recorded depth encoding have not yet passed the
necessary audit, so v1 must not claim semantic/depth-refined support merely
because those files exist.

## Overlay QA

Two interfaces support G1 visual QA without real data:

- `make_projection_overlay_data` returns JSON-serializable camera, polygon,
  depth, valid-patch and fractional-patch data;
- `render_projection_overlay_image` renders patch heat and box polygons onto
  either a supplied CPU RGB tensor or a neutral synthetic canvas.

The image renderer imports Pillow and NumPy lazily; core target projection only
depends on torch and the standard library.

The minimum real replay acceptance check is still visual: transformed RGB,
projected polygons and the 40 x 40 patch heat must align for all six cameras on
sampled frames. Passing synthetic unit tests alone is not G1 evidence.

## Covered CPU cases

The tests cover:

- a front-view visible box and direct target-aggregator consumption;
- a box entirely behind the cameras with no false visibility;
- near-plane clipping and processed-image edge clipping;
- a box crossing four patches;
- a small object occupying approximately `0.10` of one patch;
- feature-camera, matrix-camera and image-shape camera reorder rejection;
- exact production `[6,1600,J]` output and 40 x 40 mismatch rejection;
- malformed patch-valid masks and non-uniform processed shapes;
- explicit center/bottom z-origin behavior;
- explicit semantic-refiner data requirements;
- JSON overlay and synthetic overlay-image generation.

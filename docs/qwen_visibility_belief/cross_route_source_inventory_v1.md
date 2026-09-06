# Cross-route source inventory for Qwen U grounding

Date: 2026-09-06 (Asia/Shanghai)

## Outcome

The installed Bench2Drive raw subset is large and diverse enough to build the
next route-diverse grounding pilot without first running Qwen or recollecting
every scene. It is not yet a training-ready U-token dataset.

The current Qwen visibility-token corpus remains Route 151 only: 54 frames,
with all 32 frontier rows populated in every frame. The installed raw corpus
contains 100 physical routes, 24,024 indexed frames, 43 scenario types, and 12
towns. A previously frozen manifest already gives a route-disjoint
70/10/10/10 train/validation/calibration/held-out split and passed its recorded
route/folder/town leakage checks.

| Split | Routes | Indexed frames | Maximum possible `(frame,F00--F31)` pairs |
| --- | ---: | ---: | ---: |
| Train | 70 | 16,207 | 518,624 |
| Validation | 10 | 2,423 | 77,536 |
| Calibration | 10 | 2,790 | 89,280 |
| Held-out | 10 | 2,604 | 83,328 |

These pair counts are upper bounds, not promised examples. Offline
tokenization has not yet measured how many candidate frames populate every
frontier row.

## What was checked

The read-only audit selected the first, middle, and last indexed frame from
each of the 100 routes: 300 frames total. It checked 1,800 required RGB/depth
paths and parsed every sampled annotation.

- All sampled front, front-left, and front-right RGB and depth files exist.
- All 300 sampled annotations contain the three camera calibrations, ego
  state, speed, and near/far navigation commands required for a route-exposure
  preflight.
- All 900 sampled depth files have a 1600 x 900, 8-bit grayscale PNG header.
- Two routes each have one internal missing indexed frame: Route1093 frame
  322 and Route1419 frame 334. A future temporal loader must break rather than
  bridge memory across those gaps.
- Expert-assessment files are absent at the terminal indexed frame of every
  route and nowhere else in this 300-frame sample. That is expected for a
  future-looking target and does not block U geometry, but terminal frames
  cannot be assumed to have planning supervision.

This is a sampled source audit, not a checksum of every raw image. The report
sets `full_file_integrity_proven=false` explicitly.

## Depth boundary

The offline depth is not equivalent to the live Route 151 oracle depth.
Bench2Drive's collector computes metric range in metres as `float16` and then
writes it with `cv2.imwrite`. On the installed OpenCV build, an explicit
round-trip probe confirmed fallback to `uint8`: 4.4 m became 4, 4.6 m became
5, 59.5 m became 60, and 300 m saturated at 255. Thus the near-range storage
uncertainty is approximately plus or minus 0.5 m.

This matters because the accepted live grid uses 0.5 m cells and a 0.45 m
surface tolerance. Reading the PNG as exact metric depth would be false
precision and can move the visible/occupied/occluded boundary by a cell. The
offline data is suitable for a coarse tokenization preflight only after one of
these policies is frozen:

1. represent each stored depth as an interval and propagate the quantization
   ambiguity into the visibility state; or
2. use a separately recorded, larger surface tolerance and measure token
   stability; or
3. recapture selected routes with lossless 24-bit CARLA depth if the coarse
   pilot is unstable.

Collector source provenance on the server:

- `Bench2Drive/tools/data_collect.py`: SHA-256
  `be1d399100e20ba1809652beff5451a73af7c12a9d16bac00911dc1d9ac37b66`;
- `Bench2Drive/tools/utils.py`: SHA-256
  `089b083ca395db844ba3e59fc25ee5334f56049e602216c99c97fe21da2308ed`.

## Query and split feasibility

Random `F00`--`F31` queries are feasible as a data-construction operation.
They must be sampled after tokenization from valid rows, with a deterministic
seed recorded in a new immutable query manifest. The 54-frame Route 151 token
set proves that the current tokenizer can populate all 32 rows in one route;
it does not prove the same rate across the 100 offline routes.

Route and frame isolation are already available: assign routes first using the
frozen split, so no frame can cross a split. A global frontier-index holdout is
also mechanically possible, for example by reserving selected `Fxx` identities
for a separate interface-extrapolation test. It should not be confused with
the main generalization claim; the primary split must remain natural,
route-disjoint data with no repeated `(route,frame,Fxx)` query.

The training loss remains ordinary per-example supervision. There is no loss
for giving the same answer to two examples from one frame. Matched-pair and
shuffle comparisons remain diagnostics and demand a changed answer only when
the queried field crosses the declared threshold.

## Remaining gates before training

1. Freeze the integer-depth interval/tolerance policy.
2. Freeze how the offline route polyline is reconstructed from the current
   pose and near/far navigation annotations; do not silently use future
   executed motion as a planning input.
3. Tokenize a small route-diverse pilot in memory or into a new immutable
   artifact root and report valid-frontier-row coverage and tolerance
   stability.
4. Freeze the random-query and route/frame/optional-row split manifest.
5. Only then launch another Qwen grounding run.

The recommended next action is gate 1 plus a no-training tokenization preflight
over a small stratified set of routes. Full CARLA recapture is not justified
until that test shows the 8-bit depth is too unstable.

## Immutable audit artifact

- Report:
  `/public/share/lidachuan/orion_assets/qwen_visibility_grounding_runs/cross_route_source_inventory_v1/inventory.json`
- Report SHA-256:
  `f21695d91df467a83d2a66abe3ef849c24fe0fe7f6fd751abaca96392581507d`
- Infos SHA-256:
  `e50584ee0eb39df11068a726418117958a0b2f3e5cd82ba76fe20ca63814d1d3`
- Frozen route-manifest SHA-256:
  `9f2acaaf8b9ec291ac803bb3a014e880265f0399bcc22e3d1fdb66dd5a628fd3`
- Existing Route 151 token-manifest SHA-256:
  `69fd211e666f952914cb3bcc076bbccbb6f6fbe1207cb5fd0469661d64afe751`

No training was started and no offline U token was generated by this audit.

# Cross-route source inventory for Qwen U grounding

Date: 2026-09-06 (Asia/Shanghai)

## Outcome

The installed Bench2Drive raw subset was large and diverse enough to build a
bounded route-diverse grounding pilot without recollecting every scene. A
120-frame no-training preflight showed that its coarse depth preserves the
physical frontier set and minority `ON_ROUTE` labels well enough for that
pilot. The query manifest and serialized corpus have now been built and
independently audited; the subsequent single Qwen baseline is complete.

At inventory time, the Qwen visibility-token corpus was Route 151 only: 54
frames, with all 32 frontier rows populated in every frame. The installed raw
corpus contains 100 physical routes, 24,024 indexed frames, 43 scenario types,
and 12 towns. A previously frozen manifest gives a route-disjoint 70/10/10/10
train/validation/calibration/held-out split and passed its recorded
route/folder/town leakage checks. The later V1k corpus adds one audited query
from each non-calibration route.

| Split | Routes | Indexed frames | Maximum possible `(frame,F00--F31)` pairs |
| --- | ---: | ---: | ---: |
| Train | 70 | 16,207 | 518,624 |
| Validation | 10 | 2,423 | 77,536 |
| Calibration | 10 | 2,790 | 89,280 |
| Held-out | 10 | 2,604 | 83,328 |

These pair counts are upper bounds, not promised examples. The later frozen
pilot selected one valid queried row from one frame on each of 90 routes;
88 selected frames expose all 32 rows, and the remaining two expose 18 and
seven. No query addresses a masked row.

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
mechanically possible only as a slot-address interface test. The preflight
shows that `Fxx` is a row in a per-frame sorted table, not a persistent spatial
identity, so reserving an index cannot support a spatial-generalization claim.
The primary split must remain natural, route-disjoint data with no repeated
`(route,frame,Fxx)` query.

The training loss remains ordinary per-example supervision. There is no loss
for giving the same answer to two examples from one frame. Matched-pair and
shuffle comparisons remain diagnostics and demand a changed answer only when
the queried field crosses the declared threshold.

## No-training depth preflight result

The preflight used the ten frozen calibration routes only: 120 frames at the
live 2 Hz U cadence. It compared stored integer-metre depth, its minus/plus
0.5 m quantization endpoints, and a quantization-aware 0.95 m surface
tolerance. No Qwen model was loaded and no U-token corpus was serialized.

- All four variants produced 32 valid rows on all 120 frames.
- A `route label` is `ON_ROUTE` when `route_weight_mean >= 0.2`.
  `ON_ROUTE` prevalence is only 7.5%--8.3%, so all label results were checked
  separately by class.
- `nearest physical match` is the bidirectional fraction whose nearest
  cross-variant frontier center is within 2 m. The strict minus/plus 0.5 m
  endpoint result is 95.31%.
- For those nearest physical matches, source-conditioned `ON_ROUTE` label
  agreement is 93.91% at the strict endpoints; `OFF_ROUTE` agreement is
  99.14%, and their equal-class average is 96.53%.
- In contrast, comparing the same `Fxx` index at the strict endpoints gives
  only 29.24% physical agreement within 2 m, 59.56% `ON_ROUTE` agreement, and
  a 39.03 m p95 location difference. Ranking changes can reassign an index to
  a distant frontier even when the physical frontier set itself is stable.

The coarse source is therefore accepted for a bounded pilot under the
quantization-aware center-depth/0.95 m tolerance policy. A query label must be
computed from the exact current row; `Fxx` must never be treated as a stable
world-space object. Comparisons between tokenizations must spatially rematch
frontiers before comparing their attributes.

Immutable report:
`/public/share/lidachuan/orion_assets/qwen_visibility_grounding_runs/offline_depth_preflight_v1_1_label_balance/report.json`;
SHA-256
`2f694458497b6e93227ea4cb1130c239fd781763c3a77d50c88c2949c36082c5`.

## Completed training-data gates

The final corpus satisfies the four former gates: an immutable random-query
manifest identifies every example by `(route, frame, Fxx)`; the 70/10/10
train/validation/held-out routes are disjoint and exactly class-balanced;
training uses ordinary per-example labels without pair-flip or same-answer
penalties; and an independent audit verifies hashes, row masks, labels,
duplicate keys, split membership, and the optimizer schedule.

Corpus root:
`/public/share/lidachuan/orion_assets/qwen_visibility_grounding_runs/route_diverse_random_query_data_v1_2`.
The data audit SHA-256 is
`7db420926ef24b35a05ff9a79c47534dff4e0b70f88c95da603a427c8e3169fa`.

The one authorized V1k Qwen baseline subsequently completed, but true-U,
zero-U, and shuffled-U accuracy against the original queried-row targets were
all 50%. Its independent result audit passed, so this source inventory and
construction are no longer the leading explanation for failure; the current
VLM prefix/readout interface is the remaining bounded negative.

Full CARLA recapture is not justified by this result. It should be reconsidered
only if the larger serialized pilot exposes route/scenario-specific instability
that this calibration slice did not cover.

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

This inventory itself did not train Qwen or generate tokens. Those later steps
are separately recorded in `implementation.md` so their evidence is not
retroactively attributed to this read-only audit.

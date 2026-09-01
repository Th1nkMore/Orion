# Decoded actual-target exporter integration note

> Date: 2026-08-26
> Status: CPU schema/mock complete; real frozen-ORION hook **not implemented**

## Boundary

`uq_estimator/decoded_actual_target_export.py` begins after frozen ORION has
already produced decoded object, motion, and traffic outputs. It imports no
ORION/MMCV/CARLA code and does not run a model. It consumes:

- decoded centers/classes/scores and the full per-query class sigmoid vector;
- full decoded and GT LiDAR boxes, the resolved class/score/distance-gated
  one-to-one match trace, and a versioned BEV-IoU policy identifier;
- `source_query_index` retained from the query×class flattened top-k decoder;
- all four query-aligned traffic logits from a run with
  `with_light_state=True`; the v1 task component uses sigmoid probabilities
  from the first three state logits only and retains the affects-ego logit for
  audit;
- all trajectory modes, their mode scores, the explicit score-selected mode,
  occupancy rasterized from that selected mode, and the immutable rasterizer
  ID;
- privileged B2D object/future/state GT;
- audited pairwise BEV IoU;
- GT and predicted-object visible supports projected with the exact
  post-augmentation `lidar2img`, plus an explicit valid patch mask;
- checkpoint/config/class-map/decoder/camera/transform provenance and route
  chronology.

The output is a small paired target bundle containing continuous severity,
an independently thresholded failure-event map, per-component maps, matches,
projection supports, decoder audit tensors, GT occupancy/validity, the actual
selected-mode occupancy used by the motion component, `E_obs`, `E_clean`,
`DeltaE`, and optional BEV occupancy sidecars. The bridge produces a
`PairedSpatialFeatureRecord` v2 without treating severity as an event
probability.

The object-component audit tensor is `[G+N,6]`: GT rows carry miss, class,
localization, motion-occupancy and traffic-state channels; decoded-prediction
rows carry the false-positive channel. Binary events are thresholded at this
object level first, then projected with a separate minimum visible-support
threshold (`0.01` in policy v1) and OR-ed. They are not obtained by thresholding
the support-diluted continuous patch map, so a small visible pedestrian miss
is not erased merely because it occupies 10% of a patch.

The bundle also retains the per-GT distance gate, minimum prediction score,
valid-pair matrix, reciprocal match indices and matched distances. On load it
reconstructs the class/score/distance gate, continuous severity, every
component map and the binary event map; inconsistent tensors are rejected.
Pairwise BEV IoU is stored together with full boxes and an explicit policy ID
so the real hook can be compared against the repository evaluator rather than
trusting an unaudited matrix.

## Temporal pairing

Temporal provenance has two distinct identifiers:

- `branch_history_id`: the actual clean or observed temporal-memory content;
- `paired_replay_id` / `paired_history_protocol_id`: the shared replay reset,
  source sequence, seed, and schedule-prefix protocol.

Clean and corrupted histories are expected to differ after an intervention.
Pairing checks the shared protocol and route/frame/config/geometry identity; it
does not falsely require equal hidden state. Shared-history batching is only
accepted when both branches declare `history_content=shared`.

## Real hook still required

The real adapter should run each route chronologically and reset temporal
memory only at route boundaries. Repository inspection suggests a
perception-only path can call `extract_feat`, `prepare_location` /
`position_embeding`, then `pts_bbox_head.forward`, avoiding the LLM and
diffusion decoder. This path still mutates temporal memory and must be smoked
against normal inference before use.

`CustomNMSFreeCoder.decode_single` currently flattens query × class sigmoid,
takes top-k 300, derives `bbox_index`, and then drops that index; it also drops
trajectory mode scores. The hook must capture those tensors before they are
discarded. Duplicate decoded entries from one source query are legal and must
not be silently deduplicated. The exporter checks that each decoded score is
the corresponding entry of the retained full class vector and that the
declared selected motion mode equals the mode-score argmax.

Before any real target is accepted, the hook must additionally verify:

1. final decoder layer and calibrated decode/NMS thresholds;
2. `with_light_state=True`, or explicitly omit/mark traffic state invalid in a
   future schema version;
3. predicted occupancy uses ORION-selected mode, never oracle-best mode;
4. BEV IoU/rasterization reproduces the repository evaluator on fixtures;
5. projected supports use processed-image calibration and pass overlay QA;
6. clean and observed branches share source state and replay protocol while
   preserving their branch-specific temporal histories.

For v1, the privileged traffic-state label is `traffic_state[:,0]` and its
validity is restricted to loader-valid lights with `affects_ego=true` from
`traffic_state[:,1]`. The fourth prediction logit is retained but not silently
mixed into the state probability. A future joint state/affect target requires
a new explicit component policy.

## CPU smoke

Dry run, with no writes:

```bash
.venv/bin/python scripts/export_decoded_actual_targets.py --mock --dry-run
```

Write a mock bundle and bridged v2 dataset only to disposable paths:

```bash
.venv/bin/python scripts/export_decoded_actual_targets.py \
  --mock \
  --output /tmp/orion_actual_target_mock.pt \
  --record-output /tmp/orion_actual_record_mock.pt
```

The JSON summary always reports
`real_orion_hook_executed=false` for this CLI. A passing mock validates only
schema, arithmetic, serialization, and bridge wiring. It does not pass G1 and
is not evidence that real ORION task errors have been exported.

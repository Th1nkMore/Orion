# Frozen-ORION actual-target exporter integration audit

> Date: 2026-08-26
> Status: code-path audit complete; real exporter and A800 smoke not yet run
> Scope: Stage-1 actual perception-failure target only. Diffusion, adapter
> training, and closed-loop expansion are out of scope.

## 1. Recommended inference path

The actual-target exporter does not need to invoke the LLM, VAE, diffusion
decoder, route planner, or CARLA.  It can run the frozen perception path:

```text
processed multiview RGB
  -> Orion.extract_feat
  -> prepare_location / position_embeding
  -> pts_bbox_head.forward
  -> raw final-decoder object, motion-mode, and traffic-state tensors
  -> audited decode + actual error versus privileged GT
```

`pts_bbox_head.forward` already returns the raw dictionary needed by the target:

```text
all_cls_scores
all_bbox_preds
all_traj_preds
all_traj_cls_scores
all_traffic_states
```

Calling this partial path still updates the head's temporal memory, which is
required for faithful chronological inference.  It avoids language generation
but does not yet prove a lower host-memory loading envelope: until a
perception-only checkpoint loader is separately validated, request the known
full-model envelope of about 192 GB host memory for the first smoke.

## 2. Decode contract that must be fixed before G1

The configured `CustomNMSFreeCoder` applies sigmoid to final class logits,
flattens query-by-class scores, retains the top 300 entries, and then decodes
their boxes and trajectories.  Two details prevent the current public decoded
result from being used directly as the complete target input:

1. the decoder does not retain the selected source `bbox_index`, so the full
   class-probability vector and traffic-state logits cannot be joined back to
   each decoded prediction without reproducing the indexing;
2. it returns all trajectory modes for the selected query but drops
   `all_traj_cls_scores`; the actual-target policy requires the mode selected
   by ORION's own score, not an oracle-best mode.

The exporter must therefore reproduce the configured decode exactly while
retaining source query index, full sigmoid class vector, all trajectory modes,
trajectory-mode scores, selected mode, and traffic-state logits.  Detection
score threshold, class map, center-distance gate, decoder layer, top-k, and
post-center range must be versioned metadata.  Because flattened top-k can
select one query under more than one class, duplicate source-query behavior
must be preserved for evaluator parity and explicitly audited rather than
silently deduplicated.

The dependency-light adapter in `uq_estimator/orion_decode_adapter.py` now
implements this exact selection order and retains full decoded boxes plus its
selection audit. CPU fixtures pass, but the adapter has not yet consumed a
real frozen-ORION head output; that remains part of the first A800 smoke.

## 3. Temporal clean/corrupt pairing

The head mutates `memory_embedding`, reference points, timestamps, ego poses,
velocities, CAN-bus history, scene tokens, and optional scene-query memory on
every frame.  Clean and corrupt frames must never be alternated through this
single mutable state.

The first correct implementation is two chronological route replays:

1. reset detector/head/map state at the route boundary;
2. replay the complete clean route in frame order and export clean outputs;
3. reset all state again;
4. replay the same route with the preregistered corruption schedule and export
   observed outputs;
5. join only exact `(folder, frame_idx)` pairs with matching config, camera
   order, transforms, GT, and declared temporal-history policy.

The branch-specific `temporal_history_id` must hash the route, frame prefix,
clean/corrupt branch, corruption schedule, checkpoint, and inference config.
The pair builder may require different history hashes but must verify that the
two branches represent the same frame prefix and the declared intervention.
Snapshotting and restoring private memory tensors per frame is deferred until
it is shown equivalent to replay.

The info file appears grouped by folder, and the head also refreshes memory on
`scene_token` change, but the exporter must independently check strictly
increasing unique `frame_idx` values within every selected folder.  It must not
rely on dataloader batch position as identity.

## 4. Image geometry and corruption point

The evaluation pipeline first applies `ResizeCropFlipRotImage`, then a 640 x
640 resize, normalization, and padding.  Both resize stages update
`cam_intrinsic`/`lidar2img`.  Object support must use the final matrices and
processed image shape from `img_metas`; raw 1600 x 900 intrinsics cannot index
the 40 x 40 EVAViT grid.

The existing local corruption functions explicitly operate on normalized
network tensors and use the configured ImageNet mean/std to implement dark,
glare, occlusion, and dropout in RGB space.  Applying the intervention after
the deterministic geometry transforms preserves clean/corrupt projection
identity.  The exact normalized region, realized pixel box, camera names,
seed, and patch coverage must be stored.

## 5. Traffic-state GT gap in the current agent test config

`pts_bbox_head` declares `pred_traffic_light_state=True`, and raw annotations
contain light state and `affects_ego`.  However, the current
`orion_stage3_agent.py` test pipeline calls `LoadAnnotations3D` without
`with_light_state=True`.  Consequently the normal agent test batch is not a
valid source for traffic-state target labels.

The target-export config must explicitly enable light-state loading and assert
that `traffic_state` has shape `[N,2]` with a matching boolean mask after all
box filters.  If that contract is absent, traffic-state component validity must
be false; it must not be zero-filled.  This is an exporter-config issue, not a
reason to alter closed-loop results or the deployed agent.

This audit also found and fixed a formatter overwrite: after
`gt_bboxes_3d_mask`, the previous code assigned the filtered boolean
`traffic_state_mask` into `traffic_state`.  State and validity are now checked
and filtered together from their original object axis, with a dependency-light
regression test. The full runtime must still attest `[N,2]` state and `[N]`
validity on every frame.

## 6. First real smoke and stop rules

### 6.1 Occupancy rasterizer parity caveats

`PlanningMetric.get_birds_eye_view_label` is the repository reference for the
200 x 200, 0.5 m, six-step GT occupancy convention.  It converts cumulative
future offsets and yaw into the current LiDAR frame and rasterizes vehicle and
human polygons.  The exporter should match this convention on cloned fixtures
before extending it to predicted occupancy.

The reference converts `gt_agent_boxes.tensor.cpu().numpy()` and then edits yaw
in place.  On a CPU tensor NumPy can share storage, so the exporter must pass a
clone or use a side-effect-free reimplementation; target matching must never
consume boxes after an unnoticed rasterizer mutation.  The reference's
category grouping also includes class index 3 (bicycle) in its `human` layer.
Preserve that behavior only for parity reporting and record any later
safety-class correction as a new rasterizer version.

Predicted occupancy is not a native ORION head output.  It must be described as
a deterministic rasterization of the decoded current box plus the
ORION-selected trajectory mode, with the same bounds, yaw convention, future
horizon, and validity mask.  Oracle-best trajectory occupancy is diagnostic
only.

### 6.2 Resource-bounded smoke

The first A800 smoke should use one complete short route or a bounded
chronological prefix, one local corruption family, one severity, and both
branches.  It must save only small decoded/target records plus a bounded number
of feature tensors and overlays.

The selected smoke is now fixed to Town04/Route214 frames 0--63: 64 clean and
64 observed forwards, with 43 measurement pairs persisted and 21 warm-up
frames per branch forwarded but not saved. Shared-data preflight verified all
64 info states, 384 camera images and 64 annotation files. The persisted plan
remains `execution_ready=false` until the real adapters and runtime geometry /
traffic attestations are connected.

Proceed past G1 only if:

- raw-to-decoded results reproduce the repository decoder/evaluator within a
  recorded tolerance;
- ORION-selected trajectory modes are verified against stored mode scores;
- clean/observed memory histories and identities pass the pairing audit;
- all six camera projections align with processed RGB overlays;
- traffic-state labels are either correctly loaded or explicitly invalid;
- invalid or invisible patches are masked, not labeled as certain/zero-error.

Stop before training if any decoder-index, temporal-memory, camera-order,
projection, or label-validity check fails.

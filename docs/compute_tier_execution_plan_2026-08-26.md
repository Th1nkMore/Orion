# ORION uncertainty research compute-tier execution plan

Date: 2026-08-26 CST

## 1. Objective

The A800 queue must not block every adapter iteration.  Work is split into three
tiers so that the cluster is used only for operations that actually require
ORION, CARLA, or large host memory.  Precomputed feature grids are the boundary
between the expensive and portable parts of the pipeline.

This plan does not authorize Stage B, actual-target training, or a full
closed-loop matrix.  The existing Teacher gate and research stop conditions
remain in force.

## 2. Current measured constraints

- Slurm job `1062977` was scheduled early and completed successfully on `gpu5`
  at 2026-08-26 18:16:38 CST.  It ran for 15:49; final accounting recorded
  8 CPU, 64 GiB requested, and about 21.5 GiB batch MaxRSS.  The v3.1 Teacher
  failed all six frozen diagnostic checks, so the conditional adapter did not
  run.
- Full ORION initialization has previously required about 192 GiB host RAM.
  Neither the 32 GiB desktop nor the 16 GiB Mac can run that path reliably.
- The immutable v3.1 input shard is 20 GiB and contains 720 clean plus 320
  diagnostic observations, with feature shape `[6,40,40,1024]` in FP16.
- The v3.1 Teacher has about 0.31 M trainable parameters and one adapter has
  about 0.21 M.  With batch 8, raw FP32 current and previous feature tensors
  occupy about 0.59 GiB before activations.  Therefore 16 GiB VRAM is expected
  to be sufficient; host-side loading of the monolithic 20 GiB shard is the
  immediate desktop risk.

## 3. Fixed division of work

| Resource | Primary role | Allowed examples | Not a target |
|---|---|---|---|
| Mac, 16 GiB unified memory | development and evidence processing | unit tests, 8--64 frame smoke, metric aggregation, plots, GIF/BEV rendering, manifest/hash verification, config generation | full 20 GiB training, ORION, CARLA closed loop |
| Desktop, RTX 4060 Ti 16 GiB + 32 GiB RAM | portable feature-level experiments | full Teacher and adapter training, ablations, held-out-family evaluation, calibration, checkpoint selection; optionally isolated visual-backbone extraction after a separate feasibility gate | full ORION load or official closed-loop CARLA |
| A800 cluster | irreducible model/simulator work | ORION feature extraction, visual-backbone checkpoint export, actual-target diagnostics, native-engine weather/sensor data generation, later ORION/VLM tuning, bounded CARLA closed loop | repeated small-head training or idle GPU holding |

## 4. Immediate preparation after job 1062977

### 4.1 Portable feature store v2

Keep the current 20 GiB v1 shard immutable as the completed job `1062977`
input.  For all new runs, replace the monolithic `torch.save` payload with route- or chunk-sharded
FP16 tensors plus a small JSON manifest:

- target chunk size 256--768 MiB;
- lazy load only the routes needed by the current batch/evaluation split;
- preserve `sample_id`, route, frame, family, severity, mask, and previous-frame
  links in the manifest;
- record SHA256 per chunk and for the manifest;
- make train/validation/held-out ownership auditable without loading tensors;
- add deterministic sampling and resume tests.

This change is infrastructure only.  It must not alter the v3.1 labels, split,
gate thresholds, or model architecture.  A parity check will compare v1 and v2
sample tensors and a fixed-seed smoke result.

### 4.2 Desktop execution bundle

Prepare a minimal bundle containing the relevant source, frozen config, split
manifest, environment lock, launch command, and expected hashes.  The desktop
runner should support:

1. CUDA/environment preflight;
2. a 32--64 clean-frame forward/backward smoke;
3. one full epoch with peak CPU RAM/VRAM and ETA reporting;
4. progress checkpoint and resume;
5. the unchanged 24-epoch Teacher gate;
6. conditional clean-only adapter continuation only when the Teacher passes.

The full 20 GiB feature payload is copied separately and verified against
`ab8d16ce9ffe67aba192ae331b102bcda8ccf917b4bbe491e86e452367b5beac`.

### 4.3 A800-to-desktop interface

Use A800 allocations to export reusable artifacts rather than one-off metrics:

- clean and native-degradation feature chunks;
- route/sensor/event metadata and corruption-free split manifests;
- frozen Teacher/adapter checkpoints and reports;
- an isolated EVAViT/visual-encoder checkpoint if it can be exported without
  loading full ORION on the desktop;
- front-view/BEV traces needed for later evidence rendering.

Only small checkpoints, reports, and newly generated chunks should move back to
the cluster.  Full ORION weights and CARLA assets remain in shared storage.

## 5. 4060 Ti feasibility and run sequence

The desktop becomes an official feature-level runner only after these checks:

1. the same sample IDs and tensors are read from the portable shard;
2. CPU RAM stays below a safe ceiling (target at most 26 GiB on a 32 GiB host);
3. a batch-8 smoke fits in 16 GiB VRAM; otherwise use batch 4 and explicitly
   record the changed optimization configuration rather than silently claiming
   exact replication;
4. fixed-checkpoint evaluation metrics match CPU/A800 results within numerical
   tolerance;
5. checkpoint resume produces the same next-step behavior.

After the feasibility gate, run the same v3.1 Teacher first.  Do not start a
hyperparameter sweep merely because the desktop is available.  If the Teacher
passes the frozen gate, continue with the clean-only adapter.  If it fails,
analyze the Teacher and modify the observation model before training adapters.

Job `1062977` has already completed, so there is no pending duplicate to
cancel.  Desktop parity now uses its frozen checkpoint/report as the reference.

## 6. Optional visual-backbone offload gate

In a future A800 allocation, export only the visual backbone and the minimal
preprocessing needed to produce the `[6,40,40,1024]` grids.  Test one route on
the 4060 Ti:

- if the isolated backbone fits in 16 GiB VRAM and the checkpoint/preprocessing
  fits comfortably in 32 GiB RAM, synthetic offline degradation generation can
  also move to the desktop;
- if it does not fit or its features differ from the ORION path, retain all
  extraction on A800 and use the desktop only after features are cached.

This feasibility test does not support a model-independence claim.

## 7. How to use future A800 allocations

Submit one legitimate, checkpointed work package rather than an idle shell or
many speculative jobs.  Each package should:

- request the minimum CPU needed by its real workload;
- request at least 210--220 GiB host RAM for paths that fully initialize ORION,
  based on the observed 192 GiB requirement;
- place reusable caches in `/public/share/lidachuan`;
- checkpoint at route/epoch boundaries and be safely resumable;
- run pre-fixed gates in allocation and exit early on failure;
- never expand into Stage B without an explicit decision after Stage A/oracle
  evidence.

CARLA closed loop, native-engine event capture, and later ORION/VLM fine-tuning
remain cluster tasks.  Teacher/adapter repetitions should not return to A800
once desktop parity is established.

## 8. Decision order

1. Preserve job `1062977` as a failed-but-valid Teacher diagnostic and do not
   train its adapter.
2. Implement portable loading and establish the 4060 Ti smoke/memory profile.
3. Compare generator-independent temporal, cross-view, and clean-calibrated
   residual signals on the existing shard before choosing a new adapter target.
4. Use the comparison to decide the next observation-model change or
   native-engine feature extraction package.
5. Only after uncertainty prediction is credible, return to the separated
   oracle/learned closed-loop questions; Stage B remains frozen until then.

## 9. 2026-08-27 execution update

The A800 candidate-signal sequence is complete through the first post-freeze
cross-family screen.  Clean-calibrated temporal self-consistency passed all
aggregate and route-robust development checks on `local_glare`.  On the frozen,
previously unseen `local_blur` family (job `1064728`), it passed AUROC, uplift,
route-shift, per-route, previous-valid-only, and localized severity checks, but
failed the pre-registered aggregate severity-Spearman threshold on both splits
(`0.09595/0.06672 < 0.10`).  The adapter therefore remains unauthorized.

This narrows the next A800 package.  Do not spend another allocation on a sweep
of pixel-space corruption operators.  Reuse the existing CARLA/ORION runtime to
produce a bounded native-engine weather or sensor-event observation set, with
front and BEV evidence, event timing, reusable visual features, and a frozen
temporal-score report.  Full ORION paths must continue to request at least
210--220 GiB host RAM.  The package must exit after the native signal gate and
must not conditionally launch adapter training or Stage B.

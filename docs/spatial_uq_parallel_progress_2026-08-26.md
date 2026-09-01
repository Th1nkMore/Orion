# Spatial UQ parallel work log

> Started: 2026-08-26 (Asia/Shanghai)
> Status: active, living document
> Scope: uncertainty extraction, oracle behavior adaptation, and gated
> closed-loop validation. Diffusion decoding is deliberately deferred.

## 1. Research objective

Build a two-stage uncertainty-aware driving system without using ADE as the
main claim:

1. train and validate a spatial signal that estimates where frozen ORION
   perception is likely to be wrong;
2. train a bounded behavior adapter to respond to path-relevant uncertainty;
3. show in closed loop that the learned signal approaches an oracle response,
   lowers collision/violation risk, preserves route completion, and does not
   reduce to constant slowing.

The following quantities remain separate throughout the code and paper:

```text
spatial perception failure -> fixed path-risk aggregation -> behavior response
```

Density UQ remains a global anomaly/OOD baseline.  A corruption mask is only
auxiliary supervision.  A paired feature residual is only a representation
proxy.  No current result supports “the LLM understands uncertainty.”

## 2. Literature-backed decisions

The current contract is based on heteroscedastic aleatoric uncertainty,
ensemble disagreement, held-out calibration, spatial error/OOD evaluation,
probabilistic object detection, clean/adverse correspondences, corruption
robustness, and selective risk/coverage.  Primary sources and the exact claim
boundary are recorded in `docs/spatial_uq_two_stage_v1.md`.

Key decisions:

- predict local task error rather than weather, route, or corruption identity;
- keep aleatoric error scale, head-level epistemic disagreement, path risk,
  and behavior intervention distinct;
- calibrate only on a route-disjoint calibration split;
- report actual task failure and representation proxy separately;
- require on-path versus equal-area off-path counterfactuals;
- require oracle success before testing learned-UQ closed-loop claims;
- defer diffusion until these gates pass.

## 3. Completed parallel work

### 3.1 Spatial UQ primitives and Stage-1 training

Implemented:

- versioned per-view/per-patch `SpatialUQOutput`;
- heteroscedastic Gaussian NLL with bounded log variance;
- failure Brier loss and measured-error-only ranking;
- three-head ensemble variance decomposition and single-pass distillation;
- exact local blur/dark/glare/occlusion masks;
- fixed route projection and auditable top-q/CVaR path risk;
- spatial, calibration, risk-coverage, on/off-path, and temporal metrics;
- route-disjoint teacher/student training CLI and mock smoke;
- independent calibration/evaluation CLI that never fits on validation or
  held-out routes and never pools actual/proxy targets.

### 3.2 Oracle behavior adapter

Implemented:

- same-rollout oracle dataset exporter;
- six-step local displacement coordinate contract;
- identity-at-zero-risk, bounded trajectory residual adapter;
- strict training-role gate: only eligible, completed, zero-collision,
  on-path oracle controlled-stop rollouts provide imitation targets;
- failed oracle and off collision rollouts are diagnostic-only with zero loss
  weight;
- route-disjoint trainer, single-route smoke disclosure, checkpoint and audit
  summary.

### 3.3 Closed-loop evidence and held-out screen

Verified route-146 mechanism evidence:

- transient corrupt/off: 100% completion, one pedestrian collision, score 50;
- 1.5 m/s-floor oracle: still one collision, score 50;
- controlled-stop oracle: 100% completion, zero collision, score 100, with an
  8.85 s duration cost relative to corrupt/off.

This proves only that an accurate oracle signal plus a sufficiently strong
bounded response can improve this exploratory case.

The current held-out screening pool is routes 147, 151, 194, 157, 164, 168,
with 180 and 208 as reserves.  These routes are held out from mechanism
development, not proven absent from ORION pretraining.

### 3.4 Real EVAViT paired-feature smoke

Slurm job `1061171` used one A800, four CPU cores, and 128 GB requested host
memory.  The extractor completed successfully; the Slurm allocation was marked
failed only because the final one-line read-back command had a shell-quoting
error.  The artifact was independently reloaded on the login node.

Verified artifact:

```text
/public/share/lidachuan/orion_assets/spatial_uq_v1/smoke/
paired-real-1061171.pt
```

- two real records;
- tensor shape `[6 cameras, 1600 patches, 1024 dims]`;
- 157,368,832 bytes;
- corruption index 0 was resolved from annotation order as `CAM_FRONT`;
- provenance is only `paired_cosine_representation_error_proxy`;
- semantic-UQ, safety, and LLM-understanding claims are explicitly false;
- maximum observed host RSS was about 20.7 GiB.

The storage result is decisive: naive float32 caching of four corruptions and
three severities for all 12,806 installed frames would be roughly 11 TiB.
Bulk extraction in this format is prohibited.  Full training must generate
pairs online or use audited, deduplicated, lower-precision shards.

### 3.5 Target schema, object-target primitives, and route manifest

The three integration branches are now implemented and cross-checked:

- paired records/checkpoints/evaluation use schema v2, with continuous
  `error_severity_target`, a separate explicit `failure_event_target`, and
  independent observed/clean valid masks;
- representation-proxy records have severity supervision only and cannot
  enter Brier calibration or failure-probability claims;
- named object-error components retain miss, class, localization,
  ORION-selected-mode motion occupancy, and valid traffic state separately;
- class/score/distance-gated one-to-one matching and projected visible-object
  support produce auditable `[view, patch]` targets while explicitly marking
  patch attribution as non-causal;
- `E_obs`, `E_clean`, and `relu(E_obs-E_clean)` remain separate;
- weather/repetition siblings are grouped by canonical `Town/RouteN` before
  the route split, while persisted route IDs remain compatible with existing
  extracted records.

The real exploratory manifest is persisted at:

```text
/public/share/lidachuan/orion_assets/spatial_uq_v1/manifests/
b2d_val_exploratory_seed20260826.json
```

It covers 12,806 frames and 50 canonical routes: 35 train, 5 validation,
5 calibration, and 5 held-out routes.  Held-out towns are Town02 and Town05;
the leakage audit passes.  The source info SHA-256 is
`d31e151fbc1854ccc8b7288445f3585d1f6dcf660f08bbfb90f71a5660943798`.
This is a split *within the installed validation subset*, not evidence that
any route was absent from ORION pretraining or an untouched official test set.

### 3.6 Chronological actual-target pilot manifest

The bounded pilot submanifest is persisted locally and on the shared server:

```text
configs/spatial_uq_route_manifests/
b2d_val_exploratory_pilot10_seed20260826.json

/public/share/lidachuan/orion_assets/spatial_uq_v1/manifests/
b2d_val_exploratory_pilot10_seed20260826.json
```

Its local SHA-256 is
`e7fd36663901f4a85485afb13afd97bf81dbc78665fdf517878e131f55cb4270`.
It selects 10 canonical routes and 900 measurement states, balanced as 450
annotation candidates and 450 backgrounds across all four parent splits.
These are measurement states, not independent model calls.  ORION temporal
memory requires replaying every selected folder contiguously from frame 0,
resetting between folders and between clean/observed branches.  The selected
folders contain 2,301 replay frames, so clean plus one observed condition
requires at least 4,602 forwards.  No frame-independent target generation is
permitted.

The shortest selected folder is calibration route Town04/Route214 with 136
frames.  It is the preferred engineering smoke candidate; using it does not
turn calibration data into training data.  Town12 routes are acceptable for
offline recorded-data export even though the current CARLA runtime cannot run
Town12 in closed loop.

### 3.7 Decoded actual-target bundle and CPU smoke

The dependency-light exporter now consumes decoded frozen-ORION outputs,
privileged B2D targets, post-augmentation projected supports, pairwise BEV IoU,
and route chronology without importing ORION/MMCV/CARLA.  It preserves:

- source query indices from flattened query-by-class top-k, including legal
  duplicate queries;
- complete class sigmoid vectors and traffic logits;
- all trajectory modes/scores and the explicit ORION-selected mode;
- GT and selected-mode occupancy, validity masks, and rasterizer identity;
- separate clean/observed history-content IDs plus a shared replay protocol;
- continuous `E_obs/E_clean/DeltaE`, binary events, component maps, and both GT
  and unmatched-prediction projected support.

Failure events are generated at the object level first.  Six component
thresholds (miss, class, localization, selected-mode occupancy, traffic state,
and false positive) produce hard object events; those events are then
projected with an independent 0.01 minimum visible-support gate and ORed.  The
continuous patch severity remains a support-weighted noisy-OR and is not used
as an event probability.  A CPU fixture verifies both a 0.10-support missed
object and a 0.10-support false positive remain event-positive even though
their continuous patch severities are below 0.50.

Bundle validation reconstructs severity, every component map, and the event
map from stored object values/support and rejects tampering.  The mock CLI
reports `real_orion_hook_executed=false`; it validates only schema, arithmetic,
serialization, and bridge wiring, and therefore does not pass G1.

The final-head decode adapter now reproduces `CustomNMSFreeCoder`'s exact
sigmoid query-by-class top-k, box denormalization, post-center filter and
adaptive score-threshold order while retaining otherwise dropped source-query
indices and motion-mode scores.  It keeps full decoded boxes as well as the
match gates/trace and requires a caller-owned, versioned occupancy rasterizer;
the CPU fixture cannot be presented as PlanningMetric parity.

The first real replay has been reduced further to the frame-0-through-63
prefix of Town04/Route214.  Shared-data preflight verified 64 contiguous info
states, 384/384 six-camera images, 64/64 annotations, source-info SHA, and
camera insertion order.  It requires 64 clean plus 64 observed forwards and
persists only 43 measurement pairs (20 annotation candidates, 23 background);
21 warm-up frames per branch are forwarded but not saved.  The persisted plan
still reports `execution_ready=false`, `g1_passed=false`, and no job submitted.
The amended post-fix plan is
`/public/share/lidachuan/orion_assets/spatial_uq_v1/manifests/route214_prefix63_replay_plan_v2.json`
with SHA-256
`904c9d3187373bc44937d33c98f84ff406c7bec9b7d6467bbf9fa5e0d0de0538`;
the original plan was preserved rather than silently rewritten.

Preflight also exposed a real traffic-label formatter bug: filtered
`traffic_state_mask` was being written into `traffic_state`.  The local code
now validates and filters `[N,2]` state and `[N]` validity together from the
same original GT-box mask, with a dependency-light regression test.  A
dedicated exporter pipeline must still enable `with_light_state=True`, and
per-frame runtime shape/alignment attestation remains required.

## 4. Actual perception-failure target decision

The preferred v1 task target is frozen ORION object/motion-occupancy error
against privileged Bench2Drive ground truth, attributed to visible image
patches.

```text
E_obs   = frozen ORION task error on the observation shown to the UQ head
E_clean = frozen ORION task error on the paired clean observation
DeltaE  = relu(E_obs - E_clean)
```

`E_obs` is the primary target.  `E_clean` is not forced to zero.  `DeltaE` is
for corruption attribution and ranking, not a replacement for real clean
failures.  Object components are miss, class, localization, motion occupancy,
valid traffic-light state, and unmatched-prediction false positive.  The error
magnitude is task-grounded; its projection onto visible GT/predicted-object
patches remains an attribution proxy.

The installed server subset has 12,806 frames from 50 route folders with 3D
boxes/motion and aligned RGB/semantic/instance/depth views.  It can support an
exploratory route-disjoint pilot.  No `b2d_infos_train.pkl` is currently
installed, so it cannot support an untouched official-validation claim.

## 5. Verification ledger

Latest consolidated local verification:

```text
202 related tests passed
1 warning: PyTorch TypedStorage deprecation only
closed-loop mechanism verifier: verified=true
git diff --check on the new spatial-UQ work: clean
```

The v2 extractor/trainer/evaluator mock CLIs pass end to end.  The proxy
extractor emits no failure-event target, and the evaluator reports no proxy
Brier/calibration metrics.  Forty-eight schema/target/extraction tests also
pass in the server's Python 3.8 environment.  The previous real A800 v1 proxy
artifact reloads through an explicit v1-to-v2 proxy-only migration; no actual
target is inferred.

No user Slurm job is currently queued or running.

## 6. Completed integration checkpoint

The completed parallel branches are:

1. **Target schema v2:** separate error severity, failure-event probability,
   valid mask, clean targets, and named component errors; migrate v1 proxy
   records without silently upgrading their claim.
2. **Route split builder:** canonicalize route/weather/repetition identities
   and create train/validation/calibration/held-out manifests with lineage
   audit and fail-closed leakage checks.
3. **Object failure targets:** implement dependency-light per-object error,
   class-aware distance-gated matching, visible patch support/noisy-OR,
   `E_obs/E_clean/DeltaE`, valid masks, and optional BEV sidecars.
4. **Actual-target decode boundary:** retain complete boxes, match gates,
   flattened-top-k source queries, class/state/motion logits, selected-mode
   occupancy inputs and strict clean/observed chronology.
5. **Bounded replay plan:** preflight Route214 frames 0--63 and require an
   exhaustive per-frame G1 attestation before any result can pass.
6. **Traffic-state alignment:** fix and test the state/mask overwrite in the
   formatter while retaining a runtime alignment check; carry raw actor IDs
   through the same range/name/final filters so support order is auditable.
7. **Production geometry:** implement side-effect-free PlanningMetric GT
   occupancy rasterization, selected-mode predicted occupancy, continuous
   rotated pairwise BEV IoU, and six-view post-augmentation visible-support
   projection with distinct GT-bottom/predicted-center z origins.
8. **Frozen perception runner:** connect the perception-only ORION path
   (`extract_feat -> position embedding -> pts_bbox_head`), exact final-head
   decode, temporal reset/chronology checks, and EVAViT `[6,1600,C]` patch
   features while excluding LLM/VAE/diffusion execution.
9. **Concrete target builder:** synchronize the GT object axes, filter the
   preregistered safety actors, construct real branch bundles, preserve
   distinct GT/predicted rasterizer IDs, and mask invalid traffic state rather
   than treating it as a negative label.

No GPU job was submitted for these branches.

## 7. Active work and next gates

The next active unit is the single-prefix real actual-target smoke, not
Stage-1 training. Event-label policy, decoded bundle, temporal replay plan,
traffic/actor alignment, frozen perception hook, BEV geometry, projected
support, and the concrete target builder are implemented. The remaining
launch gate is a fail-closed production integration/record sink plus real
Route214 projection evidence; after that, run only the bounded A800 prefix.

Planned implementation order:

1. complete the production launcher, deterministic corruption transform,
   atomic record sink, and non-placeholder QA-evidence callbacks;
2. run the dedicated `with_light_state=True`, actor-ID-preserving dataset
   pipeline and real projection overlay preflight;
3. run and inspect one Route214-prefix A800 target-generation smoke;
4. inspect G1 overlays and G2/G3 induction/localization before any UQ-head
   training.

### Gate G0 — data lineage

- canonical route-disjoint manifest;
- weather/repetition variants cannot cross splits;
- explicit exploratory-validation limitation;
- no route/town/corruption metadata enters the UQ head.

### Gate G1 — target correctness

- exporter reproduces frozen ORION evaluator errors on a fixed smoke subset;
- occupancy rasterization matches repository fixtures;
- transformed camera geometry and 40 x 40 patch alignment pass overlay QA;
- invalid/unobserved patches are masked, never labeled zero.

### Gate G2 — failure induction/localization

- paired `DeltaE` increases on the corrupted safety-object region;
- on-path corruption is worse than matched off-path corruption;
- effects survive class/distance/town/weather stratification;
- visually dramatic corruptions that do not increase real task error are
  rejected.

### Gate G3/G4 — learning and calibration

- actual-error AUPRC/FPR95/Brier/NLL/AURC beat photometric, Density-UQ,
  corruption-mask-only, and feature-residual baselines;
- held-out corruption family and natural adverse weather remain valid;
- clean false triggers, onset, and recovery are bounded.

### Gate G5 — minimal closed loop

Only after the signal and oracle adapter pass separately, compare off,
matched constant, temporal/spatial shuffle, aligned learned UQ, and oracle on a
minimal route set.  Expand routes/seeds only if learned UQ beats the controls
and approaches oracle while preserving liveness and traffic rules.

## 8. Detailed companion documents

- `docs/spatial_uq_two_stage_v1.md` — literature and interface contract
- `docs/spatial_uq_execution_plan_2026-08-26.md` — execution order and gates
- `docs/stage1_actual_failure_target_design_2026-08-26.md` — actual target audit
- `docs/actual_target_orion_integration_audit_2026-08-26.md` — frozen ORION
  decoder, temporal-memory, geometry, and traffic-state integration audit
- `docs/decoded_actual_target_export_integration_2026-08-26.md` — decoded
  object/target bundle, event semantics, serialization, and claim boundary
- `docs/orion_decode_adapter_integration_2026-08-26.md` — exact final-head
  tensor and CustomNMSFreeCoder adapter contract
- `docs/orion_actual_target_replay_smoke_2026-08-26.md` — Route214 prefix
  preflight, runtime attestation, resource envelope, and stop conditions
- `docs/orion_actual_target_runner_2026-08-26.md` — real perception-only hook,
  pipeline mutation, actor/traffic alignment, and fail-closed execution
- `docs/orion_actual_target_builder_2026-08-26.md` — concrete GT/predicted
  branch-target construction and dual rasterizer provenance
- `docs/bev_target_rasterizer_v1.md` — GT/predicted occupancy and pairwise BEV
  IoU parity contract
- `docs/projected_visible_support_v1_2026-08-26.md` — six-view projected patch
  support and z-origin contract
- `docs/spatial_uq_real_feature_smoke_2026-08-26.md` — A800 smoke evidence
- `docs/spatial_uq_code_audit_2026-08-26.md` — legacy/new code audit
- `docs/closedloop_heldout_route_audit_2026-08-26.md` — route screening evidence
- `results/closedloop_uq_pilot/MECHANISM_PROOF_2026-08-26.md` — oracle mechanism result

## 9. Changelog

- **2026-08-26:** created the living log; consolidated literature decisions,
  implemented branches, 118-test ledger, A800 job 1061171, storage stop rule,
  actual-target decision, active schema/split/target branches, and next gates.
- **2026-08-26:** integrated target schema v2, object-failure primitives, and
  canonical route splitting; persisted the real exploratory manifest; raised
  the local ledger to 143 tests and remote Python 3.8 ledger to 48 tests; kept
  all Slurm work paused pending the actual-target exporter and G1-G3 gates.
- **2026-08-26:** completed the perception-only runner, actor-ID and traffic
  alignment, BEV raster/IoU parity primitives, post-augmentation six-view
  support projection, and concrete production branch builder; raised the
  consolidated local ledger to 202 tests and retained an empty Slurm queue
  pending the final production-launch/overlay gate.

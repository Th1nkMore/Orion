# Spatial-UQ two-stage execution plan (2026-08-26)

## Goal and claim ladder

The target claim is not that a corruption label improves open-loop trajectory
error.  The target claim is that a calibrated, spatially localized perception
failure signal can be related to the ego path and can trigger a bounded,
safer closed-loop response without causing indiscriminate slowing.

The evidence must be accumulated in this order:

1. **Signal validity:** the Stage-1 model localizes and calibrates actual or
   explicitly proxy-labelled perception failure on route-disjoint data.
2. **Behavior upper bound:** an oracle path-risk signal can teach or trigger a
   safe response on failure-inducing scenarios while preserving clean and
   off-path behavior.
3. **Composition:** frozen learned spatial UQ, the fixed path projection, and
   the behavior adapter approach the oracle result on routes that were not used
   to choose the mechanism.
4. **Only then:** a training-route-only closed-loop fine-tuning phase may
   optimize collision, TTC, completion, liveness, and traffic-rule penalties.

No result below step 3 supports “the LLM understands uncertainty.”  A paired
feature-residual target supports only a spatial frozen-representation-error
claim.  Density UQ remains a global anomaly baseline.

Diffusion decoding is intentionally outside this execution plan.  It can be
reconsidered only after the spatial signal and response mechanism pass their
separate gates.

## Parallel workstreams

### A. Stage-1 spatial uncertainty

- Extract clean/corrupt EVAViT feature pairs from the same source frame.
- Preserve the view and patch axes; do not globally pool the tokens.
- Use local blur, darkening, glare, and occlusion with exact masks.  Camera
  dropout is diagnostic-only.
- Prefer a task-grounded local failure target when it is available.  Otherwise
  label `1 - cosine(clean_feature, corrupt_feature)` as a representation proxy
  in every record and checkpoint.
- Train an independent-head ensemble and distill its epistemic variance to a
  single-pass student.
- Fit calibration parameters on the calibration split only.  Report actual and
  proxy targets separately on the held-out split.

Signal gate:

- route-disjoint split with no repetition leakage;
- improved AP/AUROC/FPR95, Brier/ECE, and risk-coverage versus global Density
  UQ and corruption-energy controls;
- positive on-path versus matched equal-area off-path contrast;
- timely event onset and recovery, with bounded clean false triggers;
- monotonicity is credited only where the measured target error itself rises.

### B. Oracle behavior adapter

- Export only same-rollout future trajectories.
- Positive imitation comes only from eligible, completed, zero-collision,
  hazard/on-path oracle controlled-stop rollouts.
- Failed oracle and off-policy collision rollouts remain diagnostic-only and
  have zero trajectory and stop loss weight.
- Risk-inactive, clean, no-hazard, and off-path frames preserve the frozen ORION
  base trajectory.
- The adapter is an exact trajectory identity at zero path risk and has a
  bounded residual at nonzero risk.

Behavior gate:

- at least two successful oracle routes before a route-disjoint training claim;
- oracle improves collision/TTC without blocked, timeout, route-completion, or
  traffic-rule regressions;
- no-hazard and off-path intervention stays near zero;
- a post-event recovery interval is present and does not remain stopped.

### C. Closed-loop scenario protocol

The screening pool is frozen in
`configs/closedloop_uq_pilot/heldout_route_candidates_v1.json`.  Routes 147,
151, and 194 are the first choices; 157, 164, and 168 are secondary; 180 and
208 are reserves.  This pool is held out from mechanism development, not proven
held out from ORION pretraining.

For each route, run conditions sequentially behind hard gates:

1. `clean_off` only.  Stop the route if the current baseline is invalid.
2. Freeze a short route-progress event window from the clean trace before
   looking at corrupt outcomes.
3. Run `corrupt_on_path_off` and matched `corrupt_off_path_off` only.
4. Proceed only if on-path corruption is reproducibly worse than both clean and
   off-path corruption.
5. Run the fixed oracle through the same adapter/controller interface.
6. Proceed to learned UQ only if the oracle improves safety and liveness.
7. Run learned, constant, and time-shuffled controls only as the final minimal
   causal comparison; do not launch a full Stage-B matrix.

Primary closed-loop endpoints are collision, near-collision/TTC, route
completion, blocked/timeout, traffic-rule violations, intervention magnitude,
and recovery.  ADE is diagnostic-only.

## Composition decision table

| Observation | Decision |
|---|---|
| Oracle cannot improve safety/liveness | Redesign the response or scenario; do not tune learned UQ around it. |
| Oracle works; learned signal fails offline | Improve Stage-1 target/model/calibration; do not spend on the learned closed-loop matrix. |
| Offline signal works; learned composition fails while oracle works | Debug path projection, latency, scale calibration, or adapter interface. |
| Learned equals constant slowing | Reject spatial/timing-value claim. |
| Learned beats off, constant, and shuffled and approaches oracle | Expand to multiple routes/seeds and no-hazard controls. |
| Clean and on-path corrupt do not differ | Failure induction failed; do not advance that route. |

## Resource order

CPU work (parallel and cheap): schema tests, route/data lineage, corruption
generation checks, exporter validation, metrics, calibration, and visualization.

A800 work (one short job at a time until gates pass):

1. frozen-backbone paired-feature extraction on a tiny real subset;
2. Stage-1 head training/calibration smoke;
3. one clean closed-loop route;
4. its matched on-path/off-path failure-induction pair;
5. oracle, then learned minimal comparison only after the preceding gates.

Full ORION/CARLA jobs request at least 192 GB host memory.  Backbone-only
extraction should be measured separately rather than inheriting that request
blindly.  Additional CPU cores may be requested for data loading only after GPU
utilization or loader timing shows a CPU bottleneck.

## Current status

- The versioned spatial UQ output, route projection, CVaR path-risk aggregation,
  corruptions, metrics, and Stage-1 teacher/student trainer are implemented.
- The oracle dataset exporter, strict role assignment, bounded trajectory
  adapter, and standalone trainer are implemented.
- Local tests and mock checkpoints validate pipeline semantics only; they are
  not model-quality or safety results.
- A real A800 smoke extracted two frozen-EVAViT paired records with shape
  `[6, 1600, 1024]`.  It also showed that the current float32 record format is
  suitable only for bounded debug shards: duplicating clean tokens for every
  corruption would exceed the shared quota at scale.
- The preferred actual-error target is now frozen conceptually as frozen
  ORION object/motion-occupancy error against Bench2Drive GT, projected to
  visible patches.  The error magnitude is task-grounded; the image-patch
  location remains an attribution proxy.
- Before actual-target training, the record contract must separate error
  severity, failure event, valid mask, and component errors, and the data must
  use a canonical route-disjoint manifest.
- Route 146 establishes an exploratory oracle mechanism upper bound for a
  transient complete front-camera loss.  It does not validate learned UQ.
- No new Stage-B or CARLA matrix job is authorized by this plan.

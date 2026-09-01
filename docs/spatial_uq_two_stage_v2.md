# Spatial observation uncertainty and risk-aware ORION: v2 contract

> Status: active architecture contract, amended 2026-08-29 after the
> Route197 native-collision diagnostics
> Replaces: `docs/spatial_uq_two_stage_v1.md`
> Main change: the Stage-1 adapter is strictly task-agnostic. Route relevance,
> path risk, and conservative behavior are learned inside the fine-tuned
> ORION/VLM rather than supplied by a fixed learned-risk label or scalar
> governor.
>
> Current execution status, exact checkpoint lineage, passed/failed gates, and
> next runnable milestone are maintained in [`docs/CURRENT_STATE.md`](CURRENT_STATE.md).
> This file owns architectural responsibility boundaries, not mutable run
> status.

## 1. Claim and module boundary

The system contains two learned stages with different semantics:

```text
multi-view observations + temporal context
                    |
                    v
       Stage-1 observation-UQ adapter
       U: [B,V,H,W,K] + compact UQ tokens
                    |
                    |  no route, actor, hazard, action, or collision label
                    v
visual tokens + UQ map/tokens + navigation + ego state
                    |
                    v
          Stage-2 fine-tuned ORION/VLM
     task relevance + risk-aware plan representation
                    |
                    v
       trajectory decoder (existing first;
       conditional diffusion is a later ablation)
```

Stage 1 answers: **where and how strongly is visual evidence unreliable?**

Stage 2 answers: **does that unreliability matter for this route and scene, and
what safe behavior should follow?**

Consequences:

- Equal-strength on-path and off-path degradation should both produce a
  localized Stage-1 response at their respective locations.
- Only Stage 2 should learn that on-path uncertainty often warrants yielding or
  slowing while matched off-path uncertainty often does not.
- A global mean score followed by a scalar speed governor was retained only as
  a mechanism diagnostic. Route197 development runs have now rejected both a
  rolling cap and a fixed-duration stop-and-hold: they delayed the collision
  and changed the colliding actor without reducing collision count. This path
  is closed; response learning moves into the trajectory/planning layer.
- The existing actual-target pipeline remains diagnostic/auxiliary supervision.
  It may test whether observation UQ predicts ORION degradation, but it is not
  the sole definition of uncertainty.

## 2. Stage-1 adapter training

### 2.1 Inputs and forbidden shortcuts

The adapter consumes current and previous multi-view visual features, view
identity, and temporal-validity state. It must not receive:

- route or navigation commands;
- actor/hazard identity or scenario class;
- planned trajectory or future collision;
- known corruption family as an input;
- action, braking, or governor state.

This protects the interpretation of its output as observation uncertainty
rather than a disguised task-risk map.

### 2.2 Supervision hierarchy

No synthetic corruption mask is called uncertainty ground truth. Training uses
several mutually constraining signals:

1. **Paired evidence-loss proxy (primary executable signal).** For the same
   simulator state, compare clean and degraded frozen visual features. Predict
   persistent direction change, magnitude change, and temporal inconsistency at
   each patch. The target is explicitly named representation/evidence loss.
2. **Clean temporal and multiview consistency.** Natural night, normal fog,
   shadows, and other valid observations are clean whenever they remain
   informative and consistent. Appearance alone is not a positive label.
3. **Counterfactual localization/equivariance.** Moving a local intervention
   should move the predicted UQ region without changing unrelated patches.
4. **Severity and recovery ordering.** Evidence loss should increase only when
   the measured feature/perception effect increases, and should decay promptly
   after evidence recovers.
5. **Corruption mask (auxiliary only).** The known region can stabilize early
   localization, but receives low weight and cannot pass the validation gate by
   itself.
6. **Actual ORION degradation (diagnostic/auxiliary).** Frozen-ORION prediction
   error, feature degradation, actor/traffic target failure, and later
   perception failures test practical relevance. They do not turn Stage 1 into
   a route-risk learner.

The current pairwise hurdle adapter is a valid pilot implementation of items
1--4. It is not yet a validated universal semantic-UQ estimator.

### 2.3 Data separation

Splits are by whole corruption family, route, Town, and native rendering
condition—not random frames alone.

- Training interventions: local blur, exposure/contrast/noise/compression and
  natural-texture occlusion, with deterministic temporal windows.
- Held-out synthetic/rendered tests: local glare, lens-like artifacts,
  intermittent frames, unseen severity, and spatial translations.
- Native tests: CARLA weather/time-of-day rerendering and sensor behavior not
  generated by the training augmentation function.
- External offline tests: adverse-condition data with clean/adverse
  correspondence where licensing and calibration permit.

Full-camera black dropout remains a pipeline sanity check only.

### 2.4 Stage-1 outputs and losses

The required output is a versioned spatial map and compact tokenization:

```text
uq_map              [B,V,H,W,K]
uq_tokens           [B,N_uq,D]
view/time metadata  [B,V,...]
```

The executable loss family is:

```text
L_stage1 = L_pairwise_evidence
         + lambda_clean * L_clean_false_positive
         + lambda_equiv * L_spatial_equivariance
         + lambda_time  * L_onset_recovery
         + lambda_rank  * L_measured_severity_order
         + lambda_mask  * L_mask_aux
         + lambda_diag  * L_actual_target_aux
```

Stage-1 success requires localization, temporal response, clean false-positive
control, and held-out-family/native-condition transfer. ADE is not a target.

## 3. Stage-2 risk-aware ORION/VLM training

Stage 2 resolves the chicken-and-egg problem with a curriculum. It does not
wait for a perfect learned UQ map, and it does not let collision gradients
silently redefine the Stage-1 signal.

### 3.1 Phase 2A: privileged/oracle-UQ behavior warm-start

Use the known spatial/time extent of simulator interventions and privileged
hazard state to provide an oracle observation-UQ map. Feed this map through the
same interface that learned UQ will later use.

Training tuples must contain matched examples:

- hazard + on-path uncertainty;
- hazard + equal-area off-path uncertainty;
- no hazard + on-path uncertainty;
- clean hazard and clean no-hazard;
- uncertainty that recovers before and after conflict resolution.

Targets combine privileged safe-expert/teacher trajectories with identity
preservation of the original ORION policy when no response is warranted. This
teaches the VLM what the signal means for driving before learned Stage-1 noise
is introduced.

The privileged teacher must label both **yield onset and release** from the
current conflict state. A fixed time window is not an acceptable substitute.
The Route197 sequential-cross-traffic failure showed that a controller can
stop correctly yet collide immediately after a time-based release. The policy
therefore re-plans a complete trajectory every frame and represents at least
`go`, `prepare_yield`, `hold`, and `release` task states.

### 3.2 Phase 2B: learned-UQ substitution and distillation

Freeze Stage 1 and replace oracle maps with frozen adapter predictions using a
scheduled mixture:

```text
oracle UQ -> oracle/learned mixture -> learned UQ
```

Distil the oracle-conditioned behavior while retaining matched off-path and
no-hazard negatives. Spatial and temporal shuffling are required controls.

The task policy may use route, visual content, ego state, actor context, and UQ
jointly. That is where task relevance is supposed to be learned.

### 3.3 Phase 2C: closed-loop fine-tuning

After the supervised warm-start is stable, use CARLA rollouts with a
constrained outcome objective:

```text
J = collision_cost
  + traffic_violation_cost
  + route_failure_cost
  + unnecessary_slowing_cost
  + prolonged_stop_cost
  + discomfort_cost
  - progress_reward
```

Practical choices are DAgger-style expert relabeling, offline/on-policy
advantage learning on a small policy adapter, or another constrained policy
optimization method. Start with LoRA/behavior adapters and a frozen visual/UQ
front end; full-model tuning is not the first experiment.

Collision and route rewards update Stage 2. They are stopped at the Stage-1 UQ
adapter boundary. If later joint tuning is attempted, retain the frozen Stage-1
checkpoint as an anchor and enforce its localization/clean/generalization
losses; otherwise the UQ map can collapse into an undocumented risk/action map.

## 4. Decoder strategy: VAE first, diffusion second

Replacing the existing trajectory decoder with conditional diffusion is
plausible and potentially useful because it can represent multiple safe
futures and allow UQ to reshape a trajectory distribution. It is not required
to establish the first causal result.

Execution order:

1. Prove the privileged task-risk and oracle/learned-UQ curriculum with the
   existing VAE decoder or a small identity-initialized trajectory adapter.
   The output must be a time-indexed trajectory, not a post-hoc throttle cap.
2. Freeze the scenario bank and evaluation protocol.
3. Add a conditional diffusion trajectory decoder as an architectural ablation,
   conditioned on ORION plan features, navigation, ego state, and UQ tokens.
4. Compare safety, route completion, latency, comfort, and compute against the
   existing decoder under the same intervention budget.

This prevents decoder capacity from being confused with the value of spatial
uncertainty. The VLM may also emit a visualizable task-risk/attention map, but
that map must be named separately from Stage-1 observation UQ.

The existing ORION code already contains VAE, MLP, and conditional-diffusion
trajectory paths. The first repair therefore does not require replacing VAE.
Diffusion becomes useful later for representing several safe modes such as
yield/go, but changing decoder family before the task-risk interface works
would confound the experiment.

## 5. Closed-loop scenario-bank gates

The primary event pattern is:

```text
clean ORION succeeds
        +
finite localized observation degradation
        -> baseline ORION fails or loses safety margin
```

For each selected route, freeze from clean evidence:

- one hazard-visible/conflict window;
- one on-path region;
- one equal-area, matched-luminance off-path region;
- hazard and no-hazard XML variants;
- family, severity, seed, and duration.

Run order is strictly gated:

1. clean replay;
2. one on-path failure-induction run with response off;
3. only if degradation is meaningful, off-path and no-hazard specificity
   controls;
4. Stage-1 spatial trace without control;
5. relevance-aware trajectory/planner oracle upper bound with state-dependent
   release;
6. Stage-2 learned ORION/VLM behavior;
7. only then constant, temporal-shuffle, spatial-shuffle, multi-route, and seed
   expansion.

Route197/Town05 is now the primary mechanism-development route because frozen
ORION reproducibly completes it with one crossing-vehicle collision. It is not
held-out evidence. Full-camera dropout and the Route197 fixed-window governor
runs are development history, not the scenario-bank main result.

## 6. Evidence required for the central claim

### Stage-1 evidence

- degraded-region versus outside-region localization contrast;
- comparable localization for matched on-path/off-path evidence loss;
- onset/recovery latency and severity ordering;
- clean/native night/weather false positives;
- held-out family/Town/route transfer;
- diagnostic prediction of frozen-ORION failure.

### Stage-2 evidence

- collision and traffic-violation reduction on induced-failure hazards;
- preserved route completion;
- unnecessary-stop time and completion-time cost;
- low response on matched off-path and no-hazard uncertainty;
- oracle, learned-aligned, no-UQ, global/constant, temporal-shuffle, and
  spatial-shuffle comparisons;
- safety--utility Pareto curves, not a collision-only table.

Only a result in which learned spatial UQ plus fine-tuned ORION approaches the
relevance-aware oracle while beating constant/global and shuffled controls
supports the intended uncertainty-aware VLM claim.

## 7. Immediate implementation state

- Native collision discovery produced valid 100%-complete collisions on
  Routes 196, 197, 201, and 210. Route197 is the primary development case.
- The frozen pairwise Stage-1 adapter checkpoint emits six 10 x 10 grids and
  was verified over all 471 Route197 frames with no control intervention and
  with the legacy Density estimator structurally absent.
- The Route197 adapter response was mixed: CAM_FRONT triggered 2.5 seconds
  before the independently anchored event, but CAM_BACK_RIGHT had the largest
  approach uplift and emphasized static facade/road regions. The result is a
  useful diagnostic, not clean task-relevant localization.
- The same checkpoint passed held-out local-glare ranking but failed the frozen
  heavy native-fog gate. It has no learned-control authority.
- Route197 clean/off collided with a Mini at 14.0 seconds. A 5-second,
  1.5-m/s task-event oracle avoided that actor but collided with a Mercedes at
  18.15 seconds. A single preregistered 9-second stop-and-hold repair then
  collided with an Audi at 22.3 seconds. Every run completed 100%, and the two
  oracle traces passed all window, signal-separation, and inactive-control
  contracts. Collision count and score remained 1 and 60.
- Fixed-window scalar governor tuning is closed. No additional duration,
  speed, constant, shuffled, or learned-control variants are authorized.

## 8. 2026-08-29 planning-layer execution amendment

### 8.1 Minimal Stage-2 interface

Stage 1 remains frozen and task-agnostic. Its multi-view component maps are
pooled or sparsified into tokens with camera, grid-position, component, and
time embeddings:

```text
frozen observation UQ [B,V,H,W,K]
             -> spatial UQ tokens [B,N_uq,D]
             -> cross-attend with ORION object/map/route/ego tokens
             -> task-risk tokens + yield state + trajectory features
             -> existing VAE trajectory decoder
```

Collision and planning gradients stop at the Stage-1 output. The Stage-2
fusion/projector, a small ORION/VLM LoRA or task adapter, and the trajectory
conditioning layers are trainable. An explicit task-risk/path map is logged
separately from the Stage-1 observation-UQ map.

### 8.2 Supervision and curriculum

Work proceeds in parallel on two datasets:

1. **Stage-1 independent observation supervision.** Expand clean/native
   temporal support and independently rendered adverse conditions; keep whole
   families, Towns, routes, and severities held out. Improve localization and
   native transfer before granting the adapter control authority.
2. **Stage-2 privileged task-response supervision.** In CARLA, use privileged
   actor boxes/futures, traffic state, route geometry, and a safe
   planner/expert to label time-indexed conflict occupancy, yield state, and a
   safe trajectory. Pair hazard/on-path, hazard/off-path, no-hazard/on-path,
   clean hazard, and recovery examples.

The supervised objective is:

```text
L_stage2 = lambda_plan     * L_safe_trajectory
         + lambda_state    * L_yield_state
         + lambda_conflict * L_future_actor_margin
         + lambda_rule     * L_traffic_and_lane
         + lambda_identity * L_no_response_identity
         + lambda_progress * L_progress
         + lambda_comfort  * L_acceleration_and_jerk
```

The collision term uses future oriented-box/occupancy geometry rather than
ADE as the central target. Matched off-path and no-hazard examples prevent a
trivial always-stop policy. DAgger-style relabeling then adds states reached by
the learned policy, including unsafe release states absent from nominal expert
data.

### 8.3 Next gates

1. Implement a dependency-light spatial-UQ token projector and task-risk/yield
   head with CPU shape, gradient-stop, identity, and serialization tests.
2. Build a privileged Route197 trajectory-label exporter that marks the
   sequential crossing conflict and release state without reading adapter
   output.
3. Show offline that the privileged planner-conditioned model produces a
   stopped/yielding trajectory while conflict remains and restores the frozen
   ORION trajectory when it clears.
4. Only then run one planner-level oracle closed loop on Route197. If it cannot
   remove the collision while completing the route, redesign the planner or
   scenario before learned-UQ substitution.
5. A successful Route197 result remains development-only; prospectively freeze
   a different route/seed before making a safety claim.

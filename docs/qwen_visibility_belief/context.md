# Qwen visibility-belief context

Last audited: 2026-09-06 (Asia/Shanghai)

## 1. Current objective

The current research direction is:

> Augment a pretrained driving VLA with an explicit, inspectable belief about
> what parts of the nearby scene cannot currently be observed, then let the
> VLM combine that evidence with RGB semantics, route, ego motion, and braking
> margin to produce safer closed-loop trajectories under occlusion.

The intended contribution is not generic semantic occupancy reconstruction.
Occupancy-style grids are the carrier for a task-agnostic physical visibility
belief. The research question is whether a pretrained driving VLM can consume
that belief and improve anticipatory behavior before a hidden road user is
revealed.

The accepted architecture and alternatives are recorded in [adr.md](adr.md).

## 2. Grill design-review trace

This section is the persistent design tree reconstructed from the interactive
grill review. It records decisions and their dependencies, not a verbatim chat
transcript. Experimental facts established after the review remain in the
evidence sections below and must not be presented as user decisions.

```text
D0  What problem are we solving?
|
+-- accepted: clean occlusion is itself scene uncertainty
|   +-- D1  What should the small estimator learn?
|   |   +-- rejected: hidden-actor probability, task action, or full semantic risk
|   |   +-- accepted: task-agnostic physical observability
|   |       +-- D2  How is near-term danger represented?
|   |           +-- accepted: keep U_vis physical and separately compute U_urgent
|   |               from route, speed, and stopping margin
|   |
|   +-- D3  What representation carries U?
|       +-- accepted: depth-derived 3D visibility -> inspectable 2.5D BEV
|           +-- accepted: global tokens + local frontier tokens
|
+-- D4  Who interprets semantic relevance?
|   +-- accepted primary: inject U into the Qwen 4B VLM
|   |   +-- D5  How is consumption trained and demonstrated?
|   |       +-- accepted: oracle-U consumer proof before predicted-U training
|   |       +-- accepted: structured grounding, then longitudinal planning LoRA
|   |       +-- accepted: zero/shuffle causal controls
|   +-- conditional fallback: direct Planning Expert injection only after a
|       bounded, valid VLM attempt fails
|
+-- D6  How is the claim evaluated?
    +-- accepted primary: Bench2Drive true closed loop
    +-- accepted secondary: NAVSIM pseudo-closed-loop corroboration
    +-- accepted: Route 151 is a motivating/plumbing case, not an untouched test
    +-- accepted: one fixed official-style baseline is sufficient initially
    +-- accepted: retain released/native image processing; no resolution ablation
```

The dependency order matters. D2 cannot redefine the estimator before D1 fixes
its responsibility; D5 cannot train a deployable depth model before D4 shows
that the intended consumer can use oracle U; and the Planning Expert fallback
cannot become the main path merely because VLM training is difficult.

### Review rounds and settled frontier

The grill proceeded through these normalized frontiers:

1. **Problem and responsibility:** accept clean visual occlusion as the central
   uncertainty case; reject asking a small adapter to infer semantic risk or a
   hidden actor directly.
2. **Physical representation:** estimate visibility from depth, preserve an
   inspectable U field, and keep distance/route/stopping urgency separate from
   observability.
3. **Consumer:** make Qwen's VLM interpret U first; keep direct Planning Expert
   conditioning as a conditional fallback because using it first would bypass
   the intended VLM claim.
4. **Training and evaluation:** oracle-first staged grounding and planning;
   Bench2Drive primary, NAVSIM secondary; a single controlled baseline; no
   artificial image-resolution reduction or resolution ablation.

The design frontier was treated as closed after the user accepted the bounded
VLM-first path and the one-baseline evaluation scope. The following items were
explicitly deferred as implementation frontiers rather than silently assumed:

- the independent predicted-depth model and its training data, gated on an
  interpretable oracle-U consumer result;
- the acceptable efficiency loss at a given safety improvement;
- exact LoRA/data-mixture parameters and held-out route/seed manifests;
- promotion of direct Planning Expert conditioning, gated on the failure rule
  in the ADR;
- use of the released RL Planning Expert, gated on provisioning and checksum
  verification of that checkpoint.

Any change to a settled node must be recorded as a new or superseding ADR. Any
new decision whose prerequisites are unsettled returns to the design frontier
instead of being chosen implicitly during implementation.

## 3. Why the direction changed

The historical Orion mainline learned observation degradation from frozen
EVAViT features and attempted to expose spatial U to Orion's language/planning
path. Its feature targets, calibration, adapters, projections, and checkpoints
are EVAViT- and Orion-specific. They cannot be treated as compatible with
Qwen-Drive merely because both systems contain visual and language backbones.

The large generated EVAViT counterfactual feature tensor was deleted with user
authorization on 2026-09-05. The retained deletion receipt records that
70,837,587,338 bytes were freed. It can be regenerated from raw data if a
historical reproduction is ever required.

Qwen-Drive provides a materially stronger native multimodal starting point than
the historical Orion language path, but the completed diagnostics do not show
that replacing the backbone alone solves uncertainty-aware planning:

- On the frozen 120-state textual-U diagnostic, Qwen-Drive reached 69.4%
  nonzero-field accuracy excluding presence and 42.19% changed-field exact
  response, versus 6.4% and 0% for the Orion v15 LoRA. Qwen still missed the
  predeclared 80% and 70% sufficiency gates.
- In Route 151, official-input clean Qwen runs repeatedly collided with the
  crossing pedestrian. A separate Qwen VQA diagnostic recognized the
  pedestrian and recommended braking, while the online Planning Expert still
  produced a maintain-speed trajectory. The strongest current diagnosis is a
  domain-sensitive perception/reasoning-to-trajectory alignment failure, not
  proof of generic visual blindness.
- Front-camera dropout made Route 151 slower and collision-free, but degraded
  Route 146 and did not rescue Route 203. This is evidence of a strong but
  context-dependent native response to evidence loss, not a calibrated safety
  policy.
- Three progressively stricter Route 151 oracle-U grounding probes are valid
  negative results. V1c learned composite answer syntax and majority fields;
  V1d factorized the fields but retained image/label shortcuts; V1e balanced
  multiple real row-addressed labels for the same images and slots. V1e reached
  80% only on the route field, while frontier, margin, and action reached 40%,
  50%, and 66.7%. Its largest true-U advantage over the stronger zero/shuffle
  control was only 16.7 percentage points. This does not show that Qwen can
  never consume continuous U, but it does show that the current projector,
  upper-layer LoRA, and bounded supervision do not yet provide causal grounding.
- V1f then isolated a single explicitly addressed `route_weight_mean` threshold.
  It trained balanced within-image ON/OFF row-swap pairs under three randomized
  decoy orders and evaluated two unseen orders. Held-out true-U accuracy was
  75%, only half of the pairs flipped correctly, spatial-shuffle target accuracy
  was 45%, and the causal gap was 20 points. This valid negative rules out the
  current generic whole-row MLP projector as a sufficient small-data adapter;
  it still does not rule out a field-typed physical adapter or Qwen consumption
  in general.
- V1g changed only that projector to a field-disjoint scalar basis while
  reusing the exact V1f data, split, schedule, LoRA, seed, and gates. It collapsed
  every true/zero/shuffle output to `OFF_ROUTE`: held-out true-U accuracy was
  50%, matched-pair accuracy 0%, and causal gap 0. This rules out scalar typing
  alone. The unresolved interface gaps are explicit G/F slot identity and
  whether adapting only two of Qwen's eight full-attention layers provides
  enough reachability; those are hypotheses, not established causes.
- V1h then changed only slot identity by appending a fixed 48-way `Gxx/Fxx`
  code. Held-out true-U accuracy rose to 95%, both route labels passed 90%,
  9/10 matched pairs were jointly correct, and the true-versus-control gap was
  40 points. However, spatial-shuffle changed-target accuracy remained 45% and
  19/20 shuffled predictions were `OFF_ROUTE`. The run is therefore a valid
  negative rather than causal-readout acceptance. It supports explicit slot
  identity as useful but not sufficient; attention reachability remains the
  next isolated interface hypothesis.
- V1i kept the V1h adapter/data fixed and expanded LoRA to all eight released
  full-attention layers. True-U accuracy and matched-pair correctness both
  reached 100%, with a 50-point control gap, but spatial-shuffle changed-target
  accuracy remained 50% and 18/20 shuffled outputs were `OFF_ROUTE`. A
  read-only feature audit found that the five real ON/OFF target-row pairs are
  reused across train/evaluation order variants and that, on those rows,
  `route_weight_mean` is perfectly confounded with multiple urgency,
  unknown-space, observation-age, and frontier-score features. The result does
  not distinguish exact field readout from memorized row templates. The next
  open frontier is therefore data/field identifiability, not more epochs or
  broader attention scope.
- V1j is the accepted data-identifiability probe. It keeps the V1i model and
  optimization settings but replaces repeated target rows with 13 training
  ON/OFF target-row pairs and five disjoint evaluation pairs. The
  `route151-step-000200` frame is evaluation-only. Spatial shuffle remains a
  fully reported control and remains absent from the optimizer, but its former
  80% changed-target threshold is no longer a hard veto; true-U/per-label,
  matched-pair, target-row leakage, and 30-point stronger-control-gap checks
  are the V1j gates.
- V1j completed as a protocol-valid negative: true-U target-row accuracy is
  50% (`ON_ROUTE` 4/10, `OFF_ROUTE` 6/10), no held-out matched pair is jointly
  correct, and the control gap is zero. The prediction is constant within each
  frame across its ON/OFF target rows, exposing a sample/global-context
  shortcut. V1i's 100% result does not generalize to unseen queried rows.

## 4. Current Qwen-to-Bench2Drive system

The active branch is `codex/qwen-drive-transition`. Milestone commits and live
run identifiers are tracked in [implementation.md](implementation.md) rather
than pinning this context document to a quickly stale HEAD.

The implemented runtime is a Qwen sidecar rather than a backbone transplant
inside Orion:

```text
Bench2Drive/CARLA Python 3.8
  -> three camera views, four timestamps, ego history, navigation command
  -> local Unix-socket RPC
Qwen Python 3.10
  -> released Qwen-Drive VLM and Planning Expert
  -> 50 x (x_forward, y_left, heading), 10 Hz, 5 s
  -> existing coordinate/time adapter and PID
  -> CARLA VehicleControl
```

The formal transport profile keeps the CARLA images lossless at 1600x900 and
lets the released Qwen processor apply its own history/current pixel budgets.
The retired low-resolution JPEG profile is not an experimental arm.

The released Qwen model has no official Bench2Drive/CARLA agent or published
Bench2Drive score. Existing closed-loop scores are results of this repository's
integration using the official Bench2Drive evaluator. Absolute failures must
therefore not be described as official Qwen benchmark results.

The existing server asset is the released SFT Planning Expert. As of
2026-09-06, the official release also includes a 2.1 GB `planner-rl` head; the
server snapshot is incomplete rather than blocked on an upstream release.
SFT direct and
SFT reasoning Route 151 runs exist. The target main comparison selected during
design review is the released RL Planning Expert in reasoning-planning mode,
one sample, and a fixed seed, after that exact checkpoint is provisioned and
verified. Until then, SFT reasoning remains an engineering baseline rather
than a silently substituted final baseline.

## 5. Verified Route 151 evidence

Route 151 is the motivating case, not an untouched test case.

- Two available SFT direct clean runs collided with the pedestrian.
- One SFT reasoning clean run also collided.
- Near contact, the planner continued to request a fast forward trajectory and
  the controller applied no brake.
- With front-camera dropout during the event, Qwen shortened its trajectory,
  slowed, avoided the pedestrian, and completed the route.
- The same low-level controller can therefore execute a conservative Qwen
  trajectory; the clean failure is upstream of the actuator command in the
  available trace.

The case supports studying occlusion-induced scene uncertainty: a parked
vehicle hides an area from which a vulnerable road user can emerge, and a safe
policy should reduce commitment before the actor becomes fully observable.

Route 151 may be used in a disposable overfit smoke to verify the new modality
path. The final checkpoint must be trained on separate parameterized scenes,
and claims of generalization must use held-out routes, seeds, scene layouts,
and occluder/actor combinations.

## 6. Scientific boundary

The project does not claim any of the following:

- first combination of occupancy and an LLM/VLA;
- complete semantic occupancy or world-model reconstruction;
- direct prediction of a hidden actor's existence from unobservable evidence;
- superiority of a VLM over every classical risk-aware planner;
- an official Qwen-Drive Bench2Drive result;
- safety improvement from one Route 151 run alone.

Nearby work already covers occupancy-language models, occupancy-supervised
VLAs, BEV-to-LLM adapters, unknown-aware occupancy planning, and classical
occlusion-aware speed control. The defensible gap is the joint combination of:

1. an explicit visibility/unknown-space belief rather than only semantic
   occupancy;
2. a task-agnostic physical estimator whose gradients are separated from
   planning;
3. injection into a pretrained driving VLM so semantic relevance is decided
   in the large model;
4. anticipatory, before-reveal closed-loop evaluation with causal U controls;
5. explicit reporting of the safety/progress trade-off.

## 7. Fixed responsibility split

```text
Independent depth/visibility estimator
  answers: what is visible, occupied, occluded, stale, or never observed?

Deterministic exposure computation
  answers: which visibility frontiers are near the route/stopping envelope?

Qwen VLM
  answers: which of those frontiers is semantically relevant now?

Released Planning Expert
  answers: what 50-waypoint trajectory should be executed?
```

The visibility estimator must not predict a hidden pedestrian/vehicle
probability, TTC to an unobserved actor, or the final driving action. Planning
gradients stop before this estimator.

`U_vis` denotes lack of current observation. `U_urgent` denotes a deterministic
exposure weighting derived from route, ego speed, and stopping margin. The two
must remain separately inspectable; distance weighting must not turn a far but
unobserved cell into a falsely "certain" cell.

## 8. Existing assets and constraints

Implemented and verified assets:

- `team_code/qwen_drive_b2d_agent.py`;
- `uq_estimator/qwen_drive_bridge.py`;
- `configs/qwen_drive_b2d_agent_v1.json`;
- official-input Qwen bridge and closed-loop launch scripts;
- clean/dropout Route 146, 151, and 203 traces;
- Route 151 VQA and reasoning-planning diagnostics;
- NAVSIM image resolver, pair manifest builder, paired runner, and integrity
  audit scaffolding.

Important constraints:

- Qwen warm inference remains substantially slower than the nominal planning
  cadence, although synchronous CARLA can still run the experiment.
- The public Qwen checkout documents inference and evaluation but does not
  expose its complete training entry point; a local trainer is required.
- The shared filesystem cannot hold the complete documented NAVSIM sensor
  archive. NAVSIM must use selected scenes/shards and a separate environment.
- The final predicted U requires a new Qwen-independent depth/visibility
  estimator. The historical EVAViT Stage-1 checkpoint cannot be reused.

## 9. Immediate evidence ladder

The next work is intentionally oracle-first:

1. Generate oracle 3D visibility from CARLA depth and calibration.
2. Collapse it to the accepted 2.5D BEV schema and render it for inspection.
3. Produce global and frontier tokens and inject them into the 4B VLM.
4. Verify structured U grounding with the Planning Expert frozen. The V1c-V1i
   bounded probes are valid negatives, so this step remains open.
5. Train a longitudinal-only trajectory response using paired robust targets.
6. Run one fixed baseline and one otherwise identical oracle-U Route 151 arm.
7. Replace oracle depth with the independent predicted-U module only after the
   consumer path is shown to work.
8. Expand to held-out scenarios and NAVSIM only after the small closed-loop
   slice is interpretable.

The first experiment may be informative without completing the full research
claim. No document may promote an oracle result, a disposable Route 151 overfit,
or an isolated collision avoidance to a learned-U generalization result.

## 10. Source documents

- `docs/qwen_drive_b2d_integration_v1.md`
- `docs/qwen_drive_u_gap_review_2026-09-05.md`
- `docs/qwen_drive_agent_fidelity_and_evaluation_review_2026-09-05.md`
- `docs/qwen_drive_official_input_dropout_screen_acceptance_2026-09-05.md`
- `docs/qwen_route151_failure_and_navsim_pair_plan_2026-09-05.md`
- `docs/qwen_drive_orion_backbone_diagnostic_2026-09-05.md`
- `docs/CURRENT_STATE.md` for the complete historical Orion evidence chain

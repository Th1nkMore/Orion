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

## 2. Why the direction changed

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

## 3. Current Qwen-to-Bench2Drive system

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

The existing server asset is the released SFT Planning Expert. SFT direct and
SFT reasoning Route 151 runs exist. The target main comparison selected during
design review is the released RL Planning Expert in reasoning-planning mode,
one sample, and a fixed seed, after that exact checkpoint is provisioned and
verified. Until then, SFT reasoning remains an engineering baseline rather
than a silently substituted final baseline.

## 4. Verified Route 151 evidence

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

## 5. Scientific boundary

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

## 6. Fixed responsibility split

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

## 7. Existing assets and constraints

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

## 8. Immediate evidence ladder

The next work is intentionally oracle-first:

1. Generate oracle 3D visibility from CARLA depth and calibration.
2. Collapse it to the accepted 2.5D BEV schema and render it for inspection.
3. Produce global and frontier tokens and inject them into the 4B VLM.
4. Verify structured U grounding with the Planning Expert frozen.
5. Train a longitudinal-only trajectory response using paired robust targets.
6. Run one fixed baseline and one otherwise identical oracle-U Route 151 arm.
7. Replace oracle depth with the independent predicted-U module only after the
   consumer path is shown to work.
8. Expand to held-out scenarios and NAVSIM only after the small closed-loop
   slice is interpretable.

The first experiment may be informative without completing the full research
claim. No document may promote an oracle result, a disposable Route 151 overfit,
or an isolated collision avoidance to a learned-U generalization result.

## 9. Source documents

- `docs/qwen_drive_b2d_integration_v1.md`
- `docs/qwen_drive_u_gap_review_2026-09-05.md`
- `docs/qwen_drive_agent_fidelity_and_evaluation_review_2026-09-05.md`
- `docs/qwen_drive_official_input_dropout_screen_acceptance_2026-09-05.md`
- `docs/qwen_route151_failure_and_navsim_pair_plan_2026-09-05.md`
- `docs/qwen_drive_orion_backbone_diagnostic_2026-09-05.md`
- `docs/CURRENT_STATE.md` for the complete historical Orion evidence chain

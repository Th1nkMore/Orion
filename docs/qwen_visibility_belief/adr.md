# ADR-001: Visibility-belief augmentation of Qwen-Drive

- Status: Accepted
- Accepted: 2026-09-06 (Asia/Shanghai)
- Branch: `codex/qwen-drive-transition`
- Supersedes: the EVAViT/Orion corruption-UQ architecture as the active research
  direction; historical evidence and reproducibility assets remain valid

## Context

The historical Orion work estimates corruption-induced observation
degradation from EVAViT features. It does not directly model the uncertainty
created by occluded space in an otherwise clean scene. Route 151 demonstrates
the latter problem: Qwen can recognize the pedestrian after it becomes visible
yet still commits to an unsafe trajectory, while front-camera dropout can
accidentally induce enough conservatism to avoid the collision.

The broad description "occupancy + uncertainty + LLM" is not a sufficient
novelty claim. Existing work already combines semantic occupancy with language
models and uses unknown-aware occupancy for planning. The design must therefore
center on explicit visibility belief, VLM consumption, causal attribution, and
before-reveal closed-loop behavior.

## Decision

### 1. Treat occupancy as a carrier, not the contribution

The method will represent physical observability rather than build a complete
semantic occupancy/world model.

The minimum `U_vis` schema contains:

- currently visible free space;
- currently visible occupied surfaces;
- occluded unknown space inside nominal sensor coverage;
- outside-FOV space as a distinct state;
- depth confidence for predicted U;
- occlusion/visibility frontiers;
- observation age distinguishing recently seen, long-unseen, and never-seen
  regions.

The visibility estimator is task-agnostic. It does not consume route, actor,
TTC, collision, action, or future-outcome labels.

### 2. Generate in 3D and consume as 2.5D BEV

CARLA depth and calibration will first be used to ray-cast visibility in 3D.
The relevant vertical volume is then collapsed into a 2.5D ego-frame BEV with
height/visibility statistics. Full 3D voxels remain supervision and diagnostic
state; they are not sent wholesale to Qwen.

This keeps occlusion geometry correct without turning the project into a large
3D generative world model.

### 3. Keep observability and urgency separate

`U_vis` represents what is unknown. It is not distance-decayed.

`U_urgent` is a deterministic, separately inspectable exposure weighting based
on route proximity, ego speed, and safe stopping margin. A representative
stopping distance is:

```text
d_stop = v * reaction_time + v^2 / (2 * safe_deceleration)
```

The relevant quantity is time/distance to a visibility frontier, not TTC to an
actor whose existence and motion are unknown. `U_urgent` may bias attention and
token selection but may not directly issue the driving action in the main
method.

### 4. Use global plus frontier tokens

The full 2.5D BEV remains visualizable and auditable. The VLM consumes:

- a small set of global U tokens preserving the whole field; and
- Top-K frontier tokens preserving local geometry, coordinates, unknown area,
  height ratio, observation age, confidence, and urgency.

Top-K selection is soft. Far regions remain represented by the global tokens
rather than being hard-deleted.

Exact grid bounds, resolution, K, and embedding width are implementation
parameters and are not fixed by this ADR.

### 5. Inject U into the 4B VLM

The primary computation path is:

```text
official camera tokens + U tokens + route/ego context
  -> Qwen 4B VLM
  -> U-fused multi-layer VLM cache
  -> released Qwen Planning Expert
  -> released 50-waypoint trajectory representation
```

U tokens enter the VLM transformer between explicit U boundary tokens. The
projector maps physical U tokens into the Qwen hidden space and preserves
metric coordinates through continuous positional features or a verified
multimodal-position scheme.

Direct injection into the Planning Expert would let the planner combine U with
VLM caches but would bypass VLM interpretation of U. It is therefore a fallback
and diagnostic, not the primary method.

### 6. Keep the U estimator independent from Qwen planning

The first predicted-U implementation will use an independent lightweight
metric-depth/visibility module. It will be trained against privileged CARLA
geometry and then frozen. Planning gradients stop before it.

The official Qwen perception head may be studied later for feature sharing or
distillation, but it is not the first implementation. This preserves physical
interpretability, error attribution, and portability to another VLA.

### 7. Validate the consumer with oracle U before training predicted U

The first vertical slice uses CARLA ground-truth depth and calibration at
inference to produce oracle U. This is an engineering and capacity upper bound,
not the final deployable method.

If oracle U cannot improve VLM grounding or the targeted trajectory response,
training a depth estimator is premature. If it succeeds, oracle U is replaced
with RGB-predicted U and the degradation is measured explicitly.

### 8. Use structured grounding followed by staged LoRA

Training proceeds in stages:

1. Freeze the official vision encoder and Planning Expert. Train the U
   tokenizer/projector and VLM upper-layer LoRA on structured U-grounded tasks.
2. Add Planning Expert cross-attention LoRA and the released flow-matching
   trajectory objective. Continue VLM LoRA at a lower learning rate.
3. Mix grounding and planning batches so trajectory training does not erase U
   meaning.

Grounding targets include frontier identity, route intersection, stopping
margin bucket, and `KEEP/SLOW/STOP`. Free-form chain-of-thought is not required.
Full-parameter VLM fine-tuning is deferred unless oracle-U grounding fails
after the token and supervision contracts are verified.

### 9. Train a longitudinal response first

The first safety teacher preserves a valid base trajectory's lateral path and
heading and changes only its longitudinal time parameterization. Near,
route-relevant frontiers inside the stopping envelope receive a slower or
stopping target. Far or irrelevant frontiers preserve the base behavior.

The first version does not teach speculative lateral swerves around a hidden
actor. Full lateral avoidance is deferred.

### 10. Use observation-causal paired supervision

Training scenes are generated in paired latent worlds with identical visible
history before reveal:

- a hidden pedestrian/vehicle is present; or
- the same occluded space is empty.

Before reveal, both variants receive the same locally robust defensive target
whenever a plausible route-relevant visibility frontier lies inside the safe
stopping envelope. After reveal, their targets may diverge based on observation.
The teacher may not use hidden-actor existence to choose different pre-reveal
actions.

Hard negatives include close but route-irrelevant occlusion, physically
disconnected space, opposite-direction occlusion, and distant occlusion with
adequate stopping margin.

### 11. Limit the first scenario scope

The first training and closed-loop suite contains two complementary families:

1. parked/static vehicles hiding pedestrians or cyclists;
2. road geometry or parked vehicles hiding cross traffic at corners and
   junctions.

Additional occlusion types are held for later generalization tests rather than
expanding the first training generator indefinitely.

### 12. Preserve one strong, controlled baseline

The first feasibility comparison uses one fixed released Qwen baseline and an
otherwise identical oracle-U arm. The target final baseline is the released RL
Planning Expert in reasoning-planning mode, one trajectory sample, and a fixed
seed. Ground-truth best-of-N selection is prohibited. If the RL checkpoint is
not yet provisioned, SFT reasoning is labeled as an engineering baseline and
must not be silently relabeled as the final configuration.

Both arms use the same official-input profile, image/history/command contract,
Planning Expert, sampler, trajectory adapter, PID, and route seed. The only
method change is the U path and its trained adapters. Planned and executed
speed are logged to detect an obvious controller mismatch; a full controller
qualification suite does not block the first oracle-U experiment.

### 13. Treat Route 151 as a motivating case

A disposable Route 151 overfit is allowed only to verify plumbing. The
checkpoint is not reportable. Formal training excludes final evaluation
routes, seeds, and scene combinations.

Evaluation has three roles:

- Route 151 as the known motivating case;
- held-out parameterized occlusion scenarios as generalization evidence;
- ordinary and semantic-negative scenarios as unnecessary-braking evidence.

Bench2Drive is the primary true closed-loop evaluation. NAVSIM is a secondary
pseudo-closed-loop corroboration after its environment and selected data are
available.

## Success criteria

Minimum success:

- oracle-U Qwen reduces collisions or avoids the Route 151 pedestrian conflict
  through an anticipatory trajectory change;
- removing, zeroing, or spatially shuffling U removes or degrades that response;
- the effect occurs before the actor becomes visible;
- route completion is not replaced by indefinite stopping.

Full success:

- predicted U retains a material portion of the oracle-U gain;
- true U beats zero and shuffled U on held-out routes/seeds;
- at comparable safety, Qwen+U has fewer unnecessary slowdowns or better
  progress than a hard visibility-triggered speed shield;
- safety, progress, blockage, lane compliance, comfort, and latency are all
  reported.

The project does not require Qwen+U to beat every classical planner. It must
show that explicit visibility belief improves the same pretrained VLA and that
the VLM performs nontrivial semantic conditioning beyond a universal slowdown.

## Fallback

If the oracle-U VLM path has a valid modality/position contract and passes
structured grounding but cannot produce the required trajectory response after
the bounded staged-LoRA pilot, inject the same U tokens directly into the
Planning Expert's cross-attention memory.

That fallback must be described accurately: the Qwen driving system consumes U,
but the 4B VLM does not explicitly interpret it. A hard speed shield remains a
comparison baseline only, not the main fallback.

## Rejected or deferred alternatives

- **Reuse the EVAViT U checkpoint:** rejected as feature-space incompatible.
- **Generic semantic occupancy as the main contribution:** rejected as crowded
  and broader than the target uncertainty question.
- **Predict hidden-actor probability in the U estimator:** rejected because it
  turns the small module into a semantic risk model.
- **Distance-decay `U_vis` itself:** rejected because relevance would be
  confused with observability.
- **Single global scalar U:** rejected because it discards location and frontier
  geometry.
- **Full 3D occupancy tokens:** deferred because of compute and scope.
- **U only in the Planning Expert:** retained only as fallback/diagnostic.
- **Hard U-triggered speed control as the main method:** rejected because it
  bypasses the VLM and cannot establish semantic U consumption.
- **Full 4B fine-tuning first:** rejected because LoRA provides a lower-memory,
  more attributable starting point.
- **Full longitudinal and lateral replanning first:** deferred; initial targets
  change longitudinal timing only.
- **Corruption-only uncertainty:** retained as a secondary robustness test, not
  the new core definition.

## Consequences and open implementation parameters

The accepted design is more expensive than direct planner conditioning. It
requires custom continuous-token insertion into Qwen, verified multimodal
position handling, a local training loop, and an independently trained depth
module. In return, the scientific claim matches the actual information path:
the VLM sees and reasons over the uncertainty evidence before planning.

The following are intentionally left to measured implementation decisions:

- BEV metric extent, cell size, and height bands;
- number of global and frontier tokens;
- observation-age decay;
- reaction time and safe-deceleration constants;
- LoRA target modules, rank, and learning rates;
- loss weights and batch mixture;
- exact held-out route/seed manifests.

Changing any responsibility boundary or promoting the Planning Expert fallback
to the main method requires a new ADR entry rather than silently editing this
decision.

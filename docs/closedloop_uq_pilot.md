# Closed-Loop UQ Causal Pilot

## Claim under test

The current pre-experiment asks two separate questions: (A) can a known,
oracle uncertainty event improve closed-loop safety through the bounded risk
governor, and (B) how closely can the existing learned Density score
approximate that oracle?

This pilot does **not** test whether the LLM understands uncertainty. It first
The learned Density score is only an aligned anomaly/OOD trigger at this
stage. It is not assumed to identify the uncertain object, image region,
camera, future-path overlap, or collision risk.

## Why this design is diagnostic

- The base planner and its trajectory are unchanged by UQ conditioning.
- Every causal condition receives the same front-camera dropout.
- The governor has a positive 1.5 m/s speed floor, so stopping forever is not
  an available solution.
- The constant control matches the mean score from the route's corrupted-off
  trace and tests generic slowing.
- The shuffled control preserves the empirical score distribution while
  breaking temporal alignment.
- Removing the hazard actor from the same route measures unnecessary slowing
  and guards against a collision-only conclusion.

## Frozen pilot parameters

The machine-readable preregistration is
`configs/closedloop_uq_pilot/protocol.json`.

- Routes:
  - 25: `VanillaSignalizedTurnEncounterRedLight`, Town12
  - 113: `PedestrianCrossing`, Town12
  - 148: `Accident`, Town10HD
- Corruption: front-camera dropout, severity 1, active for the full route.
- Density UQ calibration from the existing route-balanced audit:
  - clean median 0.331, Q75 0.404;
  - corrupted median 0.589, mean 0.621.
- Governor:
  - threshold 0.4;
  - saturation 0.8;
  - speed range 1.5--5.0 m/s;
  - maximum added brake 0.5.

These values are frozen before observing any closed-loop outcome.

## Protocol amendments

Execution-only changes made after the pilot started are recorded separately in
`configs/closedloop_uq_pilot/amendments.json`; they are not silently folded
into the preregistration. After the first runtime smoke, a uniform fast-screen
rule was added for the remaining clean baselines: cancel a route if absolute
ego speed stays below 0.25 m/s for at least 8.0 consecutive simulation
seconds. Such a cancellation only makes the route ineligible for Stage B. It
is not counted as an official collision, infraction, blocked-agent, or timeout
result.

The installed CARLA package was then found to lack Town12. Before observing
any outcome on replacement routes, the active screen was changed to official
one-scenario routes 203 (`PedestrianCrossing`, Town04), 195
(`OppositeVehicleRunningRedLight`, Town03), and 146
(`DynamicObjectCrossing`, Town01). They were selected by deterministic
shortest-route rules recorded in the amendment file. The failed Town12 launch
is an environment error and is not an experimental result.

## Stage A: failure induction screen

Run only `clean_off` and `front_corrupt_off` on the active routes. Carry a route into the
causal matrix only when:

1. `clean_off` produces a valid route result; and
2. `front_corrupt_off` worsens collision, traffic-rule, timeout, or route-completion
   behavior.

If fewer than two independent hazard families pass, apply severity 2 to all
three routes. Do not select isolated routes post hoc.

Also inspect whether the learned score responds consistently to the corrupted
front view. Full-route dropout can establish failure induction and a score
shift, but cannot by itself establish onset timing or temporal alignment.

## Oracle gate before Stage B

For the first route that passes Stage A, fix a finite corruption event window
before running further outcomes. Run a known-corruption oracle through the
same governor bounds. Outside the window its oracle score is 0; inside the
window it is 1. The event window must be identical in later off, constant,
shuffled, aligned-learned, and hazard/nohazard comparisons.

If oracle control does not improve safety, stop and redesign the governor or
scenario. If oracle succeeds but aligned learned does not, treat UQ as the
bottleneck. Do not expand the deferred matrix until this gate passes.

## Deferred Stage B matrix

For each selected route, run the same corrupted observation under:

1. `front_corrupt_off`;
2. `front_corrupt_constant`;
3. `front_corrupt_shuffled`;
4. `front_corrupt_aligned_learned`;
5. `front_corrupt_oracle` as the upper-bound reference.

Build the constant and shuffled controls only from the route's Stage-A
`corrupt_off` raw-UQ trace. Use shuffle seed 20260825. Repeat the four
conditions on the route with its scenario actors removed.

## Outcomes and pilot gate

Primary safety outcomes:

- vehicle, pedestrian, and layout collisions;
- red-light, stop-sign, and outside-lane infractions.

Efficiency guardrails:

- route completion;
- mean speed and game duration;
- minimum-speed, blocked-agent, and timeout infractions;
- fraction and magnitude of control interventions.

A learned-score result is only promising if it improves safety over off and
both placebo controls, approaches the oracle, and does not create a
blocked/timeout route. Matching constant slowing does not support temporal or
semantic value. No p-value or LLM-understanding claim will be made from this
small pilot.

## Reproducible workflow

Build paired routes:

```bash
python scripts/build_closedloop_uq_pilot_routes.py \
  --bench2drive-root /public/share/lidachuan/orion_assets/Bench2Drive \
  --route-indices 203 195 146 \
  --out-dir configs/closedloop_uq_pilot/routes
```

Submit one condition:

```bash
bash scripts/submit_closedloop_uq_pilot.sh 203 front_corrupt_off hazard
```

Build matched controls after `corrupt_off` completes:

```bash
python scripts/build_uq_score_controls.py CONTROL_TRACE.jsonl \
  --out-dir SCORE_CONTROL_DIR --seed 20260825
```

Summarize all completed runs:

```bash
python scripts/summarize_closedloop_uq_pilot.py \
  /public/share/lidachuan/orion_assets/results/uqcl_p0 \
  --out-dir results/closedloop_uq_pilot/uqcl_p0
```

# Closed-loop oracle mechanism proof — 2026-08-26

## Outcome

Route 146 (`DynamicObjectCrossing`, Town01) now provides a complete exploratory
mechanism chain under a transient five-second front-camera dropout.

| Condition | Job | Completion | Pedestrian collisions | Score | Game duration |
|---|---:|---:|---:|---:|---:|
| clean, no corruption | 1057222 | 100% | 0 | 100 | 30.95 s |
| transient dropout, response off | 1057306 | 100% | 1 | 50 | 30.10 s |
| transient dropout, oracle, 1.5 m/s floor | 1057314 | 100% | 1 | 50 | 35.40 s |
| transient dropout, oracle controlled stop | 1057566 | 100% | 0 | 100 | 38.95 s |

All four referenced outcomes have eligible terminal Bench2Drive records.  The
controlled-stop oracle has no pedestrian, vehicle, or layout collision; no red
light, stop-sign, outside-lane, route-deviation, blocked, scenario-timeout, or
route-timeout event.  Bench2Drive emits the same aggregate min-speed diagnostic
(`136.3`) for the transient conditions, but it does not reduce their official
penalty score; this diagnostic is reported rather than hidden.

Relative to transient-off, the successful oracle changes collision count from
1 to 0 and composed score from 50 to 100 while retaining 100% completion.  Its
cost is 8.85 additional simulation seconds (29.4%).

## Observed mechanism

1. Front dropout and oracle activate once at route progress 0.30121.
2. The bounded controller stops the ego near progress 0.317; maximum added
   braking remains 0.5, although the frozen base planner may independently
   command stronger braking.
3. Dropout and oracle deactivate exactly 5.00 simulation seconds after onset,
   independent of ego progress.  The longest interval below 0.25 m/s is 4.65 s,
   below the fixed 8 s liveness screen.
4. With the front camera restored, the frozen ORION policy approaches the
   crossing, stops again before the previous collision region, lets the
   pedestrian clear, resumes, and completes the route.

The failed 1.5 m/s-floor oracle is an important control: partial slowing merely
shifted the pedestrian collision from `(180.299, 330.310)` to
`(183.572, 329.991)` and left the score at 50.  Therefore the positive result is
not supported by an arbitrary claim that any slowing is sufficient; under a
complete forward-camera outage, the response had to remove unjustified forward
motion long enough for perception to recover.

## What this supports

This is a strong **oracle mechanism pre-experiment**: when transient forward
perception is known to be unavailable, an appropriate bounded conservative
response can trade travel time for lower closed-loop collision risk without
sacrificing route completion.

## What this does not support

- The oracle uses the known synthetic corruption state, not learned Density UQ.
- It does not show that Density UQ is object-, location-, path-, or
  collision-risk-grounded.
- It does not show that the VLM/LLM understands uncertainty.
- The five-second event and controlled-stop policy were selected during
  exploratory redesign after earlier route-146 failures.  A paper-level causal
  claim needs the now-frozen policy confirmed on held-out scenario realizations
  before learned/off/constant/shuffled comparisons.
- No ADE claim is made.

## Auditable artifacts

- Transient preregistration:
  `configs/closedloop_uq_pilot/transient_oracle_preregistration.json`
- Controlled-stop preregistration:
  `configs/closedloop_uq_pilot/transient_safe_stop_preregistration.json`
- Off and 1.5 m/s oracle raw results:
  `results/closedloop_uq_pilot/uqcl_p1_transient/raw/`
- Controlled-stop raw result:
  `results/closedloop_uq_pilot/uqcl_p2_safe_stop/raw/`
- Front-input, raw-front, and BEV GIFs:
  `results/closedloop_uq_pilot/uqcl_p1_transient/visualizations/` and
  `results/closedloop_uq_pilot/uqcl_p2_safe_stop/visualizations/`

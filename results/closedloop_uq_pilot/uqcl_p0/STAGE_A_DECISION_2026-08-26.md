# Stage A decision snapshot — 2026-08-26 05:05 +08:00

## Registered comparison

Route 146 (`DynamicObjectCrossing`, Town01) uses one fixed spatial corruption
window: projected route progress `[0.30, 0.55)`.  The planning model is frozen.
The oracle condition receives the known binary corruption state through the same
bounded risk governor; it does not receive missing visual content.

| Condition | Job | Eligibility | Completion | Safety outcome |
|---|---:|---|---:|---|
| clean off | 1057222 | official | 100% | no collision or listed violation; composed score 100 |
| finite-window front-corrupt off | 1057275 | official | 100% | one pedestrian collision; penalty 0.5; composed score 50 |
| finite-window front-corrupt oracle | 1057294 | diagnostic only | not completed | passed the earlier collision location, then stopped at progress 0.466800; below 0.25 m/s for 9.00 s and cancelled by the registered fast-screen rule |

## Decision

The finite-window corruption successfully induces a closed-loop safety failure:
the same route changes from a clean official completion with no collision to an
official pedestrian collision when the front input is dropped.

The current oracle governor does **not** pass the mechanism gate.  Although its
partial trace contains no collision and reaches beyond the corrupted-off
collision location, it has no terminal evaluator record and fails liveness.  It
therefore cannot be described as an official safety improvement.

Per the preregistered decision rule, stop this governor/scenario branch here.
Do not submit learned, constant, shuffled, hazard/nohazard, additional-seed, or
additional-route governor jobs.  A subsequent experiment must first redesign
the conservative response or event geometry and then show an official oracle
completion benefit before evaluating learned Density UQ.

The trace also exposes a specific design artifact: the corruption is switched
off only after the ego reaches progress 0.55.  Once conservative control and a
black front input stop the ego at progress 0.466800, the run cannot leave the
window and the visual input cannot recover.  This creates an absorbing
progress-locked failure.  The cleanest next oracle screen is therefore a new,
explicitly preregistered transient event: trigger at a fixed route/scenario
point, remain active for a fixed amount of simulation time, then restore the
camera regardless of ego progress.  Completion after recovery must be part of
the success criterion.

## Interpretation boundary

- This is useful negative evidence: accurate knowledge that corruption is active
  can reduce immediate exposure yet still cause a deadlock under a scalar
  speed-cap/brake governor.
- The result does not show that the VLM understands uncertainty.
- It does not validate Density UQ as object-, location-, path-, or risk-grounded.
- No ADE claim is made.

## Artifacts

- Machine-readable snapshot: `visualizations/stage_a_snapshot_summary.json`
- Clean views: `visualizations/route146_clean_off_front.gif`,
  `visualizations/route146_clean_off_bev.gif`
- Corrupted-off views: `visualizations/route146_front_corrupt_off_raw_front.gif`,
  `visualizations/route146_front_corrupt_off_agent_input_front.gif`,
  `visualizations/route146_front_corrupt_off_bev.gif`
- Oracle diagnostic views:
  `visualizations/route146_front_corrupt_oracle_raw_front.gif`,
  `visualizations/route146_front_corrupt_oracle_agent_input_front.gif`,
  `visualizations/route146_front_corrupt_oracle_bev.gif`

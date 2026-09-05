# Qwen visibility-belief implementation status

Last updated: 2026-09-06 (Asia/Shanghai)

## Current milestone

`O2: temporal observation memory and deterministic exposure`

Status: in progress (local implementation complete; live validation pending)

O1 is accepted. O2 now adds ego-motion-compensated observation age and a
separate route/stopping exposure view. The live gate is to verify that temporal
states persist at the same world locations and that urgency selects plausible
frontiers before Route 151's pedestrian reveal without altering Qwen output.

## Ordered implementation ladder

| ID | Deliverable | Status |
| --- | --- | --- |
| O0 | CARLA depth decoding, camera calibration, 3D visibility fusion, 2.5D BEV, rendering, unit tests | Complete (`6addb2fe`) |
| O1 | Add co-located oracle depth sensors to the Qwen agent behind an explicit oracle-only config | Complete (`c8fac0b5`; accepted by run `1165332`) |
| O2 | Observation-age memory and deterministic urgency/stopping-margin map | In progress (local implementation and tests complete) |
| O3 | Global/frontier tokenizer with serialization and causal zero/shuffle controls | Not started |
| V0 | Insert U tokens into the 4B VLM with verified positions and zero-U identity | Not started |
| V1 | Structured U-grounding warm-up with staged LoRA | Not started |
| P0 | Longitudinal trajectory retiming teacher and flow-matching training path | Not started |
| C0 | Fixed-baseline versus oracle-U Route 151 closed-loop comparison | Not started |
| E0 | Independent predicted-depth/visibility estimator | Blocked on interpretable oracle-U consumer evidence |

## 2026-09-06 start record

- Confirmed local branch `codex/qwen-drive-transition` was clean at
  `ff45841a` before implementation began.
- Confirmed SSH access to `lidachuan@172.18.18.7` and found the server checkout
  at `/public/home/lidachuan/project/Orion`.
- The server checkout is a dirty historical worktree on
  `uq-orion-wip-20260903`; it will not be overwritten or switched in place.
  A separate clean worktree will be created for this branch before remote runs.
- Confirmed the released Qwen checkpoint and source tree exist remotely. Only
  `planner-sft` is currently provisioned; the ADR's final RL/reasoning baseline
  is not yet available and must not be claimed as current state.
- Confirmed the active Qwen agent currently exposes only three RGB cameras.
  Oracle depth sensors will be introduced only through a separate explicit
  config so the fixed RGB baseline remains unchanged.
- Verified against the CARLA sensor reference that depth is encoded as a
  24-bit value in BGRA bytes and represents pixel-to-camera distance with a
  1000 m far plane.

## O0 terminal record

- Commit: `6addb2fe` (`Add oracle visibility geometry`).
- Added the NumPy-only module
  `uq_estimator/qwen_visibility_belief.py`.
- Added `tests/test_qwen_visibility_belief.py` with six physical-geometry and
  process-isolation tests.
- Regression command:
  `pytest -q tests/test_qwen_drive_bridge.py tests/test_qwen_visibility_belief.py`.
- Result: `23 passed, 1 skipped` in the local Python 3.13 environment.
- The skipped test is the existing optional OpenCV-dependent bridge test; no O0
  visibility test was skipped.
- O0 establishes deterministic geometry and an inspectable schema. It does not
  establish live CARLA sensor alignment, temporal memory, U-token consumption,
  Qwen grounding, planning change, or safety.

## O1 local record

- Added a separate oracle-only bridge config; the existing RGB and reasoning
  baseline configs remain unchanged.
- The oracle profile clones all pose, intrinsics, and resolution fields from
  the three Qwen RGB sensors into CARLA depth sensors, then records one
  compressed tensor and one rendered PNG per Qwen inference step.
- Every artifact declares `oracle_depth=true` and `used_by_qwen=false`.
  Therefore an O1 run is a sensor/alignment smoke only and is not evidence of a
  model or safety improvement.
- Local regression command:
  `pytest -q tests/test_qwen_visibility_belief.py tests/test_qwen_drive_bridge.py`.
- Result: `26 passed, 1 skipped`; Python compilation and `git diff --check`
  also passed.
- An attempted repository-wide `pytest -q` could not collect in the local
  lightweight environment: 132 existing modules require unavailable `torch`
  or `cv2`. This is an environment limitation, not a passing full-suite claim;
  the relevant suite must be rerun in the remote Orion environment.
- O1 remains open until a remote CARLA run validates sensor availability,
  coordinate alignment, artifact integrity, and acceptable runtime overhead.

## O1 remote attempt 1

- Slurm job: `1165318`; run id:
  `qwen_oracle_visibility_route151_reasoning_sft_seed42_v1`.
- The isolated server checkout was clean at `137524ec`; its targeted regression
  passed `27/27` tests in the Orion Python 3.8 environment.
- CARLA, Town02, Route 151, and the Qwen sidecar all initialized. The evaluator
  then rejected `sensor.camera.depth` before the first simulation tick because
  Bench2Drive 0.0.4 omits depth from both its official SENSORS allowlist and
  camera preprocessing branch.
- Terminal state: Slurm `FAILED`, exit `127:0`; evaluator status
  `Failed - Agent's sensors were invalid`; zero oracle artifacts. This run says
  nothing about model behavior or visibility geometry.
- Resolution: add a default-off, explicitly logged evaluator extension for the
  privileged oracle-depth experiment. Any run using it is non-official and
  ineligible as a Bench2Drive sensor-track score; the ordinary baseline remains
  on the unmodified allowlist.

## O1 remote attempt 2

- Slurm job: `1165319`; run id:
  `qwen_oracle_visibility_route151_reasoning_sft_seed42_v2`.
- Checkout `e496858f`; remote regression passed `37/37` tests. The explicit
  `[OracleDepthHarness]` marker was present and the depth allowlist/preprocessor
  patch succeeded.
- The evaluator then raised `KeyError: sensor.camera.depth` while constructing
  its display-only sensor icon list. This was again before the first tick and
  produced zero oracle artifacts.
- Resolution: extend the same default-off oracle harness to map depth to the
  existing camera icon. No ordinary sensor validation or model path changes.

## O1 remote attempt 3

- Commit: `7b5b95de`; Slurm job: `1165331`; run id:
  `qwen_oracle_visibility_route151_reasoning_sft_seed42_v3`.
- Terminal state: Slurm `COMPLETED`, exit `0:0`, elapsed `00:14:56`, peak RSS
  `5,820,032 KiB`. The route completed 100% in 25.1 s simulation time.
- Produced 51 planning traces, 51 compressed belief tensors, and 51 rendered
  maps. All tensors are finite float32 `[5,120,100]`; their four mutually
  exclusive physical channels sum to one in every cell; all metadata flags are
  `oracle_depth=true` and `used_by_qwen=false`.
- Qwen sidecar inference time was mean 4.953 s, median 4.624 s, p95 5.216 s;
  the 25.584 s maximum includes first-inference warm-up. Geometry time is not
  yet instrumented separately and must not be inferred from these numbers.
- The unchanged Qwen planner still collided with one pedestrian at
  `(100.162, 303.092)`: route score 100, penalty 0.5, driving score 50. This is
  expected because O1 records U but does not consume it. It is useful baseline
  confirmation, not an oracle-U safety result.
- Although the patched evaluator JSON writes `eligible=true`, this run used the
  explicitly logged non-official oracle-depth extension and is scientifically
  ineligible as an official SENSORS-track score.
- The live maps show plausible three-camera coverage and changing occlusion
  frontiers. O1 remains open until sparse lossless RGB/depth audit snapshots
  verify actual cross-modal alignment; belief artifacts alone are insufficient
  for that claim.

## O1 alignment-audit instrumentation

- Added sparse snapshots at steps `0, 200, 260, 280, 300`, spanning route
  entry and the pre-collision/collision interval observed in attempt 3.
- Each snapshot preserves the original 1600x900 RGB as lossless PNG and stores
  each co-located depth image as a uint16 millimetre PNG clipped at the oracle
  grid's 60 m range. These are audit copies only; Qwen input resolution and
  transport remain unchanged.
- Added `geometry_seconds` around BGRA decode plus visibility fusion, excluding
  disk I/O and Qwen inference, so the oracle path's cost is measured directly.
- Local relevant regression: `37 passed, 1 skipped`; compilation and shell
  syntax checks passed.

## O1 remote attempt 4 and acceptance

- Commit: `c8fac0b5`; Slurm job: `1165332`; run id:
  `qwen_oracle_visibility_route151_reasoning_sft_seed42_v4`.
- Terminal state: Slurm `COMPLETED`, exit `0:0`, elapsed `00:15:17`, peak RSS
  `6,206,632 KiB`. The route completed 100% in 24.55 s simulation time.
- Produced 50 Qwen plans, 50 compressed belief tensors, 50 rendered belief
  maps, and 491 controller trace rows. Every tensor is finite float32
  `[5,120,100]`; the four mutually exclusive visibility states sum to one in
  every cell; every artifact declares `oracle_depth=true` and
  `used_by_qwen=false`.
- The five requested audit directories (`0`, `200`, `260`, `280`, `300`)
  contain all 30 native sensor snapshots. RGB is lossless uint8
  `1600x900x3`; depth is uint16 `1600x900`, millimetric, and bounded by the
  configured 60 m clip. No Qwen RGB resolution or transport setting changed.
- Manual paired inspection at all five times found matching outlines for
  buildings, curbs, poles, signs, vehicles, and the pedestrian visible at step
  260. Camera directions and the front/right overlap are also consistent.
- Three files named `*_depth_vis.png` under the step-0 audit directory were
  generated manually after the run as false-colour inspection aids. They are
  derived from the native uint16 files, were not emitted or consumed by the
  agent, and are excluded from the 30-file integrity count. Later false-colour
  previews were created only under remote `/tmp`.
- Geometry timing (BGRA decode plus 3D fusion, excluding disk and Qwen) over 50
  frames: mean 0.150 s, median 0.137 s, p95 0.218 s, max 0.515 s. Qwen
  inference timing: mean 5.099 s, median 4.615 s, p95 5.915 s, max 24.113 s;
  the maximum is first-inference warm-up.
- In the collision approach, the forward-right region `x=[0,20) m`,
  `y=[-10,-1] m` contains persistent occluded-unknown/frontier evidence from
  steps 180 through 250. At step 200 it has 213 cells with unknown ratio at
  least 0.5 versus 124 in the symmetric left region, while the central
  `|y|<=2 m` corridor remains observed-free. This supports the intended
  physical interpretation, but is not yet a learned risk score or a causal
  safety result.
- The unchanged Qwen consumer still collided with one pedestrian at
  `(103.522, 302.732)`: route score 100, penalty 0.5, driving score 50. Its
  reasoning changes from recognizing a parked vehicle on the right to
  recognizing a pedestrian by step 250, but the plan/controller maintains
  speed; at step 260 the text says to decelerate while the issued control still
  has throttle 0.75 and brake 0.0. This establishes separate consumer and
  longitudinal-control gaps. O1 itself intentionally cannot improve either
  because `used_by_qwen=false`.
- As in attempt 3, the evaluator JSON's `eligible=true` field is not a valid
  official-track claim: the explicit oracle-depth harness makes this run
  scientifically ineligible for the Bench2Drive SENSORS track.
- O1 acceptance: sensor availability, RGB/depth geometry, artifact integrity,
  coordinate convention, and measured runtime overhead have passed. Claims
  about temporal U, VLM grounding, planning response, collision avoidance, and
  predicted-depth transfer remain gated by O2 and later milestones.

## O2 local implementation record

- Added an ego-motion-compensated observation memory in the NumPy-only
  visibility module. Every BEV cell is explicitly one of currently observed,
  previously observed, or never observed; age is capped at 10 s, with a
  separate never-observed mask so the cap cannot be confused with actual
  history.
- The memory uses CARLA world pose only to warp the prior task-agnostic grid.
  It consumes no route, actor, TTC, collision, action, or hidden-state label.
- Added a separate deterministic exposure object. It projects the known
  navigation route into Qwen ego coordinates and records route distance, route
  progress, stopping margin, route weight, stopping-envelope weight, and
  frontier urgency. The pilot stopping distance is
  `v * 1.0 s + v^2 / (2 * 4.0 m/s^2)`; route sigma is 2.5 m and the soft
  stopping transition is 5 m.
- Urgency is the product of current frontier strength, route proximity, and
  stopping-envelope weight. It does not overwrite or distance-decay `U_vis`,
  and it does not issue throttle, brake, steering, or trajectory changes.
- The route helper projects onto the nearest ordered route segment, keeps the
  upcoming portion, transforms CARLA world coordinates into Qwen
  `x-forward/y-left`, and interpolates an exact 60 m horizon.
- The oracle agent records the memory and exposure tensors, metadata, timing,
  and separate PNGs at each Qwen inference step while retaining
  `used_by_qwen=false`. This is instrumentation for O2 and cannot itself change
  the baseline trajectory.
- Pilot parameters are explicit in
  `configs/qwen_drive_b2d_agent_oracle_visibility_sft_v1.json`; changing them
  for a reported comparison requires a recorded configuration change.
- Local relevant regression:
  `pytest -q tests/test_qwen_visibility_belief.py tests/test_qwen_drive_bridge.py tests/test_closedloop_sensor_diagnostics.py`.
  Result: `44 passed, 1 skipped`; Python compilation and `git diff --check`
  also passed. The skip is the existing optional local OpenCV transport test.
- O2 remains open until a live run validates temporal warp, route projection,
  stopping-distance arithmetic, pre-reveal frontier selection, artifact
  integrity, and derived runtime overhead.

## Integrity constraints

- No Torch, Qwen, Orion, or CARLA import in the geometry module.
- Use Qwen ego coordinates: x forward, y left, z up.
- Keep occluded unknown distinct from outside-FOV.
- A point visible from any camera is not unknown merely because another camera
  sees it behind a surface.
- Oracle depth is an upper bound and must never be reported as predicted U.
- Every implementation milestone updates this file and is committed separately.

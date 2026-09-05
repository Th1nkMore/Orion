# Qwen visibility-belief implementation status

Last updated: 2026-09-06 (Asia/Shanghai)

## Current milestone

`O3: global/frontier U tokenizer`

Status: not started (O2 accepted by live Route 151 run `1165345`)

O2 is accepted as an interpretable representation milestone. It establishes
ego-motion-compensated observation age and a separate route/stopping exposure
view on live CARLA data, but it does not establish Qwen consumption or safety.
O3 must turn the physical fields into deterministic global and frontier tokens,
with serialization and zero/spatial-shuffle controls, before any VLM injection.

## Ordered implementation ladder

| ID | Deliverable | Status |
| --- | --- | --- |
| O0 | CARLA depth decoding, camera calibration, 3D visibility fusion, 2.5D BEV, rendering, unit tests | Complete (`6addb2fe`) |
| O1 | Add co-located oracle depth sensors to the Qwen agent behind an explicit oracle-only config | Complete (`c8fac0b5`; accepted by run `1165332`) |
| O2 | Observation-age memory and deterministic urgency/stopping-margin map | Complete (`c4f62543`; accepted by run `1165345`) |
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
  speed. At step 260 the text says to decelerate, but the generated trajectory
  still asks for about 5.60 m/s while ego speed is 4.96 m/s; the PID therefore
  applies throttle 0.75 and brake 0.0. This is a reasoning/planning-output
  mismatch, not evidence that the controller failed to execute a conservative
  trajectory. At steps 270/280 throttle becomes zero only because ego speed is
  above the configured 5 m/s cap; brake remains zero. O1 intentionally cannot
  improve this consumer gap because `used_by_qwen=false`.
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
  Result after the yaw-compensation and urgency-render audits:
  `45 passed, 1 skipped`; Python compilation, shell syntax, and
  `git diff --check`
  also passed. The skip is the existing optional local OpenCV transport test.
- O2 remained open at this point pending a live run validating temporal warp,
  route projection, stopping-distance arithmetic, frontier selection, artifact
  integrity, and derived runtime overhead.

## O2 remote attempt 1

- Slurm job: `1165344`; run id:
  `qwen_oracle_visibility_route151_reasoning_sft_seed42_o2_v1`.
- Terminal state: Slurm `FAILED`, exit `2:0`, elapsed `00:01:06`. CARLA became
  ready, but the isolated checkout had no in-tree `Bench2Drive` directory and
  the submit wrapper had not forwarded the shared external asset path. Route
  splitting therefore failed before evaluator/model initialization; no
  experimental frames were produced and this says nothing about O2 behavior.
- The submit wrapper now resolves Bench2Drive and Bench2DriveZoo from the
  configured shared asset root, validates `tools/split_xml.py`, forwards both
  paths explicitly, and prints them in dry-run provenance. Attempt 2 was
  submitted with those explicit roots as Slurm job `1165345`.

## O2 remote attempt 2 and acceptance

- Commit under test: `c4f62543`; Slurm job: `1165345`; run id:
  `qwen_oracle_visibility_route151_reasoning_sft_seed42_o2_v2`.
- Terminal state: Slurm `COMPLETED`, exit `0:0`, elapsed `00:19:05`, peak RSS
  `6,000,380 KiB`. The route completed 100% in 26.85 s simulation time and
  again received route score 100, penalty 0.5, driving score 50 after a
  pedestrian collision at `(100.723, 303.171)`.
- The 46 MB run directory contains 54 Qwen plans, 54 belief tensors, 54 base
  belief PNGs, 54 memory PNGs, 54 exposure PNGs, 537 controller trace rows,
  five sensor-audit directories, and all 30 requested native audit images.
  Every NPZ has finite float32 visibility `[5,120,100]`, observation memory
  `[4,120,100]`, and exposure `[6,120,100]` arrays. The visibility partition,
  memory partition, current-age, never-age, stopping-margin, and exact urgency
  formula invariants all pass.
- Observation memory showed 0 to 2,075 previously observed cells and retained
  the explicit never-observed state separately. The normalized previous age
  reached 1.0, corresponding to the configured 10 s cap. The added unit audit
  also verifies that a 90-degree CARLA yaw change rotates remembered world
  evidence into the correct Qwen ego-grid cell.
- Across 54 frames, geometry time was mean 0.150 s, median 0.135 s, p95
  0.253 s, max 0.451 s. The derived memory, route, and exposure stage was mean
  0.0074 s, median 0.0071 s, p95 0.0122 s, max 0.0139 s. Stopping distance
  ranged from approximately 0 to 8.73 m. Non-zero urgent-frontier support
  ranged from 52 to 1,210 cells; frame maxima ranged from `5.29e-05` to 0.650.
- The live fields select the physically relevant forward-right occlusion near
  the parked vehicle. For example, the maximum urgency moves from
  `(x=6.25,y=-4.75)` at step 220 to `(5.75,-2.75)` at step 250 and
  `(5.25,-3.75)` at step 260. The step-260 native RGB audit shows the
  pedestrian emerging beside the parked police vehicle, while Qwen's recorded
  reasoning never mentions the pedestrian in this run.
- The available native audit at step 200 shows the parked occluder but no
  pedestrian, and already has weak forward-right exposure. However, the run
  did not save native RGB at steps 220/240/250, so the stronger signal there
  must not be claimed as strictly pre-reveal evidence from this run alone.
  Future audits now include all three steps to bracket the reveal directly.
- The combined exposure PNG made low absolute urgency visually hard to audit.
  A separate fixed-scale red-only urgency renderer is therefore added for
  future runs; it uses absolute saturation 0.5 rather than per-frame
  normalization, preserving cross-frame comparability.
- Every artifact remains `used_by_qwen=false`, and the model still collided.
  This run accepts temporal memory, route projection, exposure arithmetic,
  plausible spatial selection, integrity, and runtime only. It makes no causal
  safety claim. As with O1, the explicit oracle-depth harness also makes the
  evaluator's JSON `eligible=true` field invalid as an official SENSORS score.
- O2 acceptance: the representation is stable and inspectable enough to feed
  the tokenizer. The next gate is O3 serialization and controls; V0 must still
  prove that U actually enters the VLM prefix before any behavioral claim.

## Integrity constraints

- No Torch, Qwen, Orion, or CARLA import in the geometry module.
- Use Qwen ego coordinates: x forward, y left, z up.
- Keep occluded unknown distinct from outside-FOV.
- A point visible from any camera is not unknown merely because another camera
  sees it behind a surface.
- Oracle depth is an upper bound and must never be reported as predicted U.
- Every implementation milestone updates this file and is committed separately.

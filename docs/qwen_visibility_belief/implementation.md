# Qwen visibility-belief implementation status

Last updated: 2026-09-06 (Asia/Shanghai)

## Current milestone

`V1: structured U-grounding warm-up with staged LoRA`

Status: in progress (V1a accepted; V1b full-model gradient smoke implemented locally)

O2 is accepted as an interpretable representation milestone. It establishes
ego-motion-compensated observation age and a separate route/stopping exposure
view on live CARLA data, but it does not establish Qwen consumption or safety.
O3 has produced deterministic, audited physical tokens and causal controls. V0
has established that those tokens can enter both the direct and reasoning VLM
prefixes while preserving an exact disabled-path reproduction of the released
model. This is an interface result only: the projector is untrained, the live
agent still records `used_by_qwen=false`, and no grounding, trajectory-quality,
or safety improvement has been established. V1 must create learned,
inspectable U consumption before any closed-loop claim.

## Ordered implementation ladder

| ID | Deliverable | Status |
| --- | --- | --- |
| O0 | CARLA depth decoding, camera calibration, 3D visibility fusion, 2.5D BEV, rendering, unit tests | Complete (`6addb2fe`) |
| O1 | Add co-located oracle depth sensors to the Qwen agent behind an explicit oracle-only config | Complete (`c8fac0b5`; accepted by run `1165332`) |
| O2 | Observation-age memory and deterministic urgency/stopping-margin map | Complete (`c4f62543`; accepted by run `1165345`) |
| O3 | Global/frontier tokenizer with serialization and causal zero/shuffle controls | Complete (`2d86b809`; accepted on 54-frame derived run) |
| V0 | Insert U tokens into the 4B VLM with verified positions and disabled-path identity | Complete (`4e4672ba`; direct job `1166148`, reasoning job `1166382`) |
| V1 | Structured U-grounding warm-up with staged LoRA | In progress (V1a data contract accepted) |
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

## O3 local implementation record

- The server's dedicated clean checkout was fast-forwarded to `de24aeca`
  after O2 acceptance. Its Orion-Python targeted regression passed `46/46`
  tests; the checkout remained clean. The historical dirty Orion checkout was
  not changed.
- Added a deterministic NumPy-only physical tokenizer. The pilot shape is 16
  global tokens from a complete `4x4` BEV tiling plus at most 32 frontier
  tokens, each with 23 versioned features. No learned projector or Qwen import
  is part of O3.
- Global tokens preserve the entire dense field. Frontier candidates use a
  separately inspectable score combining urgency with a 0.05 physical-frontier
  floor, followed by 2 m metric non-maximum suppression and 2 m local pooling.
  This prioritizes exposed frontiers without deleting distant uncertainty from
  the global representation.
- Each physical token includes type, normalized metric center and extent,
  visible-free/occupied/occluded/outside-FOV height ratios, frontier fraction,
  observation age/state, depth confidence, route and stopping weights,
  urgency, frontier-weighted stopping margin, and unknown area relative to the
  full grid. Oracle confidence is explicitly 1.0; it is not silently reused as
  the future predicted-depth confidence.
- The zero-U control preserves token counts and masks while zeroing every
  feature. The spatial-shuffle control preserves token types, metric slots,
  masks, and per-feature marginals, but cyclically reassigns physical content
  between slots independently inside the global and frontier families. The
  shuffle is seeded and deterministic.
- Serialization uses only numeric/string NumPy arrays and JSON metadata, loads
  with `allow_pickle=False`, and records its schema, feature order,
  normalization, control, and selection parameters. The CARLA agent writes
  true, zero, and shuffled token sets into future oracle NPZ artifacts while
  retaining `used_by_qwen=false`.
- Added `scripts/tokenize_qwen_visibility_artifacts.py` to derive token files
  from an immutable O2 run into a new refuse-to-overwrite directory. It checks
  the dense channel contracts, validates oracle/non-consumer provenance, emits
  a manifest and SHA-256 ledger, and never edits the source artifacts.
- The first local converter test exposed an accidental import of the legacy
  `uq_estimator` package, which would load Torch. The converter now loads the
  bridge and visibility files in isolation, matching the CARLA agent boundary.
- Local relevant regression after that repair: `49 passed, 1 skipped`;
  compilation, shell syntax, and `git diff --check` pass. O3 remains open until
  the converter is run over all 54 accepted O2 frames and token invariants,
  spatial selection, serialization, hashes, and runtime are audited remotely.

## O3 remote real-artifact acceptance

- Commit under test: `2d86b809`. The dedicated remote checkout targeted
  regression passed `50/50` tests and remained clean.
- Derived all 54 O2 frames into the new directory
  `/public/share/lidachuan/orion_assets/qwen_visibility_token_runs/qwen_route151_o2_v2_tokens_o3_v1`.
  The accepted O2 source directory was opened read-only by the converter and
  was not used as an output location.
- Output size is 823 KB: 54 token NPZ files plus `manifest.json` and
  `artifact_sha256.txt`. All 54 SHA-256 checks pass. Mean tokenization time,
  including construction of true, zero, and shuffled sets, is 0.0582 s; p95 is
  0.0833 s.
- Every frame has true global `[16,23]` and frontier `[32,23]` tensors, valid
  masks, finite values, and pickle-free metadata. All 32 frontier slots are
  populated on every frame. This means the frontier budget is saturated, not
  that omitted space disappears: the complete `4x4` global tiling separately
  preserves the full-field statistics.
- Across all frames, the sum of global unknown-area features equals the dense
  source field's unknown-area fraction. Frontier selection scores are sorted;
  zero-U arrays are exactly zero with unchanged masks; shuffled arrays keep the
  first six type/metric-slot features fixed and preserve every content-feature
  marginal. No invariant failure was observed.
- The strongest frontier remains physically consistent with the O2 audit:
  `(x=8.25,y=-4.25)` at step 200, `(6.25,-4.75)` at 220,
  `(5.25,-4.25)` at 240, `(5.75,-2.75)` at 250, and
  `(5.25,-3.75)` at 260. Its local urgency maximum rises from 0.011 at step
  200 to 0.650 at step 250.
- O3 acceptance is limited to deterministic compression, serialization,
  controls, spatial plausibility, and runtime. The tokens are still marked
  `used_by_qwen=false`; this run provides no grounding, trajectory, or safety
  evidence. V0 is the next gate.

## V0a interface-contract record

- Audited the provisioned Qwen-Drive and installed Transformers source. The 4B
  VLM hidden width is 2,560; the released Planning Expert consumes post-rotary
  K/V from eight full-attention VLM layers and continues waypoint mRoPE from
  the final prefix anchor.
- The runtime imports Transformers 5.14.1 although the checkpoint config names
  5.15.0. A live smoke against the exact installed stack is therefore required.
- Added `docs/qwen_visibility_belief/vlm_insertion_contract.md`. It fixes the
  insertion after the last image and before history/navigation text, preserves
  official image scatter, explicitly recomputes augmented three-axis positions,
  omits invalid frontier padding from Planning Expert cache, and distinguishes
  official disabled identity from augmented zero-U control.
- Added the sidecar-only `uq_estimator/qwen_visibility_vlm.py` with a
  zero-output-initialized 23-to-512-to-2560 projector, learned boundary vectors,
  official disabled path, augmented direct-prefill path, and explicit position
  and scene-cache audit.
- Added an actual-model smoke and Slurm submit wrapper using the accepted
  step-260 O3 token artifact and native three-camera audit images. This is V0a
  plumbing only. V0 remains open until the direct-prefill smoke passes and the
  reasoning-planning insertion path is implemented and verified.

## V0a remote attempt 1

- Commit: `b69eb859`; Slurm job: `1166130`; run id:
  `qwen_visibility_vlm_insertion_v0_step260_v1`.
- The job loaded the released 4B model, completed the official, disabled, and
  augmented prefill paths, and completed all three Planning Expert calls. It
  then failed while formatting the report because the smoke treated
  Qwen-Drive's already-NumPy `_plan_from_cache` return as a Torch tensor and
  called `.cpu()` on it.
- Terminal state: `FAILED`, exit `1:0`, elapsed `00:06:21`, peak host RSS
  `2,917,768 KiB`. This is a harness type error after the interfaces under test,
  not an accepted V0 result. The v1 output/log is retained; attempt 2 uses a new
  run id and normalizes the three returns with `np.asarray`.

## V0a remote attempt 2 and acceptance

- Commit: `e599db2f`; Slurm job: `1166148`; run id:
  `qwen_visibility_vlm_insertion_v0_step260_v2`.
- Terminal state: `COMPLETED`, exit `0:0`, elapsed `00:05:37`, peak host RSS
  `3,683,280 KiB`. The report contains no contract failure.
- The official prompt has 4,348 positions. The accepted O3 frame contributes
  48 valid physical tokens; with two learned boundary vectors the augmented
  prefix has 4,398 positions. The insertion index is 4,109, immediately after
  the final vision-end token and before the driving instruction.
- All pre-insertion three-axis mRoPE positions are unchanged; every suffix
  position advances by exactly 50; all 50 U-block positions are contiguous;
  the anchor equals the final augmented position. Every one of the eight
  Planning Expert scene-cache layers has sequence length 4,398.
- The disabled path is bit-identical to the released path in cache, anchor, and
  fixed-seed trajectory. The projector has 1,330,734 parameters and its output
  projection and boundaries are zero-initialized for this interface probe.
- Even with zero projected values, merely inserting the 50 attention slots
  changes the untrained trajectory by maximum absolute 0.12085. This confirms
  the contract's warning: augmented zero-U is a paired causal control, not a
  substitute for the released no-block baseline.
- Warm measured prefills in this three-cache diagnostic were 0.964 s for the
  released disabled call and 0.793 s for the augmented call after a 12.54 s
  first prefill. Peak allocated/reserved GPU memory was 11,255/14,024 MB.
- V0a is accepted for direct prefill only. It provides interface evidence, not
  grounding or safety evidence.

## V0b reasoning-path implementation record

- Added manual greedy continuation from an arbitrary continuous-token prefix,
  including the released minimum reasoning length, deterministic argmax,
  terminators, assistant turn closure, cache positions, and final Planning
  Expert anchor.
- Added a no-U manual reference path whose reasoning text, final cache, anchor,
  and fixed-seed trajectory must exactly match upstream
  `_prefill_with_reasoning`. The same smoke can now run in `direct` or
  `reasoning` mode.
- V0 remains open until a full-model reasoning smoke passes both the no-U
  reproduction and the augmented prompt/cache contracts. The projector remains
  untrained and behavior-neutral claims remain prohibited.

## V0b remote reasoning acceptance

- Commit under test: `4e4672ba`; Slurm job: `1166382`; run id:
  `qwen_visibility_vlm_insertion_v0_step260_reasoning_v1`.
- Terminal state: `COMPLETED`, exit `0:0`, elapsed `00:05:58`, peak host RSS
  `3,217,192 KiB`. The report contains no contract failure.
- The released upstream path, the manually reproduced no-U reference, and the
  zero-initialized augmented path all greedily generated the same sentence:
  `Accelerate to target speed along the clear lane markings.` The manual
  reference is exactly equal to upstream in final cache, anchor, and fixed-seed
  Planning Expert trajectory.
- The no-U reasoning prefix has 4,363 positions. The accepted O3 frame adds 48
  valid physical tokens and two boundary vectors, producing 4,413 positions
  before continuation. The insertion index is 4,109; all prefix, suffix-shift,
  contiguous-U-position, final-anchor, and eight-layer cache checks pass.
- After reasoning continuation and assistant-turn closure, all eight reference
  caches have length 4,380 and all eight augmented caches have length 4,430.
  This verifies that the accepted insertion remains present through reasoning
  generation and reaches the Planning Expert.
- The zero-initialized augmented trajectory differs from the official path by
  maximum absolute 0.68481. As in V0a, this confirms that zero-U is not the
  released baseline: the added attention slots and shifted suffix are already
  an intervention.
- Model load took 161.58 s. The cold upstream reasoning call took 19.66 s;
  subsequent manual reference and augmented calls took 1.60 s and 1.62 s.
  These mixed cold/warm figures are retained for diagnostics and are not a
  stable latency comparison. Peak allocated/reserved GPU memory was
  12,102/12,366 MB.
- V0 is accepted as an interface milestone. It proves exact disabled-path
  reproduction and real full-model U-block propagation through direct and
  reasoning paths. It does not prove that the untrained VLM understands U, that
  trajectories improve, or that Route 151 becomes safer. The live CARLA agent
  has not yet been wired to a trained sidecar and still writes
  `used_by_qwen=false`; those are V1/C0 deliverables.

## V1a structured-grounding data-contract record

- Added `docs/qwen_visibility_belief/grounding_contract.md` before starting an
  optimizer. It fixes the exact four-field answer, deterministic target
  thresholds, frozen-model boundary, nonclaims, and acceptance gates.
- Added a NumPy-only target/manifest module and refuse-to-overwrite builder.
  The builder joins immutable O3 true-U artifacts to native 1600x900 RGB audit
  images, hashes every input, records all numeric label evidence, and marks the
  sparse Route 151 set as a non-reportable plumbing overfit.
- O3 frontier rows are serialized in descending score order, which would make
  the maximum-score identity target trivially `F00`. V1 therefore records and
  applies a seeded complete-row permutation before deriving the frontier label.
  This changes only arbitrary sequence order; it is distinct from the
  spatial-shuffle causal control, which deliberately misaligns content.
- Zero-U and spatial-shuffle controls are prohibited from optimizer examples.
  Hidden-actor labels and the Planning Expert are also absent from this stage.
- Local target, boundary, permutation, provenance, hashing, overwrite, O3, and
  bridge regression: `48 passed, 1 skipped`. Python compilation and
  `git diff --check` pass. V1a remains open until the manifest is built and
  audited against the five retained real Route 151 sensor-audit steps.

## V1a real-manifest acceptance

- Commit under test: `3cc21145`. The dedicated remote checkout remained clean
  and its target/geometry/bridge regression passed `49/49` tests in the Orion
  Python environment.
- Built the refuse-to-overwrite manifest at
  `/public/share/lidachuan/orion_assets/qwen_visibility_grounding_runs/route151_v1a_manifest_v1/manifest.json`.
  It joins all five retained audit steps `0, 200, 260, 280, 300` to their exact
  O3 artifacts and three native RGB images.
- An independent readback verified every token and image SHA-256, all five
  complete 32-row permutations, `hidden_actor_labels_used=false`, and
  `controls_used_for_optimizer=false`; no integrity failure was found.
- The seeded permutation moves the original maximum-score row from original
  index 0 to sequence labels `F14, F11, F04, F11, F11`, so the task cannot be
  solved by emitting `F00`. The corresponding action labels are one `KEEP`,
  two `SLOW`, and two `STOP`; margins are one `CLEAR`, two `NEAR`, and two
  `INSIDE`.
- All five sparse audit records are `ON_ROUTE`. This is useful negative
  evidence: the retained Route 151 snapshots do not contain the accepted
  route-irrelevant hard-negative class. The manifest is sufficient for a
  gradient and disposable overfit plumbing probe only, not V1 learned-consumer
  acceptance or a semantic-conditioning claim.
- V1a is accepted as a data-contract milestone. V1b must prove that answer-only
  loss reaches both the projector and declared upper-VLM LoRA tensors while
  the vision encoder, base VLM, LM head, embeddings, and Planning Expert remain
  frozen.

## V1b full-model gradient-smoke implementation record

- Added a sidecar-only training module that freezes every released model
  parameter, then installs float32 rank-8 LoRA residuals only on `q/k/v/o` in
  upper full-attention layers 27 and 31. Installation fails closed if layer
  type, projection type, or trainable scope differs from the declared config.
- The 23-to-512-to-2560 projector remains fully trainable. The vision encoder,
  token embeddings, LM head, all non-LoRA VLM tensors, and complete Planning
  Expert must report zero trainable parameters before optimization.
- The supervised forward reuses native current-image processing and the V0
  insertion/position contract. Cross-entropy is computed only for the compact
  JSON assistant answer and ChatML turn ending; prompt, image, and U positions
  are never labels. Frozen image embeddings are precomputed without changing
  resolution, while the full language path remains differentiable to U.
- The bounded V1b protocol permits exactly one optimizer step on Route 151 step
  260. It records gradients before that step, evaluates true/zero/spatially
  shuffled U after it, and saves only projector/LoRA adaptation tensors plus
  provenance and SHA-256. It explicitly prohibits a grounding, trajectory, or
  safety claim.
- The Slurm wrapper requests one A800, 96 GB host memory, and leaves native
  image preprocessing intact. V1b remains unaccepted until the real 4B job
  proves finite nonzero gradients to both adaptation families, frozen-scope
  integrity, answer-only lengths, control evaluation, and a base-weight-free
  checkpoint.

## Integrity constraints

- No Torch, Qwen, Orion, or CARLA import in the geometry module.
- Use Qwen ego coordinates: x forward, y left, z up.
- Keep occluded unknown distinct from outside-FOV.
- A point visible from any camera is not unknown merely because another camera
  sees it behind a surface.
- Oracle depth is an upper bound and must never be reported as predicted U.
- Every implementation milestone updates this file and is committed separately.

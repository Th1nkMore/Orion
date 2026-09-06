# Qwen visibility-belief implementation status

Last updated: 2026-09-06 (Asia/Shanghai)

## Current milestone

`V1: structured U-grounding warm-up with staged LoRA`

Status: in progress (V1i valid negative; field-identifiability audit next)

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
| V1 | Structured U-grounding warm-up with staged LoRA | In progress (V1a/V1b accepted; V1c-V1i valid negatives) |
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

## V1b remote full-model acceptance

- Commit under test: `f285399f`; Slurm job: `1166774`; run id:
  `qwen_visibility_grounding_v1b_step260_gradient_v1`.
- Terminal state: `COMPLETED`, exit `0:0`, elapsed `00:09:42`, peak host RSS
  `1,693,340 KiB`. The model load took 175.17 s. Peak allocated/reserved GPU
  memory was 12,035/12,492 MB, so native three-camera input and a complete
  checkpointed language backward fit on the requested A800 without resolution
  reduction.
- The answer has 25 supervised tokens. The native-image prompt has 2,798
  positions; the 48 valid U tokens plus two boundaries produce 2,848 prompt
  positions and 2,873 total teacher-forced positions. Insertion remains at
  position 2,701 after the final image.
- Trainable scope is exactly 1,330,734 projector parameters plus 393,216 LoRA
  parameters. Vision, embeddings, LM head, and Planning Expert report zero
  trainable parameters. All eight LoRA-B tensors across layers 27/31 and all
  four `q/k/v/o` projections receive finite nonzero gradients before the first
  optimizer step. Projector boundary/output tensors also receive finite
  nonzero gradients; its earlier layers correctly have zero first-step
  gradients because the output projection starts at zero.
- Initial answer-only loss is 1.65026. The combined pre-clip gradient norm is
  6.727e9, driven by the projector output projection, while the largest LoRA
  gradient norm is 0.352. The original joint clipping would therefore nearly
  erase the LoRA update. This is an observed optimization defect to fix before
  the multi-step probe, not evidence for increasing model capacity.
- The one-step outputs are not grounded: true-U and shuffled-U terminate with
  an empty answer; zero-U emits a fenced JSON object with wrong frontier,
  margin, and action. Exact and per-field accuracy are zero. This is expected
  to keep V1 open: V1b accepts differentiable connectivity and frozen scope,
  not learned consumption.
- The adaptation checkpoint is 6,906,037 bytes with SHA-256
  `590dc9731269a7441b057928ace06bc4d8fb3fceb73513ead99995522205a6d7`.
  Independent `weights_only` loading confirms exactly seven projector tensors
  (1,330,734 values), sixteen LoRA tensors (393,216 values), no base keys, and
  no optimizer state.
- V1b is accepted. V1c must separate projector and LoRA gradient clipping,
  record per-family parameter updates, and attempt a bounded five-frame
  plumbing overfit. Even a successful V1c remains non-reportable because all
  five records are Route 151 and lack `OFF_ROUTE` examples.

## V1c bounded-overfit implementation record

- The trainer now clips projector and LoRA gradients independently and records
  each family's pre-clip norm, nonzero-gradient tensor count, and actual
  parameter-update norm per step. This directly fixes the V1b failure mode in
  which the projector's 6.7e9 norm scaled the already modest LoRA gradients to
  near zero under one joint clip.
- The fixed V1c protocol cycles exactly three times over all five immutable
  Route 151 plumbing records for 15 optimizer steps. It uses a lower `1e-4`
  projector rate, retains the `2e-4` LoRA rate, records true-U outputs before
  training, and evaluates true/zero/spatial-shuffle after training.
- Native three-camera image inputs, the V0 U insertion, answer-only loss,
  rank/layer scope, frozen base, and non-reportable claim boundary are
  unchanged. The wrapper permits a separately declared four-hour Slurm window;
  it does not reduce image resolution or remove camera views.
- V1c is a capacity/plumbing overfit, not V1 acceptance. A successful run can
  justify building the held-out parameterized grounding set; it cannot provide
  semantic generalization or closed-loop safety evidence.
- Added a separate fail-closed report auditor before observing V1c results. It
  first verifies frozen scope, input/control coverage, balanced optimizer use,
  finite nonzero updates in both adaptation families, and checkpoint integrity.
  It then requires at least four of five true-U exact answers, at least four of
  five correct answers for every field, and a two-of-five exact/action gap over
  the stronger zero/shuffle control. Protocol validity and causal capacity are
  reported separately, so a valid negative result cannot be promoted to a
  pass. Local evaluation-contract regression brings the relevant suite to
  `51 passed, 2 skipped`.

## V1c remote negative result

- Commit under test: `4108b62b`; Slurm job: `1166797`; run id:
  `qwen_visibility_grounding_v1c_route151_overfit_v1`.
- Terminal state: `COMPLETED`, exit `0:0`, elapsed `00:08:20`, peak host RSS
  `456,148 KiB`. Model load took 191.02 s, native-image preparation 10.28 s,
  five pre-training generations 28.16 s, all 15 optimizer steps 19.33 s, and
  fifteen post-training control generations 29.16 s. Peak allocated/reserved
  GPU memory was 12,269/12,912 MB.
- The predeclared auditor passed every protocol invariant: all five records
  appear exactly three times in the optimizer, both adaptation families have
  finite nonzero updates on every step, the base/vision/expert boundary is
  intact, every evaluation arm is present, and checkpoint integrity passes.
- Separate clipping fixed the observed optimizer defect. After step 1, all
  seven projector tensors and all sixteen LoRA tensors receive finite nonzero
  gradients. Projector update norm declines from 0.115 to 0.026; LoRA update
  norm declines from 0.090 to 0.065 rather than being globally scaled to zero.
- Teacher-forced answer loss falls from 1.60228 to 0.30945. Nevertheless,
  true-U exact accuracy is `0/5`; frontier accuracy is `0/5`, margin `2/5`,
  action `2/5`, and route `5/5`. Shuffled-U has the same action accuracy and a
  higher margin accuracy. The true-minus-control exact and action gaps are both
  zero. Audit status is `valid_run_without_causal_grounding`.
- Before training the base model emits the same fenced `F00/ON_ROUTE/INSIDE/SLOW`
  JSON for every frame. After training, true and shuffled U mostly emit the
  same compact `F10/ON_ROUTE/INSIDE/SLOW` JSON. The model learned the requested
  serialization and majority fields, not the physical token content.
- This is an accepted negative experiment, not a failed protocol and not V1
  acceptance. The 25-token composite answer devotes most CE weight to fixed
  syntax and common values; only a few tokens encode the varying supervision.
  ADR-001 and the grounding contract now require a balanced factorized
  four-field warm-up before returning to composite answers. More composite
  steps alone are not the next experiment.

## V1d factorized-overfit pre-run contract

- The V1d protocol changes only the supervision curriculum exposed by the V1c
  negative result. The physical targets, native three-camera inputs, V0 token
  insertion, projector, rank-8 LoRA scope, frozen released model boundary, and
  non-reportable Route 151 manifest remain unchanged.
- The four labels are presented as separate exact-answer questions. All 20
  `(five immutable frames, four fields)` pairs receive exactly three optimizer
  updates for 60 total steps. Zero-U and spatial-shuffle remain evaluation-only
  interventions and never enter the optimizer.
- The factorized questions state the numeric route, normalized stopping-margin,
  urgency, and action rules explicitly. This prevents success from depending
  on an unstated rule that the model could not infer from the prompt.
- A stage-specific fail-closed auditor is fixed before the run. It verifies the
  hashed protocol and manifest, exact 20/60 evaluation coverage, balanced
  optimizer pairs, finite nonzero updates in both adapter families, frozen
  scope, and adaptation-only checkpoint integrity before scoring capacity.
- Capacity requires `4/5` true accuracy for every field, a `2/5` frontier gap
  over the stronger zero/shuffle control, and `2/5` margin and action gaps over
  zero U. No route causal gap is permitted as a claim because all five targets
  are `ON_ROUTE`. Margin/action are not required to degrade under spatial
  shuffle because that control moves each frontier's complete content bundle
  and therefore preserves those values.
- A passing V1d result is still disposable capacity evidence, not V1 acceptance
  or safety evidence. A valid negative result is also retained. The remote run
  has not started at the time this contract is committed.

## V1d remote negative result

- Commit under test: `6fffebdd`; Slurm job: `1167134`; run id:
  `qwen_visibility_grounding_v1d_route151_factorized_v1`.
- Terminal state: `COMPLETED`, exit `0:0`, elapsed `00:07:55`, peak host RSS
  `2,966,824 KiB`. Model load took 164.44 s, native-image preparation 29.35 s,
  20 pre-training generations 23.47 s, all 60 optimizer steps 48.27 s, and 60
  post-training control generations 30.40 s. Peak allocated/reserved GPU memory
  was 13,399/13,832 MB; no input resolution or camera view was changed.
- The predeclared audit is protocol-valid with no failures. Every one of the 20
  frame/field pairs appears exactly three times; both adaptation families have
  finite nonzero updates at every step; the vision encoder, base VLM, embeddings,
  LM head, and Planning Expert remain frozen; all 20/60 evaluation rows exist;
  and the protocol, manifest, report, and checkpoint hashes are intact.
- Teacher-forced loss falls from 4.99236 on the first step to 0.43227 on the
  last. Per-field mean loss from epoch 1 to epoch 3 falls from 2.558 to 0.732
  for frontier, 2.743 to 0.002 for route, and 2.037 to 0.515 for action; margin
  changes from 2.544 to 1.283 after reaching 1.141 in epoch 2.
- True-U exact accuracy is frontier `3/5`, route `5/5`, margin `2/5`, and action
  `3/5`. The frontier gap over the stronger zero/shuffle control is `0/5`;
  margin and action each beat zero U by only `1/5`. Every predeclared causal
  capacity check fails, so the audit status is
  `valid_run_without_causal_grounding`.
- The failure pattern remains shortcut-dominated. All five true, zero, and
  shuffled frontier queries emit `F11`; that happens to match three of the five
  targets. Route always emits the dataset's constant `ON_ROUTE`. Margin mostly
  emits `NEAR`, and action mostly emits `SLOW`, with one learned `STOP`. The
  factorized objective reduced syntax dominance but did not force the model to
  read U.
- The adaptation checkpoint is 6,906,485 bytes with SHA-256
  `81e3ad416452e37e360f90070b8c67891561991f0e6d6797a8223bea2eb9d3a4`.
  Independent `weights_only` readback verifies seven projector tensors
  (1,330,734 values), sixteen LoRA tensors (393,216 values), no optimizer state,
  and no base-model tensors. Report and audit SHA-256 are respectively
  `731a2f57387c933dff43f5b28365d21de4bf84068c25669cfc537a9bebcad84b`
  and `cbb79c5239b74073dacf539a04e8f4f5265921737cb020767c15c14ff2ccbb29`.
- This is an accepted negative experiment, not evidence that the Qwen backbone
  cannot consume continuous U. Each image still has only one U/label tuple and
  a fixed frontier permutation, so image identity and majority answers remain
  easier shortcuts. Repeating the same protocol for more steps would not remove
  that confound. V1e must make one image/token set support multiple row-addressed
  labels from its real frontier records and balance the queried classes before
  any capacity conclusion is revisited.

## V1e row-addressed-overfit pre-run contract

- V1e derives every target from the 32 real frontier rows already present in
  each immutable V1a token artifact. It does not invent an actor, risk value,
  or action label. The existing route/margin/urgency thresholds remain the sole
  source of `ON/OFF_ROUTE`, `INSIDE/NEAR/CLEAR`, and `KEEP/SLOW/STOP`.
- Complete-row cyclic permutations place different real physical rows at the
  same query slot. Route always queries `F00`, margin `F01`, and action `F02`;
  within a field, the same image/slot receives different labels. Frontier-max
  examples move the true maximum to balanced `F03`, `F13`, and `F23` slots.
  Because coordinates and content move together, this is an arbitrary sequence
  reordering rather than the O3 spatial-shuffle intervention.
- Candidate rows are selected only when their spatial-shuffle target differs
  from their true target. Selection is deterministic, uses the highest-score
  qualifying row per sample/class, and balances the 43 unique examples as 15
  frontier, 10 route, 12 margin, and 6 action questions. Label counts are
  respectively `5/5/5`, `5/5`, `4/4/4`, and `2/2/2`.
- The immutable curriculum records every question, source row, complete
  permutation, true/zero/shuffle expected answer, and an explicit 360-step
  schedule. Each field receives 90 optimizer updates and every label within a
  field receives equal updates. Controls remain evaluation-only.
- The stage-specific auditor is fixed before the run. It verifies the protocol,
  base manifest, curriculum, schedule, per-example evaluation coverage, frozen
  model scope, finite nonzero adapter updates, and adaptation-only checkpoint.
  Capacity requires at least 80% true-U accuracy for every field, at least 80%
  for every label within every field, and a true-minus-stronger-control gap of
  at least 30 percentage points for every field.
- Native three-camera image processing, V0 insertion, projector, rank-8 LoRA,
  and the Planning Expert freeze are unchanged. Passing remains a disposable
  Route 151 plumbing result; the remote curriculum and training run have not
  started at the time this contract is committed.

## V1e remote negative result

- Commit under test: `9e9dc348`; Slurm job: `1167453`; run id:
  `qwen_visibility_grounding_v1e_route151_row_addressed_v1`.
- Terminal state: `COMPLETED`, exit `0:0`, elapsed `00:12:18`, peak host RSS
  `2,828,268 KiB`. Model load took 170.31 s, native-image preparation 25.68 s,
  43 pre-training generations 49.83 s, all 360 optimizer steps 252.93 s, and
  129 post-training control generations 63.47 s. Peak allocated/reserved GPU
  memory was 15,133/15,580 MB; native image processing and all three camera
  views were retained.
- The immutable curriculum SHA-256 is
  `036ad03b11c715e0ba69c0ce171e43172fbb1117cd97e14797dafb637070d7fe`.
  Its 43 real-row examples and explicit schedule give every field 90 updates
  and balance every label. The predeclared auditor reports no protocol failure;
  controls never enter the optimizer, hidden-actor labels are absent, the
  Planning Expert and released base remain frozen, and both adaptation families
  have a finite nonzero update at every optimizer step.
- Pre-training exact accuracy is zero for all four fields. Post-training true-U
  accuracy is frontier `6/15`, route `8/10`, margin `6/12`, and action `4/6`.
  Per-label accuracy is frontier `F03 0/5`, `F13 5/5`, `F23 1/5`; route
  `ON_ROUTE 4/5`, `OFF_ROUTE 4/5`; margin `INSIDE 0/4`, `NEAR 3/4`,
  `CLEAR 3/4`; and action `KEEP 1/2`, `SLOW 2/2`, `STOP 1/2`.
- The true-U advantage over the stronger zero/spatial-shuffle arm is only 0,
  10.0, 8.3, and 16.7 percentage points for frontier, route, margin, and action.
  All three preregistered capacity checks therefore fail, and the audit status
  is `valid_run_without_causal_grounding`.
- Training is real but insufficient. Overall first/last-step loss is 5.01959
  to 1.38632. Mean loss over the first versus last 30 updates of each field is
  frontier 0.847 to 0.246, route 0.468 to 0.159, margin 0.800 to 0.327, and
  action 0.652 to 0.277. Projector and LoRA update norms are nonzero on all 360
  steps. Predictions are more class-diverse than V1d, but the model still
  overpredicts `F13`, never learns `INSIDE`, and mostly preserves the original
  target under target-changing spatial shuffles.
- The adaptation checkpoint is 6,906,677 bytes with SHA-256
  `36ba2281848901c8eb122558cd5c64963c25365086a3e41c94648b79650b3a0d`.
  Independent `weights_only` readback verifies seven projector tensors
  (1,330,734 values), sixteen LoRA tensors (393,216 values), no optimizer state,
  and no base-model tensors. Report and audit SHA-256 are respectively
  `d5dd61925a845b06037491f2f0fb98aae9290c50622f03160293d12202fcffc9`
  and `bbb552b1339cb488d73eb2589c8f49323823d08d7106898dc1ec9bc72ea1e503`.
- V1e closes the fixed-image, fixed-slot, and globally imbalanced-label
  explanations, but each label is still correlated with one deterministic
  cyclic ordering of all non-queried rows. The negative result therefore
  rejects this bounded consumer protocol, not continuous-U consumption in
  general. The next bounded diagnostic must randomize non-queried complete-row
  order while holding the queried row and label fixed, then test unseen row
  orders and target-changing swaps. Blindly extending V1e is not justified.

## V1f matched-pair route-readout pre-run contract

- V1f is a deliberately narrower causal-capacity diagnostic, not a fourth
  attempt to claim full grounding. It tests the simplest physical readout that
  V1e partially learned: whether `route_weight_mean` in the complete frontier
  record at `F00` is at least 0.2.
- Each of the five immutable Route 151 frames contributes one real
  shuffle-sensitive `ON_ROUTE` row and one real shuffle-sensitive `OFF_ROUTE`
  row. A matched pair has the same image, question, complete 32-row multiset,
  and non-query order. Its two members differ only by swapping the row at
  `F00` with the paired row at one other slot. No individual feature, actor,
  risk, or action label is synthesized.
- Every frame has three training pair variants and two evaluation-only pair
  variants. The non-query 31-row order is independently seeded per variant.
  The 30 training examples are balanced 15/15 and each appears exactly eight
  times in a 240-step alternating-label schedule. The 20 held-out-order
  examples are balanced 10/10 and never enter the optimizer.
- The Planning Expert, vision encoder, base VLM weights, embeddings, and LM
  head remain frozen. Only the unchanged 1,330,734-parameter projector and
  393,216 upper-layer LoRA parameters train. Native three-camera processing,
  V0 insertion, separate gradient clipping, zero U, and the O3 spatial-shuffle
  control remain unchanged.
- The stage-specific audit is fixed before the run. It requires at least 90%
  held-out-order true-U accuracy overall and for each route label, at least 80%
  of held-out matched pairs to flip correctly, at least 80% accuracy against
  the spatial-shuffle arm's changed control target, and at least a 30-point
  true-U gap over the stronger zero/shuffle true-target accuracy. It also
  fail-closes on split leakage, pair structure, hashes, trainable scope,
  gradient/update connectivity, coverage, and checkpoint contents.
- Passing establishes only that the current VLM path can causally read one
  continuous scalar from a locally addressed U row under unseen decoy orders.
  It does not accept full structured grounding, planning, or safety. Failure
  motivates an explicit typed/feature-aware modality adapter while retaining
  VLM-first consumption; it does not automatically promote direct Planning
  Expert injection.

## V1f remote negative result

- Commit under test: `eb91f55a`; Slurm job: `1167927`; run id:
  `qwen_visibility_grounding_v1f_route151_route_readout_v1`.
- Terminal state: `COMPLETED`, exit `0:0`, elapsed `00:10:03`, peak host RSS
  `2,890,012 KiB`. Model load took 164.02 s, preparation 34.99 s, 20
  pre-training held-out generations 23.71 s, all 240 optimizer steps 173.74 s,
  and 60 post-training held-out/control generations 30.57 s. Peak
  allocated/reserved GPU memory was 15,637/15,974 MB; native three-camera
  processing was unchanged.
- The real curriculum SHA-256 is
  `63bb8fbe6fc929591b5e97bbd55e30a1437c309a58b507765085f00b4566d592`.
  Independent readback verifies 25 matched pairs, a 30-example training split,
  a disjoint 20-example held-out-order split, balanced 120/120 optimizer label
  counts, complete-row permutations, two-position pair swaps, and no
  control/hidden-actor/Planning-Expert optimizer input.
- The stage-specific audit is protocol-valid with no failures. Pre-training
  held-out accuracy is `0/20`. Post-training true-U accuracy is `15/20`:
  `ON_ROUTE 9/10` and `OFF_ROUTE 6/10`. Only `5/10` matched pairs are correct
  on both members. Spatial shuffle reaches `9/20` on its changed target, and
  the true-U gap over the stronger control is 20 percentage points. Every
  preregistered causal-capacity check fails; status is
  `valid_run_without_causal_route_readout`.
- Loss and updates confirm a real optimization path rather than an execution
  failure. First/last-step loss is 6.56275 to 0.14613; first/last 30-step mean
  loss is 0.81566 to 0.17091. Projector and LoRA update norms are finite and
  nonzero on every step. On unseen orders, both pairs for steps 000000 and
  000200 collapse to `ON_ROUTE`, while one step-000280 pair collapses to
  `OFF_ROUTE`; this is consistent with an image/order fallback rather than a
  stable local F00 readout.
- The adaptation checkpoint is 6,906,677 bytes with SHA-256
  `1ae5b805e91e8522dd711105b9cedd25607911cd06c5c2ab6b370a50d517e586`.
  Independent `weights_only` readback again verifies seven projector tensors
  (1,330,734 values), sixteen LoRA tensors (393,216 values), no optimizer state,
  and no base-model tensors. Report and audit SHA-256 are respectively
  `bccc36c59e681fee7acba1f5ea784fd5d80c34349dc1743dc1f2749c09f25144`
  and `3687ff9bd95d380db5cdabe01689b347f753db0725ceeb20d7195a1aea6d850b`.
- V1f rejects the unchanged generic whole-row MLP plus upper-layer-LoRA recipe
  as a sufficient bounded small-data consumer. The next experiment must change
  the modality adapter so physical field identity and scalar value structure
  are explicit while Qwen remains the semantic consumer. More V1e/V1f epochs
  are not the next step, and the Planning Expert fallback remains inactive.

## V1g typed-scalar route-readout pre-run contract

- V1g changes one component relative to V1f: the physical-token projector.
  It reuses the exact immutable curriculum, training/held-out split, 240-step
  schedule, questions, controls, LoRA scope, native camera inputs, optimizer,
  seed, and predeclared capacity gates.
- The rejected generic projector first applied row-wise LayerNorm across all
  23 heterogeneous physical fields, then mixed them with one linear layer.
  V1g applies no cross-field input normalization. Each field is deterministically
  expanded in its own disjoint four-channel scalar block as
  `[x, x^2, sin(pi*x), cos(pi*x)]`; only then are the typed blocks learnedly
  mixed, hidden-normalized, and projected to Qwen's 2,560-wide space. It still
  emits one token per physical row and uses the unchanged V0 insertion path.
- The typed projector has 1,367,040 trainable parameters, 36,306 more than the
  generic projector. The difference is entirely inside the modality adapter;
  it does not add actor, route-decision, risk, or action prediction to the small
  module. The vision encoder, base VLM, embeddings, LM head, and Planning
  Expert remain frozen; the same 393,216 upper-layer LoRA parameters train.
- The V1f held-out-order gate is reused verbatim: overall and per-label true-U
  accuracy at least 90%, matched-pair correctness at least 80%, changed
  spatial-shuffle target accuracy at least 80%, and a true-U causal gap of at
  least 30 points. The auditor additionally verifies the typed-projector config
  and exact trainable parameter count.
- Passing would accept only typed route-scalar readout as a plumbing lower
  bound and justify restoring margin/frontier/action tasks around the typed
  adapter. Failure would leave structured VLM consumption open and motivate a
  stronger semantically anchored or query-based adapter review; it would not
  by itself satisfy the ADR's Planning Expert fallback condition.

## V1g remote negative result

- Commit under test: `6cc097f9`; Slurm job: `1168178`; run id:
  `qwen_visibility_grounding_v1g_route151_typed_route_readout_v1`.
- Terminal state: `COMPLETED`, exit `0:0`, elapsed `00:10:00`, peak host RSS
  `2,965,416 KiB`. Model load took 170.45 s, preparation 27.19 s, pre-training
  held-out generation 22.95 s, 240 optimizer steps 171.23 s, and post-training
  evaluation 32.09 s. Peak allocated/reserved GPU memory was
  15,638/15,974 MB; native inputs were unchanged.
- The V1f curriculum was reused byte-for-byte at SHA-256
  `63bb8fbe6fc929591b5e97bbd55e30a1437c309a58b507765085f00b4566d592`.
  The stage-specific audit is protocol-valid with no failures and verifies the
  typed projector's exact 1,367,040 parameters, frozen released-model boundary,
  held-out split, pair structure, schedule, controls, updates, and checkpoint.
- All 20 true-U, 20 zero-U, and 20 spatial-shuffle held-out generations emit
  `OFF_ROUTE`. True-U accuracy is therefore 50% (`OFF_ROUTE 10/10`,
  `ON_ROUTE 0/10`), matched-pair accuracy is 0%, spatial-shuffle changed-target
  accuracy is 50%, and the true-U causal gap is zero. Every predeclared gate
  fails; status is `valid_run_without_causal_typed_route_readout`.
- The optimization path is active: first/last-step loss is 6.56275 to 0.10740,
  first/last 30-step mean loss is 0.87931 to 0.20084, and both projector and
  LoRA update norms are finite and nonzero on all 240 steps. The low training
  loss together with complete held-out collapse is memorization, not evidence
  of a working physical readout.
- The adaptation checkpoint is 7,052,021 bytes with SHA-256
  `d82997c22f00bdb49659585af70599cb7f92dc90b4b106480003c5a7f9fb9743`.
  Independent `weights_only` readback verifies seven typed-projector tensors
  (1,367,040 values), sixteen LoRA tensors (393,216 values), no optimizer state,
  and no base-model tensors. Report and audit SHA-256 are respectively
  `a2cc8c5881f46052a21c9ed6e68569d1a9529dd7b2133b95292b11e160702847`
  and `9c25f87dcfeb80732cd812ec20a59b38371caf8790be97195d5a27b06da22a39`.
- V1g rejects scalar typing alone. The next work is an interface audit before
  another run: the physical U sequence currently carries no explicit `Gxx/Fxx`
  slot identity, while only full-attention layers 27 and 31 receive LoRA even
  though the released VLM exposes eight full-attention layers. These are
  competing hypotheses and must be isolated rather than changed together.

## V1h explicit-slot route-readout pre-run contract

- The released Qwen config records eight full-attention layers at indices
  `[3, 7, 11, 15, 19, 23, 27, 31]`. V1g adapts only `[27, 31]`. V1h leaves
  that LoRA scope unchanged and tests the other open interface hypothesis:
  missing explicit G/F row identity.
- V1h reuses V1g's typed scalar basis and the exact V1f curriculum, split,
  schedule, optimizer, seed, questions, controls, LoRA, native inputs, and
  numerical gates. Its only change is to append a fixed 48-way one-hot slot
  code before the learned hidden projection. Sequence slots `0..15` identify
  `G00..G15`; slots `16..47` identify `F00..F31`. No semantic label or action
  is computed by the adapter.
- The resulting slot-typed projector has 1,391,616 trainable parameters and
  seven saved tensors, adding 24,576 input-projection weights relative to V1g.
  The one-hot itself is deterministic and untrained. The auditor verifies the
  exact projector type/count while retaining all V1f/V1g leakage, scope,
  checkpoint, held-out-order, pair, and causal checks.
- Passing would support the slot-addressing hypothesis and show that the two
  upper adapted full-attention layers are sufficient for the minimal route
  readout. Failure would justify a separate V1i test that expands attention
  LoRA while keeping this slot-typed adapter fixed. Neither outcome establishes
  full grounding, planning, or safety, and neither activates the direct
  Planning Expert fallback.

## V1h remote negative result

- Commit under test: `4a1f1156`; Slurm job: `1168609`; run id:
  `qwen_visibility_grounding_v1h_route151_slot_typed_route_readout_v1`.
- Terminal state: `COMPLETED`, exit `0:0`, elapsed `00:10:12`, peak host RSS
  `2,513,340 KiB`. Model load took 163.14 s, preparation 32.73 s,
  pre-training held-out generation 24.92 s, 240 optimizer steps 177.55 s, and
  post-training evaluation 29.78 s. Peak allocated/reserved GPU memory was
  15,638/15,976 MB; native image processing was unchanged.
- The V1f curriculum was reused byte-for-byte at SHA-256
  `63bb8fbe6fc929591b5e97bbd55e30a1437c309a58b507765085f00b4566d592`.
  The fail-closed audit is protocol-valid with no integrity failures and
  verifies the slot-typed projector's exact 1,391,616 parameters, frozen model
  boundary, held-out split, pair structure, schedule, controls, updates, and
  checkpoint scope.
- Pre-training true-U accuracy was `0/20`. Post-training true-U accuracy is
  `19/20`: `OFF_ROUTE 10/10` and `ON_ROUTE 9/10`. Nine of ten held-out matched
  pairs are jointly correct, and the true-U gap over the stronger control is
  40 percentage points. These four gates pass.
- The remaining predeclared gate fails: spatial-shuffle changed-target
  accuracy is `9/20` (45%). The shuffled arm emits `OFF_ROUTE` on 19/20
  examples, even though its targets are balanced 10/10, and changes its answer
  relative to true U on only 8/20 examples. Status is therefore
  `valid_run_without_causal_slot_typed_route_readout`; high true-U accuracy may
  not be promoted to causal readout acceptance.
- The optimization path is active. First/last-step loss is 6.56275 to 0.000152;
  first/last 30-step mean loss is 0.82809 to 0.04764. Projector and LoRA
  gradient and update norms are finite and nonzero on all 240 steps.
- The adaptation checkpoint is 7,150,389 bytes with SHA-256
  `1f99aa21fdd715b1215d3401c8fa81eed4eba1539b7367a4a37681f2673c24f8`.
  Independent `weights_only` readback verifies seven slot-typed-projector
  tensors (1,391,616 values), sixteen LoRA tensors (393,216 values), no
  optimizer state, and no base-model tensors. Report and audit SHA-256 are
  respectively
  `30c337055aea39e54582bb101270481d4e287755fa2c90f35c86d6933061c654`
  and `58232a5ad69a814bb8c05f8891f41a639cfe331c85f3e89e225a7651f80ce5ea`.
- Relative to the otherwise identical V1g run, explicit slot identity is a
  material interface improvement but not sufficient for the complete causal
  gate. The next controlled diagnostic may change only attention scope: keep
  the slot-typed adapter, data, split, schedule, optimizer, seed, questions,
  and gates fixed while expanding LoRA from `[27, 31]` to all eight released
  full-attention layers `[3, 7, 11, 15, 19, 23, 27, 31]`. The Planning Expert
  remains frozen and its fallback remains inactive.

## V1i full-attention route-readout pre-run contract

- V1i changes exactly one factor relative to V1h: LoRA coverage expands from
  full-attention layers `[27, 31]` to all released full-attention layers
  `[3, 7, 11, 15, 19, 23, 27, 31]`. The four target modules, rank 8, alpha 16,
  zero dropout, and both learning rates remain unchanged.
- The exact V1h slot-typed projector remains fixed at 1,391,616 trainable
  parameters and seven saved tensors. The immutable V1f curriculum, 30/20
  train/held-out split, 240-step balanced schedule, images, questions, paired
  row swaps, controls, seed, native processing, insertion positions, mRoPE,
  optimizer, clipping, and claim boundary are unchanged.
- Full attention adds 32 LoRA-wrapped modules and 64 saved LoRA tensors,
  totaling 1,572,864 trainable LoRA parameters. The vision encoder, base VLM,
  embeddings, LM head, and Planning Expert remain frozen. The protocol loader
  and auditor fail closed on the exact layer list, module list, parameter
  names/counts, tensor count, and all unchanged V1h settings.
- The five numerical gates remain verbatim: at least 90% true-U held-out
  accuracy overall and per label, at least 80% matched-pair correctness, at
  least 80% spatial-shuffle changed-target accuracy, and at least a 30-point
  true-U gap over the stronger control. Passing establishes only that the
  slot-typed, full-attention VLM path can causally read this one route scalar.
  Failure rejects this bounded interface/training recipe; neither result is
  full grounding, planning, safety, or permission to activate the Planning
  Expert fallback.

## V1i remote negative result

- Commit under test: `b6bbd245`; Slurm job: `1168669`; run id:
  `qwen_visibility_grounding_v1i_route151_slot_typed_full_attention_route_readout_v1`.
- Terminal state: `COMPLETED`, exit `0:0`, elapsed `00:09:54`, peak host RSS
  `2,923,844 KiB`. Model load took 153.68 s, preparation 25.75 s,
  pre-training held-out generation 24.36 s, 240 optimizer steps 183.87 s, and
  post-training evaluation 29.69 s. Peak allocated/reserved GPU memory was
  15,656/16,000 MB; native image processing was unchanged.
- The V1f curriculum was reused byte-for-byte at SHA-256
  `63bb8fbe6fc929591b5e97bbd55e30a1437c309a58b507765085f00b4566d592`.
  The V1i-specific fail-closed audit is protocol-valid with no failures. It
  verifies all eight layer indices, 32 wrapped modules, 64 LoRA parameter
  names/tensors, exact unchanged V1h settings, held-out split, schedule,
  controls, updates, and frozen released-model boundary.
- Pre-training true-U accuracy was `0/20`. Post-training true-U accuracy is
  `20/20`, both labels are `10/10`, all ten matched pairs are jointly correct,
  and the true-U gap over the stronger control is 50 points. Those four gates
  pass, modestly improving V1h's 19/20 and 9/10 results.
- Spatial-shuffle changed-target accuracy is only `10/20` (50%), so the fifth
  gate fails. The shuffled arm emits `OFF_ROUTE` on 18/20 examples despite
  balanced targets. Status is
  `valid_run_without_causal_slot_typed_full_attention_route_readout`; perfect
  true-U order accuracy is not causal-readout acceptance.
- First/last-step loss is 6.56275 to 0.0000743; first/last 30-step mean loss is
  0.96975 to 0.03331. Projector and LoRA gradients and update norms are finite
  and nonzero on all 240 steps. Expanding attention scope did not repair the
  unchanged control failure.
- The adaptation checkpoint is 11,886,709 bytes with SHA-256
  `8bf1de866f8c963b0e7a473e8db50addbe47d982df44a189f7d8ceee7adc9903`.
  Independent `weights_only` readback verifies seven projector tensors
  (1,391,616 values), 64 LoRA tensors (1,572,864 values) at exactly layers
  `[3, 7, 11, 15, 19, 23, 27, 31]`, no other tensors, no optimizer state, and
  no base-model tensors. Report and audit SHA-256 are respectively
  `aa7915735cf93f2281bad4013149b4f91886048f6ef64559ae816affce7ca43f`
  and `d574a43ca804165f5163d659dd12337db567c1591c8a4cf546db7167d8e9b10d`.
- A read-only post-run feature audit validates every control target directly
  from the serialized F00 row: thresholding `route_weight_mean` at 0.2 gives
  20/20 correct targets for both true and shuffled U. The model agrees on
  20/20 true rows but only 10/20 shuffled rows. On the ten distinct true target
  rows reused across order variants, multiple other individual features also
  separate ON/OFF perfectly, including urgency, unknown-area,
  occluded-unknown, observation-age, never-observed, and frontier-score fields.
  Thus the existing curriculum does not identify the requested route field
  independently of real-row templates and correlated physical covariates.
- V1i rejects broader attention as a sufficient repair. More layers or epochs
  are not the next step. Before another full-model run, the real-row inventory
  must show that a training/held-out-target-row split can balance route labels
  while breaking the identified covariates. Any new curriculum must keep
  complete physical rows, reserve disjoint target rows for evaluation, retain
  target-changing controls, and continue to keep the Planning Expert frozen.

## V1j target-row-disjoint route-readout pre-run contract

- User decision: proceed with V1j and relax spatial shuffle. Shuffle examples
  do not enter optimization; their complete results remain reportable, but the
  former 80% changed-target gate is not a V1j veto.
- V1j holds the complete V1i model side fixed: 1,391,616-parameter slot-typed
  projector; all eight full-attention layers with 1,572,864 LoRA parameters;
  240 steps; the same learning rates, clipping, seed, native images, question,
  U insertion, frozen vision encoder, base VLM weights, LM head, embeddings,
  and Planning Expert.
- The immutable split has 13 training target pairs and five evaluation target
  pairs. Three randomized non-query orders per training pair produce 78
  examples (39 per label); two orders per evaluation pair produce 20 examples
  (10 per label). Queried target rows never cross the split, and
  `route151-step-000200` is fully evaluation-only.
- The predeclared hard numerical gates are: true-U evaluation accuracy at
  least 90% overall and per label; at least 80% of matched pairs jointly
  correct; and a true-U advantage of at least 30 percentage points over the
  stronger zero/shuffle control when scored against the original target.
  Spatial-shuffle changed-target accuracy is reported separately without a
  threshold.
- Passing is evidence only that the V1i consumer can read the route scalar on
  unseen queried rows in this small oracle-U slice. It is not full structured
  grounding, Planning Expert conditioning, closed-loop safety, or deployable
  predicted U.

## Qwen RL and Robusto-2 external review (2026-09-06)

- Qwen's official release now lists both 2.1 GB planning heads. `planner-rl`
  is public and must be used in reasoning-planning mode; our shared checkpoint
  directory contains only `planner-sft`, so this is an incomplete local
  snapshot rather than an upstream-release wait. The shared filesystem has
  sufficient capacity to provision it without deleting SFT.
- Robusto-2 is useful as a secondary VLM/OOD diagnostic: 20 curated dashcam
  clips (10 Lima, 10 New York), 20 questions per clip in factual, rating,
  counterfactual, and reasoning blocks, plus repeated outputs from 10 VLMs and
  answers from 20 humans. Its open data and analysis code can support a
  human-alignment and scenario-reasoning comparison.
- Robusto-2 is not a planning or closed-loop benchmark: it has no trajectory
  target, collision exposure, route progress, intervention timing, or
  clean/U paired actuation. It may therefore supplement but not replace the
  Bench2Drive primary and NAVSIM secondary evaluation. The relevant reference
  is its question taxonomy and repeated human/VLM comparison, especially for
  visibility, hazard, counterfactual-crash, and reasoning prompts.
- Primary sources: Qwen official README
  <https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/README.md>, Robusto-2
  paper <https://arxiv.org/abs/2606.20980>, and official dataset/code card
  <https://huggingface.co/datasets/Artificio/robusto-2>.

## V1j remote negative result

- Commit under test: `fbe0c72f`; Slurm job: `1168982`; run id:
  `qwen_visibility_grounding_v1j_route151_target_row_route_readout_v1`.
- Terminal state: `COMPLETED`, exit `0:0`, elapsed `00:10:22`, peak host RSS
  `2,692,276 KiB`. Model load took 166.22 s, preparation 46.00 s,
  pre-training evaluation 24.50 s, 240 optimizer steps 183.40 s, and
  post-training evaluation 29.13 s. Peak allocated/reserved GPU memory was
  19,269/19,786 MB; native input processing was unchanged.
- The immutable V1j curriculum contains 13 training target pairs under three
  orders and five evaluation target pairs under two orders, for 78/20
  examples. `route151-step-000200` is evaluation-only. Its SHA-256 is
  `0171a247b4c390ff9ba389a9815961c13c5937774770a1a1e02f2324414bf30e`.
  The independent audit is protocol-valid with no failures and verifies the
  exact split, no queried-row leakage, balanced 240-step schedule, controls,
  frozen boundary, trainable scope, updates, and checkpoint.
- Pre-training true-U accuracy is 0/20. Post-training true-U accuracy is 10/20:
  `ON_ROUTE` 4/10 and `OFF_ROUTE` 6/10. No one of the ten matched pairs has
  both answers correct, and the true-U gap over the stronger control is zero.
  All four V1j hard gates fail. Spatial-shuffle target accuracy is reported at
  10/20 but is not a hard gate. Status is
  `valid_run_without_target_row_disjoint_route_readout`.
- The failure is structured rather than random. For true U the model emits
  `OFF_ROUTE` for all four evaluation examples in each of steps 0, 260, and
  280, and `ON_ROUTE` for all four examples in steps 200 and 300. It therefore
  ignores the ON/OFF F00 swap within a frame and follows sample/global context.
- First/last-step loss is 6.56275 to 0.18998; first/last 30-step mean loss is
  0.92821 to 0.17751. Projector and LoRA updates are finite and nonzero on all
  240 steps, excluding a disconnected optimizer as the explanation.
- The adaptation checkpoint is 11,886,709 bytes with SHA-256
  `74ae028c2f2911ed2b56e178ab3f29f7390e5ee0fd95cd3bbd7a7cba4125bd56`.
  Independent `weights_only` readback verifies seven projector tensors
  (1,391,616 values), 64 LoRA tensors (1,572,864 values) at layers
  `[3, 7, 11, 15, 19, 23, 27, 31]`, no optimizer state, and no base weights.
  Report and audit SHA-256 are respectively
  `87ae2091b1119197c9de1c8e884c5e69721a0517aca71a4878a5f1dd9f5a20be`
  and `4f8f363588d9e181ab3554a9924d4182370ab2483df840109c9f5db4718c1bd6`.
- V1j invalidates V1i's apparent target-row generality. The immediate open
  question is whether to redesign the supervision/interface so the answer is
  extracted from one addressed token (for example an explicit query-token
  interaction), or to stop spending on the minimal route QA probe. Moving to
  trajectory training or direct Planning Expert injection now would bypass the
  accepted structured-grounding prerequisite and requires a new decision.

## Cross-route grounding source inventory

- User review rejected a proposed same-frame answer-equality penalty as too
  mechanical: two U inputs may legitimately imply the same answer. The ADR now
  prohibits pair-flip supervision. The accepted next step is data inventory,
  not another training run.
- Added `scripts/audit_qwen_visibility_grounding_source_inventory.py` and its
  isolated test. Local result: `1 passed`; compilation and `git diff --check`
  pass. Commit under audit: `b15a810d`.
- The read-only remote run consumed the immutable 100-route infos and frozen
  route manifest, did not load Qwen, did not generate U tokens, and did not
  start training. Its report is
  `/public/share/lidachuan/orion_assets/qwen_visibility_grounding_runs/cross_route_source_inventory_v1/inventory.json`,
  4,752 bytes, SHA-256
  `f21695d91df467a83d2a66abe3ef849c24fe0fe7f6fd751abaca96392581507d`.
- Source capacity is 100 routes, 24,024 indexed frames, 43 scenario types, and
  12 towns. The already frozen, leakage-checked split is 70/10/10/10 routes
  and 16,207/2,423/2,790/2,604 frames for
  train/validation/calibration/held-out. At most this exposes
  518,624/77,536/89,280/83,328 `(frame,F00--F31)` query pairs before filtering
  invalid frontier rows.
- The sampled file audit selected first/middle/last from every route: 300
  frames and 1,800 required camera-file checks. No required front RGB/depth
  file was missing and all sampled annotations passed camera calibration plus
  ego/navigation state checks. The 900 depth headers were uniformly 1600 x
  900, 8-bit grayscale. Two source routes have one indexed-frame gap each;
  expert assessment is absent only from the 100 sampled terminal frames.
- Existing Qwen U tokens still cover only Route 151: 54 frames, all with 32
  valid frontier rows. Random `F00`--`F31` query construction is feasible, but
  route-diverse frontier-row coverage remains unmeasured until offline
  tokenization.
- Bench2Drive collection source computes metric `float16` depth and writes it
  through OpenCV. A server round-trip probe confirmed the OpenCV PNG fallback
  rounds to `uint8` metres and saturates at 255. This source is not equivalent
  to the live 24-bit oracle and cannot silently use the live 0.45 m surface
  tolerance on a 0.5 m grid. A quantization-interval/tolerance policy and route
  reconstruction policy must be frozen before a small, no-training
  tokenization preflight.
- Full CARLA recapture is deferred. It becomes justified only if that bounded
  preflight shows unstable U rows under the stored-depth ambiguity.

## Integrity constraints

- No Torch, Qwen, Orion, or CARLA import in the geometry module.
- Use Qwen ego coordinates: x forward, y left, z up.
- Keep occluded unknown distinct from outside-FOV.
- A point visible from any camera is not unknown merely because another camera
  sees it behind a surface.
- Oracle depth is an upper bound and must never be reported as predicted U.
- Every implementation milestone updates this file and is committed separately.

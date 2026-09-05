# Qwen visibility-belief implementation status

Last updated: 2026-09-06 (Asia/Shanghai)

## Current milestone

`O1: live oracle visibility capture`

Status: in progress (local implementation complete; remote CARLA smoke pending)

The pure geometry has passed its local tests. The current gate is to prove that
three CARLA depth cameras, explicitly co-located with Qwen's unchanged RGB
cameras, produce aligned and auditable artifacts in a real Bench2Drive run.

## Ordered implementation ladder

| ID | Deliverable | Status |
| --- | --- | --- |
| O0 | CARLA depth decoding, camera calibration, 3D visibility fusion, 2.5D BEV, rendering, unit tests | Complete (`6addb2fe`) |
| O1 | Add co-located oracle depth sensors to the Qwen agent behind an explicit oracle-only config | In progress (local complete; live smoke pending) |
| O2 | Observation-age memory and deterministic urgency/stopping-margin map | Not started |
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

## Integrity constraints

- No Torch, Qwen, Orion, or CARLA import in the geometry module.
- Use Qwen ego coordinates: x forward, y left, z up.
- Keep occluded unknown distinct from outside-FOV.
- A point visible from any camera is not unknown merely because another camera
  sees it behind a surface.
- Oracle depth is an upper bound and must never be reported as predicted U.
- Every implementation milestone updates this file and is committed separately.

# Qwen visibility-belief implementation status

Last updated: 2026-09-06 (Asia/Shanghai)

## Current milestone

`O0: oracle visibility geometry`

Status: in progress

The first implementation milestone is intentionally below the VLM boundary. It
must establish a correct, inspectable, NumPy-only physical visibility map before
custom Qwen tokens or LoRA training are introduced.

## Ordered implementation ladder

| ID | Deliverable | Status |
| --- | --- | --- |
| O0 | CARLA depth decoding, camera calibration, 3D visibility fusion, 2.5D BEV, rendering, unit tests | In progress |
| O1 | Add co-located oracle depth sensors to the Qwen agent behind an explicit oracle-only config | Not started |
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

## Integrity constraints

- No Torch, Qwen, Orion, or CARLA import in the geometry module.
- Use Qwen ego coordinates: x forward, y left, z up.
- Keep occluded unknown distinct from outside-FOV.
- A point visible from any camera is not unknown merely because another camera
  sees it behind a surface.
- Oracle depth is an upper bound and must never be reported as predicted U.
- Every implementation milestone updates this file and is committed separately.

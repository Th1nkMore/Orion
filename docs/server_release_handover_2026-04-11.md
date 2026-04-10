# Server Release Handover (2026-04-11)

This document is the release and migration handover before shutting down the
current server instance. It records the current reproducibility state, the
official-vs-internal evaluation split, the main findings from `R2-A`, and the
recommended next steps for the next provider such as Vast.ai.

## Current State Summary

### Code and workflow state

- The workflow is now explicitly split into:
  - local-only analysis, plotting, and documentation
  - server-side training and open-loop evaluation
- `closed-loop` is reserved for paper-aligned official CARLA evaluation only.
- Bench2Drive replay is diagnostic-only and must not be reported as the main
  closed-loop result.
- The official agent path now supports loading an additional FiLM checkpoint on
  top of the base ORION checkpoint through
  `team_code/orion_b2d_agent.py`.

### Branch state

- Working branch: `dev`
- Local branch contains the official-closed-loop alignment work and is ahead of
  `origin/dev`
- Policy:
  - keep code and docs on `dev`
  - keep generated artifacts on a dedicated artifact branch

## What Was Actually Reproduced

### Internal round-2 path

The repo can run the internal round-2 path consisting of:

1. `scripts/train_film.py`
2. `scripts/eval_openloop.py`

This is the current FiLM/UQ comparison gate.

### Official upstream-style open-loop path

The upstream official open-loop entry was reproduced on the server with:

```bash
./adzoo/orion/orion_dist_eval.sh adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth 1
```

Observed final metrics from the server log:

- `plan_L2_1s = 0.2673`
- `plan_L2_2s = 0.6299`
- `plan_L2_3s = 1.1282`
- `mAP = 0.6380`
- `NDS = 0.7176`

Interpretation:

- The upstream open-loop entry can run in the ORION environment.
- The result scale is close enough to the original ORION numbers to treat this
  path as the horizontal-comparison gate for future work.

### Official closed-loop path

The Python/import/code side is largely aligned, but paper-aligned official
CARLA closed-loop was not made runnable on the tested AutoDL instances.

The blocker is infrastructure/runtime, not repository logic:

- `vulkaninfo --summary` falls back to `llvmpipe`
- forcing the NVIDIA ICD still fails
- the container images exposed CUDA compute but not a usable NVIDIA Vulkan
  runtime for CARLA and Unreal

This was observed on multiple AutoDL instances, including a newer Ubuntu 22.04
instance with a healthy `nvidia-smi` but still broken Vulkan ICD behavior.

## R2-A Result Summary

Primary artifact:

- `results/round2/R2-A/openloop_summary.json`

Comparison baseline:

- `results/eval_openloop_v3.json`

### Key deltas versus the current internal baseline

All split:

- `L2@3s`: `1.9060 -> 2.2253` (`+16.75%`)
- `Col@3s`: `1.2689% -> 0.9449%` (`-25.54%`)

Normal split:

- `L2@3s`: `2.3791 -> 2.4837` (`+4.40%`)
- `Col@3s`: `0.0123% -> 0.0062%`

Adverse split:

- `L2@3s`: `1.7791 -> 2.1560` (`+21.18%`)
- `Col@3s`: `1.6061% -> 1.1967%` (`-25.49%`)

UQ semantic quality:

- `AUROC(adverse)`: `0.9536 -> 0.7184`
- normal/adverse `uq_mean` gap: `0.8699 -> 0.3754`

### Interpretation

`R2-A` reduces the collision proxy but degrades trajectory quality, especially
under adverse conditions, and also damages the separation quality of the UQ
signal. This is not a clean improvement. It is more consistent with a
conservative-shortcut effect than with a robust planning gain.

Decision:

- keep `R2-A` as a negative-result reference and ablation anchor
- do not treat it as a winning training recipe

## Reproducibility Rules Going Forward

### Accepted meaning of each evaluation term

- `Round-2 open-loop`: internal offline comparison with
  `scripts/eval_openloop.py`
- `Official open-loop`: upstream-style comparison with
  `adzoo/orion/orion_dist_eval.sh`
- `Closed-loop`: official CARLA + Bench2Drive online evaluation only
- `Replay`: diagnostic only, not headline evaluation

### Required order

1. reproduce official open-loop on the new machine or provider
2. verify the result is in the expected ORION ballpark
3. run the modified model through the same official open-loop path
4. only then attempt official CARLA closed-loop

## Migration Plan For Vast.ai Or Another Provider

### What the next machine must satisfy

The next candidate machine should be validated in this exact order:

1. `nvidia-smi`
2. `vulkaninfo --summary`
3. `CarlaUE4.sh -RenderOffScreen ...`
4. official closed-loop env import check
5. one-route official smoke

Do not spend time on ORION-side closed-loop debugging until steps 1-3 pass.

### Practical requirement split

- Training and large-model inference:
  - A100-class GPU remains the safe default
- Official CARLA server:
  - must expose a real NVIDIA Vulkan runtime
  - compute-only GPU access is not enough

### Recommended next sequence

1. Rent a new provider instance that explicitly supports Vulkan, Unreal, or
   CARLA.
2. Re-run:
   - `scripts/check_official_carla_vulkan.sh`
   - `scripts/check_official_closedloop_env.sh`
3. Reproduce the official open-loop baseline first.
4. Run the one-route official closed-loop smoke.
5. Only after that, decide whether to continue `R2-B/C` or switch to a
   different model-side direction.

## Current Insight Inventory

### Research-side insights

- Pure collision-aware FiLM fine-tuning is not enough.
- The main risk is conservative shortcut behavior rather than a real safety
  gain.
- Internal replay-style diagnostics are useful, but they are not publishable
  substitutes for official closed-loop.
- Official open-loop must be treated as the minimum comparison gate before any
  claim of alignment with the original ORION setup.

### Infrastructure-side insights

- AutoDL instances tested so far appear suitable for ORION training and custom
  open-loop evaluation.
- They did not provide a reliable NVIDIA Vulkan runtime for CARLA.
- A healthy `nvidia-smi` alone is insufficient; `vulkaninfo --summary` must
  enumerate a real NVIDIA GPU rather than `llvmpipe`.
- Do not assume a CUDA-capable rental machine can also run Unreal or CARLA.

## Artifact Preservation Policy

### Keep on `dev`

- code changes
- docs
- runbooks
- operation logs

### Keep off `dev`

- generated logs
- generated figures
- generated binary evaluation outputs such as `openloop.pt`

These should be preserved either on:

- a dedicated artifact branch, or
- a GitHub release asset

For the current shutdown, a dedicated artifact branch is the simplest choice.

### Current local artifact set

Small enough to preserve in GitHub if needed:

- `results/round2/R2-A/openloop.pt` (`~1.9MB`)
- `results/round2/R2-A/openloop_summary.json`
- `results/round2/R2-A/manifest.json`
- local dashboard summaries and figures under `results/round2_dashboard/`

The training checkpoint `R2-A.pt` was not synced into the local repo before
shutdown. Do not assume it is preserved unless separately copied elsewhere.

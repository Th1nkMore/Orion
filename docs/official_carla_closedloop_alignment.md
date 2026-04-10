# Official CARLA Closed-Loop Alignment

This runbook defines the only accepted meaning of `closed-loop` for this repo:
paper-aligned official CARLA evaluation, using the ORION agent interface rather
than offline replay metrics.

## Goal

After `R2-A` finishes its training and open-loop evaluation, first reproduce
the upstream-style official open-loop baseline, then shift the main effort to
making the official CARLA closed-loop path runnable and comparable to the
original ORION setup.

## Upstream Alignment Facts

The upstream repository is `xiaomi-mlab/Orion`.

From the upstream README:

- Suggested base environment:
  - `python=3.8`
  - `torch==2.4.1+cu118`
  - `torchvision==0.19.1+cu118`
  - `torchaudio==2.4.1+cu118`
- Official open-loop command:
  - `./adzoo/orion/orion_dist_eval.sh adzoo/orion/configs/orion_stage3_infer.py [CHECKPOINT] 1`
- Official close-loop guidance:
  - use Bench2Drive evaluation tools plus CARLA
  - set `TEAM_CONFIG=adzoo/orion/configs/orion_stage3_agent.py+[CHECKPOINT_PATH]`

Local extension kept close to upstream:

- `TEAM_CONFIG=adzoo/orion/configs/orion_stage3_agent.py+[BASE_CHECKPOINT]+[FILM_CHECKPOINT]`
- The extra FiLM checkpoint is optional and is loaded after the base ORION
  checkpoint inside `team_code/orion_b2d_agent.py`.

From the upstream Bench2Drive README:

- Target CARLA version: `0.9.15`
- CARLA Python API must be visible in the active env
- Evaluation toolkit layout includes:
  - `leaderboard/`
  - `scenario_runner/`
  - `tools/`
- Debug command:
  - `bash leaderboard/scripts/run_evaluation_debug.sh`
- Multi-process command:
  - `bash leaderboard/scripts/run_evaluation_multi_uniad.sh`
- Expected benchmark route count for final metrics: `220`

These upstream facts are the baseline for future environment alignment.

## Stage 0 Gate: Official Open-Loop Reproduction

Before calling any CARLA result paper-aligned, reproduce the upstream
open-loop entry with the base ORION checkpoint:

```bash
./adzoo/orion/orion_dist_eval.sh adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth 1
```

This stage exists for one reason: the repo already has strong internal
open-loop baselines from `scripts/eval_openloop.py`, but that custom script
adds UQ hooks, weather splits, and internal diagnostics. Those results are
useful for FiLM/UQ analysis, yet they are not the strict upstream evaluation
entry used by the original ORION repo.

Therefore the gating order is:

1. reproduce the upstream official open-loop baseline
2. verify the result is in the expected ballpark of the original ORION numbers
3. wire `R2-A` into the same official open-loop path
4. only then spend effort on official CARLA closed-loop environment alignment

Accepted artifact names for this stage:

- `results/openloop_official/baseline.log`
- `results/openloop_official/baseline.pkl`
- `results/openloop_official/r2a.log`
- `results/openloop_official/r2a.pkl`

## Repo Anchors

The repo already contains the key code entry points for the official path:

- Agent entry: `team_code/orion_b2d_agent.py`
- Agent config: `adzoo/orion/configs/orion_stage3_agent.py`

Important details from the code:

- The agent implements `leaderboard.autoagents.autonomous_agent.AutonomousAgent`.
- It imports `carla`, `leaderboard.autoagents`, and Bench2Drive helpers from
  `Bench2DriveZoo.team_code`.
- It expects a config-plus-checkpoint string in `setup()` and uses the
  stage-3 agent config to build ORION for runtime inference.

## Acceptance Criteria

Only call a result `closed-loop` when all of these are true:

- CARLA runtime is actually used.
- `team_code/orion_b2d_agent.py` is the evaluation agent entry.
- The runtime stack includes `carla`, `leaderboard`, and `scenario_runner`.
- Bench2Drive/leaderboard route and scenario files are provided explicitly.
- Output comes from the official online evaluation flow, not from
  `scripts/eval_closedloop_replay.py`.

## Current Server Status

Verified on `autodl`:

- Present:
  - `/root/Orion/team_code/orion_b2d_agent.py`
  - `/root/Orion/adzoo/orion/configs/orion_stage3_agent.py`
- Missing from Python runtime:
  - `carla`
  - `leaderboard`
  - `scenario_runner`
- Not found on disk during quick scan:
  - `Bench2DriveZoo`
  - `CarlaUE4.sh`

This means the current blocker is environment alignment, not missing ORION-side
agent code.

## Known Code-Level Gap For R2-A

The official ORION agent currently expects:

- `TEAM_CONFIG=adzoo/orion/configs/orion_stage3_agent.py+[FULL_CHECKPOINT]`

The current `R2-A` artifact is not a full ORION checkpoint. It stores only the
trained FiLM weights from `scripts/train_film.py`.

Therefore, before official closed-loop evaluation can use `R2-A`, one of these
must be implemented:

1. extend `team_code/orion_b2d_agent.py` to accept an extra FiLM checkpoint and
   load it after the base ORION checkpoint
2. merge `R2-A.pt` into a full ORION checkpoint before evaluation

To stay close to upstream while minimizing friction, option 1 is preferred.

This repo now implements option 1.

## Required Environment Pieces

The target machine must provide:

- CARLA binary/runtime compatible with the original evaluation stack
- Python API package `carla`
- `leaderboard`
- `scenario_runner`
- `Bench2DriveZoo` checkout or equivalent runtime dependency tree
- Route file(s) and scenario file(s) matching the original protocol
- A working way to launch CARLA and then launch leaderboard evaluation against
  `team_code/orion_b2d_agent.py`

## Immediate Plan After R2-A

1. Let `R2-A` finish training plus custom open-loop only.
2. Pause `R2-B` and `R2-C`.
3. Reproduce the upstream official open-loop baseline with:
   - `adzoo/orion/orion_dist_eval.sh`
   - `adzoo/orion/configs/orion_stage3_infer.py`
   - `ckpts/Orion.pth`
4. Once the baseline is confirmed, run the same official open-loop path again
   with the FiLM checkpoint injected through the existing local extension.
5. Only after the official open-loop path is validated, align the server to the
   official closed-loop runtime:
   - install/import `carla`
   - install/import `leaderboard`
   - install/import `scenario_runner`
   - place or clone `Bench2DriveZoo`
   - identify exact routes/scenarios used for comparison
6. Run environment validation with:
   - `bash scripts/check_official_closedloop_env.sh`
7. Once the environment check passes, write the concrete launch command for the
   official evaluation and test it on the smallest possible route subset.

## Rules

- Do not substitute replay metrics for official closed-loop.
- Do not claim paper-level horizontal comparison until the upstream official
  open-loop path has been reproduced on the current server/runtime.
- Do not continue `R2-B` or `R2-C` until the official closed-loop path is at
  least environment-complete, unless priorities change explicitly.
- Keep official closed-loop outputs separate from `results/round2/`.

Recommended output namespace:

- `results/closedloop_official/`
- `results/closedloop_official_smoke/`

## Validation Command

On the target machine:

```bash
cd /path/to/Orion
bash scripts/check_official_closedloop_env.sh
```

With explicit locations:

```bash
PROJECT_ROOT=/root/Orion \
BENCH2DRIVE_ROOT=/root/Bench2DriveZoo \
CARLA_ROOT=/root/CARLA_0.9.x \
ROUTES_PATH=/path/to/routes.xml \
SCENARIOS_PATH=/path/to/scenarios.json \
bash scripts/check_official_closedloop_env.sh
```

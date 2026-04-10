# Operation Log

This file is the persistent trace for repository changes made during active
development. Use it together with git history.

## Policy

- One log entry per logical change batch.
- Update this file before creating the corresponding git commit.
- Keep commit messages in English.
- Use the relevant runbook or workflow document for implementation details; use
  this file for concise traceability.

## 2026-04-10

### Server release handover and artifact split

- Added a dedicated shutdown/migration handover document for releasing the
  current server instance.
- Recorded the current reproducibility boundary:
  - internal round-2 open-loop is reproducible locally from synced outputs
  - upstream-style official open-loop was reproduced on the server
  - official CARLA closed-loop remains blocked by provider runtime/Vulkan
- Recorded the artifact policy split:
  - keep code/docs on `dev`
  - preserve generated outputs on a dedicated artifact branch instead of
    polluting the main development branch

### Bench2Drive vs Bench2DriveZoo split correction

- Corrected a repository-layout mistake in the closed-loop bootstrap flow:
  Bench2Drive and Bench2DriveZoo must be treated as separate upstream repos.
- Updated the bootstrap and validation scripts so:
  - `Bench2Drive` provides `leaderboard/`, `scenario_runner/`, and route files
  - `Bench2DriveZoo` provides the `team_code` utilities required by
    `team_code/orion_b2d_agent.py`
- Documented the new split explicitly in the official closed-loop runbook.
- Added a local fallback in `team_code/orion_b2d_agent.py` so the official
  agent can use vendored `team_code/pid_controller.py` and `team_code/planner.py`
  when `Bench2DriveZoo.team_code` is not yet available.

### Flash attention fallback and CARLA root correction

- Added a runtime fallback in `mmcv/models/utils/attention.py` so the repo can
  import and run smoke-level paths even when `flash_attn` is unavailable in the
  isolated closed-loop env.
- Kept the native flash-attention path when the package is installed; the
  fallback only activates on missing dependency.
- Added a dedicated CARLA extraction/import helper script and corrected the
  documented `CARLA_ROOT` to the real extracted layout under
  `/root/autodl-tmp/carla`.

### Py3.8 mmcv smoke-build path

- Added a controlled setup switch in `setup.py` so the isolated official
  closed-loop env can build `mmcv._ext` without pulling in the point-cloud CUDA
  extensions.
- Added `scripts/build_closedloop_mmcv_ext.sh` to standardize that py3.8 build
  step on the server.
- Extended the environment check to attempt `team_code.orion_b2d_agent`
  import directly, so the next closed-loop blocker is always surfaced by the
  same validation command.

### Official one-route smoke launcher

- Added `scripts/run_official_closedloop_smoke.sh` to lock down the first
  paper-aligned online smoke command shape against Bench2Drive's
  `leaderboard_evaluator.py`.
- Standardized absolute `TEAM_AGENT` / `TEAM_CONFIG` wiring for ORION so the
  smoke path does not depend on the current working directory.
- Standardized one-route XML splitting from `bench2drive220.xml` and output
  placement under `results/closedloop_official_smoke/`.
- Avoided Bench2Drive's `run_evaluation.sh` helper for the smoke path because
  it hardcodes the `carla-0.9.15-py3.7` egg, which conflicts with the isolated
  Python 3.8 official env used on `autodl`.

### Runtime import cleanup for the official path

- Removed an unused `IPython` import from
  `mmcv/datasets/samplers/group_sampler.py` so the official agent import path
  does not depend on a debug-only package.
- Added fallback bridges from legacy `iou3d_det` / `roiaware_pool3d` modules to
  the py3.8 `mmcv._ext` build, which keeps the official smoke path moving
  without rebuilding the old standalone point-cloud extensions.

### Vulkan runtime gating for official CARLA smoke

- Added `scripts/check_official_carla_vulkan.sh` to validate whether the
  current machine exposes a usable GPU Vulkan runtime for CARLA instead of
  silently falling back to Mesa llvmpipe.
- Extended `scripts/check_official_closedloop_env.sh` with an opt-in
  `RUN_VULKAN_RUNTIME_CHECK=1` gate so the import/layout check can also enforce
  the graphics-runtime prerequisite.
- Updated `scripts/run_official_closedloop_smoke.sh` to run the Vulkan gate
  before launching `leaderboard_evaluator.py`, which prevents misleading smoke
  failures when the server cannot create a valid NVIDIA Vulkan instance.
- Recorded the current `autodl` blocker in the runbook:
  - ORION imports and official agent wiring are ready
  - CARLA still cannot start because the container only exposes llvmpipe and
    the NVIDIA ICD returns `ERROR_INCOMPATIBLE_DRIVER`
  - even an isolated unpacked NVIDIA user-space prefix does not change that
    outcome, which points to an infrastructure/runtime issue rather than a repo
    code issue

### Closed-loop environment bootstrap on `autodl`

- Added a repeatable bootstrap path for the official closed-loop environment.
- Recorded the server-side compatibility split:
  - the existing `uq` env can run ORION, but cannot install `carla==0.9.15`
    through pip on Python 3.11
  - a new `orion-cl` Python 3.8 env can install `carla==0.9.15`, but still
    needs the ORION runtime layer completed
- Prepared separate upstream roots on the server:
  - `Bench2Drive` for evaluation/runtime tooling
  - `Bench2DriveZoo` for model helper code
- Added automatic path injection for `leaderboard`, `scenario_runner`, and the
  project root in the isolated closed-loop env.
- Tightened the environment check script so it validates the actual import
  chain used by the official agent.

### Official open-loop reproduction gate

- Recorded the decision that official CARLA closed-loop alignment must be
  preceded by an upstream-style official open-loop reproduction step.
- Locked the immediate comparison gate to the upstream
  `adzoo/orion/orion_dist_eval.sh` entry, using the base ORION checkpoint
  before introducing FiLM checkpoints.
- Clarified that the existing `scripts/eval_openloop.py` results remain valid
  for internal UQ / FiLM analysis, but are not the strict paper-aligned
  horizontal-comparison protocol.

### Round-2 workflow reset and official closed-loop priority

- Re-defined `closed-loop` to mean paper-aligned official CARLA evaluation only.
- Demoted Bench2Drive replay control checks to diagnostic-only status.
- Updated the round-2 server bundle so the default flow is `train + open-loop`;
  replay is now opt-in.
- Added an environment validation script for the official closed-loop path.
- Added a dedicated runbook for aligning the official CARLA stack with the
  upstream ORION repository.
- Updated automation so that after `R2-A` finishes open-loop, `R2-B/C` are
  paused and the main priority shifts to official closed-loop alignment.
- Extended `team_code/orion_b2d_agent.py` so official evaluation can optionally
  load a separate FiLM checkpoint on top of the base ORION checkpoint.

### FiLM round-2 training support

- Extended `scripts/train_film.py` with round-2 loss controls for:
  - low-UQ FiLM amplitude regularization
  - under-progress penalty
  - comfort / smoothness penalty
- Kept the training target as FiLM-only fine-tuning with frozen UQ and ORION
  backbone weights.

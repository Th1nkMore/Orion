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

### Closed-loop environment bootstrap on `autodl`

- Added a repeatable bootstrap path for the official closed-loop environment.
- Recorded the server-side compatibility split:
  - the existing `uq` env can run ORION, but cannot install `carla==0.9.15`
    through pip on Python 3.11
  - a new `orion-cl` Python 3.8 env can install `carla==0.9.15`, but still
    needs the ORION runtime layer completed
- Cloned Bench2Drive under `/root/autodl-tmp/Bench2DriveZoo` and linked it into
  the project tree as `Bench2DriveZoo`.
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

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

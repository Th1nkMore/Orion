# Round-2 Local / Server Workflow

This workflow assumes a hard split:

- Local machine: consume existing JSON/TXT outputs, generate figures, write summaries, and compare experiments.
- Server: train FiLM variants and run ORION open-loop evaluation. Paper-aligned closed-loop is a separate future track.

This document is the operational source of truth for round-2 work.

## Terminology

- `Open-loop`: offline validation with `scripts/eval_openloop.py`. This is the current round-2 comparison gate.
- `Closed-loop`: reserved for the paper-aligned official CARLA evaluation only.
- `Replay control check`: `scripts/eval_closedloop_replay.py` on Bench2Drive recordings. This is diagnostic only and must not be reported as the main closed-loop result.

## Server Environment

Current verified server setup:

- Host alias: `autodl`
- Project root: `/root/Orion`
- Conda init: `source /root/miniconda3/etc/profile.d/conda.sh`
- Runtime env: `conda activate uq`
- Python: `3.11.5`
- Key packages confirmed: `torch`, `transformers`, `matplotlib`, `scipy`

Quick connect:

```bash
ssh autodl
cd /root/Orion
source /root/miniconda3/etc/profile.d/conda.sh
conda activate uq
```

For automation, non-interactive probes, and rsync-based sync, prefer:

```bash
ssh -o ClearAllForwardings=yes autodl
```

This avoids noisy `RemoteForward` warnings from the interactive SSH config.

## Local Phase

Build the current round-2 baseline dashboard without loading ORION:

```bash
python scripts/build_round2_dashboard.py --out-dir results/round2_dashboard
```

Outputs:

- `results/round2_dashboard/dashboard_summary.json`
- `results/round2_dashboard/current_best_table.csv`
- `results/round2_dashboard/summary.md`
- `results/round2_dashboard/fig_current_best_overview.png`
- `results/round2_dashboard/fig_safety_efficiency_tradeoff.png`
- `results/round2_dashboard/fig_conservative_shortcut_evidence.png`
- `results/round2_dashboard/fig_bev_feasibility_snapshot.png`

These files should be used as the baseline reference before interpreting any new server experiment.

## Round-2 Experiment Bundle

Server-side output layout is fixed:

- Results: `results/round2/<EXP_ID>/`
- Checkpoint: `checkpoints/film_round2/<EXP_ID>.pt`

Mandatory files per default experiment:

- `manifest.json`
- `train.log`
- `openloop.log`
- `openloop.pt`
- `openloop_summary.json`
- `<EXP_ID>.pt`

Optional diagnostic replay files, only when explicitly enabled with `RUN_REPLAY_CLOSED_LOOP=1`:

- `replay_closedloop.log`
- `replay_closedloop.json`

Planned experiment IDs:

| Exp ID | Goal | Notes |
|---|---|---|
| `R2-A` | Stronger UQ gating | Minimal round-2 variant |
| `R2-B` | UQ gating + FiLM amplitude constraint | Main regularized variant |
| `R2-C` | UQ gating + FiLM amplitude constraint + efficiency/comfort constraint | Full round-2 variant |

Recommended training flags:

| Exp ID | `TRAIN_ENV_VARS` | `TRAIN_EXTRA_ARGS` |
|---|---|---|
| `R2-A` | `USE_FILM_L1L2=1` | `--epochs 3 --lr 1e-3 --max-samples 3000 --lambda-col 0.5 --col-margin 4.0` |
| `R2-B` | `USE_FILM_L1L2=1` | `--epochs 3 --lr 1e-3 --max-samples 3000 --lambda-col 0.5 --col-margin 4.0 --lambda-film-reg 0.05` |
| `R2-C` | `USE_FILM_L1L2=1` | `--epochs 3 --lr 1e-3 --max-samples 3000 --lambda-col 0.5 --col-margin 4.0 --lambda-film-reg 0.05 --lambda-progress 0.1 --lambda-comfort 0.02` |

## Server Run Template

Run a single experiment on the server:

```bash
ssh autodl
cd /root/Orion
source /root/miniconda3/etc/profile.d/conda.sh
conda activate uq

EXP_ID=R2-A \
TRAIN_EXTRA_ARGS="--epochs 3 --lr 1e-3 --lambda-col 0.5 --col-margin 4.0" \
bash scripts/round2_server_bundle.sh
```

The bundle script standardizes all output paths and logs. If round-2 code adds new flags later, pass them through `TRAIN_EXTRA_ARGS`, `OPEN_EXTRA_ARGS`, or `REPLAY_EXTRA_ARGS` instead of changing the folder layout.

Optional environment overrides:

```bash
EXP_ID=R2-B \
PROJECT_ROOT=/root/Orion \
ORION_CKPT=ckpts/Orion.pth \
ANN_FILE=data/infos/b2d_infos_val.pkl \
TRAIN_ENV_VARS="USE_FILM_L1L2=1" \
TRAIN_EXTRA_ARGS="--epochs 5 --lr 1e-3 --max-samples 3000 --lambda-col 0.5 --col-margin 4.0" \
OPEN_EXTRA_ARGS="" \
RUN_REPLAY_CLOSED_LOOP=0 \
bash scripts/round2_server_bundle.sh
```

Concrete examples:

```bash
EXP_ID=R2-A \
TRAIN_ENV_VARS="USE_FILM_L1L2=1" \
TRAIN_EXTRA_ARGS="--epochs 3 --lr 1e-3 --max-samples 3000 --lambda-col 0.5 --col-margin 4.0" \
bash scripts/round2_server_bundle.sh

EXP_ID=R2-B \
TRAIN_ENV_VARS="USE_FILM_L1L2=1" \
TRAIN_EXTRA_ARGS="--epochs 3 --lr 1e-3 --max-samples 3000 --lambda-col 0.5 --col-margin 4.0 --lambda-film-reg 0.05" \
bash scripts/round2_server_bundle.sh

EXP_ID=R2-C \
TRAIN_ENV_VARS="USE_FILM_L1L2=1" \
TRAIN_EXTRA_ARGS="--epochs 3 --lr 1e-3 --max-samples 3000 --lambda-col 0.5 --col-margin 4.0 --lambda-film-reg 0.05 --lambda-progress 0.1 --lambda-comfort 0.02" \
bash scripts/round2_server_bundle.sh
```

If you explicitly need a diagnostic replay check for debugging, run it as an opt-in side task:

```bash
EXP_ID=R2-A \
RUN_REPLAY_CLOSED_LOOP=1 \
REPLAY_EXTRA_ARGS="--max-scenarios 50" \
bash scripts/round2_server_bundle.sh
```

## Result Sync Back To Local

Sync one experiment:

```bash
scripts/fetch_round2_results.sh autodl /root/Orion R2-A
```

The sync script already defaults to:

```bash
ssh -o ClearAllForwardings=yes
```

Then refresh the local dashboard or downstream comparison scripts using the returned summary files.

## Decision Loop

1. Generate `results/round2_dashboard/` locally from the current baseline assets.
2. Run one server experiment bundle into `results/round2/<EXP_ID>/`.
3. Sync back lightweight artifacts only.
4. Compare the new open-loop outputs against the baseline dashboard.
5. After `R2-A`, pause `R2-B/C` and switch the main priority to the official CARLA closed-loop alignment track documented in `docs/official_carla_closedloop_alignment.md`.

## Official Closed-Loop Track

This is not part of the default round-2 bundle.

- Goal: reproduce the paper-aligned CARLA closed-loop configuration for direct horizontal comparison.
- Required environment: A100-class GPU plus CARLA runtime, route/scenario setup matching the original evaluation protocol.
- Expected output target: a dedicated result file such as `results/closedloop_official_<tag>.json`, separate from `results/round2/`.
- Rule: do not label replay outputs as `closed-loop` in summaries, dashboards, or decisions.
- Current priority rule: once `R2-A` finishes open-loop, this track takes precedence over `R2-B` and `R2-C`.
- Operational runbook: `docs/official_carla_closedloop_alignment.md`

## Notes

- Do not treat the local machine as an ORION runtime target.
- Do not rely on ad-hoc output paths; use the fixed `results/round2/<EXP_ID>/` layout.
- Do not overwrite baseline files in `results/`; write all new outputs under `results/round2/`.
- Do not use replay control checks as the headline closed-loop comparison.

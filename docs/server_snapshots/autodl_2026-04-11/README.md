# AutoDL Server Snapshot (2026-04-11)

This directory captures the minimum reference snapshot before releasing the
current AutoDL server instance.

## Included Files

- `conda_uq_env.yml`
  - exported from `/root/autodl-tmp/conda/envs/uq`
- `conda_orion-cl_env.yml`
  - exported from `/root/autodl-tmp/conda/envs/orion-cl`
- `storage_layout.txt`
  - top-level layout snapshot for `/root/Orion` and `/root/autodl-tmp`
  - includes selected `du -sh` totals for the major storage roots

## Why These Snapshots Matter

- `uq` is the working ORION training/open-loop environment.
- `orion-cl` is the isolated Python 3.8 environment prepared for official
  closed-loop compatibility checks.
- The storage layout snapshot records where the main checkpoints, CARLA
  runtime, Bench2Drive repos, and data caches lived on the released server.

## Key Files Still Worth Preserving From The Server

The following files were still present on the server at snapshot time and were
worth copying out before shutdown:

- `/root/Orion/checkpoints/uq/best.pt`
- `/root/Orion/checkpoints/film/best_l1l2_col_v4.pt`
- `/root/Orion/checkpoints/film_round2/R2-A.pt`
- `/root/Orion/results/openloop_official/baseline.log`

These binary/log artifacts are preserved separately on the artifact branch
instead of `dev`.

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${ORION_REPO_ROOT:-/public/home/lidachuan/project/Orion}"
ASSET_ROOT="${ORION_ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
RUN_SCRIPT="${REPO_ROOT}/scripts/run_b2d_expansion100_infos.sh"
LOG_ROOT="${ASSET_ROOT}/observation_uq_v3/plans/b2d_expansion100_seed20260827"

test -x "${RUN_SCRIPT}"
mkdir -p "${LOG_ROOT}"

sbatch --parsable \
  --partition=Nvidia_A800 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=02:00:00 \
  --job-name=b2d_expand_infos \
  --chdir="${REPO_ROOT}" \
  --output="${LOG_ROOT}/infos-slurm-%j.out" \
  --export="ALL,ORION_REPO_ROOT=${REPO_ROOT},ORION_ASSET_ROOT=${ASSET_ROOT}" \
  "${RUN_SCRIPT}"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
HOME_WORK_ROOT="${HOME_WORK_ROOT:-/public/home/lidachuan/orion_work}"
RUN_ID="${RUN_ID:-counterfactual_evidence_fp16_routes_expansion100_seed20260827_r1}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_counterfactual_evidence_fp16_route_expansion100.sh"
LOG_ROOT="${HOME_WORK_ROOT}/observation_uq_v3/logs"

test -x "${RUN_SCRIPT}"
mkdir -p "${LOG_ROOT}"

sbatch --parsable \
  --partition=Nvidia_A800 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=160G \
  --time=04:00:00 \
  --job-name=cf_fp16_routes100 \
  --chdir="${PROJECT_ROOT}" \
  --output="${LOG_ROOT}/${RUN_ID}-%j.out" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},HOME_WORK_ROOT=${HOME_WORK_ROOT},RUN_ID=${RUN_ID},RESUME=${RESUME:-0}" \
  "${RUN_SCRIPT}"

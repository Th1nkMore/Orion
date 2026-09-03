#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
HOME_WORK_ROOT="${HOME_WORK_ROOT:-/public/home/lidachuan/orion_work}"
AUDIT_RUN_ID="${AUDIT_RUN_ID:-counterfactual_evidence_fp16_route_audit_expansion100_seed20260827_r1}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_counterfactual_evidence_fp16_route_audit.sh"
LOG_ROOT="${HOME_WORK_ROOT}/observation_uq_v3/logs"

test -x "${RUN_SCRIPT}"
mkdir -p "${LOG_ROOT}"

dependency_option=""
if [[ -n "${DEPENDENCY_JOB:-}" ]]; then
  if [[ ! "${DEPENDENCY_JOB}" =~ ^[0-9]+$ ]]; then
    echo "[FAIL] DEPENDENCY_JOB must be a numeric Slurm job id" >&2
    exit 1
  fi
  dependency_option="--dependency=afterok:${DEPENDENCY_JOB}"
fi

sbatch --parsable \
  ${dependency_option:+"${dependency_option}"} \
  --partition=Nvidia_A800 \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=160G \
  --time=02:00:00 \
  --job-name=cf_fp16_route_audit \
  --chdir="${PROJECT_ROOT}" \
  --output="${LOG_ROOT}/${AUDIT_RUN_ID}-%j.out" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},HOME_WORK_ROOT=${HOME_WORK_ROOT},AUDIT_RUN_ID=${AUDIT_RUN_ID}" \
  "${RUN_SCRIPT}"

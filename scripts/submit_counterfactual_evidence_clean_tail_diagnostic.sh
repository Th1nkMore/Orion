#!/usr/bin/env bash
set -euo pipefail

submit=0
case "${1:-}" in
  --submit) submit=1 ;;
  ""|--dry-run) ;;
  *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
HOME_WORK_ROOT="${HOME_WORK_ROOT:-/public/home/lidachuan/orion_work}"
RUN_ID="${RUN_ID:-counterfactual_evidence_clean_tail_seed20260827_r1}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/observation_uq_counterfactual_clean_tail_diagnostic_v1.json}"
LOG_ROOT="${LOG_ROOT:-${HOME_WORK_ROOT}/observation_uq_v3/logs}"
PARTITION="${SLURM_PARTITION:-Nvidia_A800}"

wrapped=(env \
  "PROJECT_ROOT=${PROJECT_ROOT}" \
  "HOME_WORK_ROOT=${HOME_WORK_ROOT}" \
  "RUN_ID=${RUN_ID}" \
  "CONFIG_PATH=${CONFIG_PATH}" \
  bash "${PROJECT_ROOT}/scripts/run_counterfactual_evidence_clean_tail_diagnostic.sh")
printf -v wrapped_command '%q ' "${wrapped[@]}"
args=(sbatch --parsable \
  "--partition=${PARTITION}" \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=00:20:00 \
  --job-name=cf_clean_tail \
  "--output=${LOG_ROOT}/${RUN_ID}-%j.out" \
  --export=ALL \
  --wrap "${wrapped_command}")

if [[ "${submit}" != 1 ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RESOURCE_CONTRACT=partition:${PARTITION},gpu:1,cpus:4,mem:32G,time:00:20:00"
  echo "TRAINING=0"
  echo "VALIDATION_ROUTES=10"
  echo "HELDOUT_GLARE_TENSOR_VALUES_ACCESSED=0"
  echo "HELDOUT_SPLIT_TENSOR_VALUES_ACCESSED=0"
  echo "NATIVE_WEATHER_READ=0"
  echo "ORION_FINETUNING=0"
  echo "STAGE_B=0"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${args[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${LOG_ROOT}"
"${args[@]}"

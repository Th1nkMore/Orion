#!/usr/bin/env bash
set -euo pipefail

submit=0
case "${1:-}" in
  --submit) submit=1 ;;
  ""|--dry-run) ;;
  *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
esac

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
work_root="${WORK_ROOT:-/public/home/lidachuan/orion_work/observation_uq_v3}"
run_id="${RUN_ID:-counterfactual_pairwise_glare_seed20260828_r1}"
log_root="${LOG_ROOT:-${work_root}/logs}"
wrapped=(env
  "PROJECT_ROOT=${project_root}"
  "WORK_ROOT=${work_root}"
  "TRAINING_RUN_ID=${TRAINING_RUN_ID:-counterfactual_pairwise_native_repair_seed20260828_r1}"
  "RUN_ID=${run_id}"
  bash "${project_root}/scripts/run_counterfactual_pairwise_glare.sh")
printf -v wrapped_command '%q ' "${wrapped[@]}"
args=(sbatch --parsable
  --partition=Nvidia_A800
  --gres=gpu:1
  --cpus-per-task=2
  --mem=24G
  --time=00:30:00)
if [[ -n "${SLURM_DEPENDENCY:-}" ]]; then
  args+=("--dependency=${SLURM_DEPENDENCY}")
fi
args+=(
  --job-name=cf_pair_glare
  "--output=${log_root}/${run_id}-%j.out"
  --export=ALL
  --wrap "${wrapped_command}")

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RESOURCE_CONTRACT=partition:Nvidia_A800,gpu:1,cpus:2,mem:24G,time:00:30:00"
  echo "TRAINING=0"
  echo "FAMILY_READ=local_glare"
  echo "SPLITS=validation,held_out"
  echo "NATIVE_FINAL_HELDOUT_READ=0"
  echo "STAGE_B=0"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${args[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${log_root}"
"${args[@]}"

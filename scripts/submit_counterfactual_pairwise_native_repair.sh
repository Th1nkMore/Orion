#!/usr/bin/env bash
set -euo pipefail

submit=0
case "${1:-}" in
  --submit) submit=1 ;;
  ""|--dry-run) ;;
  *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
esac

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
home_work_root="${HOME_WORK_ROOT:-/public/home/lidachuan/orion_work}"
run_id="${RUN_ID:-counterfactual_pairwise_native_repair_seed20260828_r1}"
log_root="${LOG_ROOT:-${home_work_root}/observation_uq_v3/logs}"
partition="${SLURM_PARTITION:-Nvidia_A800}"
cpus_per_task="${SLURM_CPUS_PER_TASK:-2}"

wrapped=(env
  "PROJECT_ROOT=${project_root}"
  "HOME_WORK_ROOT=${home_work_root}"
  "ASSET_ROOT=${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
  "RUN_ID=${run_id}"
  "NATIVE_RUN_ID=${NATIVE_RUN_ID:-counterfactual_pairwise_native_train_seed20260828_r2}"
  "CONFIG_PATH=${CONFIG_PATH:-${project_root}/configs/observation_uq_counterfactual_pairwise_native_training_v4.json}"
  bash "${project_root}/scripts/run_counterfactual_pairwise_native_repair.sh")
printf -v wrapped_command '%q ' "${wrapped[@]}"

args=(sbatch --parsable
  "--partition=${partition}"
  --gres=gpu:1
  "--cpus-per-task=${cpus_per_task}"
  --mem=64G
  --time=02:00:00
  --job-name=cf_pair_repair
  "--output=${log_root}/${run_id}-%j.out"
  --export=ALL
  --wrap "${wrapped_command}")

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RESOURCE_CONTRACT=partition:${partition},gpu:1,cpus:${cpus_per_task},mem:64G,time:02:00:00"
  echo "PAIRWISE_DELTA_SUPERVISION=1"
  echo "BLANKET_REFERENCE_ZERO_LOSS=0"
  echo "FINAL_NATIVE_HELDOUT_ROUTES_READ=0"
  echo "MAX_EPOCHS=4"
  echo "STAGE_B=0"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${args[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${log_root}"
"${args[@]}"

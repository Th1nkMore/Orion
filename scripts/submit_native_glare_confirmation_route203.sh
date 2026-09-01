#!/usr/bin/env bash
set -euo pipefail

submit=0
case "${1:-}" in
  --submit) submit=1 ;;
  ""|--dry-run) ;;
  *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
esac

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-route203_native_glare_same_tick_v1}"
output_root="${OUTPUT_ROOT:-${asset_root}/scenario_factory/glare_confirmations/${run_id}}"
log_root="${asset_root}/scenario_factory/logs/glare_confirmations"
job_name="glare203_triplet"

if [[ -e "${output_root}" ]]; then
  echo "refusing to reuse glare confirmation output root: ${output_root}" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active glare confirmation" >&2
  exit 1
fi

wrapped=(
  env
  "PROJECT_ROOT=${project_root}"
  "ASSET_ROOT=${asset_root}"
  "RUN_ID=${run_id}"
  "OUTPUT_ROOT=${output_root}"
  bash "${project_root}/scripts/run_native_glare_confirmation_route203.sh"
)
printf -v wrapped_command '%q ' "${wrapped[@]}"
sbatch_args=(
  sbatch --parsable --partition=Nvidia_A800 --gres=gpu:1
  --cpus-per-task=2 --mem=16G --time=00:45:00 --exclude=gpu5
  --job-name="${job_name}"
  "--output=${log_root}/${run_id}-%j.out"
  --export=ALL --wrap "${wrapped_command}"
)

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RUN_ID=${run_id}"
  echo "RESOURCE_CONTRACT=partition:Nvidia_A800,gpu:1,cpus:2,mem:16G,time:00:45:00"
  echo "ORION_LOAD=0"
  echo "ADAPTER_LOAD=0"
  echo "AUTOMATIC_RETRY=0"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${log_root}"
job_id=$("${sbatch_args[@]}")
printf 'GLARE_CONFIRMATION_JOB_ID=%s\n' "${job_id}"

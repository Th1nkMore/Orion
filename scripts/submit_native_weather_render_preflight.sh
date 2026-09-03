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
run_id="${RUN_ID:-native_weather_epic_visual_preflight_r1}"
output_root="${OUTPUT_ROOT:-${asset_root}/observation_uq_v3/runs/${run_id}}"
log_root="${LOG_ROOT:-${asset_root}/logs/observation_uq_v3}"

wrapped=(
  env
  "PROJECT_ROOT=${project_root}"
  "ASSET_ROOT=${asset_root}"
  "RUN_ID=${run_id}"
  "OUTPUT_ROOT=${output_root}"
  bash "${project_root}/scripts/run_native_weather_render_preflight.sh"
)
printf -v wrapped_command '%q ' "${wrapped[@]}"

nodelist_args=()
if [[ -n "${SLURM_NODELIST:-}" ]]; then
  nodelist_args+=("--nodelist=${SLURM_NODELIST}")
fi
sbatch_args=(
  sbatch --parsable --partition=Nvidia_A800 --gres=gpu:1
  --cpus-per-task=2 --mem=16G --time=00:15:00
  "${nodelist_args[@]}"
  --job-name=fog_epic_qa
  "--output=${log_root}/${run_id}-%j.out"
  --export=ALL --wrap "${wrapped_command}"
)

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RUN_ID=${run_id}"
  echo "RESOURCE_CONTRACT=partition:Nvidia_A800,gpu:1,cpus:2,mem:16G,time:00:15:00"
  echo "FEATURE_EXTRACTION=0"
  echo "ADAPTER_TRAINING=0"
  echo "STAGE_B=0"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi

if [[ -e "${output_root}" ]]; then
  echo "refusing to reuse Epic preflight output root: ${output_root}" >&2
  exit 1
fi
mkdir -p "${log_root}"
"${sbatch_args[@]}"

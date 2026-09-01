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
run_id="${RUN_ID:-route151_native_motion_blur_clean_revalidation_v1}"
output_root="${OUTPUT_ROOT:-${asset_root}/scenario_factory/corruption_hardcase_diagnostics/${run_id}}"
log_root="${asset_root}/scenario_factory/logs/corruption_hardcase_diagnostics"
job_name="hc151_blur_reval"

if [[ -e "${output_root}" ]]; then
  echo "refusing to reuse native-motion-blur output root: ${output_root}" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active native-motion-blur revalidation" >&2
  exit 1
fi

wrapped=(
  env
  "PROJECT_ROOT=${project_root}"
  "ASSET_ROOT=${asset_root}"
  "RUN_ID=${run_id}"
  "OUTPUT_ROOT=${output_root}"
  bash "${project_root}/scripts/run_native_motion_blur_clean_revalidation_route151.sh"
)
printf -v wrapped_command '%q ' "${wrapped[@]}"
sbatch_args=(
  sbatch --parsable --partition=Nvidia_A800 --gres=gpu:1
  --cpus-per-task=2 --mem=16G --time=00:25:00 --exclude=gpu5
  --job-name="${job_name}"
  "--output=${log_root}/${run_id}-%j.out"
  --export=ALL --wrap "${wrapped_command}"
)

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RUN_ID=${run_id}"
  echo "RESOURCE_CONTRACT=partition:Nvidia_A800,gpu:1,cpus:2,mem:16G,time:00:25:00"
  echo "ORION_LOAD=0"
  echo "PROFILES=none,medium"
  printf 'SBATCH_COMMAND='; printf '%q ' "${sbatch_args[@]}"; printf '\n'
  exit 0
fi

mkdir -p "${log_root}"
"${sbatch_args[@]}"

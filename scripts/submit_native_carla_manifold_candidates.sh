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
run_id="${RUN_ID:-native_manifold_candidates_seed20260826_r1}"
output_root="${OUTPUT_ROOT:-${asset_root}/observation_uq_v3/runs/${run_id}}"
log_root="${LOG_ROOT:-${asset_root}/logs/observation_uq_v3}"
partition="${SLURM_PARTITION:-Nvidia_A800}"

wrapped=(env "PROJECT_ROOT=${project_root}" "ASSET_ROOT=${asset_root}" "RUN_ID=${run_id}" "OUTPUT_ROOT=${output_root}" bash "${project_root}/scripts/run_native_carla_manifold_candidates.sh")
printf -v wrapped_command '%q ' "${wrapped[@]}"
sbatch_args=(
  sbatch --parsable "--partition=${partition}" --gres=gpu:1 --cpus-per-task=8 --mem=64G
  --time=00:30:00
)
if [[ -n "${SLURM_NODELIST:-}" ]]; then
  sbatch_args+=("--nodelist=${SLURM_NODELIST}")
fi
sbatch_args+=(
  --job-name=native_manifold
  "--output=${log_root}/${run_id}-%j.out" --export=ALL --wrap "${wrapped_command}"
)

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RUN_ID=${run_id}"
  echo "RESOURCE_CONTRACT=partition:${partition},gpu:1,cpus:8,mem:64G,time:00:30:00"
  echo "CANDIDATE_TRAINING=0"
  echo "ADAPTER_TRAINING=0"
  echo "STAGE_B=0"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi
if [[ -e "${output_root}" ]]; then
  echo "refusing to reuse manifold audit output root: ${output_root}" >&2
  exit 1
fi
mkdir -p "${log_root}"
"${sbatch_args[@]}"

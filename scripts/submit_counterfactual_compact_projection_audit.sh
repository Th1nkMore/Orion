#!/usr/bin/env bash
set -euo pipefail
submit=0
case "${1:-}" in --submit) submit=1 ;; ""|--dry-run) ;; *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;; esac
project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-counterfactual_compact_projection_audit_seed20260827_r1}"
log_root="${LOG_ROOT:-${asset_root}/logs/observation_uq_v3}"
partition="${SLURM_PARTITION:-Nvidia_A800}"
wrapped=(env "PROJECT_ROOT=${project_root}" "ASSET_ROOT=${asset_root}" "RUN_ID=${run_id}" bash "${project_root}/scripts/run_counterfactual_compact_projection_audit.sh")
printf -v wrapped_command '%q ' "${wrapped[@]}"
args=(sbatch --parsable "--partition=${partition}" --gres=gpu:1 --cpus-per-task=4 --mem=140G --time=00:30:00 --job-name=cf_compact_audit "--output=${log_root}/${run_id}-%j.out" --export=ALL --wrap "${wrapped_command}")
if [[ "${submit}" != 1 ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RESOURCE_CONTRACT=partition:${partition},gpu:1,cpus:4,mem:140G,time:00:30:00"
  echo "OUTPUT_FEATURE_SHARD_WRITTEN=0"
  echo "AUTOMATIC_ADAPTER_TRAINING=0"
  echo "STAGE_B=0"
  printf 'SBATCH_COMMAND='; printf '%q ' "${args[@]}"; printf '\n'; exit 0
fi
mkdir -p "${log_root}"
"${args[@]}"

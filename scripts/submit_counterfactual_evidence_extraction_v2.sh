#!/usr/bin/env bash
set -euo pipefail

submit=0
case "${1:-}" in --submit) submit=1 ;; ""|--dry-run) ;; *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;; esac
project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-counterfactual_evidence_features_windowcycle_seed20260827_r1}"
output_root="${OUTPUT_ROOT:-${asset_root}/observation_uq_v3/runs/${run_id}}"
log_root="${LOG_ROOT:-${asset_root}/logs/observation_uq_v3}"
partition="${SLURM_PARTITION:-Nvidia_A800}"

wrapped=(env "PROJECT_ROOT=${project_root}" "ASSET_ROOT=${asset_root}" "RUN_ID=${run_id}" "OUTPUT_ROOT=${output_root}" bash "${project_root}/scripts/run_counterfactual_evidence_extraction_v2.sh")
printf -v wrapped_command '%q ' "${wrapped[@]}"
args=(sbatch --parsable "--partition=${partition}" --gres=gpu:1 --cpus-per-task=8 --mem=160G --time=03:00:00 --job-name=cf_evidence_v2_extract "--output=${log_root}/${run_id}-%j.out" --export=ALL --wrap "${wrapped_command}")
if [[ "${submit}" != 1 ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RESOURCE_CONTRACT=partition:${partition},gpu:1,cpus:8,mem:160G,time:03:00:00"
  echo "PROJECTED_FEATURES=3600"
  echo "PROJECTED_STORAGE_GIB=65.92"
  echo "VIEW_SCHEDULE=route_condition_window_cycle/v2"
  echo "EXACT_NONZERO_PRESENCE_LABEL=0"
  echo "ADAPTER_TRAINING=0"
  echo "STAGE_B=0"
  printf 'SBATCH_COMMAND='; printf '%q ' "${args[@]}"; printf '\n'
  exit 0
fi
if [[ -e "${output_root}" ]]; then echo "refusing to reuse ${output_root}" >&2; exit 1; fi
mkdir -p "${log_root}"
"${args[@]}"

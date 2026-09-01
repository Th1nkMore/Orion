#!/usr/bin/env bash
set -euo pipefail
submit=0
case "${1:-}" in --submit) submit=1 ;; ""|--dry-run) ;; *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;; esac
project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-counterfactual_evidence_high_support_hurdle_seed20260827_r1}"
config_path="${CONFIG_PATH:-${project_root}/configs/observation_uq_counterfactual_high_support_hurdle_smoke_v1.json}"
log_root="${LOG_ROOT:-${asset_root}/logs/observation_uq_v3}"
partition="${SLURM_PARTITION:-Nvidia_A800}"
wrapped=(env "PROJECT_ROOT=${project_root}" "ASSET_ROOT=${asset_root}" "RUN_ID=${run_id}" "CONFIG_PATH=${config_path}" bash "${project_root}/scripts/run_counterfactual_evidence_high_support_hurdle_smoke.sh")
printf -v wrapped_command '%q ' "${wrapped[@]}"
args=(sbatch --parsable "--partition=${partition}" --gres=gpu:1 --cpus-per-task=4 --mem=140G --time=01:00:00 --job-name=cf_high_support "--output=${log_root}/${run_id}-%j.out" --export=ALL --wrap "${wrapped_command}")
if [[ "${submit}" != 1 ]]; then
  echo "DRY_RUN_ONLY=1"; echo "RESOURCE_CONTRACT=partition:${partition},gpu:1,cpus:4,mem:140G,time:01:00:00"
  echo "SUPPORT=TRAIN_RESPONSIVE_Q80"; echo "EXACT_NONZERO_PRESENCE_LABEL=0"
  echo "HELDOUT_GLARE_READ=0"; echo "NATIVE_FOG_READ=0"; echo "AUTOMATIC_FULL_TRAINING=0"; echo "STAGE_B=0"
  printf 'SBATCH_COMMAND='; printf '%q ' "${args[@]}"; printf '\n'; exit 0
fi
mkdir -p "${log_root}"
"${args[@]}"

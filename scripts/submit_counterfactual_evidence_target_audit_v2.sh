#!/usr/bin/env bash
set -euo pipefail

submit=0
case "${1:-}" in --submit) submit=1 ;; ""|--dry-run) ;; *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;; esac
project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
feature_run_id="${FEATURE_RUN_ID:-counterfactual_evidence_features_windowcycle_seed20260827_r1}"
audit_run_id="${AUDIT_RUN_ID:-counterfactual_evidence_target_audit_windowcycle_seed20260827_r1}"
dependency_job="${DEPENDENCY_JOB:-1071011}"
log_root="${LOG_ROOT:-${asset_root}/logs/observation_uq_v3}"
partition="${SLURM_PARTITION:-Nvidia_A800}"
wrapped=(env "PROJECT_ROOT=${project_root}" "ASSET_ROOT=${asset_root}" "FEATURE_RUN_ID=${feature_run_id}" "AUDIT_RUN_ID=${audit_run_id}" bash "${project_root}/scripts/run_counterfactual_evidence_target_audit_v2.sh")
printf -v wrapped_command '%q ' "${wrapped[@]}"
args=(sbatch --parsable "--partition=${partition}" --gres=gpu:1 --cpus-per-task=2 --mem=120G --time=00:45:00 --job-name=cf_target_audit_v2 "--dependency=afterok:${dependency_job}" "--output=${log_root}/${audit_run_id}-%j.out" --export=ALL --wrap "${wrapped_command}")
if [[ "${submit}" != 1 ]]; then
  echo "DRY_RUN_ONLY=1"; echo "DEPENDENCY=afterok:${dependency_job}"
  echo "RESOURCE_CONTRACT=partition:${partition},gpu:1,cpus:2,mem:120G,time:00:45:00"
  echo "CONTINUOUS_TARGET_AUDIT=1"; echo "EXACT_NONZERO_PRESENCE_LABEL=0"
  echo "ADAPTER_TRAINING=0"; echo "VALIDATION_READ=0"; echo "STAGE_B=0"
  printf 'SBATCH_COMMAND='; printf '%q ' "${args[@]}"; printf '\n'; exit 0
fi
mkdir -p "${log_root}"
"${args[@]}"

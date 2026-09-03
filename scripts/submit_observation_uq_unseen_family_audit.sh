#!/usr/bin/env bash
set -euo pipefail

submit=0
if [[ "${1:-}" == "--submit" ]]; then
  submit=1
elif [[ -n "${1:-}" && "${1:-}" != "--dry-run" ]]; then
  echo "usage: $0 [--dry-run|--submit]" >&2
  exit 2
fi

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
family="${DIAGNOSTIC_FAMILY:-local_blur}"
run_id="${RUN_ID:-observation_uq_unseen_${family}_seed20260827_r1}"
output_root="${OUTPUT_ROOT:-${asset_root}/observation_uq_v3/runs/${run_id}}"
log_root="${LOG_ROOT:-${asset_root}/logs/observation_uq_v3}"

wrapped_parts=(
  env
  "PROJECT_ROOT=${project_root}"
  "ASSET_ROOT=${asset_root}"
  "RUN_ID=${run_id}"
  "OUTPUT_ROOT=${output_root}"
  "DIAGNOSTIC_FAMILY=${family}"
  "DIAGNOSTIC_SEED=${DIAGNOSTIC_SEED:-20260827}"
  "PYTHONPATH=${project_root}:${PYTHONPATH:-}"
  "COMPAT_PYTHON_BIN=${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
  "COMPAT_GLIBC_SYSROOT=${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
  "COMPAT_LIBRARY_PATH=${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
  bash
  "${project_root}/scripts/run_observation_uq_unseen_family_audit.sh"
)
printf -v wrapped_command '%q ' "${wrapped_parts[@]}"

sbatch_args=(
  sbatch --parsable
  --partition=Nvidia_A800
  --gres=gpu:1
  --cpus-per-task=8
  --mem=96G
  --time=01:00:00
  --job-name=obs_uq_unseen
  "--output=${log_root}/${run_id}-%j.out"
  --export=ALL
  --wrap "${wrapped_command}"
)

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RUN_ID=${run_id}"
  echo "DIAGNOSTIC_FAMILY=${family}"
  echo "TRAIN_CORRUPTION_OBSERVATIONS=0"
  echo "CROSS_FAMILY_SCREEN_ONLY=1"
  echo "NATIVE_ENGINE_EVIDENCE=0"
  echo "ADAPTER_TRAINING=0"
  echo "STAGE_B=0"
  echo "RESOURCE_CONTRACT=partition:Nvidia_A800,gpu:1,cpus:8,mem:96G,time:01:00:00"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -f "${project_root}/scripts/run_observation_uq_unseen_family_audit.sh" ]]; then
  echo "missing remote runner" >&2
  exit 1
fi
if [[ -e "${output_root}" ]]; then
  echo "refusing to reuse output root: ${output_root}" >&2
  exit 1
fi
mkdir -p "${log_root}"
"${sbatch_args[@]}"


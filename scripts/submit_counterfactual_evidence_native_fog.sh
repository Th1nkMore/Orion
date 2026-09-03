#!/usr/bin/env bash
set -euo pipefail

submit=0
case "${1:-}" in
  --submit) submit=1 ;;
  ""|--dry-run) ;;
  *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
esac

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
work_root="${WORK_ROOT:-/public/home/lidachuan/orion_work/observation_uq_v3}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-counterfactual_evidence_no_view_native_fog_seed20260827_r1}"
output_root="${OUTPUT_ROOT:-${work_root}/runs/${run_id}}"
log_root="${LOG_ROOT:-${work_root}/logs}"
wrapped_parts=(
  env
  "PROJECT_ROOT=${project_root}"
  "WORK_ROOT=${work_root}"
  "ASSET_ROOT=${asset_root}"
  "RUN_ID=${run_id}"
  "OUTPUT_ROOT=${output_root}"
  "PYTHONPATH=${project_root}:${PYTHONPATH:-}"
  "COMPAT_PYTHON_BIN=${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
  "COMPAT_GLIBC_SYSROOT=${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
  "COMPAT_LIBRARY_PATH=${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
  bash
  "${project_root}/scripts/run_counterfactual_evidence_native_fog.sh"
)
printf -v wrapped_command '%q ' "${wrapped_parts[@]}"
sbatch_args=(
  sbatch --parsable
  --partition=Nvidia_A800
  --gres=gpu:1
  --cpus-per-task=2
  --mem=16G
  --time=00:15:00
  --job-name=uq_native_fog
  "--output=${log_root}/${run_id}-%j.out"
  --export=ALL
  --wrap "${wrapped_command}"
)

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RUN_ID=${run_id}"
  echo "UPSTREAM_GLARE_PASS_REQUIRED=1"
  echo "CARLA_RERENDER=0"
  echo "TRAINING=0"
  echo "CHECKPOINT_UPDATE=0"
  echo "STAGE_B=0"
  echo "RESOURCE_CONTRACT=partition:Nvidia_A800,gpu:1,cpus:2,mem:16G,time:00:15:00"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi

[[ -f "${project_root}/scripts/run_counterfactual_evidence_native_fog.sh" ]] || {
  echo "missing remote native-fog runner" >&2
  exit 1
}
[[ ! -e "${output_root}" ]] || { echo "refusing to reuse output root: ${output_root}" >&2; exit 1; }
mkdir -p "${log_root}"
"${sbatch_args[@]}"

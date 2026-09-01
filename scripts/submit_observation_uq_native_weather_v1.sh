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
run_id="${RUN_ID:-observation_uq_native_weather_seed20260826_r1}"
output_root="${OUTPUT_ROOT:-${asset_root}/observation_uq_v3/runs/${run_id}}"
log_root="${LOG_ROOT:-${asset_root}/logs/observation_uq_v3}"
cpus_per_task="${SLURM_CPUS_PER_TASK:-2}"

wrapped=(
  env
  "PROJECT_ROOT=${project_root}"
  "ASSET_ROOT=${asset_root}"
  "RUN_ID=${run_id}"
  "OUTPUT_ROOT=${output_root}"
  "REUSE_CAPTURE_ROUTE146=${REUSE_CAPTURE_ROUTE146:-}"
  "PYTHONPATH=${project_root}:${PYTHONPATH:-}"
  "COMPAT_PYTHON_BIN=${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
  "COMPAT_GLIBC_SYSROOT=${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
  "COMPAT_LIBRARY_PATH=${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
  bash
  "${project_root}/scripts/run_observation_uq_native_weather_v1.sh"
)
printf -v wrapped_command '%q ' "${wrapped[@]}"

nodelist_args=()
if [[ -n "${SLURM_NODELIST:-}" ]]; then
  nodelist_args+=("--nodelist=${SLURM_NODELIST}")
fi

sbatch_args=(
  sbatch --parsable
  --partition=Nvidia_A800
  --gres=gpu:1
  "--cpus-per-task=${cpus_per_task}"
  --mem=96G
  --time=01:00:00
  "${nodelist_args[@]}"
  --job-name=obs_uq_native
  "--output=${log_root}/${run_id}-%j.out"
  --export=ALL
  --wrap "${wrapped_command}"
)

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RUN_ID=${run_id}"
  echo "NATIVE_CARLA_WEATHER=1"
  echo "PAIRED_WORLD_POSE=1"
  echo "PIXEL_CORRUPTION_GENERATOR=0"
  echo "ADAPTER_TRAINING=0"
  echo "ACTUAL_TARGET_TRAINING=0"
  echo "STAGE_B=0"
  echo "RESOURCE_CONTRACT=partition:Nvidia_A800,gpu:1,cpus:${cpus_per_task},mem:96G,time:01:00:00"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -f "${project_root}/scripts/run_observation_uq_native_weather_v1.sh" ]]; then
  echo "missing remote native-weather runner" >&2
  exit 1
fi
if [[ -e "${output_root}" ]]; then
  echo "refusing to reuse native-weather output root: ${output_root}" >&2
  exit 1
fi
mkdir -p "${log_root}"
"${sbatch_args[@]}"

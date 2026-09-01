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
source_run="${SOURCE_RUN:-observation_uq_teacher560_seed20260826_r1}"
teacher_run="${TEACHER_RUN:-observation_uq_teacher560_v31_seed20260826_r1}"
run_id="${RUN_ID:-observation_uq_signal_audit_v12_route_robust_teacher560_v31_seed20260826_r1}"
shard="${FEATURE_SHARD:-${asset_root}/observation_uq_v3/runs/${source_run}/clean_first_features.pt}"
teacher="${TEACHER_CHECKPOINT:-${asset_root}/observation_uq_v3/runs/${teacher_run}/teacher_v31.pt}"
output_root="${OUTPUT_ROOT:-${asset_root}/observation_uq_v3/runs/${run_id}}"
output="${output_root}/signal_audit_v12.json"
log_root="${LOG_ROOT:-${asset_root}/logs/observation_uq_v3}"

wrapped_parts=(
  env
  "PYTHONPATH=${project_root}:${PYTHONPATH:-}"
  "COMPAT_PYTHON_BIN=${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
  "COMPAT_GLIBC_SYSROOT=${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
  "COMPAT_LIBRARY_PATH=${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
  bash -lc
  "set -euo pipefail; mkdir -p '${output_root}'; sha256sum '${shard}' '${teacher}' '${project_root}/uq_estimator/observation_uq_signal_audit.py' '${project_root}/scripts/audit_observation_uq_signals_v1.py' > '${output_root}/source_sha256.txt'; '${project_root}/scripts/run_compat_python.sh' '${project_root}/scripts/audit_observation_uq_signals_v1.py' --shard '${shard}' --teacher '${teacher}' --output '${output}' --batch-size 8 --device cuda; sha256sum '${output}' > '${output_root}/artifact_sha256.txt'"
)
printf -v wrapped_command '%q ' "${wrapped_parts[@]}"

sbatch_args=(
  sbatch --parsable
  --partition=Nvidia_A800
  --gres=gpu:1
  --cpus-per-task=2
  --mem=48G
  --time=00:30:00
  --job-name=obs_uq_sig_audit
  "--output=${log_root}/${run_id}-%j.out"
  --export=ALL
  --wrap "${wrapped_command}"
)

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RUN_ID=${run_id}"
  echo "FEATURE_SHARD=${shard}"
  echo "TEACHER_CHECKPOINT=${teacher}"
  echo "CALIBRATION_INPUTS=clean_train_only"
  echo "CORRUPTION_METADATA=evaluation_only"
  echo "PAIRED_DELTA=diagnostic_oracle_only"
  echo "ADAPTER_TRAINING=0"
  echo "STAGE_B=0"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -f "${shard}" || ! -f "${teacher}" ]]; then
  echo "missing immutable input" >&2
  exit 1
fi
if [[ -e "${output_root}" ]]; then
  echo "refusing to reuse output root: ${output_root}" >&2
  exit 1
fi
mkdir -p "${log_root}"
"${sbatch_args[@]}"

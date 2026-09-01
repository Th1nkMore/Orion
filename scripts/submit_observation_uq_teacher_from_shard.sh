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
shard="${FEATURE_SHARD:-${asset_root}/observation_uq_v3/runs/${source_run}/clean_first_features.pt}"
run_id="${RUN_ID:-observation_uq_teacher560_v31_seed20260826_r1}"
output_root="${OUTPUT_ROOT:-${asset_root}/observation_uq_v3/runs/${run_id}}"
log_root="${LOG_ROOT:-${asset_root}/logs/observation_uq_v3}"
teacher_output="${output_root}/teacher_v31.pt"
teacher_epochs="${TEACHER_EPOCHS:-24}"
mask_block_size="${MASK_BLOCK_SIZE:-4}"
mask_halo="${MASK_HALO:-2}"

wrapped_parts=(
  env
  "PYTHONPATH=${project_root}:${PYTHONPATH:-}"
  "COMPAT_PYTHON_BIN=${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
  "COMPAT_GLIBC_SYSROOT=${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
  "COMPAT_LIBRARY_PATH=${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
  bash -lc
  "set -euo pipefail; mkdir -p '${output_root}'; sha256sum '${shard}' '${project_root}/uq_estimator/observation_uq_v3.py' '${project_root}/scripts/train_observation_uq_teacher_v3.py' > '${output_root}/source_sha256.txt'; '${project_root}/scripts/run_compat_python.sh' '${project_root}/scripts/train_observation_uq_teacher_v3.py' --shard '${shard}' --output '${teacher_output}' --heldout-family local_glare --feature-dim 1024 --hidden-dim 64 --teacher-members 2 --teacher-epochs '${teacher_epochs}' --batch-size 8 --learning-rate 0.002 --mask-block-size '${mask_block_size}' --mask-halo '${mask_halo}' --seed 20260826 --device cuda; sha256sum '${teacher_output}' '${teacher_output%.pt}.report.json' > '${output_root}/artifact_sha256.txt'"
)
printf -v wrapped_command '%q ' "${wrapped_parts[@]}"
sbatch_args=(
  sbatch --parsable
  --partition=Nvidia_A800
  --gres=gpu:1
  --cpus-per-task=8
  --mem=64G
  --time=01:00:00
  --job-name=obs_uq_teacher560r
  "--output=${log_root}/${run_id}-%j.out"
  --export=ALL
  --wrap "${wrapped_command}"
)
if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "REUSES_IMMUTABLE_SHARD=${shard}"
  echo "NO_EXTRACTION=1"
  echo "NO_ADAPTER_TRAINING=1"
  echo "NO_STAGE_B=1"
  echo "RUN_ID=${run_id}"
  echo "TEACHER_EPOCHS=${teacher_epochs}"
  echo "MASK_BLOCK_SIZE=${mask_block_size}"
  echo "MASK_HALO=${mask_halo}"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi
if [[ ! -f "${shard}" ]]; then
  echo "missing source shard: ${shard}" >&2
  exit 1
fi
if [[ -e "${output_root}" ]]; then
  echo "refusing to reuse output root: ${output_root}" >&2
  exit 1
fi
mkdir -p "${log_root}"
"${sbatch_args[@]}"

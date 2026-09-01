#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-native_appearance_candidates_seed20260826_r1}"
output_root="${OUTPUT_ROOT:-${asset_root}/observation_uq_v3/runs/${run_id}}"
calibration_shard="${CALIBRATION_SHARD:-${asset_root}/observation_uq_v3/runs/observation_uq_teacher560_seed20260826_r1/clean_first_features.pt}"
native_features="${NATIVE_FEATURES:-${asset_root}/observation_uq_v3/runs/observation_uq_native_weather_epic_seed20260826_r5_gpu4/native_weather_features.pt}"
config="${project_root}/configs/observation_uq_native_appearance_candidates_v1.json"
report="${output_root}/native_appearance_candidates.json"

if [[ -e "${output_root}" ]]; then
  echo "[FAIL] refusing to reuse output root: ${output_root}" >&2
  exit 1
fi
mkdir -p "${output_root}"
export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "RUN_ID=${run_id}"
echo "SCOPE=clean_only_native_appearance_candidate_audit"
echo "GPU_REQUIRED=1"
echo "CANDIDATE_TRAINING=0"
echo "ADAPTER_TRAINING=0"
echo "STAGE_B=0"
sha256sum \
  "${config}" \
  "${project_root}/uq_estimator/native_appearance_audit.py" \
  "${project_root}/scripts/audit_native_carla_appearance_candidates.py" \
  "${native_features}" > "${output_root}/source_sha256.txt"

"${project_root}/scripts/run_compat_python.sh" \
  "${project_root}/scripts/audit_native_carla_appearance_candidates.py" \
  --clean-calibration-shard "${calibration_shard}" \
  --native-features "${native_features}" \
  --config "${config}" \
  --output "${report}" \
  --batch-size 4 \
  --device cuda

sha256sum "${report}" > "${output_root}/artifact_sha256.txt"
echo "NATIVE_APPEARANCE_CANDIDATE_AUDIT_OK=1"

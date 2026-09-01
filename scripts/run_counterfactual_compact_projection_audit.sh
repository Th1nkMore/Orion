#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-counterfactual_compact_projection_audit_seed20260827_r1}"
output_root="${asset_root}/observation_uq_v3/runs/${run_id}"
feature_root="${asset_root}/observation_uq_v3/runs/counterfactual_evidence_features_windowcycle_seed20260827_r1"
config="${project_root}/configs/observation_uq_counterfactual_compact_projection_audit_v1.json"

if [[ -e "${output_root}" ]]; then echo "refusing to reuse ${output_root}" >&2; exit 1; fi
mkdir -p "${output_root}"
export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "RUN_ID=${run_id}"
echo "SCOPE=NO_TRAINING_COMPACT_PROJECTION_AUDIT"
echo "OUTPUT_FEATURE_SHARD_WRITTEN=0"
echo "SPATIAL_GRID_CHANGED=0"
echo "HELDOUT_GLARE_EVALUATED=0"
echo "NATIVE_WEATHER_EVALUATED=0"
echo "AUTOMATIC_ADAPTER_TRAINING=0"
echo "STAGE_B=0"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
sha256sum "${config}" \
  "${project_root}/uq_estimator/counterfactual_compaction.py" \
  "${project_root}/scripts/audit_counterfactual_compact_projection.py" \
  > "${output_root}/source_sha256.txt"

"${project_root}/scripts/run_compat_python.sh" \
  "${project_root}/scripts/audit_counterfactual_compact_projection.py" \
  --feature-shard "${feature_root}/counterfactual_evidence_features.pt" \
  --config "${config}" \
  --output "${output_root}/counterfactual_compact_projection_audit.json" \
  --device cuda

sha256sum "${output_root}/counterfactual_compact_projection_audit.json" \
  > "${output_root}/artifact_sha256.txt"
echo "COUNTERFACTUAL_COMPACT_PROJECTION_AUDIT_JOB_OK=1"

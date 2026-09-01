#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
feature_run_id="${FEATURE_RUN_ID:-counterfactual_evidence_features_seed20260826_r1}"
target_audit_run_id="${TARGET_AUDIT_RUN_ID:-counterfactual_evidence_target_audit_seed20260826_r2}"
run_id="${RUN_ID:-counterfactual_evidence_spatial_support_seed20260827_r1}"
output_root="${asset_root}/observation_uq_v3/runs/${run_id}"
log_root="${asset_root}/logs/observation_uq_v3"

if [[ -e "${output_root}" ]]; then
  echo "[FAIL] refusing to reuse spatial-support output root: ${output_root}" >&2
  exit 1
fi
mkdir -p "${output_root}"
export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

echo "RUN_ID=${run_id}"
echo "SCOPE=train_target_spatial_support_audit_only"
echo "CORRUPTION_MASK_OPTIMIZER_WEIGHT=0"
echo "ADAPTER_TRAINING=0"
echo "VALIDATION_READ=0"
echo "STAGE_B=0"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
sha256sum \
  "${project_root}/uq_estimator/counterfactual_evidence_training.py" \
  "${project_root}/scripts/audit_counterfactual_evidence_spatial_support.py" \
  > "${output_root}/source_sha256.txt"

"${project_root}/scripts/run_compat_python.sh" \
  "${project_root}/scripts/audit_counterfactual_evidence_spatial_support.py" \
  --feature-shard "${asset_root}/observation_uq_v3/runs/${feature_run_id}/counterfactual_evidence_features.pt" \
  --target-audit "${asset_root}/observation_uq_v3/runs/${target_audit_run_id}/counterfactual_evidence_target_audit.json" \
  --protocol "${project_root}/configs/observation_uq_counterfactual_evidence_v1.json" \
  --output "${output_root}/counterfactual_evidence_spatial_support.json" \
  --device cuda --batch-size 2 --mask-label-floor 0.25

sha256sum "${output_root}/counterfactual_evidence_spatial_support.json" \
  > "${output_root}/artifact_sha256.txt"
echo "COUNTERFACTUAL_EVIDENCE_SPATIAL_SUPPORT_JOB_OK=1"

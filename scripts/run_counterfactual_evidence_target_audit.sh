#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
feature_run_id="${FEATURE_RUN_ID:-counterfactual_evidence_features_seed20260826_r1}"
audit_run_id="${AUDIT_RUN_ID:-counterfactual_evidence_target_audit_seed20260826_r1}"
feature_root="${asset_root}/observation_uq_v3/runs/${feature_run_id}"
output_root="${asset_root}/observation_uq_v3/runs/${audit_run_id}"

if [[ -e "${output_root}" ]]; then
  echo "[FAIL] refusing to reuse target-audit output root: ${output_root}" >&2
  exit 1
fi
mkdir -p "${output_root}"
export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

echo "AUDIT_RUN_ID=${audit_run_id}"
echo "SCOPE=counterfactual_train_target_distribution_only"
echo "ADAPTER_TRAINING=0"
echo "VALIDATION_READ=0"
echo "ACTUAL_TARGET_TRAINING=0"
echo "STAGE_B=0"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
sha256sum \
  "${project_root}/configs/observation_uq_counterfactual_evidence_v1.json" \
  "${project_root}/uq_estimator/counterfactual_evidence.py" \
  "${project_root}/uq_estimator/counterfactual_evidence_training.py" \
  "${project_root}/scripts/audit_counterfactual_evidence_targets.py" \
  > "${output_root}/source_sha256.txt"

"${project_root}/scripts/run_compat_python.sh" \
  "${project_root}/scripts/audit_counterfactual_evidence_targets.py" \
  --feature-shard "${feature_root}/counterfactual_evidence_features.pt" \
  --artifact-sha256 "${feature_root}/artifact_sha256.txt" \
  --protocol "${project_root}/configs/observation_uq_counterfactual_evidence_v1.json" \
  --output "${output_root}/counterfactual_evidence_target_audit.json" \
  --device cuda \
  --batch-size 2 \
  --quantile 0.95 \
  --response-floor 1e-6

sha256sum "${output_root}/counterfactual_evidence_target_audit.json" \
  > "${output_root}/artifact_sha256.txt"
echo "COUNTERFACTUAL_EVIDENCE_TARGET_AUDIT_JOB_OK=1"

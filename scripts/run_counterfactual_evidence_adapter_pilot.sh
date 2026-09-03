#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-counterfactual_evidence_adapter_seed20260827_r1}"
output_root="${asset_root}/observation_uq_v3/runs/${run_id}"
feature_root="${asset_root}/observation_uq_v3/runs/counterfactual_evidence_features_seed20260826_r1"
native_root="${asset_root}/observation_uq_v3/runs/observation_uq_native_weather_epic_seed20260826_r5_gpu4"
target_audit_root="${asset_root}/observation_uq_v3/runs/counterfactual_evidence_target_audit_seed20260826_r2"
spatial_audit_root="${asset_root}/observation_uq_v3/runs/counterfactual_evidence_spatial_support_seed20260827_r1"
training_config="${project_root}/configs/observation_uq_counterfactual_evidence_training_run_v1.json"

if [[ -e "${output_root}" ]]; then
  echo "[FAIL] refusing to reuse adapter-pilot output root: ${output_root}" >&2
  exit 1
fi
mkdir -p "${output_root}"
export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "RUN_ID=${run_id}"
echo "SCOPE=bounded_counterfactual_evidence_adapter_stage1_only"
echo "EPOCHS=24"
echo "CORRUPTION_MASK_OPTIMIZER_WEIGHT=0"
echo "ACTUAL_TARGET_OPTIMIZER_WEIGHT=0"
echo "ORION_FINETUNING=0"
echo "STAGE_B=0"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
sha256sum \
  "${training_config}" \
  "${project_root}/uq_estimator/counterfactual_evidence.py" \
  "${project_root}/uq_estimator/counterfactual_evidence_training.py" \
  "${project_root}/scripts/train_counterfactual_evidence_adapter.py" \
  > "${output_root}/source_sha256.txt"

"${project_root}/scripts/run_compat_python.sh" \
  "${project_root}/scripts/train_counterfactual_evidence_adapter.py" \
  --feature-shard "${feature_root}/counterfactual_evidence_features.pt" \
  --native-features "${native_root}/native_weather_features.pt" \
  --target-audit "${target_audit_root}/counterfactual_evidence_target_audit.json" \
  --spatial-audit "${spatial_audit_root}/counterfactual_evidence_spatial_support.json" \
  --training-config "${training_config}" \
  --output "${output_root}/counterfactual_evidence_adapter.pt" \
  --device cuda

sha256sum \
  "${output_root}/counterfactual_evidence_adapter.pt" \
  "${output_root}/counterfactual_evidence_adapter.report.json" \
  > "${output_root}/artifact_sha256.txt"
echo "COUNTERFACTUAL_EVIDENCE_ADAPTER_PILOT_JOB_OK=1"

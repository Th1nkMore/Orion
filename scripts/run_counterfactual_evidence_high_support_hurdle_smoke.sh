#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-counterfactual_evidence_high_support_hurdle_seed20260827_r1}"
output_root="${asset_root}/observation_uq_v3/runs/${run_id}"
feature_root="${asset_root}/observation_uq_v3/runs/counterfactual_evidence_features_windowcycle_seed20260827_r1"
target_root="${asset_root}/observation_uq_v3/runs/counterfactual_evidence_target_audit_windowcycle_seed20260827_r1"
spatial_root="${asset_root}/observation_uq_v3/runs/counterfactual_evidence_spatial_support_windowcycle_seed20260827_r1"
config="${CONFIG_PATH:-${project_root}/configs/observation_uq_counterfactual_high_support_hurdle_smoke_v1.json}"

if [[ -e "${output_root}" ]]; then echo "refusing to reuse ${output_root}" >&2; exit 1; fi
mkdir -p "${output_root}"
export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "RUN_ID=${run_id}"
echo "SCOPE=route_only_high_support_hurdle_smoke"
echo "SUPPORT=TRAIN_RESPONSIVE_Q80"
echo "EXACT_NONZERO_PRESENCE_LABEL=0"
echo "HELDOUT_GLARE_READ=0"
echo "NATIVE_FOG_READ=0"
echo "AUTOMATIC_FULL_TRAINING=0"
echo "STAGE_B=0"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
sha256sum "${config}" "${project_root}/uq_estimator/counterfactual_evidence.py" \
  "${project_root}/uq_estimator/counterfactual_evidence_training.py" \
  "${project_root}/scripts/train_counterfactual_evidence_hurdle_smoke.py" \
  > "${output_root}/source_sha256.txt"

"${project_root}/scripts/run_compat_python.sh" \
  "${project_root}/scripts/train_counterfactual_evidence_hurdle_smoke.py" \
  --feature-shard "${feature_root}/counterfactual_evidence_features.pt" \
  --target-audit "${target_root}/counterfactual_evidence_target_audit.json" \
  --spatial-audit "${spatial_root}/counterfactual_evidence_spatial_support.json" \
  --config "${config}" \
  --output "${output_root}/counterfactual_evidence_high_support_hurdle_smoke.pt" \
  --device cuda

sha256sum "${output_root}/counterfactual_evidence_high_support_hurdle_smoke.pt" \
  "${output_root}/counterfactual_evidence_high_support_hurdle_smoke.report.json" \
  > "${output_root}/artifact_sha256.txt"
echo "COUNTERFACTUAL_EVIDENCE_HIGH_SUPPORT_HURDLE_SMOKE_JOB_OK=1"

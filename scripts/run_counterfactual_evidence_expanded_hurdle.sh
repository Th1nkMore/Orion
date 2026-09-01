#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
HOME_WORK_ROOT="${HOME_WORK_ROOT:-/public/home/lidachuan/orion_work}"
DATASET_RUN_ID="${DATASET_RUN_ID:-counterfactual_evidence_fp16_routes_expansion100_seed20260827_r1}"
AUDIT_RUN_ID="${AUDIT_RUN_ID:-counterfactual_evidence_fp16_route_audit_expansion100_seed20260827_r1}"
RUN_ID="${RUN_ID:-counterfactual_evidence_expanded_hurdle_hidden128_seed20260827_r1}"
DATASET_ROOT="${HOME_WORK_ROOT}/observation_uq_v3/runs/${DATASET_RUN_ID}"
AUDIT_ROOT="${HOME_WORK_ROOT}/observation_uq_v3/runs/${AUDIT_RUN_ID}"
OUTPUT_ROOT="${HOME_WORK_ROOT}/observation_uq_v3/runs/${RUN_ID}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/observation_uq_counterfactual_expanded_hurdle_training_v1.json}"

test -s "${DATASET_ROOT}/manifest.json"
test -s "${AUDIT_ROOT}/counterfactual_evidence_target_audit.json"
test -s "${AUDIT_ROOT}/counterfactual_evidence_spatial_support.json"
test -s "${CONFIG_PATH}"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-/public/share/lidachuan/orion_assets/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-/public/share/lidachuan/orion_assets/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-/public/share/lidachuan/orion_assets/envs/orion-cl/lib}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "RUN_ID=${RUN_ID}"
echo "SCOPE=expanded_70_route_hidden128_hurdle_only"
echo "TRAIN_ROUTES=70"
echo "VALIDATION_ROUTES=10"
echo "OPTIMIZER_FAMILIES=local_blur,local_dark"
echo "VALIDATION_GLARE_TENSOR_VALUES_ACCESSED=0"
echo "HELDOUT_SPLIT_TENSOR_VALUES_ACCESSED=0"
echo "NATIVE_WEATHER_READ=0"
echo "ORION_FINETUNING=0"
echo "STAGE_B=0"
echo "AUTOMATIC_FOLLOWUP=0"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

sha256sum \
  "${CONFIG_PATH}" \
  "${PROJECT_ROOT}/uq_estimator/counterfactual_evidence.py" \
  "${PROJECT_ROOT}/uq_estimator/counterfactual_evidence_training.py" \
  "${PROJECT_ROOT}/uq_estimator/counterfactual_sharded_dataset.py" \
  "${PROJECT_ROOT}/scripts/train_counterfactual_evidence_expanded_hurdle.py" \
  > "${OUTPUT_ROOT}/source_sha256.txt"

"${PROJECT_ROOT}/scripts/run_compat_python.sh" \
  "${PROJECT_ROOT}/scripts/train_counterfactual_evidence_expanded_hurdle.py" \
  --dataset-manifest "${DATASET_ROOT}/manifest.json" \
  --target-audit "${AUDIT_ROOT}/counterfactual_evidence_target_audit.json" \
  --spatial-audit "${AUDIT_ROOT}/counterfactual_evidence_spatial_support.json" \
  --config "${CONFIG_PATH}" \
  --output "${OUTPUT_ROOT}/counterfactual_evidence_expanded_hurdle.pt" \
  --device cuda

sha256sum \
  "${OUTPUT_ROOT}/counterfactual_evidence_expanded_hurdle.pt" \
  "${OUTPUT_ROOT}/counterfactual_evidence_expanded_hurdle.report.json" \
  > "${OUTPUT_ROOT}/artifact_sha256.txt"
echo "COUNTERFACTUAL_EVIDENCE_EXPANDED_HURDLE_JOB_OK=1"

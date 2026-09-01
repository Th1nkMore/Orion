#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
HOME_WORK_ROOT="${HOME_WORK_ROOT:-/public/home/lidachuan/orion_work}"
DATASET_RUN_ID="${DATASET_RUN_ID:-counterfactual_evidence_fp16_routes_expansion100_seed20260827_r1}"
AUDIT_RUN_ID="${AUDIT_RUN_ID:-counterfactual_evidence_fp16_route_audit_expansion100_seed20260827_r1}"
DATASET_MANIFEST="${HOME_WORK_ROOT}/observation_uq_v3/runs/${DATASET_RUN_ID}/manifest.json"
PROTOCOL="${PROJECT_ROOT}/configs/observation_uq_counterfactual_evidence_expanded_v3.json"
OUTPUT_DIR="${HOME_WORK_ROOT}/observation_uq_v3/runs/${AUDIT_RUN_ID}"

export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-/public/share/lidachuan/orion_assets/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-/public/share/lidachuan/orion_assets/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-/public/share/lidachuan/orion_assets/envs/orion-cl/lib}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

test -s "${DATASET_MANIFEST}"
test -s "${PROTOCOL}"
test ! -e "${OUTPUT_DIR}"

echo "SCOPE=expanded_train_target_and_spatial_audit_only"
echo "ADAPTER_TRAINING=0"
echo "HELDOUT_FEATURES_READ=0"
echo "NATIVE_WEATHER_READ=0"
echo "ORION_FINETUNING=0"
echo "STAGE_B=0"

"${PROJECT_ROOT}/scripts/run_compat_python.sh" \
  "${PROJECT_ROOT}/scripts/audit_counterfactual_evidence_fp16_route_dataset.py" \
  --dataset-manifest "${DATASET_MANIFEST}" \
  --protocol "${PROTOCOL}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cpu \
  --batch-size 2 \
  --quantile 0.95 \
  --response-floor 1e-6 \
  --mask-label-floor 0.25

sha256sum "${OUTPUT_DIR}"/*.json > "${OUTPUT_DIR}/artifact_sha256.txt"
echo "COUNTERFACTUAL_EVIDENCE_FP16_ROUTE_AUDIT_RUN_OK=1"

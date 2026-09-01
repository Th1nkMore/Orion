#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
ASSET_ROOT="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
HOME_WORK_ROOT="${HOME_WORK_ROOT:-/public/home/lidachuan/orion_work}"
RUN_ID="${RUN_ID:-counterfactual_evidence_fp16_routes_expansion100_seed20260827_r1}"
OUTPUT_DIR="${HOME_WORK_ROOT}/observation_uq_v3/runs/${RUN_ID}"
INFOS="${ASSET_ROOT}/data/infos_expansion100_seed20260827/b2d_infos_val.pkl"
ROUTE_MANIFEST="${PROJECT_ROOT}/configs/spatial_uq_route_manifests/b2d_expansion100_seed20260827.json"
PROTOCOL="${PROJECT_ROOT}/configs/observation_uq_counterfactual_evidence_expanded_v3.json"
CHECKPOINT="${ASSET_ROOT}/checkpoints/Orion.pth"

export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${ASSET_ROOT}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${ASSET_ROOT}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${ASSET_ROOT}/envs/orion-cl/lib}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

for required in "${INFOS}" "${ROUTE_MANIFEST}" "${PROTOCOL}" "${CHECKPOINT}"; do
  test -s "${required}"
done

resume_flag=""
if [[ "${RESUME:-0}" == "1" ]]; then
  resume_flag="--resume"
fi

echo "RUN_ID=${RUN_ID}"
echo "SCOPE=direct_fp16_route_feature_extraction_only"
echo "OUTPUT_STORAGE=personal_home"
echo "ROUTES=train70_validation10_heldout10_calibration_reserved10"
echo "MONOLITHIC_INTERMEDIATE=0"
echo "CORRUPTION_MASK_OPTIMIZER_WEIGHT=0"
echo "EXACT_NONZERO_PRESENCE_LABEL=0"
echo "ADAPTER_TRAINING=0"
echo "ORION_FINETUNING=0"
echo "STAGE_B=0"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"${PROJECT_ROOT}/scripts/run_compat_python.sh" \
  "${PROJECT_ROOT}/scripts/extract_counterfactual_evidence_fp16_route_shards.py" \
  --config "${PROJECT_ROOT}/adzoo/orion/configs/orion_stage3_agent.py" \
  --checkpoint "${CHECKPOINT}" \
  --ann-file "${INFOS}" \
  --route-manifest "${ROUTE_MANIFEST}" \
  --protocol "${PROTOCOL}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 2 \
  --workers 4 \
  --seed 20260827 \
  --train-routes 70 \
  --validation-routes 10 \
  --heldout-routes 10 \
  --frames-per-route 16 \
  --target-batch-size 2 \
  --max-output-gb 150 \
  ${resume_flag:+"${resume_flag}"}

sha256sum "${OUTPUT_DIR}/extraction_contract.json" "${OUTPUT_DIR}/manifest.json" \
  > "${OUTPUT_DIR}/dataset_control_artifacts.sha256"
echo "COUNTERFACTUAL_EVIDENCE_FP16_ROUTE_EXPANSION100_OK=1"

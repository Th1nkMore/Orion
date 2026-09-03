#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-counterfactual_evidence_features_windowcycle_seed20260827_r1}"
output_root="${OUTPUT_ROOT:-${asset_root}/observation_uq_v3/runs/${run_id}}"
checkpoint="${ORION_CHECKPOINT:-${asset_root}/checkpoints/Orion.pth}"
infos="${B2D_INFOS:-${asset_root}/data/infos/b2d_infos_val.pkl}"
manifest="${ROUTE_MANIFEST:-${project_root}/configs/spatial_uq_route_manifests/b2d_val_exploratory_seed20260826.json}"
protocol="${project_root}/configs/observation_uq_counterfactual_evidence_v2.json"
reference_shard="${REFERENCE_SHARD:-${asset_root}/observation_uq_v3/runs/observation_uq_teacher560_seed20260826_r1/clean_first_features.pt}"
feature_shard="${output_root}/counterfactual_evidence_features.pt"

if [[ -e "${output_root}" ]]; then echo "[FAIL] refusing to reuse output root: ${output_root}" >&2; exit 1; fi
mkdir -p "${output_root}"
export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "RUN_ID=${run_id}"
echo "SCOPE=counterfactual_evidence_v2_feature_extraction_only"
echo "VIEW_SCHEDULE=route_condition_window_cycle/v2"
echo "WINDOW_FRAMES=4"
echo "EXACT_NONZERO_PRESENCE_LABEL=0"
echo "ADAPTER_TRAINING=0"
echo "ACTUAL_TARGET_TRAINING=0"
echo "STAGE_B=0"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
sha256sum \
  "${protocol}" \
  "${manifest}" \
  "${project_root}/uq_estimator/counterfactual_evidence.py" \
  "${project_root}/uq_estimator/counterfactual_evidence_extraction.py" \
  "${project_root}/scripts/extract_counterfactual_evidence_feature_shard.py" \
  "${reference_shard}" > "${output_root}/source_sha256.txt"

"${project_root}/scripts/run_compat_python.sh" \
  "${project_root}/scripts/extract_counterfactual_evidence_feature_shard.py" \
  --config "${project_root}/adzoo/orion/configs/orion_stage3_agent.py" \
  --checkpoint "${checkpoint}" \
  --ann-file "${infos}" \
  --route-manifest "${manifest}" \
  --protocol "${protocol}" \
  --reference-feature-shard "${reference_shard}" \
  --output "${feature_shard}" \
  --batch-size 2 \
  --workers 4 \
  --seed 20260827 \
  --max-output-gb 75

sha256sum "${feature_shard}" > "${output_root}/artifact_sha256.txt"
echo "COUNTERFACTUAL_EVIDENCE_V2_EXTRACTION_OK=1"

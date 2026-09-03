#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_runner="${PYTHON_RUNNER:-${project_root}/scripts/run_compat_python.sh}"
run_id="${RUN_ID:-observation_uq_teacher_scaled}"
output_root="${OUTPUT_ROOT:-${asset_root}/observation_uq_v3/runs/${run_id}}"
checkpoint="${ORION_CHECKPOINT:-${asset_root}/checkpoints/Orion.pth}"
infos="${B2D_INFOS:-${asset_root}/data/infos/b2d_infos_val.pkl}"
manifest="${ROUTE_MANIFEST:-${project_root}/configs/spatial_uq_route_manifests/b2d_val_exploratory_seed20260826.json}"
shard="${output_root}/clean_first_features.pt"
teacher_output="${output_root}/teacher_v3.pt"

required_files=(
  "${python_runner}"
  "${project_root}/scripts/extract_observation_uq_feature_shard.py"
  "${project_root}/scripts/train_observation_uq_teacher_v3.py"
  "${project_root}/uq_estimator/observation_uq_v3.py"
  "${project_root}/uq_estimator/observation_uq_shard.py"
  "${checkpoint}"
  "${infos}"
  "${manifest}"
)
for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[FAIL] missing required file: ${path}" >&2
    exit 1
  fi
done
if [[ -e "${output_root}" ]]; then
  echo "[FAIL] refusing to reuse output root: ${output_root}" >&2
  exit 1
fi
mkdir -p "${output_root}"

echo "RUN_ID=${run_id}"
echo "OUTPUT_ROOT=${output_root}"
echo "HOST=$(hostname)"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
sha256sum \
  "${project_root}/uq_estimator/observation_uq_v3.py" \
  "${project_root}/uq_estimator/observation_uq_shard.py" \
  "${project_root}/scripts/extract_observation_uq_feature_shard.py" \
  "${project_root}/scripts/train_observation_uq_teacher_v3.py" \
  "${manifest}" > "${output_root}/source_sha256.txt"

"${python_runner}" "${project_root}/scripts/extract_observation_uq_feature_shard.py" \
  --config "${project_root}/adzoo/orion/configs/orion_stage3_agent.py" \
  --checkpoint "${checkpoint}" \
  --ann-file "${infos}" \
  --route-manifest "${manifest}" \
  --output "${shard}" \
  --split-route-quota train=35 \
  --split-route-quota validation=5 \
  --split-route-quota held_out=5 \
  --samples-per-route 16 \
  --diagnostic-split validation \
  --diagnostic-split held_out \
  --diagnostic-corruption local_glare \
  --severities 1 3 \
  --view-indices 0 \
  --batch-size 2 \
  --workers 4 \
  --seed 20260826 \
  --max-output-gb 25

"${python_runner}" "${project_root}/scripts/train_observation_uq_teacher_v3.py" \
  --shard "${shard}" \
  --output "${teacher_output}" \
  --heldout-family local_glare \
  --feature-dim 1024 \
  --hidden-dim 64 \
  --teacher-members 2 \
  --teacher-epochs 12 \
  --batch-size 8 \
  --learning-rate 0.002 \
  --seed 20260826 \
  --device cuda

sha256sum "${shard}" "${teacher_output}" \
  "${teacher_output%.pt}.report.json" > "${output_root}/artifact_sha256.txt"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "OBSERVATION_UQ_TEACHER_SCALED_OK=1"

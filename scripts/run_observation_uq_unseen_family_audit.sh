#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_runner="${PYTHON_RUNNER:-${project_root}/scripts/run_compat_python.sh}"
run_id="${RUN_ID:-observation_uq_unseen_local_blur_seed20260827_r1}"
output_root="${OUTPUT_ROOT:-${asset_root}/observation_uq_v3/runs/${run_id}}"
checkpoint="${ORION_CHECKPOINT:-${asset_root}/checkpoints/Orion.pth}"
teacher="${TEACHER_CHECKPOINT:-${asset_root}/observation_uq_v3/runs/observation_uq_teacher560_v31_seed20260826_r1/teacher_v31.pt}"
infos="${B2D_INFOS:-${asset_root}/data/infos/b2d_infos_val.pkl}"
manifest="${ROUTE_MANIFEST:-${project_root}/configs/spatial_uq_route_manifests/b2d_val_exploratory_seed20260826.json}"
family="${DIAGNOSTIC_FAMILY:-local_blur}"
seed="${DIAGNOSTIC_SEED:-20260827}"
shard="${output_root}/clean_first_${family}_features.pt"
audit="${output_root}/signal_audit_v12_${family}.json"

case "${family}" in
  local_blur|local_dark|local_occlusion) ;;
  *) echo "[FAIL] unseen audit family is not approved: ${family}" >&2; exit 2 ;;
esac

required_files=(
  "${python_runner}"
  "${project_root}/scripts/extract_observation_uq_feature_shard.py"
  "${project_root}/scripts/audit_observation_uq_signals_v1.py"
  "${project_root}/uq_estimator/observation_uq_signal_audit.py"
  "${checkpoint}"
  "${teacher}"
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
echo "DIAGNOSTIC_FAMILY=${family}"
echo "HOST=$(hostname)"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
sha256sum \
  "${project_root}/uq_estimator/observation_uq_v3.py" \
  "${project_root}/uq_estimator/observation_uq_signal_audit.py" \
  "${project_root}/scripts/extract_observation_uq_feature_shard.py" \
  "${project_root}/scripts/audit_observation_uq_signals_v1.py" \
  "${teacher}" \
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
  --diagnostic-corruption "${family}" \
  --severities 1 3 \
  --view-indices 0 \
  --batch-size 2 \
  --workers 6 \
  --seed "${seed}" \
  --max-output-gb 25

"${python_runner}" "${project_root}/scripts/audit_observation_uq_signals_v1.py" \
  --shard "${shard}" \
  --teacher "${teacher}" \
  --output "${audit}" \
  --batch-size 8 \
  --device cuda

sha256sum "${shard}" "${audit}" > "${output_root}/artifact_sha256.txt"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "OBSERVATION_UQ_UNSEEN_FAMILY_AUDIT_OK=1"


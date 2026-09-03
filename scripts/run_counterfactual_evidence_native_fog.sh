#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
work_root="${WORK_ROOT:-/public/home/lidachuan/orion_work/observation_uq_v3}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-counterfactual_evidence_no_view_native_fog_seed20260827_r1}"
output_root="${OUTPUT_ROOT:-${work_root}/runs/${run_id}}"
checkpoint="${CHECKPOINT:-${work_root}/runs/counterfactual_evidence_no_view_repair_hidden128_seed20260827_r1/counterfactual_evidence_no_view_repair.pt}"
native_features="${NATIVE_FEATURES:-${asset_root}/observation_uq_v3/runs/observation_uq_native_weather_epic_seed20260826_r5_gpu4/native_weather_features.pt}"
glare_report="${UPSTREAM_GLARE_REPORT:-${work_root}/runs/counterfactual_evidence_no_view_glare_heldout_seed20260827_r1/counterfactual_evidence_glare_heldout.report.json}"
config="${CONFIG:-${project_root}/configs/observation_uq_counterfactual_native_fog_eval_v1.json}"
output="${output_root}/counterfactual_evidence_native_fog.report.json"
python_runner="${PYTHON_RUNNER:-${project_root}/scripts/run_compat_python.sh}"

for path in "${checkpoint}" "${native_features}" "${glare_report}" "${config}" "${python_runner}"; do
  [[ -f "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 1; }
done
[[ ! -e "${output_root}" ]] || { echo "refusing to reuse output root: ${output_root}" >&2; exit 1; }
mkdir -p "${output_root}"

echo "RUN_ID=${run_id}"
echo "HOST=$(hostname)"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "CARLA_RERENDER=0"
echo "EXISTING_EXACT_POSE_EPIC_FEATURE_REUSE=1"
echo "TRAINING=0"
echo "CHECKPOINT_UPDATE=0"
echo "STAGE_B=0"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
sha256sum \
  "${project_root}/scripts/eval_counterfactual_evidence_native_fog.py" \
  "${project_root}/uq_estimator/native_appearance_audit.py" \
  "${config}" "${checkpoint}" "${native_features}" "${glare_report}" \
  > "${output_root}/source_sha256.txt"

"${python_runner}" "${project_root}/scripts/eval_counterfactual_evidence_native_fog.py" \
  --checkpoint "${checkpoint}" \
  --native-features "${native_features}" \
  --upstream-glare-report "${glare_report}" \
  --config "${config}" \
  --output "${output}" \
  --batch-size 2 \
  --device cuda

sha256sum "${output}" > "${output_root}/artifact_sha256.txt"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "COUNTERFACTUAL_EVIDENCE_NATIVE_FOG_OK=1"

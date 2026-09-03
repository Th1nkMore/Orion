#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
work_root="${WORK_ROOT:-/public/home/lidachuan/orion_work/observation_uq_v3}"
run_id="${RUN_ID:-counterfactual_evidence_no_view_glare_heldout_seed20260827_r1}"
dataset_manifest="${DATASET_MANIFEST:-${work_root}/runs/counterfactual_evidence_fp16_routes_expansion100_seed20260827_r1/manifest.json}"
checkpoint="${CHECKPOINT:-${work_root}/runs/counterfactual_evidence_no_view_repair_hidden128_seed20260827_r1/counterfactual_evidence_no_view_repair.pt}"
training_report="${TRAINING_REPORT:-${work_root}/runs/counterfactual_evidence_no_view_repair_hidden128_seed20260827_r1/counterfactual_evidence_no_view_repair.report.json}"
amendment="${AMENDMENT:-${project_root}/configs/observation_uq_counterfactual_reference_semantics_amendment_v1.json}"
output_root="${OUTPUT_ROOT:-${work_root}/runs/${run_id}}"
output="${output_root}/counterfactual_evidence_glare_heldout.report.json"
python_runner="${PYTHON_RUNNER:-${project_root}/scripts/run_compat_python.sh}"

for path in "${dataset_manifest}" "${checkpoint}" "${training_report}" "${amendment}" "${python_runner}"; do
  [[ -f "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 1; }
done
[[ ! -e "${output_root}" ]] || { echo "refusing to reuse output root: ${output_root}" >&2; exit 1; }
mkdir -p "${output_root}"

echo "RUN_ID=${run_id}"
echo "HOST=$(hostname)"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
sha256sum \
  "${project_root}/uq_estimator/counterfactual_evidence_heldout.py" \
  "${project_root}/scripts/eval_counterfactual_evidence_heldout.py" \
  "${amendment}" "${checkpoint}" "${training_report}" "${dataset_manifest}" \
  > "${output_root}/source_sha256.txt"

"${python_runner}" "${project_root}/scripts/eval_counterfactual_evidence_heldout.py" \
  --dataset-manifest "${dataset_manifest}" \
  --checkpoint "${checkpoint}" \
  --training-report "${training_report}" \
  --amendment "${amendment}" \
  --output "${output}" \
  --batch-size 2 \
  --device cuda

sha256sum "${output}" > "${output_root}/artifact_sha256.txt"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "COUNTERFACTUAL_EVIDENCE_HELDOUT_FAMILY_OK=1"

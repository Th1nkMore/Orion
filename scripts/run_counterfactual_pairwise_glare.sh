#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
work_root="${WORK_ROOT:-/public/home/lidachuan/orion_work/observation_uq_v3}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
training_run_id="${TRAINING_RUN_ID:-counterfactual_pairwise_native_repair_seed20260828_r1}"
run_id="${RUN_ID:-counterfactual_pairwise_glare_seed20260828_r1}"
dataset_manifest="${work_root}/runs/counterfactual_evidence_fp16_routes_expansion100_seed20260827_r1/manifest.json"
checkpoint="${work_root}/runs/${training_run_id}/counterfactual_evidence_pairwise_native_repair.pt"
training_report="${work_root}/runs/${training_run_id}/counterfactual_evidence_pairwise_native_repair.report.json"
protocol="${project_root}/configs/observation_uq_counterfactual_pairwise_native_v4.json"
output_root="${work_root}/runs/${run_id}"
output="${output_root}/counterfactual_evidence_pairwise_glare.report.json"
python_runner="${PYTHON_RUNNER:-${project_root}/scripts/run_compat_python.sh}"

for path in "${dataset_manifest}" "${checkpoint}" "${training_report}" "${protocol}" "${python_runner}"; do
  [[ -f "${path}" ]] || { echo "missing required pairwise glare input: ${path}" >&2; exit 1; }
done
[[ ! -e "${output_root}" ]] || { echo "refusing to reuse pairwise glare root: ${output_root}" >&2; exit 1; }
mkdir -p "${output_root}"

export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

echo "RUN_ID=${run_id}"
echo "HOST=$(hostname)"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SCOPE=frozen_pairwise_unseen_glare_gate"
echo "TRAINING=0"
echo "CHECKPOINT_UPDATE=0"
echo "NATIVE_FINAL_HELDOUT_READ=0"
echo "STAGE_B=0"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
sha256sum \
  "${project_root}/scripts/eval_counterfactual_pairwise_glare.py" \
  "${protocol}" "${checkpoint}" "${training_report}" "${dataset_manifest}" \
  > "${output_root}/source_sha256.txt"

"${python_runner}" "${project_root}/scripts/eval_counterfactual_pairwise_glare.py" \
  --dataset-manifest "${dataset_manifest}" \
  --checkpoint "${checkpoint}" \
  --training-report "${training_report}" \
  --protocol "${protocol}" \
  --output "${output}" \
  --batch-size 2 \
  --device cuda

sha256sum "${output}" > "${output_root}/artifact_sha256.txt"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "COUNTERFACTUAL_PAIRWISE_GLARE_JOB_OK=1"

#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
work_root="${WORK_ROOT:-/public/home/lidachuan/orion_work/observation_uq_v3}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
training_run_id="${TRAINING_RUN_ID:-counterfactual_pairwise_native_repair_seed20260828_r1}"
glare_run_id="${GLARE_RUN_ID:-counterfactual_pairwise_glare_seed20260828_r1}"
run_id="${RUN_ID:-counterfactual_pairwise_native_fog_seed20260828_r1}"
score_calibration="${SCORE_CALIBRATION:-absolute}"
checkpoint="${work_root}/runs/${training_run_id}/counterfactual_evidence_pairwise_native_repair.pt"
native_features="${asset_root}/observation_uq_v3/runs/observation_uq_native_weather_epic_seed20260826_r5_gpu4/native_weather_features.pt"
glare_report="${work_root}/runs/${glare_run_id}/counterfactual_evidence_pairwise_glare.report.json"
config="${CONFIG_PATH:-${project_root}/configs/observation_uq_counterfactual_pairwise_native_fog_eval_v4.json}"
output_root="${work_root}/runs/${run_id}"
output="${output_root}/counterfactual_evidence_pairwise_native_fog.report.json"
python_runner="${PYTHON_RUNNER:-${project_root}/scripts/run_compat_python.sh}"

for path in "${checkpoint}" "${native_features}" "${glare_report}" "${config}" "${python_runner}"; do
  [[ -f "${path}" ]] || { echo "missing required pairwise native-fog input: ${path}" >&2; exit 1; }
done
[[ ! -e "${output_root}" ]] || { echo "refusing to reuse pairwise native-fog root: ${output_root}" >&2; exit 1; }
mkdir -p "${output_root}"

export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

echo "RUN_ID=${run_id}"
echo "HOST=$(hostname)"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SCOPE=frozen_pairwise_final_native_fog_gate"
echo "TRAINING=0"
echo "CHECKPOINT_UPDATE=0"
echo "HELDOUT_ROUTES=Town01/Route146,Town04/Route203"
echo "SCORE_CALIBRATION=${score_calibration}"
echo "STAGE_B=0"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
sha256sum \
  "${project_root}/scripts/eval_counterfactual_pairwise_native_fog.py" \
  "${config}" "${checkpoint}" "${native_features}" "${glare_report}" \
  > "${output_root}/source_sha256.txt"

"${python_runner}" "${project_root}/scripts/eval_counterfactual_pairwise_native_fog.py" \
  --checkpoint "${checkpoint}" \
  --native-features "${native_features}" \
  --upstream-glare-report "${glare_report}" \
  --config "${config}" \
  --output "${output}" \
  --batch-size 2 \
  --score-calibration "${score_calibration}" \
  --device cuda

sha256sum "${output}" > "${output_root}/artifact_sha256.txt"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "COUNTERFACTUAL_PAIRWISE_NATIVE_FOG_JOB_OK=1"

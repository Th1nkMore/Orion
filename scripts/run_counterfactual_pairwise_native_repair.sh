#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
home_work_root="${HOME_WORK_ROOT:-/public/home/lidachuan/orion_work}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
dataset_run_id="${DATASET_RUN_ID:-counterfactual_evidence_fp16_routes_expansion100_seed20260827_r1}"
native_run_id="${NATIVE_RUN_ID:-counterfactual_pairwise_native_train_seed20260828_r2}"
run_id="${RUN_ID:-counterfactual_pairwise_native_repair_seed20260828_r1}"
dataset_manifest="${home_work_root}/observation_uq_v3/runs/${dataset_run_id}/manifest.json"
native_features="${asset_root}/observation_uq_v3/runs/${native_run_id}/native_weather_train_features.pt"
initial_checkpoint="${home_work_root}/observation_uq_v3/runs/counterfactual_evidence_no_view_repair_hidden128_seed20260827_r1/counterfactual_evidence_no_view_repair.pt"
protocol="${project_root}/configs/observation_uq_counterfactual_pairwise_native_v4.json"
config="${CONFIG_PATH:-${project_root}/configs/observation_uq_counterfactual_pairwise_native_training_v4.json}"
output_root="${home_work_root}/observation_uq_v3/runs/${run_id}"
output="${output_root}/counterfactual_evidence_pairwise_native_repair.pt"
python_runner="${PYTHON_RUNNER:-${project_root}/scripts/run_compat_python.sh}"

for path in \
  "${dataset_manifest}" \
  "${native_features}" \
  "${initial_checkpoint}" \
  "${protocol}" \
  "${config}" \
  "${python_runner}"; do
  [[ -f "${path}" ]] || { echo "missing required pairwise repair input: ${path}" >&2; exit 1; }
done
[[ ! -e "${output_root}" ]] || { echo "refusing to reuse pairwise repair root: ${output_root}" >&2; exit 1; }
mkdir -p "${output_root}"

export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

echo "RUN_ID=${run_id}"
echo "HOST=$(hostname)"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SCOPE=bounded_pairwise_synthetic_native_stage1_repair"
echo "SYNTHETIC_TRAIN_ROUTES=70"
echo "SYNTHETIC_VALIDATION_ROUTES=10"
echo "NATIVE_OPTIMIZER_ROUTE=Town10HD/Route148"
echo "NATIVE_DEVELOPMENT_ROUTE=Town03/Route195"
echo "FINAL_NATIVE_HELDOUT_ROUTES_READ=0"
echo "BLANKET_REFERENCE_ZERO_LOSS=0"
echo "ORION_FINETUNING=0"
echo "STAGE_B=0"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
sha256sum \
  "${config}" \
  "${protocol}" \
  "${project_root}/uq_estimator/counterfactual_evidence_pairwise.py" \
  "${project_root}/scripts/train_counterfactual_pairwise_native_repair.py" \
  "${dataset_manifest}" \
  "${native_features}" \
  "${initial_checkpoint}" > "${output_root}/source_sha256.txt"

"${python_runner}" "${project_root}/scripts/train_counterfactual_pairwise_native_repair.py" \
  --dataset-manifest "${dataset_manifest}" \
  --native-features "${native_features}" \
  --initial-checkpoint "${initial_checkpoint}" \
  --protocol "${protocol}" \
  --config "${config}" \
  --output "${output}" \
  --device cuda

sha256sum \
  "${output}" \
  "${output_root}/counterfactual_evidence_pairwise_native_repair.report.json" \
  "${output_root}/counterfactual_evidence_pairwise_native_repair.training_state.pt" \
  > "${output_root}/artifact_sha256.txt"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "COUNTERFACTUAL_PAIRWISE_NATIVE_REPAIR_JOB_OK=1"

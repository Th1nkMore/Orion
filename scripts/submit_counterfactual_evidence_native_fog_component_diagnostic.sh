#!/usr/bin/env bash
set -euo pipefail

submit=0
case "${1:-}" in
  --submit) submit=1 ;;
  ""|--dry-run) ;;
  *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
esac

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
work_root="${WORK_ROOT:-/public/home/lidachuan/orion_work/observation_uq_v3}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-counterfactual_evidence_no_view_native_fog_components_seed20260827_r1}"
output_root="${OUTPUT_ROOT:-${work_root}/runs/${run_id}}"
checkpoint="${work_root}/runs/counterfactual_evidence_no_view_repair_hidden128_seed20260827_r1/counterfactual_evidence_no_view_repair.pt"
native_features="${asset_root}/observation_uq_v3/runs/observation_uq_native_weather_epic_seed20260826_r5_gpu4/native_weather_features.pt"
failed_report="${work_root}/runs/counterfactual_evidence_no_view_native_fog_seed20260827_r1/counterfactual_evidence_native_fog.report.json"
output="${output_root}/counterfactual_evidence_native_fog_components.report.json"
log_root="${work_root}/logs"
wrapped_parts=(
  env
  "PYTHONPATH=${project_root}:${PYTHONPATH:-}"
  "COMPAT_PYTHON_BIN=${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
  "COMPAT_GLIBC_SYSROOT=${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
  "COMPAT_LIBRARY_PATH=${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
  "${project_root}/scripts/run_compat_python.sh"
  "${project_root}/scripts/diagnose_counterfactual_evidence_native_fog_components.py"
  --checkpoint "${checkpoint}"
  --native-features "${native_features}"
  --failed-report "${failed_report}"
  --output "${output}"
  --batch-size 2
)
printf -v wrapped_command '%q ' "${wrapped_parts[@]}"
sbatch_args=(
  sbatch --parsable
  --partition=Nvidia_A800
  --gres=gpu:1
  --cpus-per-task=2
  --mem=16G
  --time=00:15:00
  --job-name=uq_fog_diag
  "--output=${log_root}/${run_id}-%j.out"
  --export=ALL
  --wrap "set -euo pipefail; mkdir -p '${output_root}'; ${wrapped_command}; sha256sum '${output}' > '${output_root}/artifact_sha256.txt'"
)

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RUN_ID=${run_id}"
  echo "POST_FAILURE_DIAGNOSTIC_ONLY=1"
  echo "CHECKPOINT_UPDATE=0"
  echo "CLOSED_LOOP_AUTHORIZED=0"
  echo "RESOURCE_CONTRACT=partition:Nvidia_A800,gpu:1,cpus:2,mem:16G,time:00:15:00"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi

[[ ! -e "${output_root}" ]] || { echo "refusing to reuse output root: ${output_root}" >&2; exit 1; }
mkdir -p "${log_root}"
"${sbatch_args[@]}"

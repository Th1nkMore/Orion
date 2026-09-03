#!/usr/bin/env bash
set -euo pipefail

submit=0
runner_mode="--execute"
for argument in "$@"; do
  case "${argument}" in
    --submit)
      submit=1
      ;;
    --dry-run)
      submit=0
      ;;
    --model-init-only)
      runner_mode="--model-init-only"
      ;;
    *)
      echo "unknown argument: ${argument}" >&2
      echo "usage: $0 [--dry-run] [--submit] [--model-init-only]" >&2
      exit 2
      ;;
  esac
done

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_bin="${PYTHON_BIN:-${project_root}/scripts/run_compat_python.sh}"
compat_python_bin="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
compat_glibc_sysroot="${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
compat_library_path="${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
config_path="${CONFIG_PATH:-${project_root}/adzoo/orion/configs/orion_stage3_agent.py}"
pilot_manifest="${PILOT_MANIFEST:-${project_root}/configs/spatial_uq_route_manifests/b2d_val_exploratory_pilot10_seed20260826.json}"
checkpoint_path="${ORION_CHECKPOINT:-${asset_root}/checkpoints/Orion.pth}"
infos_path="${B2D_INFOS:-${asset_root}/data/infos/b2d_infos_val.pkl}"
dataset_root="${B2D_DATASET_ROOT:-${asset_root}/data/bench2drive}"
map_root="${B2D_MAP_ROOT:-${dataset_root}/maps}"
map_file="${B2D_MAP_FILE:-${asset_root}/data/infos/b2d_map_infos.pkl}"
llm_path="${ORION_QFORMER_PATH:-${asset_root}/checkpoints/pretrain_qformer}"
integration_factory="${INTEGRATION_FACTORY:-uq_estimator.orion_route214_production_integration:build_route214_production_integration_v1}"
qa_root="${QA_EVIDENCE_ROOT:-${asset_root}/spatial_uq_v1/evidence/route214}"
run_id="${RUN_ID:-route214_g1_$(date +%Y%m%dT%H%M%S)}"
output_root="${OUTPUT_ROOT:-${asset_root}/spatial_uq_v1/runs/${run_id}}"
log_root="${LOG_ROOT:-${asset_root}/logs/orion_actual_target_route214}"

runner_args=(
  "${project_root}/scripts/run_orion_actual_target_route214.py"
  "${runner_mode}"
  --config "${config_path}"
  --pilot-manifest "${pilot_manifest}"
  --checkpoint "${checkpoint_path}"
  --infos "${infos_path}"
  --dataset-root "${dataset_root}"
  --map-root "${map_root}"
  --map-file "${map_file}"
  --llm-path "${llm_path}"
  --output-root "${output_root}"
  --integration-factory "${integration_factory}"
  --decoder-parity-evidence "${qa_root}/decoder_parity.json"
  --selected-mode-evidence "${qa_root}/selected_motion_mode.json"
  --projection-overlay-evidence "${qa_root}/projection_overlay.json"
  --gt-axis-evidence "${qa_root}/gt_axis_alignment.json"
  --workers 4
  --seed 20260826
)
if [[ -n "${CHECKPOINT_SHA256:-}" ]]; then
  runner_args+=(--checkpoint-sha256 "${CHECKPOINT_SHA256}")
fi

wrapped_parts=(
  env
  "PYTHONPATH=${project_root}:${PYTHONPATH:-}"
  "COMPAT_PYTHON_BIN=${compat_python_bin}"
  "COMPAT_GLIBC_SYSROOT=${compat_glibc_sysroot}"
  "COMPAT_LIBRARY_PATH=${compat_library_path}"
  "${python_bin}"
  "${runner_args[@]}"
)
printf -v wrapped_command '%q ' "${wrapped_parts[@]}"

sbatch_args=(
  sbatch
  --parsable
  "--partition=${SLURM_PARTITION:-Nvidia_A800}"
  --gres=gpu:1
  --cpus-per-task=8
  --mem=220G
  --time=02:00:00
  --job-name=orion_g1_route214
  "--output=${log_root}/${run_id}-%j.out"
  --export=ALL
)
if [[ -n "${SLURM_NODELIST:-}" ]]; then
  sbatch_args+=("--nodelist=${SLURM_NODELIST}")
fi
sbatch_args+=(--wrap "${wrapped_command}")

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "No Slurm job was submitted. Re-run with --submit after reviewing paths and QA evidence."
  echo "RESOURCE_CONTRACT=partition:${SLURM_PARTITION:-Nvidia_A800},gpu:1,cpus:8,mem:220G,time:02:00:00"
  echo "RUNNER_MODE=${runner_mode}"
  echo "OUTPUT_ROOT=${output_root}"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi

required_files=(
  "${python_bin}"
  "${compat_python_bin}"
  "${config_path}"
  "${pilot_manifest}"
  "${checkpoint_path}"
  "${infos_path}"
  "${map_file}"
)
if [[ "${runner_mode}" == "--execute" ]]; then
  required_files+=(
    "${qa_root}/decoder_parity.json"
    "${qa_root}/selected_motion_mode.json"
    "${qa_root}/projection_overlay.json"
    "${qa_root}/gt_axis_alignment.json"
  )
fi
for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "required file is missing: ${path}" >&2
    exit 1
  fi
done
for path in \
  "${dataset_root}" \
  "${map_root}" \
  "${llm_path}" \
  "${compat_glibc_sysroot}" \
  "${compat_library_path}"; do
  if [[ ! -d "${path}" ]]; then
    echo "required directory is missing: ${path}" >&2
    exit 1
  fi
done
if [[ -e "${output_root}" ]]; then
  echo "refusing to reuse existing OUTPUT_ROOT: ${output_root}" >&2
  exit 1
fi
mkdir -p "${log_root}"
"${sbatch_args[@]}"

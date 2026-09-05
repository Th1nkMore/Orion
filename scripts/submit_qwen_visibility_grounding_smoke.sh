#!/usr/bin/env bash
set -euo pipefail

submit=0
case "${1:-}" in
  --submit) submit=1 ;;
  ""|--dry-run) ;;
  *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
esac

project_root="${PROJECT_ROOT:-/public/home/lidachuan/orion_work/qwen_visibility_belief}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_bin="${asset_root}/envs/qwen-drive-py310/bin/python"
glibc_sysroot="${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot"
glibc_loader="${glibc_sysroot}/lib64/ld-linux-x86-64.so.2"
runtime_library_path="${glibc_sysroot}/lib64:${glibc_sysroot}/usr/lib64:${asset_root}/envs/qwen-drive-py310/lib"
protocol="${project_root}/configs/qwen_visibility_grounding_smoke_v1.json"
trainer="${project_root}/scripts/train_qwen_visibility_grounding_smoke.py"
manifest="${asset_root}/qwen_visibility_grounding_runs/route151_v1a_manifest_v1/manifest.json"
run_id="${RUN_ID:-qwen_visibility_grounding_v1b_step260_gradient_v1}"
run_root="${asset_root}/qwen_visibility_grounding_runs/${run_id}"
log_root="${asset_root}/qwen_visibility_grounding_runs/logs"
job_name="${JOB_NAME:-qwen_visibility_v1b}"
node_list="${NODELIST:-gpu4}"

for prerequisite in \
  "${python_bin}" "${glibc_loader}" "${protocol}" "${trainer}" "${manifest}"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "missing Qwen visibility-grounding prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${run_root}" ]]; then
  echo "refusing to reuse Qwen visibility-grounding output: ${run_root}" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active Qwen visibility-grounding smoke" >&2
  exit 1
fi

run_parts=(
  env "PYTHONPATH=${project_root}:${asset_root}/third_party/Qwen-Drive-1.0/src:${PYTHONPATH:-}"
  "${glibc_loader}" --library-path "${runtime_library_path}"
  "${python_bin}" "${trainer}"
  --protocol "${protocol}"
  --output-dir "${run_root}"
)
printf -v run_command '%q ' "${run_parts[@]}"
sbatch_args=(
  sbatch --parsable --partition=Nvidia_A800 --gres=gpu:1
  --cpus-per-task=8 --mem=96G --time=01:30:00
  --job-name="${job_name}"
  --output="${log_root}/${run_id}-%j.out"
  --export=ALL --nodelist="${node_list}"
)
sbatch_args+=(--wrap "${run_command}")

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RUN_ID=${run_id}"
  echo "OUTPUT=${run_root}"
  echo "NODELIST=${node_list}"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${log_root}"
"${sbatch_args[@]}"


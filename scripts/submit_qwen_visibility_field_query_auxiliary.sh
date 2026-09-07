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
manifest="${asset_root}/qwen_visibility_grounding_runs/route_diverse_random_query_data_v1_2/manifest.json"
trainer="${project_root}/scripts/train_qwen_visibility_field_query_auxiliary.py"
run_id="${RUN_ID:-qwen_visibility_field_query_auxiliary_v1}"
run_root="${asset_root}/qwen_visibility_grounding_runs/${run_id}"
log_root="${asset_root}/qwen_visibility_grounding_runs/logs"
job_name="${JOB_NAME:-qwen_u_a1a}"
node_list="${NODELIST:-gpu6}"

for prerequisite in "${python_bin}" "${glibc_loader}" "${manifest}" "${trainer}"; do
  [[ -f "${prerequisite}" ]] || { echo "missing A1a prerequisite: ${prerequisite}" >&2; exit 2; }
done
[[ ! -e "${run_root}" ]] || { echo "refusing to reuse A1a output: ${run_root}" >&2; exit 1; }

run_parts=(
  env "PYTHONPATH=${project_root}:${PYTHONPATH:-}"
  "${glibc_loader}" --library-path "${runtime_library_path}"
  "${python_bin}" "${trainer}"
  --manifest "${manifest}"
  --expected-manifest-sha256 "67d330c7018c693eaac703f83909a32f610f24c6073fa2f6cc573c34d05ae79d"
  --output-dir "${run_root}"
  --device cuda --epochs 30 --seed 20260907 --learning-rate 3e-4
)
printf -v run_command '%q ' "${run_parts[@]}"
sbatch_args=(
  sbatch --parsable --partition=Nvidia_A800 --gres=gpu:1
  --cpus-per-task=4 --mem=16G --time=00:30:00
  --job-name="${job_name}" --output="${log_root}/${run_id}-%j.out"
  --export=ALL --nodelist="${node_list}" --wrap "${run_command}"
)

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RUN_ID=${run_id}"
  printf 'SBATCH_COMMAND='; printf '%q ' "${sbatch_args[@]}"; printf '\n'
  exit 0
fi
mkdir -p "${log_root}"
"${sbatch_args[@]}"

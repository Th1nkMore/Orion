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
qwen_code="${asset_root}/third_party/Qwen-Drive-1.0"
manifest="${asset_root}/qwen_visibility_grounding_runs/route_diverse_random_query_data_v1_2/manifest.json"
data_root="${asset_root}/qwen_visibility_grounding_runs/field_language_data_v1"
curriculum="${data_root}/curriculum.json"
builder="${project_root}/scripts/build_qwen_visibility_field_language_curriculum.py"
trainer="${project_root}/scripts/train_qwen_visibility_grounding_smoke.py"
protocol="${project_root}/configs/qwen_visibility_field_language_v1.json"
run_id="${RUN_ID:-qwen_visibility_field_language_v1}"
run_root="${asset_root}/qwen_visibility_grounding_runs/${run_id}"
log_root="${asset_root}/qwen_visibility_grounding_runs/logs"
node_list="${NODELIST:-gpu6}"

for prerequisite in "${python_bin}" "${glibc_loader}" "${manifest}" "${builder}" "${trainer}" "${protocol}"; do
  [[ -f "${prerequisite}" ]] || { echo "missing A1b prerequisite: ${prerequisite}" >&2; exit 2; }
done
[[ ! -e "${data_root}" ]] || { echo "refusing to reuse A1b curriculum: ${data_root}" >&2; exit 1; }
[[ ! -e "${run_root}" ]] || { echo "refusing to reuse A1b output: ${run_root}" >&2; exit 1; }

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "CURRICULUM=${curriculum}"
  echo "RUN_ID=${run_id}"
  exit 0
fi
mkdir -p "${data_root}" "${log_root}"
"${glibc_loader}" --library-path "${runtime_library_path}" "${python_bin}" "${builder}" \
  --manifest "${manifest}" \
  --expected-manifest-sha256 "67d330c7018c693eaac703f83909a32f610f24c6073fa2f6cc573c34d05ae79d" \
  --output "${curriculum}" --seed 202609071

run_parts=(
  env "PYTHONPATH=${qwen_code}/src:${project_root}:${PYTHONPATH:-}"
  "${glibc_loader}" --library-path "${runtime_library_path}"
  "${python_bin}" "${trainer}" --protocol "${protocol}" --output-dir "${run_root}"
)
printf -v run_command '%q ' "${run_parts[@]}"
sbatch --parsable --partition=Nvidia_A800 --gres=gpu:1 --cpus-per-task=4 \
  --mem=80G --time=03:00:00 --job-name=qwen_u_a1b \
  --output="${log_root}/${run_id}-%j.out" --export=ALL \
  --nodelist="${node_list}" --wrap "${run_command}"

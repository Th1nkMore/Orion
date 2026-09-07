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
model="${asset_root}/checkpoints/Qwen-Drive-1.0-4B"
data_root="${asset_root}/qwen_visibility_grounding_runs/route_diverse_random_query_data_v1_2"
manifest="${data_root}/manifest.json"
curriculum="${data_root}/curriculum.json"
evaluator="${project_root}/scripts/evaluate_qwen_visibility_numeric_text_upper_bound.py"
run_id="${RUN_ID:-qwen_visibility_numeric_text_upper_bound_v1}"
run_root="${asset_root}/qwen_visibility_grounding_runs/${run_id}"
output="${run_root}/report.json"
log_root="${asset_root}/qwen_visibility_grounding_runs/logs"
job_name="${JOB_NAME:-qwen_u_text_ub}"
node_list="${NODELIST:-gpu4}"

for prerequisite in \
  "${python_bin}" "${glibc_loader}" "${qwen_code}/src/qwen_drive/__init__.py" \
  "${model}/config.json" "${model}/model.safetensors" \
  "${manifest}" "${curriculum}" "${evaluator}"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "missing Qwen numeric text upper-bound prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${run_root}" ]]; then
  echo "refusing to reuse Qwen numeric text upper-bound output: ${run_root}" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active Qwen numeric text upper-bound job" >&2
  exit 1
fi

run_parts=(
  env "PYTHONPATH=${qwen_code}/src:${project_root}:${PYTHONPATH:-}"
  "${glibc_loader}" --library-path "${runtime_library_path}"
  "${python_bin}" "${evaluator}"
  --model "${model}"
  --manifest "${manifest}"
  --curriculum "${curriculum}"
  --expected-model-sha256 "b9de4bf448f57485fdaa45c60b1eea8e41a4b6ae82ec0cee8855a1e0301caccc"
  --expected-manifest-sha256 "67d330c7018c693eaac703f83909a32f610f24c6073fa2f6cc573c34d05ae79d"
  --expected-curriculum-sha256 "3bb13ed37890fef3be3e3c7611f9dc23c12e243e2edd105c796fd959cb7f13d5"
  --output "${output}"
  --device cuda
  --dtype bfloat16
)
printf -v run_command '%q ' "${run_parts[@]}"
sbatch_args=(
  sbatch --parsable --partition=Nvidia_A800 --gres=gpu:1
  --cpus-per-task=4 --mem=64G --time=00:45:00
  --job-name="${job_name}"
  --output="${log_root}/${run_id}-%j.out"
  --export=ALL --nodelist="${node_list}"
  --wrap "${run_command}"
)

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RUN_ID=${run_id}"
  echo "OUTPUT=${output}"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${run_root}" "${log_root}"
"${sbatch_args[@]}"

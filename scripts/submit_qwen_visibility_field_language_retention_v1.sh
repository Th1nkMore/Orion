#!/usr/bin/env bash
set -euo pipefail

submit=0
case "${1:-}" in --submit) submit=1 ;; ""|--dry-run) ;; *) exit 2 ;; esac
project_root="${PROJECT_ROOT:-/public/home/lidachuan/orion_work/qwen_visibility_belief}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_bin="${asset_root}/envs/qwen-drive-py310/bin/python"
glibc_sysroot="${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot"
glibc_loader="${glibc_sysroot}/lib64/ld-linux-x86-64.so.2"
runtime_library_path="${glibc_sysroot}/lib64:${glibc_sysroot}/usr/lib64:${asset_root}/envs/qwen-drive-py310/lib"
qwen_code="${asset_root}/third_party/Qwen-Drive-1.0"
curriculum="${asset_root}/qwen_visibility_grounding_runs/field_language_data_v1/curriculum.json"
protocol="${project_root}/configs/qwen_visibility_field_language_retention_v1.json"
trainer="${project_root}/scripts/train_qwen_visibility_grounding_smoke.py"
run_id="${RUN_ID:-qwen_visibility_field_language_retention_v1}"
run_root="${asset_root}/qwen_visibility_grounding_runs/${run_id}"
log_root="${asset_root}/qwen_visibility_grounding_runs/logs"
[[ -f "${curriculum}" ]] || { echo "missing immutable A1b curriculum" >&2; exit 2; }
[[ "$(sha256sum "${curriculum}" | awk '{print $1}')" == "c8c644a9be060fa79d2b3b4e0f6f5c21afbcdd14dfabfca161c0b35731eb2e43" ]] || { echo "A1b curriculum hash changed" >&2; exit 2; }
[[ ! -e "${run_root}" ]] || { echo "refusing to reuse A1b retention output" >&2; exit 1; }
if [[ "${submit}" != 1 ]]; then echo "DRY_RUN_ONLY=1"; echo "RUN_ID=${run_id}"; exit 0; fi
run_parts=(env "PYTHONPATH=${qwen_code}/src:${project_root}:${PYTHONPATH:-}" "${glibc_loader}" --library-path "${runtime_library_path}" "${python_bin}" "${trainer}" --protocol "${protocol}" --output-dir "${run_root}")
printf -v run_command '%q ' "${run_parts[@]}"
mkdir -p "${log_root}"
sbatch --parsable --partition=Nvidia_A800 --gres=gpu:1 --cpus-per-task=4 --mem=80G --time=03:00:00 --job-name=qwen_u_a1b_fix --output="${log_root}/${run_id}-%j.out" --export=ALL --nodelist="${NODELIST:-gpu6}" --wrap "${run_command}"

#!/usr/bin/env bash
set -euo pipefail

project_root="/public/home/lidachuan/project/Orion"
asset_root="/public/share/lidachuan/orion_assets"
python_bin="${asset_root}/envs/qwen-drive-py310/bin/python"
glibc_sysroot="${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot"
glibc_loader="${glibc_sysroot}/lib64/ld-linux-x86-64.so.2"
runtime_library_path="${glibc_sysroot}/lib64:${glibc_sysroot}/usr/lib64:${asset_root}/envs/qwen-drive-py310/lib"
qwen_code="${asset_root}/third_party/Qwen-Drive-1.0"
model="${asset_root}/checkpoints/Qwen-Drive-1.0-4B"
orion_report="${asset_root}/scenario_factory/stage2l_smokes/v15_2_text_oracle_localization_120dev_v1/report.json"
evaluator="${project_root}/scripts/evaluate_qwen_drive_text_oracle_v1.py"
triton_cc="${project_root}/scripts/triton_cc_c99.sh"
run_name="v15_3_qwen_drive_text_oracle_120dev_v1"
job_name="s2l_v153_qwen"
run_root="${asset_root}/scenario_factory/stage2l_smokes/${run_name}"
output="${run_root}/report.json"
log_root="${asset_root}/scenario_factory/logs/stage2l_smokes"

for prerequisite in \
  "${python_bin}" "${glibc_loader}" "${qwen_code}/src/qwen_drive/__init__.py" \
  "${model}/config.json" "${model}/model.safetensors" \
  "${orion_report}" "${evaluator}" "${triton_cc}"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "missing Qwen text-oracle prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${output}" ]]; then
  echo "refusing to overwrite an existing Qwen text-oracle report" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active Qwen text-oracle job" >&2
  exit 1
fi
if ! env "PYTHONPATH=${qwen_code}/src:${project_root}:${PYTHONPATH:-}" \
  "${glibc_loader}" --library-path "${runtime_library_path}" \
  "${python_bin}" -c \
  'from importlib.metadata import version; assert version("flash-linear-attention") == "0.5.1"; assert version("fla-core") == "0.5.1"'; then
  echo "flash-linear-attention preflight failed" >&2
  exit 2
fi

mkdir -p "${run_root}" "${log_root}"
run_parts=(
  env "PYTHONPATH=${qwen_code}/src:${project_root}:${PYTHONPATH:-}" "CC=${triton_cc}"
  "${glibc_loader}" --library-path "${runtime_library_path}"
  "${python_bin}" "${evaluator}"
  --model "${model}"
  --orion-report "${orion_report}"
  --expected-orion-report-sha256 "46c30afc8a41123e5eee45386e187824c7dad1dfb99ecf282479b0084fc21b72"
  --expected-model-sha256 "b9de4bf448f57485fdaa45c60b1eea8e41a4b6ae82ec0cee8855a1e0301caccc"
  --model-revision "9d2eb187c2fe03d0e30fd58c0638058980ee6267"
  --qwen-code-revision "28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0"
  --invalidated-job-id "1155299"
  --output "${output}"
  --device cuda
  --dtype bfloat16
)
printf -v run_command '%q ' "${run_parts[@]}"
job_id="$(sbatch --parsable \
  --partition=Nvidia_A800 \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=64G \
  --time=04:00:00 \
  --job-name="${job_name}" \
  --output="${log_root}/${run_name}-%j.out" \
  --export=ALL \
  --wrap "${run_command}")"
printf '%s\n' "${job_id}"

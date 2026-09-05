#!/usr/bin/env bash
set -euo pipefail

project_root="/public/home/lidachuan/project/Orion"
asset_root="/public/share/lidachuan/orion_assets"
python_bin="${asset_root}/envs/qwen-drive-py310/bin/python"
glibc_sysroot="${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot"
glibc_loader="${glibc_sysroot}/lib64/ld-linux-x86-64.so.2"
runtime_library_path="${glibc_sysroot}/lib64:${glibc_sysroot}/usr/lib64:${asset_root}/envs/qwen-drive-py310/lib"
config="${project_root}/configs/qwen_drive_b2d_agent_v1.json"
smoke="${project_root}/scripts/smoke_qwen_drive_b2d_bridge.py"
planner="${asset_root}/checkpoints/Qwen-Drive-1.0-4B/planner-sft/model.safetensors"
capture="${asset_root}/observation_uq_v3/runs/observation_uq_native_weather_seed20260826_r2_gpu4/capture/clear/Town01_Route146"
front="${capture}/rgb_front/0015.png"
front_left="${capture}/rgb_front_left/0015.png"
front_right="${capture}/rgb_front_right/0015.png"
run_root="${asset_root}/qwen_drive_b2d_smokes/bridge_flash_source_v1"
output="${run_root}/report.json"
log_root="${asset_root}/qwen_drive_b2d_smokes/logs"
job_name="qwen_b2d_flash"

for prerequisite in \
  "${python_bin}" "${glibc_loader}" "${config}" "${smoke}" "${planner}" \
  "${front}" "${front_left}" "${front_right}"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "missing Qwen-Drive bridge-smoke prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${output}" ]]; then
  echo "refusing to overwrite existing bridge-smoke report: ${output}" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active Qwen-Drive bridge-smoke job" >&2
  exit 1
fi

mkdir -p "${run_root}" "${log_root}"
run_parts=(
  env "PYTHONPATH=${project_root}:${PYTHONPATH:-}"
  "${glibc_loader}" --library-path "${runtime_library_path}"
  "${python_bin}" "${smoke}"
  --config "${config}"
  --front "${front}"
  --front-left "${front_left}"
  --front-right "${front_right}"
  --output "${output}"
)
printf -v run_command '%q ' "${run_parts[@]}"
job_id="$(sbatch --parsable \
  --partition=Nvidia_A800 \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=64G \
  --time=01:00:00 \
  --job-name="${job_name}" \
  --output="${log_root}/bridge_flash_source_v1-%j.out" \
  --export=ALL \
  --wrap "${run_command}")"
printf '%s\n' "${job_id}"

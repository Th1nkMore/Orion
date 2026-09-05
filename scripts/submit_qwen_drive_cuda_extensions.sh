#!/usr/bin/env bash
set -euo pipefail

project_root="/public/home/lidachuan/project/Orion"
asset_root="/public/share/lidachuan/orion_assets"
log_root="${asset_root}/qwen_drive_b2d_smokes/logs"
job_name="qwen_cuda_build"

if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active Qwen-Drive CUDA build" >&2
  exit 1
fi

mkdir -p "${log_root}"
job_id="$(sbatch --parsable \
  --partition=Nvidia_A800 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=02:00:00 \
  --job-name="${job_name}" \
  --output="${log_root}/cuda-build-%j.out" \
  --export=ALL \
  --wrap="cd '${project_root}' && bash scripts/build_qwen_drive_cuda_extensions.sh")"
printf '%s\n' "${job_id}"

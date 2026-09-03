#!/usr/bin/env bash
set -euo pipefail

project_root="/public/home/lidachuan/project/Orion"
asset_root="/public/share/lidachuan/orion_assets"
python_bin="${asset_root}/envs/orion-cl-centos7/bin/python"
base_root="${asset_root}/scenario_factory/stage2p_smokes/v1_controlled_k_route147_80step_v1"
output_root="${base_root}/training"
preflight="${base_root}/trainer_preflight.json"
source_root="${asset_root}/stage2/route147_stage2a_optimization_smoke_v1"
semantic_root="${asset_root}/scenario_factory/stage2l_smokes/v122_vertical_slice_semantic_17event_40_v1/training"
trainer="${project_root}/scripts/train_stage2p_v1_controlled_k_interface_smoke.py"
protocol="${project_root}/configs/scenario_factory/stage2p_v1_controlled_k_interface_protocol_v1.json"
launch="${project_root}/configs/scenario_factory/amendments/20260901_stage2p_v1_controlled_k_interface_retry1_launch_v1.json"
semantic_terminal="${project_root}/configs/scenario_factory/amendments/20260901_stage2l_v122_vertical_slice_semantic_terminal_v1.json"
job_name="s2p_v1_k_iface"
log_root="${asset_root}/scenario_factory/logs/stage2p_smokes"

for path in \
  "${python_bin}" "${trainer}" "${protocol}" "${launch}" \
  "${semantic_terminal}" "${preflight}" \
  "${source_root}/stage2_optimization_smoke_manifest.jsonl" \
  "${source_root}/build_report.json" "${semantic_root}/report.json" \
  "${semantic_root}/v122_semantic_bridge.pt" \
  "${semantic_root}/spatial_u_r_k_maps.pt"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing Stage2-P prerequisite: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${output_root}" ]] && find "${output_root}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty Stage2-P output" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active Stage2-P job" >&2
  exit 1
fi

mkdir -p "${log_root}"
parts=(
  env "PYTHONPATH=${project_root}"
  "${python_bin}" "${trainer}"
  --source-manifest "${source_root}/stage2_optimization_smoke_manifest.jsonl"
  --source-build-report "${source_root}/build_report.json"
  --semantic-terminal "${semantic_terminal}"
  --semantic-report "${semantic_root}/report.json"
  --semantic-bridge "${semantic_root}/v122_semantic_bridge.pt"
  --spatial-u-r-k-maps "${semantic_root}/spatial_u_r_k_maps.pt"
  --training-protocol "${protocol}"
  --trainer-preflight "${preflight}"
  --launch-amendment "${launch}"
  --output-dir "${output_root}"
  --device cuda
)
printf -v command '%q ' "${parts[@]}"
sbatch --parsable \
  --partition=Nvidia_A800 \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=32G \
  --time=01:00:00 \
  --exclude=gpu5 \
  --no-requeue \
  --job-name="${job_name}" \
  --output="${log_root}/v1_controlled_k_route147_80step_v1-%j.out" \
  --export=ALL \
  --wrap "${command}"

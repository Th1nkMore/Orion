#!/usr/bin/env bash
set -euo pipefail

submit=0
case "${1:-}" in
  --submit) submit=1 ;;
  ""|--dry-run) ;;
  *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
esac

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
task_id="${TASK_ID:-203}"
run_id="${RUN_ID:-closedloop_route${task_id}_flash_gpu4_v4}"
output_root="${OUTPUT_ROOT:-${asset_root}/qwen_drive_b2d_smokes/${run_id}}"
log_root="${asset_root}/qwen_drive_b2d_smokes/logs"
job_name="${JOB_NAME:-qwen_b2d_route${task_id}}"
node_list="${NODELIST:-gpu4}"
carla_port="${PORT:-30000}"
traffic_manager_port="${TM_PORT:-50000}"
agent_config_path="${AGENT_CONFIG_PATH:-${project_root}/configs/qwen_drive_b2d_agent_v1.json}"

for prerequisite in \
  "${project_root}/team_code/qwen_drive_b2d_agent.py" \
  "${agent_config_path}" \
  "${project_root}/scripts/run_official_closedloop_job.sh" \
  "${asset_root}/carla/CARLA_0.9.15/CarlaUE4.sh" \
  "${asset_root}/envs/orion-cl-centos7/bin/python"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "missing Qwen-Drive closed-loop prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${output_root}" ]]; then
  echo "refusing to reuse Qwen-Drive closed-loop output: ${output_root}" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active Qwen-Drive closed-loop smoke" >&2
  exit 1
fi

wrapped=(
  env
  "PROJECT_ROOT=${project_root}"
  "ASSET_ROOT=${asset_root}"
  "CARLA_ROOT=${asset_root}/carla/CARLA_0.9.15"
  "PYTHON_BIN=${project_root}/scripts/run_compat_python.sh"
  "COMPAT_PYTHON_BIN=${asset_root}/envs/orion-cl-centos7/bin/python"
  "COMPAT_GLIBC_SYSROOT=${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot"
  "COMPAT_LIBRARY_PATH=${asset_root}/envs/orion-cl/lib"
  "NVIDIA_RUNTIME_PREFIX=${asset_root}/nvidia/nvidia_driver-linux-x86_64-535.104.05-archive"
  "VULKAN_LOADER_LIBDIR=${asset_root}/envs/vulkan-1.3.250/lib"
  "VULKANINFO_BIN=${asset_root}/envs/vulkan-1.3.250/bin/vulkaninfo"
  "BENCH2DRIVE_MANAGES_CARLA=0"
  "BENCH2DRIVE_EXTERNAL_CARLA=1"
  "TEAM_AGENT_PATH=${project_root}/team_code/qwen_drive_b2d_agent.py"
  "AGENT_CONFIG_PATH=${agent_config_path}"
  "BASE_CHECKPOINT_PATH=qwen-drive-sidecar-no-orion-checkpoint"
  "ALGO=qwen_drive"
  "PLANNER_TYPE=traj"
  "TASK_ID=${task_id}"
  "OUTPUT_ROOT=${output_root}"
  "PORT=${carla_port}"
  "TM_PORT=${traffic_manager_port}"
  "ORION_CARLA_RPC_TIMEOUT_SECONDS=1200"
  "RESUME=False"
  bash "${project_root}/scripts/run_official_closedloop_job.sh"
)
printf -v wrapped_command '%q ' "${wrapped[@]}"
sbatch_args=(
  sbatch --parsable --partition=Nvidia_A800 --gres=gpu:1
  --nodelist="${node_list}"
  --cpus-per-task=8 --mem=96G --time=02:00:00
  --job-name="${job_name}"
  --output="${log_root}/${run_id}-%j.out"
  --export=ALL --wrap "${wrapped_command}"
)

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RUN_ID=${run_id}"
  echo "ROUTE=Bench2Drive220 task ${task_id}"
  echo "NODELIST=${node_list}"
  echo "PORT=${carla_port}"
  echo "TM_PORT=${traffic_manager_port}"
  echo "AGENT_CONFIG_PATH=${agent_config_path}"
  echo "ORION_MODEL_LOAD=0"
  echo "RESOURCE_CONTRACT=partition:Nvidia_A800,gpu:1,cpus:8,mem:96G,time:02:00:00"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${log_root}"
"${sbatch_args[@]}"

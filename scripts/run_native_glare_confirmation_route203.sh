#!/usr/bin/env bash
set -euo pipefail

# One ORION-free Route203 run with co-located clean/medium/heavy RGB cameras.
# Every comparison is made on the exact same simulator tick.

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-route203_native_glare_same_tick_v1}"
output_root="${OUTPUT_ROOT:-${asset_root}/scenario_factory/glare_confirmations/${run_id}}"
protocol="${project_root}/configs/native_glare_independent_confirmation_route203_v1.json"
source_route="${project_root}/configs/closedloop_uq_pilot/routes/route_203_hazard.xml"
derived_route="${output_root}/inputs/route_203_native_low_sun.xml"
route_manifest="${output_root}/inputs/route_derivation.json"
capture_root="${output_root}/capture"
python_runner="${project_root}/scripts/run_compat_python.sh"
carla_root="${asset_root}/carla/CARLA_0.9.15"
job_tag="${SLURM_JOB_ID:-$$}"
port_slot=$((job_tag % 1000))
port_offset=$((port_slot * 10))
port="${PORT:-$((22000 + port_offset))}"
tm_port="${TM_PORT:-$((42000 + port_offset))}"
server_log="${output_root}/logs/carla-server.log"

for prerequisite in \
  "${protocol}" "${source_route}" \
  "${project_root}/team_code/glare_triplet_capture_agent.py" \
  "${project_root}/scripts/prepare_native_glare_confirmation_route.py" \
  "${project_root}/scripts/validate_native_glare_triplet_capture.py" \
  "${project_root}/scripts/analyze_native_glare_triplet_capture.py"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "[FAIL] missing glare confirmation prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${output_root}" ]]; then
  echo "[FAIL] refusing to reuse output root: ${output_root}" >&2
  exit 1
fi
mkdir -p "${output_root}/inputs" "${output_root}/logs" "${capture_root}"

export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
export NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX:-${asset_root}/nvidia/nvidia_driver-linux-x86_64-535.104.05-archive}"
export VULKAN_LOADER_LIBDIR="${VULKAN_LOADER_LIBDIR:-${asset_root}/envs/vulkan-1.3.250/lib}"
export VULKANINFO_BIN="${VULKANINFO_BIN:-${asset_root}/envs/vulkan-1.3.250/bin/vulkaninfo}"
export LD_LIBRARY_PATH="${VULKAN_LOADER_LIBDIR}:${NVIDIA_RUNTIME_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${NVIDIA_RUNTIME_PREFIX}/etc/nvidia_icd.json"
export XDG_RUNTIME_DIR="/tmp/native-glare-confirm-${USER:-unknown}-${job_tag}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"
export PYTHONPATH="${carla_root}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${asset_root}/Bench2Drive:${asset_root}/Bench2Drive/leaderboard:${asset_root}/Bench2Drive/scenario_runner:${asset_root}/Bench2DriveZoo:${PYTHONPATH:-}"

"${python_runner}" "${project_root}/scripts/prepare_native_glare_confirmation_route.py" \
  --source-route "${source_route}" \
  --protocol "${protocol}" \
  --output-route "${derived_route}" \
  --manifest "${route_manifest}"

echo "RUN_ID=${run_id}"
echo "SCOPE=route203_same_tick_native_glare_renderer_confirmation_only"
echo "PROFILES=clean,medium,heavy"
echo "CONTROLLER=carla_basic_agent"
echo "ORION_LOAD=0"
echo "ADAPTER_LOAD=0"
echo "CLOSED_LOOP_MATRIX=0"

server_pid=""
server_live=0
stop_carla() {
  if [[ "${server_live}" != "1" ]]; then
    return
  fi
  if [[ "${server_pid}" =~ ^[0-9]+$ && "${server_pid}" -gt 1 ]]; then
    kill -TERM -- "-${server_pid}" >/dev/null 2>&1 || true
    for _ in {1..20}; do
      if ! kill -0 -- "-${server_pid}" >/dev/null 2>&1; then
        break
      fi
      sleep 0.5
    done
    if kill -0 -- "-${server_pid}" >/dev/null 2>&1; then
      kill -KILL -- "-${server_pid}" >/dev/null 2>&1 || true
    fi
  fi
  wait "${server_pid}" >/dev/null 2>&1 || true
  server_live=0
}
trap stop_carla EXIT

NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX}" VULKANINFO_BIN="${VULKANINFO_BIN}" \
  "${project_root}/scripts/check_official_carla_vulkan.sh"
setsid "${carla_root}/CarlaUE4.sh" \
  -vulkan -RenderOffScreen -nosound -quality-level=Epic \
  -carla-rpc-port="${port}" -stdout -FullStdOutLogOutput \
  >"${server_log}" 2>&1 &
server_pid=$!
server_live=1

ready=0
deadline=$((SECONDS + 180))
while (( SECONDS < deadline )); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "[FAIL] CARLA exited during glare confirmation startup" >&2
    tail -n 120 "${server_log}" >&2 || true
    exit 1
  fi
  if "${python_runner}" -c '
import sys, carla
c=carla.Client("127.0.0.1", int(sys.argv[1])); c.set_timeout(3.0)
print(c.get_world().get_map().name)
' "${port}" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "${ready}" != "1" ]]; then
  echo "[FAIL] CARLA glare confirmation server did not become ready" >&2
  exit 1
fi

if ! timeout --signal=TERM --kill-after=30s 600s env \
  PROJECT_ROOT="${project_root}" \
  BENCH2DRIVE_ROOT="${asset_root}/Bench2Drive" \
  BENCH2DRIVE_ZOO_ROOT="${asset_root}/Bench2DriveZoo" \
  CARLA_ROOT="${carla_root}" \
  PYTHON_BIN="${python_runner}" \
  PORT="${port}" TM_PORT="${tm_port}" \
  TEAM_AGENT_PATH="${project_root}/team_code/glare_triplet_capture_agent.py" \
  AGENT_CONFIG_PATH="${protocol}" \
  BASE_CHECKPOINT_PATH="/dev/null" \
  ROUTE_SPLIT_OVERRIDE="${derived_route}" \
  OUTPUT_ROOT="${capture_root}" \
  RUN_VULKAN_PRECHECK=0 \
  BENCH2DRIVE_EXTERNAL_CARLA=1 \
  RESUME=False \
  ALGO=glare_triplet_capture \
  PLANNER_TYPE=npc \
  GLARE_CAPTURE_FINISH_EXTENSION_M=6 \
  ORION_CARLA_RPC_TIMEOUT_SECONDS=90 \
  ORION_SCENARIO_TRACEBACK_INTERVAL_SECONDS=45 \
  "${project_root}/scripts/run_official_closedloop_smoke.sh" \
  >"${output_root}/logs/evaluator.log" 2>&1; then
  echo "[FAIL] Route203 native glare confirmation evaluator failed" >&2
  tail -n 180 "${output_root}/logs/evaluator.log" >&2 || true
  exit 1
fi

"${python_runner}" "${project_root}/scripts/validate_native_glare_triplet_capture.py" \
  --root "${capture_root}" \
  --protocol "${protocol}"
"${python_runner}" "${project_root}/scripts/analyze_native_glare_triplet_capture.py" \
  --root "${capture_root}" \
  --protocol "${protocol}" \
  --output "${output_root}/visual_analysis"

stop_carla
sha256sum \
  "${project_root}/team_code/glare_triplet_capture_agent.py" \
  "${project_root}/scripts/prepare_native_glare_confirmation_route.py" \
  "${project_root}/scripts/validate_native_glare_triplet_capture.py" \
  "${project_root}/scripts/analyze_native_glare_triplet_capture.py" \
  "${protocol}" "${derived_route}" \
  >"${output_root}/artifact_sha256.txt"
echo "NATIVE_GLARE_SAME_TICK_CONFIRMATION_OK=1"

#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-native_weather_epic_visual_preflight_r1}"
output_root="${OUTPUT_ROOT:-${asset_root}/observation_uq_v3/runs/${run_id}}"
python_runner="${PYTHON_RUNNER:-${project_root}/scripts/run_compat_python.sh}"
carla_root="${CARLA_ROOT:-${asset_root}/carla/CARLA_0.9.15}"
port="${PORT:-$((34000 + ${SLURM_JOB_ID:-0} % 10000))}"
capture_root="${output_root}/capture/Town04_Route203"
server_log="${output_root}/carla-server.log"

if [[ -e "${output_root}" ]]; then
  echo "[FAIL] refusing to reuse output root: ${output_root}" >&2
  exit 1
fi
mkdir -p "${output_root}"

export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
export NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX:-${asset_root}/nvidia/nvidia_driver-linux-x86_64-535.104.05-archive}"
export VULKAN_LOADER_LIBDIR="${VULKAN_LOADER_LIBDIR:-${asset_root}/envs/vulkan-1.3.250/lib}"
export VULKANINFO_BIN="${VULKANINFO_BIN:-${asset_root}/envs/vulkan-1.3.250/bin/vulkaninfo}"
export PYTHONPATH="${carla_root}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${VULKAN_LOADER_LIBDIR}:${NVIDIA_RUNTIME_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${NVIDIA_RUNTIME_PREFIX}/etc/nvidia_icd.json"
export XDG_RUNTIME_DIR="/tmp/native-weather-preflight-${USER:-unknown}-${SLURM_JOB_ID:-$$}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

echo "RUN_ID=${run_id}"
echo "SCOPE=native_weather_epic_visual_preflight_only"
echo "POSITIONS=3"
echo "FEATURE_EXTRACTION=0"
echo "ADAPTER_TRAINING=0"
echo "STAGE_B=0"

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
    echo "[FAIL] CARLA exited during Epic preflight startup" >&2
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
  echo "[FAIL] Epic preflight server did not become ready" >&2
  exit 1
fi

"${python_runner}" "${project_root}/scripts/capture_native_carla_weather.py" \
  --port "${port}" \
  --route "Town04/Route203=${project_root}/configs/closedloop_uq_pilot/routes/route_203_nohazard.xml" \
  --positions-per-route 3 \
  --renderer-quality Epic \
  --output "${capture_root}"
stop_carla

sha256sum \
  "${project_root}/scripts/capture_native_carla_weather.py" \
  "${capture_root}/capture_manifest.json" > "${output_root}/artifact_sha256.txt"
echo "NATIVE_WEATHER_EPIC_VISUAL_PREFLIGHT_OK=1"

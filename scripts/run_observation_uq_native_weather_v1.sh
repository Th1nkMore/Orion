#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-observation_uq_native_weather_seed20260826_r1}"
output_root="${OUTPUT_ROOT:-${asset_root}/observation_uq_v3/runs/${run_id}}"
python_runner="${PYTHON_RUNNER:-${project_root}/scripts/run_compat_python.sh}"
carla_root="${CARLA_ROOT:-${asset_root}/carla/CARLA_0.9.15}"
checkpoint="${ORION_CHECKPOINT:-${asset_root}/checkpoints/Orion.pth}"
calibration_shard="${CALIBRATION_SHARD:-${asset_root}/observation_uq_v3/runs/observation_uq_teacher560_seed20260826_r1/clean_first_features.pt}"
port="${PORT:-$((22000 + ${SLURM_JOB_ID:-0} % 10000))}"
reuse_capture_route146="${REUSE_CAPTURE_ROUTE146:-}"
capture_route146="${reuse_capture_route146:-${output_root}/capture/Town01_Route146}"
capture_route203="${output_root}/capture/Town04_Route203"
features="${output_root}/native_weather_features.pt"
report="${output_root}/native_weather_uq_audit.json"

required=(
  "${carla_root}/CarlaUE4.sh"
  "${checkpoint}"
  "${calibration_shard}"
  "${project_root}/configs/observation_uq_native_weather_v2.json"
  "${project_root}/configs/closedloop_uq_pilot/routes/route_146_nohazard.xml"
  "${project_root}/configs/closedloop_uq_pilot/routes/route_203_nohazard.xml"
  "${project_root}/scripts/capture_native_carla_weather.py"
  "${project_root}/scripts/extract_native_carla_weather_features.py"
  "${project_root}/scripts/audit_native_carla_weather_uq.py"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[FAIL] missing required native-weather input: ${path}" >&2
    exit 1
  fi
done
if [[ -n "${reuse_capture_route146}" && ! -f "${reuse_capture_route146}/capture_manifest.json" ]]; then
  echo "[FAIL] reused Route146 capture lacks a complete manifest: ${reuse_capture_route146}" >&2
  exit 1
fi
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
export XDG_RUNTIME_DIR="/tmp/native-weather-${USER:-unknown}-${SLURM_JOB_ID:-$$}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

echo "RUN_ID=${run_id}"
echo "HOST=$(hostname)"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SCOPE=paired_native_weather_signal_gate_only"
echo "ADAPTER_TRAINING=0"
echo "ACTUAL_TARGET_TRAINING=0"
echo "STAGE_B=0"
echo "REUSE_CAPTURE_ROUTE146=${reuse_capture_route146:-0}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
sha256sum \
  "${project_root}/configs/observation_uq_native_weather_v2.json" \
  "${project_root}/uq_estimator/native_weather_audit.py" \
  "${project_root}/scripts/capture_native_carla_weather.py" \
  "${project_root}/scripts/extract_native_carla_weather_features.py" \
  "${project_root}/scripts/audit_native_carla_weather_uq.py" \
  "${calibration_shard}" > "${output_root}/source_sha256.txt"

NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX}" \
VULKANINFO_BIN="${VULKANINFO_BIN}" \
  "${project_root}/scripts/check_official_carla_vulkan.sh"

server_pid=""
server_live=0
cleanup() {
  if [[ "${server_live}" == "1" ]]; then
    stop_carla
  fi
}
trap cleanup EXIT

start_carla() {
  local server_port="$1"
  local server_log="$2"
  # CarlaUE4.sh launches the UE4 binary as a child. A dedicated process group
  # makes shutdown cover the launcher and every descendant deterministically.
  setsid "${carla_root}/CarlaUE4.sh" \
    -vulkan -RenderOffScreen -nosound -quality-level=Epic \
    -carla-rpc-port="${server_port}" -stdout -FullStdOutLogOutput \
    >"${server_log}" 2>&1 &
  server_pid=$!
  server_live=1
  local deadline=$((SECONDS + 180))
  local ready=0
  while (( SECONDS < deadline )); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "[FAIL] CARLA exited during native-weather startup" >&2
      tail -n 120 "${server_log}" >&2 || true
      return 1
    fi
    if "${python_runner}" -c '
import sys, carla
c=carla.Client("127.0.0.1", int(sys.argv[1])); c.set_timeout(3.0)
print(c.get_world().get_map().name)
' "${server_port}" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "${ready}" != "1" ]]; then
    echo "[FAIL] CARLA native-weather server did not become ready" >&2
    tail -n 120 "${server_log}" >&2 || true
    return 1
  fi
}

stop_carla() {
  if [[ "${server_live}" == "1" ]]; then
    if [[ "${server_pid}" =~ ^[0-9]+$ && "${server_pid}" -gt 1 ]]; then
      kill -TERM -- "-${server_pid}" >/dev/null 2>&1 || true
    fi
    wait "${server_pid}" >/dev/null 2>&1 || true
    server_live=0
    server_pid=""
  fi
}

if [[ -z "${reuse_capture_route146}" ]]; then
  start_carla "${port}" "${output_root}/carla-server-route146.log"
  "${python_runner}" "${project_root}/scripts/capture_native_carla_weather.py" \
    --port "${port}" \
    --route "Town01/Route146=${project_root}/configs/closedloop_uq_pilot/routes/route_146_nohazard.xml" \
    --positions-per-route 16 \
    --renderer-quality Epic \
    --output "${capture_route146}"
  stop_carla
else
  echo "ROUTE146_CAPTURE_REUSED=1"
  sha256sum "${capture_route146}/capture_manifest.json"
fi

# CARLA reserves more than the RPC socket. Keep the second server's full port
# range disjoint from the first instead of shifting by one.
route203_port=$((port + 1000))
start_carla "${route203_port}" "${output_root}/carla-server-route203.log"
"${python_runner}" "${project_root}/scripts/capture_native_carla_weather.py" \
  --port "${route203_port}" \
  --route "Town04/Route203=${project_root}/configs/closedloop_uq_pilot/routes/route_203_nohazard.xml" \
  --positions-per-route 16 \
  --renderer-quality Epic \
  --output "${capture_route203}"
stop_carla

"${python_runner}" "${project_root}/scripts/extract_native_carla_weather_features.py" \
  --capture-root "${capture_route146}" \
  --capture-root "${capture_route203}" \
  --config "${project_root}/adzoo/orion/configs/orion_stage3_agent.py" \
  --checkpoint "${checkpoint}" \
  --output "${features}" \
  --batch-size 2

"${python_runner}" "${project_root}/scripts/audit_native_carla_weather_uq.py" \
  --clean-calibration-shard "${calibration_shard}" \
  --native-features "${features}" \
  --output "${report}" \
  --batch-size 8

sha256sum \
  "${features}" \
  "${report}" \
  "${capture_route146}/capture_manifest.json" \
  "${capture_route203}/capture_manifest.json" > "${output_root}/artifact_sha256.txt"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "OBSERVATION_UQ_NATIVE_WEATHER_V1_OK=1"

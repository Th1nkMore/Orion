#!/usr/bin/env bash
set -euo pipefail

# Start the validated CARLA runtime and stress the exact ORION sensor suite
# without loading ORION. The caller must provide a Slurm GPU allocation.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
ASSET_ROOT="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
CARLA_ROOT="${CARLA_ROOT:-${ASSET_ROOT}/carla/CARLA_0.9.15}"
CARLA_MAP="${CARLA_MAP:-Town05}"
SENSOR_TICKS="${SENSOR_TICKS:-1600}"
PORT="${PORT:-$((30000 + ${SLURM_JOB_ID:-$$} % 10000))}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ASSET_ROOT}/results/carla_runtime_repair/sensor_gate-${SLURM_JOB_ID:-$$}}"
CARLA_STARTUP_TIMEOUT="${CARLA_STARTUP_TIMEOUT:-240}"
CARLA_EXTRA_ARGS="${CARLA_EXTRA_ARGS:--stdout -FullStdOutLogOutput}"

export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${ASSET_ROOT}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${ASSET_ROOT}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${ASSET_ROOT}/envs/orion-cl/lib}"
export NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX:-${ASSET_ROOT}/nvidia/nvidia_driver-linux-x86_64-535.104.05-archive}"
export VULKAN_LOADER_LIBDIR="${VULKAN_LOADER_LIBDIR:-${ASSET_ROOT}/envs/vulkan-1.3.250/lib}"
export VULKANINFO_BIN="${VULKANINFO_BIN:-${ASSET_ROOT}/envs/vulkan-1.3.250/bin/vulkaninfo}"

mkdir -p "${OUTPUT_ROOT}"
server_log="${OUTPUT_ROOT}/carla-server.log"
client_log="${OUTPUT_ROOT}/carla-client.log"
result_json="${OUTPUT_ROOT}/sensor-runtime-gate.json"

if [[ ! -x "${CARLA_ROOT}/CarlaUE4.sh" ]]; then
  echo "[FAIL] missing CARLA server: ${CARLA_ROOT}/CarlaUE4.sh" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${VULKAN_LOADER_LIBDIR}:${NVIDIA_RUNTIME_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "${NVIDIA_RUNTIME_PREFIX}/etc/nvidia_icd.json" ]]; then
  export VK_ICD_FILENAMES="${NVIDIA_RUNTIME_PREFIX}/etc/nvidia_icd.json"
fi
if [[ -z "${XDG_RUNTIME_DIR:-}" || ! -w "${XDG_RUNTIME_DIR:-/nonexistent}" ]]; then
  export XDG_RUNTIME_DIR="/tmp/carla-runtime-${USER:-unknown}-${SLURM_JOB_ID:-$$}"
fi
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

carla_egg="${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
export PYTHONPATH="${carla_egg}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "[Runtime] host=$(hostname) job=${SLURM_JOB_ID:-none} map=${CARLA_MAP} ticks=${SENSOR_TICKS}"
echo "[Runtime] cpus_allowed=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX}" VULKANINFO_BIN="${VULKANINFO_BIN}" \
  "${PROJECT_ROOT}/scripts/check_official_carla_vulkan.sh"

read -r -a carla_extra_args <<<"${CARLA_EXTRA_ARGS}"
"${CARLA_ROOT}/CarlaUE4.sh" \
  -vulkan -RenderOffScreen -nosound -quality-level=Low \
  -carla-rpc-port="${PORT}" "${carla_extra_args[@]}" \
  >"${server_log}" 2>&1 &
server_pid=$!

cleanup() {
  kill "${server_pid}" >/dev/null 2>&1 || true
  wait "${server_pid}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

deadline=$((SECONDS + CARLA_STARTUP_TIMEOUT))
while (( SECONDS < deadline )); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "[FAIL] CARLA exited during startup" >&2
    tail -n 120 "${server_log}" >&2 || true
    exit 1
  fi
  if "${PROJECT_ROOT}/scripts/run_compat_python.sh" -c '
import sys, carla
client = carla.Client("127.0.0.1", int(sys.argv[1]))
client.set_timeout(3.0)
print(client.get_world().get_map().name)
' "${PORT}" >"${client_log}" 2>&1; then
    echo "[OK] CARLA RPC ready: $(tail -n 1 "${client_log}")"
    break
  fi
  sleep 2
done

if ! kill -0 "${server_pid}" 2>/dev/null; then
  echo "[FAIL] CARLA died before the sensor client" >&2
  tail -n 120 "${server_log}" >&2 || true
  exit 1
fi

"${PROJECT_ROOT}/scripts/run_compat_python.sh" \
  "${PROJECT_ROOT}/scripts/stress_carla_sensor_runtime.py" \
  --port "${PORT}" --map "${CARLA_MAP}" --ticks "${SENSOR_TICKS}" \
  --output "${result_json}"

if ! kill -0 "${server_pid}" 2>/dev/null; then
  echo "[FAIL] CARLA died after the sensor client" >&2
  tail -n 160 "${server_log}" >&2 || true
  exit 1
fi

cat "${result_json}"
echo "CARLA_SENSOR_GATE_JOB_OK"

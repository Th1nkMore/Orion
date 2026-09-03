#!/usr/bin/env bash
set -euo pipefail

# Run CARLA and the official Bench2Drive evaluator in the same allocation.
# The caller is responsible for requesting a GPU (for example via sbatch).

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CARLA_ROOT="${CARLA_ROOT:-}"
BENCH2DRIVE_ROOT="${BENCH2DRIVE_ROOT:-${PROJECT_ROOT}/Bench2Drive}"
BENCH2DRIVE_ZOO_ROOT="${BENCH2DRIVE_ZOO_ROOT:-${PROJECT_ROOT}/Bench2DriveZoo}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PORT="${PORT:-30000}"
TM_PORT="${TM_PORT:-50000}"
CARLA_STARTUP_TIMEOUT="${CARLA_STARTUP_TIMEOUT:-180}"
CARLA_EXTRA_ARGS="${CARLA_EXTRA_ARGS:--stdout -FullStdOutLogOutput}"
CARLA_QUALITY_LEVEL="${CARLA_QUALITY_LEVEL:-Epic}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/closedloop_official_smoke}"
NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX:-}"
VULKAN_LOADER_LIBDIR="${VULKAN_LOADER_LIBDIR:-}"
VULKANINFO_BIN="${VULKANINFO_BIN:-vulkaninfo}"
BENCH2DRIVE_MANAGES_CARLA="${BENCH2DRIVE_MANAGES_CARLA:-1}"

if [[ -z "${CARLA_ROOT}" || ! -x "${CARLA_ROOT}/CarlaUE4.sh" ]]; then
  echo "[FAIL] CARLA_ROOT does not contain CarlaUE4.sh: ${CARLA_ROOT:-<unset>}" >&2
  exit 1
fi

if [[ -n "${VULKAN_LOADER_LIBDIR}" ]]; then
  export LD_LIBRARY_PATH="${VULKAN_LOADER_LIBDIR}:${LD_LIBRARY_PATH:-}"
fi

if [[ -n "${NVIDIA_RUNTIME_PREFIX}" ]]; then
  nvidia_lib="${NVIDIA_RUNTIME_PREFIX}/lib"
  nvidia_icd="${NVIDIA_RUNTIME_PREFIX}/etc/nvidia_icd.json"
  export LD_LIBRARY_PATH="${nvidia_lib}:${LD_LIBRARY_PATH:-}"
  if [[ -f "${nvidia_icd}" ]]; then
    export VK_ICD_FILENAMES="${nvidia_icd}"
  fi
fi

if command -v nvidia-modprobe >/dev/null 2>&1; then
  nvidia-modprobe -m >/dev/null 2>&1 || true
  nvidia-modprobe -u -c=0 >/dev/null 2>&1 || true
fi

if [[ -z "${XDG_RUNTIME_DIR:-}" || ! -w "${XDG_RUNTIME_DIR:-/nonexistent}" ]]; then
  export XDG_RUNTIME_DIR="/tmp/carla-runtime-${USER:-unknown}-${SLURM_JOB_ID:-$$}"
fi
mkdir -p "${XDG_RUNTIME_DIR}" "${OUTPUT_ROOT}"
chmod 700 "${XDG_RUNTIME_DIR}"

NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX}" \
VULKANINFO_BIN="${VULKANINFO_BIN}" \
  "${PROJECT_ROOT}/scripts/check_official_carla_vulkan.sh"

carla_egg="${CARLA_PYTHON_EGG:-${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg}"
export PYTHONPATH="${carla_egg}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${PROJECT_ROOT}:${BENCH2DRIVE_ROOT}:${BENCH2DRIVE_ROOT}/leaderboard:${BENCH2DRIVE_ROOT}/scenario_runner:${BENCH2DRIVE_ZOO_ROOT}:${PYTHONPATH:-}"

# Bench2Drive 0.0.4's evaluator launches and owns its CARLA server.  Avoid a
# redundant outer server (and its GPU memory) for that code path, while keeping
# the legacy mode available for evaluator variants that expect an existing RPC
# endpoint.
if [[ "${BENCH2DRIVE_MANAGES_CARLA}" == "1" ]]; then
  echo "[INFO] Bench2Drive evaluator will launch and manage CARLA"
  exec env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    BENCH2DRIVE_ROOT="${BENCH2DRIVE_ROOT}" \
    BENCH2DRIVE_ZOO_ROOT="${BENCH2DRIVE_ZOO_ROOT}" \
    CARLA_ROOT="${CARLA_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    PORT="${PORT}" \
    TM_PORT="${TM_PORT}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    RUN_VULKAN_PRECHECK=0 \
    "${PROJECT_ROOT}/scripts/run_official_closedloop_smoke.sh"
fi

server_log="${CARLA_SERVER_LOG:-${OUTPUT_ROOT}/carla-server-${SLURM_JOB_ID:-$$}.log}"
client_log="${CARLA_CLIENT_LOG:-${OUTPUT_ROOT}/carla-client-${SLURM_JOB_ID:-$$}.log}"
read -r -a carla_extra_args <<<"${CARLA_EXTRA_ARGS}"
if [[ "${CARLA_QUALITY_LEVEL}" != "Epic" && "${CARLA_QUALITY_LEVEL}" != "Low" ]]; then
  echo "[FAIL] CARLA_QUALITY_LEVEL must be Epic or Low" >&2
  exit 2
fi

echo "[START] CARLA port=${PORT} quality=${CARLA_QUALITY_LEVEL} log=${server_log}"
"${CARLA_ROOT}/CarlaUE4.sh" \
  -vulkan -RenderOffScreen -nosound -quality-level="${CARLA_QUALITY_LEVEL}" \
  -carla-rpc-port="${PORT}" "${carla_extra_args[@]}" \
  >"${server_log}" 2>&1 &
server_pid=$!

cleanup() {
  kill "${server_pid}" >/dev/null 2>&1 || true
  wait "${server_pid}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

deadline=$((SECONDS + CARLA_STARTUP_TIMEOUT))
rpc_ready=0
while (( SECONDS < deadline )); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "[FAIL] CARLA server exited during startup" >&2
    tail -n 120 "${server_log}" >&2 || true
    exit 1
  fi
  if "${PYTHON_BIN}" -c '
import sys
import carla
client = carla.Client("127.0.0.1", int(sys.argv[1]))
client.set_timeout(3.0)
print(client.get_world().get_map().name)
' "${PORT}" >"${client_log}" 2>&1; then
    echo "[OK] CARLA RPC ready: $(tail -n 1 "${client_log}")"
    rpc_ready=1
    break
  fi
  sleep 2
done

if [[ "${rpc_ready}" != "1" ]]; then
  echo "[FAIL] CARLA RPC did not become ready" >&2
  tail -n 120 "${client_log}" >&2 || true
  tail -n 120 "${server_log}" >&2 || true
  exit 1
fi

env \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  BENCH2DRIVE_ROOT="${BENCH2DRIVE_ROOT}" \
  BENCH2DRIVE_ZOO_ROOT="${BENCH2DRIVE_ZOO_ROOT}" \
  CARLA_ROOT="${CARLA_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  PORT="${PORT}" \
  TM_PORT="${TM_PORT}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  RUN_VULKAN_PRECHECK=0 \
  "${PROJECT_ROOT}/scripts/run_official_closedloop_smoke.sh"

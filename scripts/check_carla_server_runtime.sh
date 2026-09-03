#!/usr/bin/env bash
set -euo pipefail

# Start a CARLA server, connect with its Python API, and report the loaded map.
# COMPAT_GLIBC_SYSROOT can point at a newer sysroot when the host glibc is too
# old for the official CARLA Python extension (for example, CentOS 7).

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CARLA_ROOT="${CARLA_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CARLA_HOST="${CARLA_HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
CARLA_MAP="${CARLA_MAP:-}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-120}"
CARLA_SERVER_LOG="${CARLA_SERVER_LOG:-/tmp/carla-server-${PORT}.log}"
CARLA_CLIENT_LOG="${CARLA_CLIENT_LOG:-/tmp/carla-client-${PORT}.log}"
CARLA_EXTRA_ARGS="${CARLA_EXTRA_ARGS:--stdout -FullStdOutLogOutput}"
RUN_VULKAN_PRECHECK="${RUN_VULKAN_PRECHECK:-1}"
NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX:-}"
VULKAN_LOADER_LIBDIR="${VULKAN_LOADER_LIBDIR:-}"
COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-}"
COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-}"

if [[ -z "${CARLA_ROOT}" || ! -x "${CARLA_ROOT}/CarlaUE4.sh" ]]; then
  echo "[FAIL] CARLA_ROOT does not contain CarlaUE4.sh: ${CARLA_ROOT:-<unset>}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[FAIL] Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

carla_egg="${CARLA_PYTHON_EGG:-${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg}"
if [[ ! -e "${carla_egg}" ]]; then
  echo "[FAIL] CARLA Python API is missing: ${carla_egg}" >&2
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
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

run_python() {
  if [[ -n "${COMPAT_GLIBC_SYSROOT}" ]]; then
    local loader="${COMPAT_GLIBC_SYSROOT}/lib64/ld-linux-x86-64.so.2"
    local python_prefix
    python_prefix="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
    if [[ ! -x "${loader}" ]]; then
      echo "[FAIL] compatibility loader is missing: ${loader}" >&2
      return 1
    fi
    "${loader}" --library-path \
      "${COMPAT_GLIBC_SYSROOT}/lib64:${COMPAT_GLIBC_SYSROOT}/usr/lib64:${python_prefix}/lib:${COMPAT_LIBRARY_PATH}:/lib64:/usr/lib64" \
      "${PYTHON_BIN}" "$@"
  else
    "${PYTHON_BIN}" "$@"
  fi
}

if [[ "${RUN_VULKAN_PRECHECK}" == "1" ]]; then
  NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX}" \
    "${PROJECT_ROOT}/scripts/check_official_carla_vulkan.sh"
fi

export PYTHONPATH="${carla_egg}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH:-}"

echo "[START] CARLA host=${CARLA_HOST} port=${PORT}"
read -r -a carla_extra_args <<<"${CARLA_EXTRA_ARGS}"
"${CARLA_ROOT}/CarlaUE4.sh" \
  -vulkan -RenderOffScreen -nosound -quality-level=Low \
  -carla-rpc-port="${PORT}" "${carla_extra_args[@]}" \
  >"${CARLA_SERVER_LOG}" 2>&1 &
server_pid=$!

cleanup() {
  kill "${server_pid}" >/dev/null 2>&1 || true
  wait "${server_pid}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

deadline=$((SECONDS + STARTUP_TIMEOUT))
while (( SECONDS < deadline )); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "[FAIL] CARLA server exited during startup" >&2
    set +e
    wait "${server_pid}"
    server_status=$?
    set -e
    echo "[FAIL] CARLA server exit status: ${server_status}" >&2
    tail -n 100 "${CARLA_SERVER_LOG}" >&2 || true
    exit 1
  fi

  if run_python -c '
import sys
import carla

client = carla.Client(sys.argv[1], int(sys.argv[2]))
client.set_timeout(3.0)
if sys.argv[3]:
    client.set_timeout(120.0)
    client.load_world(sys.argv[3])
    client.set_timeout(3.0)
world = client.get_world()
print(f"client_version={client.get_client_version()}")
print(f"server_version={client.get_server_version()}")
print(f"map={world.get_map().name}")
print(f"actors={len(world.get_actors())}")
print("CARLA_SERVER_SMOKE_OK")
' "${CARLA_HOST}" "${PORT}" "${CARLA_MAP}" >"${CARLA_CLIENT_LOG}" 2>&1; then
    cat "${CARLA_CLIENT_LOG}"
    exit 0
  fi
  sleep 2
done

echo "[FAIL] CARLA did not accept a client connection within ${STARTUP_TIMEOUT}s" >&2
tail -n 100 "${CARLA_CLIENT_LOG}" >&2 || true
tail -n 100 "${CARLA_SERVER_LOG}" >&2 || true
exit 1

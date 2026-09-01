#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-counterfactual_pairwise_native_train_seed20260828_r1}"
output_root="${OUTPUT_ROOT:-${asset_root}/observation_uq_v3/runs/${run_id}}"
python_runner="${PYTHON_RUNNER:-${project_root}/scripts/run_compat_python.sh}"
carla_root="${CARLA_ROOT:-${asset_root}/carla/CARLA_0.9.15}"
checkpoint="${ORION_CHECKPOINT:-${asset_root}/checkpoints/Orion.pth}"
port="${PORT:-$((24000 + ${SLURM_JOB_ID:-0} % 8000))}"
capture_route148="${output_root}/capture/Town10HD_Route148"
capture_route195="${output_root}/capture/Town03_Route195"
features="${output_root}/native_weather_train_features.pt"

required=(
  "${carla_root}/CarlaUE4.sh"
  "${checkpoint}"
  "${project_root}/configs/observation_uq_counterfactual_pairwise_native_v4.json"
  "${project_root}/configs/closedloop_uq_pilot/routes/route_148_nohazard.xml"
  "${project_root}/configs/closedloop_uq_pilot/routes/route_195_nohazard.xml"
  "${project_root}/scripts/capture_native_carla_weather.py"
  "${project_root}/scripts/extract_native_carla_weather_features.py"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[FAIL] missing pairwise native-training input: ${path}" >&2
    exit 1
  fi
done
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
export XDG_RUNTIME_DIR="/tmp/pairwise-native-${USER:-unknown}-${SLURM_JOB_ID:-$$}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

echo "RUN_ID=${run_id}"
echo "HOST=$(hostname)"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SCOPE=frozen_pairwise_native_training_capture_only"
echo "TRAIN_ROUTES=Town10HD/Route148,Town03/Route195"
echo "HELD_OUT_ROUTES_READ=0"
echo "PIXEL_CORRUPTION_GENERATOR=0"
echo "ADAPTER_TRAINING=0"
echo "ORION_FINETUNING=0"
echo "STAGE_B=0"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
sha256sum \
  "${project_root}/configs/observation_uq_counterfactual_pairwise_native_v4.json" \
  "${project_root}/scripts/capture_native_carla_weather.py" \
  "${project_root}/scripts/extract_native_carla_weather_features.py" \
  "${project_root}/configs/closedloop_uq_pilot/routes/route_148_nohazard.xml" \
  "${project_root}/configs/closedloop_uq_pilot/routes/route_195_nohazard.xml" \
  "${checkpoint}" > "${output_root}/source_sha256.txt"

NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX}" \
VULKANINFO_BIN="${VULKANINFO_BIN}" \
  "${project_root}/scripts/check_official_carla_vulkan.sh"

server_pid=""
server_live=0
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
cleanup() {
  stop_carla
}
trap cleanup EXIT

start_carla() {
  local server_port="$1"
  local server_log="$2"
  setsid "${carla_root}/CarlaUE4.sh" \
    -vulkan -RenderOffScreen -nosound -quality-level=Epic \
    -carla-rpc-port="${server_port}" -stdout -FullStdOutLogOutput \
    >"${server_log}" 2>&1 &
  server_pid=$!
  server_live=1
  local deadline=$((SECONDS + 180))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "[FAIL] CARLA exited during pairwise native startup" >&2
      tail -n 120 "${server_log}" >&2 || true
      return 1
    fi
    if "${python_runner}" -c '
import sys, carla
c=carla.Client("127.0.0.1", int(sys.argv[1])); c.set_timeout(3.0)
print(c.get_world().get_map().name)
' "${server_port}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "[FAIL] CARLA pairwise native server did not become ready" >&2
  tail -n 120 "${server_log}" >&2 || true
  return 1
}

start_carla "${port}" "${output_root}/carla-server-route148.log"
"${python_runner}" "${project_root}/scripts/capture_native_carla_weather.py" \
  --port "${port}" \
  --route "Town10HD/Route148=${project_root}/configs/closedloop_uq_pilot/routes/route_148_nohazard.xml" \
  --positions-per-route 16 \
  --renderer-quality Epic \
  --output "${capture_route148}"
stop_carla

route195_port=$((port + 1000))
start_carla "${route195_port}" "${output_root}/carla-server-route195.log"
"${python_runner}" "${project_root}/scripts/capture_native_carla_weather.py" \
  --port "${route195_port}" \
  --route "Town03/Route195=${project_root}/configs/closedloop_uq_pilot/routes/route_195_nohazard.xml" \
  --positions-per-route 16 \
  --renderer-quality Epic \
  --output "${capture_route195}"
stop_carla

"${python_runner}" "${project_root}/scripts/extract_native_carla_weather_features.py" \
  --capture-root "${capture_route148}" \
  --capture-root "${capture_route195}" \
  --config "${project_root}/adzoo/orion/configs/orion_stage3_agent.py" \
  --checkpoint "${checkpoint}" \
  --output "${features}" \
  --batch-size 2

sha256sum \
  "${features}" \
  "${capture_route148}/capture_manifest.json" \
  "${capture_route195}/capture_manifest.json" > "${output_root}/artifact_sha256.txt"
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "COUNTERFACTUAL_PAIRWISE_NATIVE_TRAIN_CAPTURE_OK=1"

#!/usr/bin/env bash
set -euo pipefail

# Four bounded Route151 visual captures under a shared low-sun route.  This
# uses CARLA BasicAgent and camera postprocessing only; ORION is never loaded.

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-route151_native_glare_bakeoff_v2}"
output_root="${OUTPUT_ROOT:-${asset_root}/scenario_factory/glare_bakeoffs/${run_id}}"
protocol="${project_root}/configs/glare_visual_bakeoff_route151_v1.json"
source_route="${project_root}/configs/closedloop_scenario_bank/routes/route_151_hazard.xml"
derived_route="${output_root}/inputs/route_151_native_low_sun.xml"
route_manifest="${output_root}/inputs/route_derivation.json"
python_runner="${project_root}/scripts/run_compat_python.sh"
carla_root="${asset_root}/carla/CARLA_0.9.15"
job_tag="${SLURM_JOB_ID:-$$}"
port_slot=$((job_tag % 1000))
port_offset=$((port_slot * 10))
port="${PORT:-$((22000 + port_offset))}"
tm_port="${TM_PORT:-$((42000 + port_offset))}"
server_log="${output_root}/logs/carla-server.log"
automold_path="${asset_root}/external/3D_Corruptions_AD-48c23f77/Automold.py"

for prerequisite in \
  "${protocol}" "${source_route}" \
  "${project_root}/team_code/glare_capture_agent.py" \
  "${project_root}/scripts/prepare_native_glare_route.py" \
  "${project_root}/scripts/validate_native_glare_capture.py" \
  "${project_root}/scripts/analyze_native_glare_bakeoff.py" \
  "${project_root}/scripts/render_glare_method_bakeoff.py" \
  "${automold_path}"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "[FAIL] missing native glare prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${output_root}" ]]; then
  echo "[FAIL] refusing to reuse output root: ${output_root}" >&2
  exit 1
fi
mkdir -p "${output_root}/inputs" "${output_root}/logs"

export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
export NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX:-${asset_root}/nvidia/nvidia_driver-linux-x86_64-535.104.05-archive}"
export VULKAN_LOADER_LIBDIR="${VULKAN_LOADER_LIBDIR:-${asset_root}/envs/vulkan-1.3.250/lib}"
export VULKANINFO_BIN="${VULKANINFO_BIN:-${asset_root}/envs/vulkan-1.3.250/bin/vulkaninfo}"
export LD_LIBRARY_PATH="${VULKAN_LOADER_LIBDIR}:${NVIDIA_RUNTIME_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${NVIDIA_RUNTIME_PREFIX}/etc/nvidia_icd.json"
export XDG_RUNTIME_DIR="/tmp/native-glare-bakeoff-${USER:-unknown}-${job_tag}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"
export PYTHONPATH="${carla_root}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${asset_root}/Bench2Drive:${asset_root}/Bench2Drive/leaderboard:${asset_root}/Bench2Drive/scenario_runner:${asset_root}/Bench2DriveZoo:${PYTHONPATH:-}"

"${python_runner}" "${project_root}/scripts/prepare_native_glare_route.py" \
  --source-route "${source_route}" \
  --protocol "${protocol}" \
  --output-route "${derived_route}" \
  --manifest "${route_manifest}"

echo "RUN_ID=${run_id}"
echo "SCOPE=route151_native_glare_visual_bakeoff_only"
echo "PROFILES=clean,light,medium,heavy"
echo "CONTROLLER=carla_basic_agent"
echo "ORION_LOAD=0"
echo "ADAPTER_TRAINING=0"
echo "COLLISION_PARAMETER_SELECTION=0"

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
    echo "[FAIL] CARLA exited during native glare startup" >&2
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
  echo "[FAIL] CARLA native glare server did not become ready" >&2
  exit 1
fi

for profile in clean light medium heavy; do
  profile_root="${output_root}/captures/${profile}"
  mkdir -p "${profile_root}"
  echo "[PROFILE_START] ${profile}"
  if ! timeout --signal=TERM --kill-after=30s 300s env \
    PROJECT_ROOT="${project_root}" \
    BENCH2DRIVE_ROOT="${asset_root}/Bench2Drive" \
    BENCH2DRIVE_ZOO_ROOT="${asset_root}/Bench2DriveZoo" \
    CARLA_ROOT="${carla_root}" \
    PYTHON_BIN="${python_runner}" \
    PORT="${port}" TM_PORT="${tm_port}" \
    TEAM_AGENT_PATH="${project_root}/team_code/glare_capture_agent.py" \
    AGENT_CONFIG_PATH="${protocol}" \
    BASE_CHECKPOINT_PATH="/dev/null" \
    ROUTE_SPLIT_OVERRIDE="${derived_route}" \
    OUTPUT_ROOT="${profile_root}" \
    RUN_VULKAN_PRECHECK=0 \
    BENCH2DRIVE_EXTERNAL_CARLA=1 \
    RESUME=False \
    ALGO=glare_capture \
    PLANNER_TYPE=npc \
    GLARE_CAPTURE_PROFILE="${profile}" \
    GLARE_CAPTURE_STRIDE=5 \
    GLARE_CAPTURE_FINISH_EXTENSION_M=6 \
    ORION_CARLA_RPC_TIMEOUT_SECONDS=90 \
    ORION_SCENARIO_TRACEBACK_INTERVAL_SECONDS=45 \
    "${project_root}/scripts/run_official_closedloop_smoke.sh" \
    >"${output_root}/logs/${profile}-evaluator.log" 2>&1; then
    echo "[FAIL] native glare evaluator failed for ${profile}" >&2
    tail -n 160 "${output_root}/logs/${profile}-evaluator.log" >&2 || true
    exit 1
  fi
  "${python_runner}" "${project_root}/scripts/validate_native_glare_capture.py" \
    --root "${profile_root}" \
    --protocol "${protocol}" \
    --profile "${profile}" \
    --minimum-frames 20
  echo "[PROFILE_OK] ${profile}"
done

"${python_runner}" "${project_root}/scripts/analyze_native_glare_bakeoff.py" \
  --root "${output_root}" \
  --protocol "${protocol}" \
  --output "${output_root}/visual_analysis"
"${python_runner}" "${project_root}/scripts/render_glare_method_bakeoff.py" \
  --root "${output_root}" \
  --protocol "${protocol}" \
  --automold "${automold_path}" \
  --output "${output_root}/method_bakeoff"

stop_carla
sha256sum \
  "${project_root}/team_code/glare_capture_agent.py" \
  "${project_root}/scripts/prepare_native_glare_route.py" \
  "${project_root}/scripts/validate_native_glare_capture.py" \
  "${project_root}/scripts/analyze_native_glare_bakeoff.py" \
  "${project_root}/scripts/render_glare_method_bakeoff.py" \
  "${automold_path}" \
  "${protocol}" "${derived_route}" \
  >"${output_root}/artifact_sha256.txt"
echo "NATIVE_GLARE_BAKEOFF_CAPTURE_OK=1"
echo "GLARE_METHOD_VISUAL_BAKEOFF_RENDERED=1"

#!/usr/bin/env bash
set -euo pipefail

# No-ORION Route151 none/medium native-motion-blur clean revalidation.

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-route151_native_motion_blur_clean_revalidation_v1}"
output_root="${OUTPUT_ROOT:-${asset_root}/scenario_factory/corruption_hardcase_diagnostics/${run_id}}"
protocol="${project_root}/configs/scenario_factory/corruption_hardcase_native_motion_blur_revalidation_route151_v1.json"
source_route="${project_root}/configs/closedloop_scenario_bank/routes/route_151_hazard.xml"
gate_config="${project_root}/configs/scenario_factory/corruption_hardcase_clean_render_artifact_gate_v1.json"
python_runner="${project_root}/scripts/run_compat_python.sh"
carla_root="${asset_root}/carla/CARLA_0.9.15"
job_tag="${SLURM_JOB_ID:-$$}"
port_slot=$((job_tag % 1000))
port_offset=$((port_slot * 10))
port="${PORT:-$((25000 + port_offset))}"
tm_port="${TM_PORT:-$((45000 + port_offset))}"

if [[ -e "${output_root}" ]]; then
  echo "[FAIL] refusing to reuse native-motion-blur revalidation root: ${output_root}" >&2
  exit 1
fi
for prerequisite in \
  "${protocol}" "${source_route}" "${gate_config}" \
  "${project_root}/team_code/glare_capture_agent.py" \
  "${project_root}/team_code/orion_native_motion_blur.py" \
  "${project_root}/scripts/evaluate_clean_render_artifacts.py" \
  "${project_root}/scripts/render_native_motion_blur_clean_revalidation.py"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "[FAIL] missing native-motion-blur prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
mkdir -p "${output_root}/logs" "${output_root}/quality_audit"

export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
export NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX:-${asset_root}/nvidia/nvidia_driver-linux-x86_64-535.104.05-archive}"
export VULKAN_LOADER_LIBDIR="${VULKAN_LOADER_LIBDIR:-${asset_root}/envs/vulkan-1.3.250/lib}"
export VULKANINFO_BIN="${VULKANINFO_BIN:-${asset_root}/envs/vulkan-1.3.250/bin/vulkaninfo}"
export LD_LIBRARY_PATH="${VULKAN_LOADER_LIBDIR}:${NVIDIA_RUNTIME_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${NVIDIA_RUNTIME_PREFIX}/etc/nvidia_icd.json"
export PYTHONPATH="${carla_root}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${asset_root}/Bench2Drive:${asset_root}/Bench2Drive/leaderboard:${asset_root}/Bench2Drive/scenario_runner:${asset_root}/Bench2DriveZoo:${PYTHONPATH:-}"

"${python_runner}" - "${protocol}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["schema"] == "orion.native_motion_blur_clean_revalidation_protocol.v1"
assert [row["profile"] for row in p["conditions"]] == ["none", "medium"]
assert p["capture_contract"]["fresh_carla_server_per_profile"] is True
assert p["capture_contract"]["orion_loaded"] is False
assert p["locks"]["orion_screen"] is False
print("NATIVE_MOTION_BLUR_REVALIDATION_PREFLIGHT_OK=1")
PY

echo "RUN_ID=${run_id}"
echo "SCOPE=route151_native_motion_blur_visual_revalidation_only"
echo "ORION_LOAD=0"
echo "PROFILES=none medium"
echo "FRESH_CARLA_SERVER_PER_PROFILE=1"

server_pid=""
server_live=0
stop_carla() {
  if [[ "${server_live}" != "1" ]]; then return; fi
  if [[ "${server_pid}" =~ ^[0-9]+$ && "${server_pid}" -gt 1 ]]; then
    kill -TERM -- "-${server_pid}" >/dev/null 2>&1 || true
    for _ in {1..20}; do
      if ! kill -0 -- "-${server_pid}" >/dev/null 2>&1; then break; fi
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

for profile in none medium; do
  export XDG_RUNTIME_DIR="/tmp/native-blur-revalidation-${USER:-unknown}-${job_tag}-${profile}"
  mkdir -p "${XDG_RUNTIME_DIR}"
  chmod 700 "${XDG_RUNTIME_DIR}"
  server_log="${output_root}/logs/${profile}-carla-server.log"
  evaluator_log="${output_root}/logs/${profile}-evaluator.log"
  profile_root="${output_root}/captures/${profile}"
  mkdir -p "${profile_root}"
  echo "[PROFILE_START] ${profile}"
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
      echo "[FAIL] CARLA exited during ${profile} startup" >&2
      tail -n 120 "${server_log}" >&2 || true
      exit 1
    fi
    if "${python_runner}" -c '
import sys, carla
c=carla.Client("127.0.0.1", int(sys.argv[1])); c.set_timeout(3.0)
print(c.get_world().get_map().name)
' "${port}" >/dev/null 2>&1; then
      ready=1; break
    fi
    sleep 2
  done
  if [[ "${ready}" != "1" ]]; then
    echo "[FAIL] CARLA did not become ready for ${profile}" >&2
    exit 1
  fi
  if ! timeout --signal=TERM --kill-after=30s 420s env \
    PROJECT_ROOT="${project_root}" \
    BENCH2DRIVE_ROOT="${asset_root}/Bench2Drive" \
    BENCH2DRIVE_ZOO_ROOT="${asset_root}/Bench2DriveZoo" \
    CARLA_ROOT="${carla_root}" \
    PYTHON_BIN="${python_runner}" \
    PORT="${port}" TM_PORT="${tm_port}" \
    TEAM_AGENT_PATH="${project_root}/team_code/glare_capture_agent.py" \
    AGENT_CONFIG_PATH="${protocol}" \
    BASE_CHECKPOINT_PATH="/dev/null" \
    ROUTE_SPLIT_OVERRIDE="${source_route}" \
    OUTPUT_ROOT="${profile_root}" \
    RUN_VULKAN_PRECHECK=0 \
    BENCH2DRIVE_EXTERNAL_CARLA=1 \
    RESUME=False \
    ALGO=native_blur_revalidation \
    PLANNER_TYPE=npc \
    HARDCAPTURE_FAMILY=native_motion_blur \
    MOTION_BLUR_CAPTURE_PROFILE="${profile}" \
    GLARE_CAPTURE_STRIDE=1 \
    HARDCAPTURE_START_PROGRESS=0.36 \
    HARDCAPTURE_END_PROGRESS=0.43 \
    GLARE_CAPTURE_FINISH_EXTENSION_M=6 \
    ORION_CARLA_RPC_TIMEOUT_SECONDS=90 \
    ORION_SCENARIO_TRACEBACK_INTERVAL_SECONDS=45 \
    "${project_root}/scripts/run_official_closedloop_smoke.sh" \
    >"${evaluator_log}" 2>&1; then
    echo "[FAIL] native-motion-blur capture failed for ${profile}" >&2
    tail -n 160 "${evaluator_log}" >&2 || true
    exit 1
  fi
  stop_carla
  echo "[PROFILE_OK] ${profile}"
done

for profile in none medium; do
  "${COMPAT_PYTHON_BIN}" "${project_root}/scripts/evaluate_clean_render_artifacts.py" \
    --label "route151_native_motion_blur_${profile}" \
    --glob "${output_root}/captures/${profile}/records_*/rgb_front/*.png" \
    --gate-config "${gate_config}" \
    --output "${output_root}/quality_audit/${profile}.json"
done

"${COMPAT_PYTHON_BIN}" "${project_root}/scripts/render_native_motion_blur_clean_revalidation.py" \
  --root "${output_root}" \
  --protocol "${protocol}" \
  --none-audit "${output_root}/quality_audit/none.json" \
  --medium-audit "${output_root}/quality_audit/medium.json" \
  --output "${output_root}/visual_review"

sha256sum \
  "${project_root}/team_code/glare_capture_agent.py" \
  "${project_root}/team_code/orion_native_motion_blur.py" \
  "${project_root}/scripts/evaluate_clean_render_artifacts.py" \
  "${project_root}/scripts/render_native_motion_blur_clean_revalidation.py" \
  "${protocol}" "${gate_config}" "${source_route}" \
  >"${output_root}/artifact_sha256.txt"
echo "NATIVE_MOTION_BLUR_CLEAN_REVALIDATION_OK=1"

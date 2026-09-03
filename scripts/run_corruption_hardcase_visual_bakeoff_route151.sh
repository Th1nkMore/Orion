#!/usr/bin/env bash
set -euo pipefail

# No-ORION Route151 capture/render job for stale, waterdrop, and native blur.

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-route151_corruption_hardcase_visual_bakeoff_v1}"
output_root="${OUTPUT_ROOT:-${asset_root}/scenario_factory/corruption_hardcase_bakeoffs/${run_id}}"
protocol="${project_root}/configs/scenario_factory/corruption_hardcase_visual_bakeoff_route151_v1.json"
source_route="${project_root}/configs/closedloop_scenario_bank/routes/route_151_hazard.xml"
python_runner="${project_root}/scripts/run_compat_python.sh"
carla_root="${asset_root}/carla/CARLA_0.9.15"
job_tag="${SLURM_JOB_ID:-$$}"
recovery_mode="${HARDCAPTURE_RECOVERY:-0}"
port_slot=$((job_tag % 1000))
port_offset=$((port_slot * 10))
port="${PORT:-$((23000 + port_offset))}"
tm_port="${TM_PORT:-$((43000 + port_offset))}"
if [[ "${recovery_mode}" == "1" ]]; then
  server_log="${output_root}/logs/carla-server-recovery-${job_tag}.log"
else
  server_log="${output_root}/logs/carla-server.log"
fi

for prerequisite in \
  "${protocol}" "${source_route}" \
  "${project_root}/team_code/glare_capture_agent.py" \
  "${project_root}/team_code/orion_native_motion_blur.py" \
  "${project_root}/uq_estimator/lens_waterdrop.py" \
  "${project_root}/scripts/render_corruption_hardcase_bakeoff.py"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "[FAIL] missing hard-case bake-off prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${output_root}" ]]; then
  if [[ "${recovery_mode}" != "1" ]]; then
    echo "[FAIL] refusing to reuse hard-case bake-off root: ${output_root}" >&2
    exit 1
  fi
  if [[ "${HARDCAPTURE_PROFILES:-}" != "heavy" ]]; then
    echo "[FAIL] recovery mode permits only HARDCAPTURE_PROFILES=heavy" >&2
    exit 2
  fi
  for completed_profile in clean light medium; do
    completed_count=$(find "${output_root}/captures/${completed_profile}" -type f -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
    if [[ "${completed_count}" != "156" ]]; then
      echo "[FAIL] recovery prerequisite ${completed_profile} has ${completed_count} PNGs, expected 156" >&2
      exit 2
    fi
  done
  if find "${output_root}/captures/heavy" -type f -name '*.png' -print -quit 2>/dev/null | grep -q .; then
    echo "[FAIL] recovery refuses to overwrite an existing heavy visual capture" >&2
    exit 2
  fi
  failed_parent_job="${HARDCAPTURE_PARENT_JOB_ID:-unknown}"
  failed_heavy_archive="${output_root}/captures/heavy_failed_job${failed_parent_job}"
  if [[ -e "${failed_heavy_archive}" ]]; then
    echo "[FAIL] failed-heavy archive already exists: ${failed_heavy_archive}" >&2
    exit 2
  fi
  if [[ -d "${output_root}/captures/heavy" ]]; then
    mv "${output_root}/captures/heavy" "${failed_heavy_archive}"
  fi
  echo "RECOVERY_MODE=1"
  echo "FAILED_HEAVY_ARCHIVE=${failed_heavy_archive}"
elif [[ "${recovery_mode}" == "1" ]]; then
  echo "[FAIL] recovery mode requires an existing partial output root" >&2
  exit 2
fi
mkdir -p "${output_root}/logs"

export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
export NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX:-${asset_root}/nvidia/nvidia_driver-linux-x86_64-535.104.05-archive}"
export VULKAN_LOADER_LIBDIR="${VULKAN_LOADER_LIBDIR:-${asset_root}/envs/vulkan-1.3.250/lib}"
export VULKANINFO_BIN="${VULKANINFO_BIN:-${asset_root}/envs/vulkan-1.3.250/bin/vulkaninfo}"
export LD_LIBRARY_PATH="${VULKAN_LOADER_LIBDIR}:${NVIDIA_RUNTIME_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${NVIDIA_RUNTIME_PREFIX}/etc/nvidia_icd.json"
export XDG_RUNTIME_DIR="/tmp/hardcase-bakeoff-${USER:-unknown}-${job_tag}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"
export PYTHONPATH="${carla_root}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${asset_root}/Bench2Drive:${asset_root}/Bench2Drive/leaderboard:${asset_root}/Bench2Drive/scenario_runner:${asset_root}/Bench2DriveZoo:${PYTHONPATH:-}"

"${python_runner}" - "${protocol}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
assert value["schema"] == "orion.corruption_hardcase_visual_bakeoff.v1"
route = value["route"]
assert route["route_id"] == 151
assert route["capture_stride_simulation_steps"] == 1
assert route["capture_start_progress"] == 0.34
assert route["capture_end_progress"] == 0.50
assert value["acceptance"]["no_orion_loaded"] is True
assert value["acceptance"]["no_collision_outcome_used_for_parameter_selection"] is True
print("HARDCAPTURE_PROTOCOL_PREFLIGHT_OK=1")
PY

echo "RUN_ID=${run_id}"
echo "SCOPE=route151_stale_waterdrop_native_motion_blur_visual_only"
echo "ORION_LOAD=0"
echo "CAPTURE_HZ=20"
echo "CAPTURE_PROGRESS=0.34,0.50"
profile_spec="${HARDCAPTURE_PROFILES:-clean light medium heavy}"
read -r -a capture_profiles <<<"${profile_spec}"
if [[ "${#capture_profiles[@]}" -eq 0 ]]; then
  echo "[FAIL] HARDCAPTURE_PROFILES resolved to an empty list" >&2
  exit 2
fi
declare -A seen_profiles=()
for profile in "${capture_profiles[@]}"; do
  case "${profile}" in
    clean|light|medium|heavy) ;;
    *) echo "[FAIL] unsupported HARDCAPTURE_PROFILES entry: ${profile}" >&2; exit 2 ;;
  esac
  if [[ -n "${seen_profiles[${profile}]:-}" ]]; then
    echo "[FAIL] duplicate HARDCAPTURE_PROFILES entry: ${profile}" >&2
    exit 2
  fi
  seen_profiles["${profile}"]=1
done
echo "PROFILES=${capture_profiles[*]}"

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
    echo "[FAIL] CARLA exited during hard-case startup" >&2
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
  echo "[FAIL] CARLA hard-case server did not become ready" >&2
  exit 1
fi

for profile in "${capture_profiles[@]}"; do
  profile_root="${output_root}/captures/${profile}"
  if [[ "${recovery_mode}" == "1" ]]; then
    evaluator_log="${output_root}/logs/${profile}-recovery-${job_tag}-evaluator.log"
  else
    evaluator_log="${output_root}/logs/${profile}-evaluator.log"
  fi
  mkdir -p "${profile_root}"
  echo "[PROFILE_START] ${profile}"
  if ! timeout --signal=TERM --kill-after=30s 360s env \
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
    ALGO=hardcase_capture \
    PLANNER_TYPE=npc \
    HARDCAPTURE_FAMILY=native_motion_blur \
    MOTION_BLUR_CAPTURE_PROFILE="${profile}" \
    GLARE_CAPTURE_STRIDE=1 \
    HARDCAPTURE_START_PROGRESS=0.34 \
    HARDCAPTURE_END_PROGRESS=0.50 \
    GLARE_CAPTURE_FINISH_EXTENSION_M=6 \
    ORION_CARLA_RPC_TIMEOUT_SECONDS=90 \
    ORION_SCENARIO_TRACEBACK_INTERVAL_SECONDS=45 \
    "${project_root}/scripts/run_official_closedloop_smoke.sh" \
    >"${evaluator_log}" 2>&1; then
    echo "[FAIL] hard-case capture failed for ${profile}" >&2
    tail -n 160 "${evaluator_log}" >&2 || true
    exit 1
  fi
  echo "[PROFILE_OK] ${profile}"
done

"${python_runner}" "${project_root}/scripts/render_corruption_hardcase_bakeoff.py" \
  --root "${output_root}" \
  --protocol "${protocol}" \
  --output "${output_root}/visual_review"

stop_carla
sha256sum \
  "${project_root}/team_code/glare_capture_agent.py" \
  "${project_root}/team_code/orion_native_motion_blur.py" \
  "${project_root}/uq_estimator/temporal_corruptions.py" \
  "${project_root}/uq_estimator/lens_waterdrop.py" \
  "${project_root}/scripts/render_corruption_hardcase_bakeoff.py" \
  "${protocol}" "${source_route}" \
  >"${output_root}/artifact_sha256.txt"
echo "CORRUPTION_HARDCASE_VISUAL_BAKEOFF_OK=1"

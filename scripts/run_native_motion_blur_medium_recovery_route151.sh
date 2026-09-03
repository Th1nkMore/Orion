#!/usr/bin/env bash
set -euo pipefail

# One-shot, medium-only recovery after job 1115765 stalled before any capture.

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-route151_native_motion_blur_medium_recovery_v2}"
output_root="${OUTPUT_ROOT:-${asset_root}/scenario_factory/corruption_hardcase_diagnostics/${run_id}}"
protocol="${project_root}/configs/scenario_factory/corruption_hardcase_native_motion_blur_revalidation_route151_v1.json"
recovery_amendment="${project_root}/configs/scenario_factory/amendments/20260831_native_motion_blur_medium_recovery_prereg_v2.json"
source_route="${project_root}/configs/closedloop_scenario_bank/routes/route_151_hazard.xml"
gate_config="${project_root}/configs/scenario_factory/corruption_hardcase_clean_render_artifact_gate_v1.json"
clean_diagnostic_root="${asset_root}/scenario_factory/corruption_hardcase_diagnostics/route151_clean_render_diagnostic_v1"
clean_profile_root="${clean_diagnostic_root}/captures/none"
clean_calibration_audit="${clean_diagnostic_root}/quality_audit/diagnostic_good_none.json"
clean_audit="${output_root}/quality_audit/none.json"
python_runner="${project_root}/scripts/run_compat_python.sh"
carla_root="${asset_root}/carla/CARLA_0.9.15"
job_tag="${SLURM_JOB_ID:-$$}"
port_slot=$((job_tag % 1000))
port_offset=$((port_slot * 10))
port="${PORT:-$((26000 + port_offset))}"
tm_port="${TM_PORT:-$((46000 + port_offset))}"

if [[ -e "${output_root}" ]]; then
  echo "[FAIL] refusing to reuse recovery root: ${output_root}" >&2
  exit 1
fi
for prerequisite in \
  "${protocol}" "${recovery_amendment}" "${source_route}" "${gate_config}" \
  "${project_root}/team_code/glare_capture_agent.py" \
  "${project_root}/team_code/orion_native_motion_blur.py" \
  "${project_root}/scripts/evaluate_clean_render_artifacts.py" \
  "${project_root}/scripts/render_native_motion_blur_clean_revalidation.py" \
  "${clean_calibration_audit}"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "[FAIL] missing medium-recovery prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ ! -d "${clean_profile_root}" ]]; then
  echo "[FAIL] missing immutable clean profile root: ${clean_profile_root}" >&2
  exit 2
fi
mkdir -p "${output_root}/logs" "${output_root}/quality_audit" "${output_root}/captures/medium"

export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
export COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot}"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${asset_root}/envs/orion-cl/lib}"
export NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX:-${asset_root}/nvidia/nvidia_driver-linux-x86_64-535.104.05-archive}"
export VULKAN_LOADER_LIBDIR="${VULKAN_LOADER_LIBDIR:-${asset_root}/envs/vulkan-1.3.250/lib}"
export VULKANINFO_BIN="${VULKANINFO_BIN:-${asset_root}/envs/vulkan-1.3.250/bin/vulkaninfo}"
export LD_LIBRARY_PATH="${VULKAN_LOADER_LIBDIR}:${NVIDIA_RUNTIME_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${NVIDIA_RUNTIME_PREFIX}/etc/nvidia_icd.json"
export PYTHONPATH="${carla_root}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${asset_root}/Bench2Drive:${asset_root}/Bench2Drive/leaderboard:${asset_root}/Bench2Drive/scenario_runner:${asset_root}/Bench2DriveZoo:${PYTHONPATH:-}"

"${COMPAT_PYTHON_BIN}" "${project_root}/scripts/evaluate_clean_render_artifacts.py" \
  --label route151_immutable_none_medium_recovery \
  --glob "${clean_profile_root}/records_*/rgb_front/*.png" \
  --gate-config "${gate_config}" \
  --output "${clean_audit}"

clean_png_manifest_sha256="$({ find "${clean_profile_root}" -type f -name '*.png' -print0 | sort -z | xargs -0 sha256sum; } | sha256sum | awk '{print $1}')"
clean_trace="$(find "${clean_profile_root}" -type f -name capture_trace.jsonl -print -quit)"

"${python_runner}" - "${protocol}" "${recovery_amendment}" "${clean_audit}" "${clean_calibration_audit}" "${clean_trace}" "${clean_png_manifest_sha256}" <<'PY'
import hashlib, json, pathlib, sys
protocol = json.load(open(sys.argv[1]))
recovery = json.load(open(sys.argv[2]))
audit_path = pathlib.Path(sys.argv[3])
audit = json.loads(audit_path.read_text())
calibration_path = pathlib.Path(sys.argv[4])
trace_path = pathlib.Path(sys.argv[5])
png_manifest_sha256 = sys.argv[6]
assert protocol["schema"] == "orion.native_motion_blur_clean_revalidation_protocol.v1"
assert recovery["schema"] == "orion.native_motion_blur_medium_recovery_prereg.v2"
assert recovery["maximum_capture_attempts"] == 1
assert recovery["orion_loaded"] is False
assert audit["status"] == "passed_clean_render_artifact_gate"
assert audit["gate"]["passed"] is True
assert audit["gate"]["suspicious_frame_count"] == 0
assert audit["gate"]["frame_count"] == 34
assert hashlib.sha256(calibration_path.read_bytes()).hexdigest() == recovery["immutable_clean_reference"]["calibration_audit_sha256"]
assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == recovery["immutable_clean_reference"]["trace_sha256"]
assert png_manifest_sha256 == recovery["immutable_clean_reference"]["png_manifest_sha256"]
print("NATIVE_MOTION_BLUR_MEDIUM_RECOVERY_PREFLIGHT_OK=1")
PY

echo "RUN_ID=${run_id}"
echo "SCOPE=route151_native_motion_blur_medium_only_visual_recovery"
echo "ORION_LOAD=0"
echo "CLEAN_REFERENCE_REUSED=1"
echo "MAXIMUM_SUBMISSIONS=1"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "PREFLIGHT_ONLY_OK=1"
  exit 0
fi

export XDG_RUNTIME_DIR="/tmp/native-blur-medium-recovery-${USER:-unknown}-${job_tag}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"
server_log="${output_root}/logs/medium-carla-server.log"
evaluator_log="${output_root}/logs/medium-evaluator.log"
profile_root="${output_root}/captures/medium"

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
    echo "[FAIL] CARLA exited during medium recovery startup" >&2
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
  echo "[FAIL] CARLA did not become ready for medium recovery" >&2
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
  ALGO=native_blur_medium_recovery \
  PLANNER_TYPE=npc \
  HARDCAPTURE_FAMILY=native_motion_blur \
  MOTION_BLUR_CAPTURE_PROFILE=medium \
  GLARE_CAPTURE_STRIDE=1 \
  HARDCAPTURE_START_PROGRESS=0.36 \
  HARDCAPTURE_END_PROGRESS=0.43 \
  GLARE_CAPTURE_FINISH_EXTENSION_M=6 \
  ORION_CARLA_RPC_TIMEOUT_SECONDS=90 \
  ORION_SCENARIO_TRACEBACK_INTERVAL_SECONDS=45 \
  "${project_root}/scripts/run_official_closedloop_smoke.sh" \
  >"${evaluator_log}" 2>&1; then
  echo "[FAIL] native-motion-blur medium recovery failed" >&2
  tail -n 160 "${evaluator_log}" >&2 || true
  exit 1
fi
stop_carla

"${COMPAT_PYTHON_BIN}" "${project_root}/scripts/evaluate_clean_render_artifacts.py" \
  --label route151_native_motion_blur_medium_recovery \
  --glob "${profile_root}/records_*/rgb_front/*.png" \
  --gate-config "${gate_config}" \
  --output "${output_root}/quality_audit/medium.json"

"${COMPAT_PYTHON_BIN}" "${project_root}/scripts/render_native_motion_blur_clean_revalidation.py" \
  --root "${output_root}" \
  --protocol "${protocol}" \
  --none-root "${clean_profile_root}" \
  --medium-root "${profile_root}" \
  --none-audit "${clean_audit}" \
  --medium-audit "${output_root}/quality_audit/medium.json" \
  --output "${output_root}/visual_review"

sha256sum \
  "${project_root}/team_code/glare_capture_agent.py" \
  "${project_root}/team_code/orion_native_motion_blur.py" \
  "${project_root}/scripts/evaluate_clean_render_artifacts.py" \
  "${project_root}/scripts/render_native_motion_blur_clean_revalidation.py" \
  "${protocol}" "${recovery_amendment}" "${gate_config}" "${source_route}" \
  >"${output_root}/artifact_sha256.txt"
echo "NATIVE_MOTION_BLUR_MEDIUM_RECOVERY_OK=1"

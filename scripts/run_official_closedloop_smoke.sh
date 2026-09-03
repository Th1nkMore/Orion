#!/usr/bin/env bash
set -euo pipefail

# Launch a one-route official Bench2Drive evaluation against the ORION agent.
# This script standardizes the smoke-test command shape but does not start the
# CARLA server; launch CARLA separately on the same PORT before use.
#
# We call `leaderboard_evaluator.py` directly instead of forwarding through the
# upstream `run_evaluation.sh`, because that helper hardcodes the py3.7 CARLA
# egg path while our isolated closed-loop env is Python 3.8.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
BENCH2DRIVE_ROOT="${BENCH2DRIVE_ROOT:-${PROJECT_ROOT}/Bench2Drive}"
BENCH2DRIVE_ZOO_ROOT="${BENCH2DRIVE_ZOO_ROOT:-${PROJECT_ROOT}/Bench2DriveZoo}"
CARLA_ROOT="${CARLA_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PORT="${PORT:-30000}"
TM_PORT="${TM_PORT:-50000}"
GPU_RANK="${GPU_RANK:-0}"
TASK_ID="${TASK_ID:-0}"
PLANNER_TYPE="${PLANNER_TYPE:-traj}"
ALGO="${ALGO:-orion}"
BASE_ROUTES="${BASE_ROUTES:-${BENCH2DRIVE_ROOT}/leaderboard/data/bench2drive220}"
ROUTE_SPLIT_OVERRIDE="${ROUTE_SPLIT_OVERRIDE:-}"
AGENT_CONFIG_PATH="${AGENT_CONFIG_PATH:-${PROJECT_ROOT}/adzoo/orion/configs/orion_stage3_agent.py}"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-${PROJECT_ROOT}/ckpts/Orion.pth}"
FILM_CHECKPOINT_PATH="${FILM_CHECKPOINT_PATH:-}"
TEAM_AGENT_PATH="${TEAM_AGENT_PATH:-${PROJECT_ROOT}/team_code/orion_b2d_agent.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/closedloop_official_smoke}"
RUN_VULKAN_PRECHECK="${RUN_VULKAN_PRECHECK:-1}"
NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX:-}"
VULKANINFO_BIN="${VULKANINFO_BIN:-vulkaninfo}"
BENCH2DRIVE_EXTERNAL_CARLA="${BENCH2DRIVE_EXTERNAL_CARLA:-0}"
CLOSEDLOOP_PROGRESS_WATCHDOG_SECONDS="${CLOSEDLOOP_PROGRESS_WATCHDOG_SECONDS:-0}"
CLOSEDLOOP_PROGRESS_STARTUP_GRACE_SECONDS="${CLOSEDLOOP_PROGRESS_STARTUP_GRACE_SECONDS:-900}"
CLOSEDLOOP_PROGRESS_WATCHDOG_POLL_SECONDS="${CLOSEDLOOP_PROGRESS_WATCHDOG_POLL_SECONDS:-15}"

if [[ -z "${CARLA_ROOT}" ]]; then
  echo "[FAIL] CARLA_ROOT must be set to the extracted CARLA runtime" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[FAIL] missing python binary: ${PYTHON_BIN}" >&2
  exit 1
fi

TEAM_CONFIG="${AGENT_CONFIG_PATH}+${BASE_CHECKPOINT_PATH}"
if [[ -n "${FILM_CHECKPOINT_PATH}" ]]; then
  TEAM_CONFIG="${TEAM_CONFIG}+${FILM_CHECKPOINT_PATH}"
fi

ROUTE_SPLIT="${ROUTE_SPLIT_OVERRIDE:-${BASE_ROUTES}_${TASK_ID}_${ALGO}_${PLANNER_TYPE}.xml}"
CHECKPOINT_ENDPOINT="${OUTPUT_ROOT}/eval_${ALGO}_${PLANNER_TYPE}_${TASK_ID}.json"
SAVE_PATH="${OUTPUT_ROOT}/records_${ALGO}_${PLANNER_TYPE}_${TASK_ID}"

mkdir -p "${OUTPUT_ROOT}" "${SAVE_PATH}"
export SAVE_PATH

if [[ "${RUN_VULKAN_PRECHECK}" == "1" ]]; then
  NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX}" \
  VULKANINFO_BIN="${VULKANINFO_BIN}" \
  "${PROJECT_ROOT}/scripts/check_official_carla_vulkan.sh"
fi

if [[ ! -f "${ROUTE_SPLIT}" ]]; then
  echo "[SETUP] create single-route splits from ${BASE_ROUTES}.xml"
  "${PYTHON_BIN}" "${BENCH2DRIVE_ROOT}/tools/split_xml.py" "${BASE_ROUTES}" 220 "${ALGO}" "${PLANNER_TYPE}"
fi

export CARLA_ROOT
export CARLA_SERVER="${CARLA_ROOT}/CarlaUE4.sh"
export SCENARIO_RUNNER_ROOT="${BENCH2DRIVE_ROOT}/scenario_runner"
export LEADERBOARD_ROOT="${BENCH2DRIVE_ROOT}/leaderboard"
export CHALLENGE_TRACK_CODENAME="SENSORS"
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export RESUME="${RESUME:-True}"
export IS_BENCH2DRIVE=True
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${PROJECT_ROOT}:${BENCH2DRIVE_ROOT}:${BENCH2DRIVE_ROOT}/leaderboard:${BENCH2DRIVE_ROOT}/scenario_runner:${BENCH2DRIVE_ZOO_ROOT}:${PYTHONPATH:-}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

echo "== Official Closed-Loop Smoke =="
echo "BENCH2DRIVE_ROOT=${BENCH2DRIVE_ROOT}"
echo "BENCH2DRIVE_ZOO_ROOT=${BENCH2DRIVE_ZOO_ROOT}"
echo "CARLA_ROOT=${CARLA_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "PORT=${PORT}"
echo "TM_PORT=${TM_PORT}"
echo "GPU_RANK=${GPU_RANK}"
echo "ROUTE_SPLIT=${ROUTE_SPLIT}"
echo "TEAM_AGENT_PATH=${TEAM_AGENT_PATH}"
echo "TEAM_CONFIG=${TEAM_CONFIG}"
echo "CHECKPOINT_ENDPOINT=${CHECKPOINT_ENDPOINT}"
echo "SAVE_PATH=${SAVE_PATH}"

cd "${BENCH2DRIVE_ROOT}"

EVALUATOR_ENTRY="${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py"
if [[ "${BENCH2DRIVE_EXTERNAL_CARLA}" == "1" ]]; then
  EVALUATOR_ENTRY="${PROJECT_ROOT}/scripts/run_bench2drive_external_server.py"
  echo "[INFO] Evaluator will reuse the externally managed CARLA server"
fi

# Slurm may expose a physical device through an allocation-specific
# CUDA_VISIBLE_DEVICES value.  Preserve that binding and use GPU_RANK only as
# the fallback used by direct, non-Slurm launches.
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-${GPU_RANK}}"
evaluator_command=(
  "${PYTHON_BIN}"
  "${EVALUATOR_ENTRY}"
  --routes="${ROUTE_SPLIT}"
  --repetitions="${REPETITIONS}"
  --track="${CHALLENGE_TRACK_CODENAME}"
  --checkpoint="${CHECKPOINT_ENDPOINT}"
  --agent="${TEAM_AGENT_PATH}"
  --agent-config="${TEAM_CONFIG}"
  --debug="${DEBUG_CHALLENGE}"
  --resume="${RESUME}"
  --port="${PORT}"
  --traffic-manager-port="${TM_PORT}"
  --gpu-rank="${GPU_RANK}"
  --timeout="${ORION_CARLA_RPC_TIMEOUT_SECONDS:-300}"
)

if (( CLOSEDLOOP_PROGRESS_WATCHDOG_SECONDS <= 0 )); then
  exec env CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}" "${evaluator_command[@]}"
fi

echo "[INFO] Closed-loop progress watchdog enabled: stall=${CLOSEDLOOP_PROGRESS_WATCHDOG_SECONDS}s startup_grace=${CLOSEDLOOP_PROGRESS_STARTUP_GRACE_SECONDS}s poll=${CLOSEDLOOP_PROGRESS_WATCHDOG_POLL_SECONDS}s"
rm -f "${OUTPUT_ROOT}/runtime_progress_watchdog.json"
env CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}" "${evaluator_command[@]}" &
evaluator_pid=$!
watchdog_start_epoch=$(date +%s)

while kill -0 "${evaluator_pid}" 2>/dev/null; do
  sleep "${CLOSEDLOOP_PROGRESS_WATCHDOG_POLL_SECONDS}"
  kill -0 "${evaluator_pid}" 2>/dev/null || break

  now_epoch=$(date +%s)
  trace_path=$(find "${SAVE_PATH}" -type f -name control_trace.jsonl -print -quit 2>/dev/null || true)
  watchdog_reason=""
  trace_mtime_epoch=""
  if [[ -n "${trace_path}" ]]; then
    trace_mtime_epoch=$(python3 -c 'import os, sys; print(int(os.path.getmtime(sys.argv[1])))' "${trace_path}")
    if (( now_epoch - trace_mtime_epoch > CLOSEDLOOP_PROGRESS_WATCHDOG_SECONDS )); then
      watchdog_reason="control_trace_stalled"
    fi
  elif (( now_epoch - watchdog_start_epoch > CLOSEDLOOP_PROGRESS_STARTUP_GRACE_SECONDS )); then
    watchdog_reason="control_trace_not_created_within_startup_grace"
  fi

  if [[ -z "${watchdog_reason}" ]]; then
    continue
  fi

  echo "[FAIL] Closed-loop progress watchdog: reason=${watchdog_reason} trace=${trace_path:-<missing>}" >&2
  python3 - \
    "${OUTPUT_ROOT}/runtime_progress_watchdog.json" \
    "${watchdog_reason}" \
    "${trace_path}" \
    "${watchdog_start_epoch}" \
    "${now_epoch}" \
    "${trace_mtime_epoch}" \
    "${CLOSEDLOOP_PROGRESS_WATCHDOG_SECONDS}" \
    "${CLOSEDLOOP_PROGRESS_STARTUP_GRACE_SECONDS}" <<'PY'
import json
import sys

(
    output_path,
    reason,
    trace_path,
    start_epoch,
    detected_epoch,
    trace_mtime_epoch,
    stall_seconds,
    startup_grace_seconds,
) = sys.argv[1:]
payload = {
    "schema": "orion.closedloop_runtime_progress_watchdog.v1",
    "reason": reason,
    "trace_path": trace_path or None,
    "watchdog_start_epoch": int(start_epoch),
    "detected_epoch": int(detected_epoch),
    "trace_mtime_epoch": int(trace_mtime_epoch) if trace_mtime_epoch else None,
    "stall_threshold_seconds": int(stall_seconds),
    "startup_grace_seconds": int(startup_grace_seconds),
    "scientific_classification": "runtime_environment_invalid",
}
with open(output_path, "w") as outfile:
    json.dump(payload, outfile, indent=2, sort_keys=True)
    outfile.write("\n")
PY
  kill -TERM "${evaluator_pid}" 2>/dev/null || true
  sleep 10
  kill -KILL "${evaluator_pid}" 2>/dev/null || true
  wait "${evaluator_pid}" 2>/dev/null || true
  exit 124
done

wait "${evaluator_pid}"

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
AGENT_CONFIG_PATH="${AGENT_CONFIG_PATH:-${PROJECT_ROOT}/adzoo/orion/configs/orion_stage3_agent.py}"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-${PROJECT_ROOT}/ckpts/Orion.pth}"
FILM_CHECKPOINT_PATH="${FILM_CHECKPOINT_PATH:-}"
TEAM_AGENT_PATH="${TEAM_AGENT_PATH:-${PROJECT_ROOT}/team_code/orion_b2d_agent.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/closedloop_official_smoke}"
RUN_VULKAN_PRECHECK="${RUN_VULKAN_PRECHECK:-1}"
NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX:-}"
VULKANINFO_BIN="${VULKANINFO_BIN:-vulkaninfo}"

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

ROUTE_SPLIT="${BASE_ROUTES}_${TASK_ID}_${ALGO}_${PLANNER_TYPE}.xml"
CHECKPOINT_ENDPOINT="${OUTPUT_ROOT}/eval_${ALGO}_${PLANNER_TYPE}_${TASK_ID}.json"
SAVE_PATH="${OUTPUT_ROOT}/records_${ALGO}_${PLANNER_TYPE}_${TASK_ID}"

mkdir -p "${OUTPUT_ROOT}" "${SAVE_PATH}"

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
export RESUME=True
export IS_BENCH2DRIVE=True
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

CUDA_VISIBLE_DEVICES="${GPU_RANK}" "${PYTHON_BIN}" \
  "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
  --routes="${ROUTE_SPLIT}" \
  --repetitions="${REPETITIONS}" \
  --track="${CHALLENGE_TRACK_CODENAME}" \
  --checkpoint="${CHECKPOINT_ENDPOINT}" \
  --agent="${TEAM_AGENT_PATH}" \
  --agent-config="${TEAM_CONFIG}" \
  --debug="${DEBUG_CHALLENGE}" \
  --resume="${RESUME}" \
  --port="${PORT}" \
  --traffic-manager-port="${TM_PORT}" \
  --gpu-rank="${GPU_RANK}"

#!/usr/bin/env bash
set -euo pipefail

# Validate whether the current machine is ready for paper-aligned official
# CARLA closed-loop evaluation. This is a pure environment check script.
#
# Usage:
#   bash scripts/check_official_closedloop_env.sh
#   PROJECT_ROOT=/root/Orion bash scripts/check_official_closedloop_env.sh

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
BENCH2DRIVE_ROOT="${BENCH2DRIVE_ROOT:-${PROJECT_ROOT}/Bench2DriveZoo}"
CARLA_ROOT="${CARLA_ROOT:-}"
ROUTES_PATH="${ROUTES_PATH:-}"
SCENARIOS_PATH="${SCENARIOS_PATH:-}"

status=0

check_path() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    printf '[OK] %s: %s\n' "${label}" "${path}"
  else
    printf '[MISS] %s: %s\n' "${label}" "${path}"
    status=1
  fi
}

check_python_module() {
  local module="$1"
  if python - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("${module}")
PY
  then
    printf '[OK] python module: %s\n' "${module}"
  else
    printf '[MISS] python module: %s\n' "${module}"
    status=1
  fi
}

echo "== Official Closed-Loop Environment Check =="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "BENCH2DRIVE_ROOT=${BENCH2DRIVE_ROOT}"
echo "CARLA_ROOT=${CARLA_ROOT:-<unset>}"
echo "ROUTES_PATH=${ROUTES_PATH:-<unset>}"
echo "SCENARIOS_PATH=${SCENARIOS_PATH:-<unset>}"

check_path "project root" "${PROJECT_ROOT}"
check_path "official agent" "${PROJECT_ROOT}/team_code/orion_b2d_agent.py"
check_path "agent config" "${PROJECT_ROOT}/adzoo/orion/configs/orion_stage3_agent.py"

check_python_module "carla"
check_python_module "leaderboard"
check_python_module "scenario_runner"

if [[ -n "${CARLA_ROOT}" ]]; then
  check_path "CarlaUE4 launcher" "${CARLA_ROOT}/CarlaUE4.sh"
fi

if [[ -n "${BENCH2DRIVE_ROOT}" ]]; then
  check_path "Bench2DriveZoo root" "${BENCH2DRIVE_ROOT}"
  check_path "Bench2Drive planner" "${BENCH2DRIVE_ROOT}/team_code/planner.py"
  check_path "Bench2Drive pid_controller" "${BENCH2DRIVE_ROOT}/team_code/pid_controller.py"
fi

if [[ -n "${ROUTES_PATH}" ]]; then
  check_path "routes file" "${ROUTES_PATH}"
fi

if [[ -n "${SCENARIOS_PATH}" ]]; then
  check_path "scenarios file" "${SCENARIOS_PATH}"
fi

echo "== Result =="
if [[ "${status}" -eq 0 ]]; then
  echo "Official closed-loop prerequisites look available."
else
  echo "Official closed-loop prerequisites are incomplete."
fi

exit "${status}"

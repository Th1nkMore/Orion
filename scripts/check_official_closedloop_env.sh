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
PYTHON_BIN="${PYTHON_BIN:-python}"

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

check_python_snippet() {
  local label="$1"
  local snippet="$2"
  if "${PYTHON_BIN}" - <<PY >/dev/null 2>&1
${snippet}
PY
  then
    printf '[OK] python import: %s\n' "${label}"
  else
    printf '[MISS] python import: %s\n' "${label}"
    status=1
  fi
}

echo "== Official Closed-Loop Environment Check =="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "BENCH2DRIVE_ROOT=${BENCH2DRIVE_ROOT}"
echo "CARLA_ROOT=${CARLA_ROOT:-<unset>}"
echo "ROUTES_PATH=${ROUTES_PATH:-<unset>}"
echo "SCENARIOS_PATH=${SCENARIOS_PATH:-<unset>}"
echo "PYTHON_BIN=${PYTHON_BIN}"

check_path "project root" "${PROJECT_ROOT}"
check_path "official agent" "${PROJECT_ROOT}/team_code/orion_b2d_agent.py"
check_path "agent config" "${PROJECT_ROOT}/adzoo/orion/configs/orion_stage3_agent.py"

check_python_snippet "carla" "import carla"
check_python_snippet "leaderboard.autoagents.autonomous_agent" \
  "from leaderboard.autoagents import autonomous_agent"
check_python_snippet "srunner" "import srunner"

if [[ -n "${CARLA_ROOT}" ]]; then
  check_path "CarlaUE4 launcher" "${CARLA_ROOT}/CarlaUE4.sh"
  check_path "CARLA PythonAPI root" "${CARLA_ROOT}/PythonAPI"
  check_path "CARLA agents package" "${CARLA_ROOT}/PythonAPI/carla/agents"
fi

if [[ -n "${BENCH2DRIVE_ROOT}" ]]; then
  check_path "Bench2DriveZoo root" "${BENCH2DRIVE_ROOT}"
  check_path "leaderboard root" "${BENCH2DRIVE_ROOT}/leaderboard"
  check_path "scenario_runner root" "${BENCH2DRIVE_ROOT}/scenario_runner"
  check_path "220-route file" "${BENCH2DRIVE_ROOT}/leaderboard/data/bench2drive220.xml"
fi

if [[ -n "${ROUTES_PATH}" ]]; then
  check_path "routes file" "${ROUTES_PATH}"
fi

if [[ -n "${SCENARIOS_PATH}" ]]; then
  check_path "scenarios file" "${SCENARIOS_PATH}"
else
  echo "[INFO] SCENARIOS_PATH unset: Bench2Drive route XML may be sufficient for smoke tests."
fi

echo "== Result =="
if [[ "${status}" -eq 0 ]]; then
  echo "Official closed-loop prerequisites look available."
else
  echo "Official closed-loop prerequisites are incomplete."
fi

exit "${status}"

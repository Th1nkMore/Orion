#!/usr/bin/env bash
set -euo pipefail

# Import the complete Bench2Drive ORION agent inside a compute allocation.
# Keeping this separate from the broad environment checker preserves the
# Python traceback and makes memory-related scheduler failures visible.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
BENCH2DRIVE_ROOT="${BENCH2DRIVE_ROOT:-${PROJECT_ROOT}/Bench2Drive}"
BENCH2DRIVE_ZOO_ROOT="${BENCH2DRIVE_ZOO_ROOT:-${PROJECT_ROOT}/Bench2DriveZoo}"
CARLA_ROOT="${CARLA_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CARLA_PYTHON_EGG="${CARLA_PYTHON_EGG:-${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg}"

if [[ -z "${CARLA_ROOT}" ]]; then
  echo "[FAIL] CARLA_ROOT must be set" >&2
  exit 1
fi

export PYTHONPATH="${CARLA_PYTHON_EGG}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${PROJECT_ROOT}:${BENCH2DRIVE_ROOT}:${BENCH2DRIVE_ROOT}/leaderboard:${BENCH2DRIVE_ROOT}/scenario_runner:${BENCH2DRIVE_ZOO_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"
echo "[IMPORT] host=$(hostname) python=${PYTHON_BIN}"
"${PYTHON_BIN}" -X faulthandler - <<'PY'
import torch

print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
from team_code.orion_b2d_agent import OrionAgent
print(f"ORION_AGENT_IMPORT_OK={OrionAgent.__name__}", flush=True)
from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
print(
    f"BENCH2DRIVE_EVALUATOR_IMPORT_OK={LeaderboardEvaluator.__name__}",
    flush=True,
)
PY

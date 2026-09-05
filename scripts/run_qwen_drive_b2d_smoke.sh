#!/usr/bin/env bash
set -euo pipefail

# Reuse the repository's official one-route Bench2Drive launcher while replacing
# only the agent/config pair.  qwen_drive_b2d_agent.py intentionally ignores the
# second legacy TEAM_CONFIG component appended by that launcher.

project_root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
agent="${project_root}/team_code/qwen_drive_b2d_agent.py"
config="${project_root}/configs/qwen_drive_b2d_agent_v1.json"

for prerequisite in "${agent}" "${config}" "${project_root}/scripts/run_official_closedloop_smoke.sh"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "missing Qwen-Drive closed-loop prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done

PROJECT_ROOT="${project_root}" \
TEAM_AGENT_PATH="${agent}" \
AGENT_CONFIG_PATH="${config}" \
BASE_CHECKPOINT_PATH="qwen-drive-sidecar-no-orion-checkpoint" \
ALGO="${ALGO:-qwen_drive}" \
PLANNER_TYPE="${PLANNER_TYPE:-traj}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-${project_root}/results/qwen_drive_b2d_smoke}" \
ORION_CARLA_RPC_TIMEOUT_SECONDS="${ORION_CARLA_RPC_TIMEOUT_SECONDS:-1200}" \
exec "${project_root}/scripts/run_official_closedloop_smoke.sh"

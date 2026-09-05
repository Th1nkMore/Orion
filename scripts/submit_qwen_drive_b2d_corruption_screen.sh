#!/usr/bin/env bash
set -euo pipefail

mode="${1:---dry-run}"
if [[ "${mode}" != "--dry-run" && "${mode}" != "--submit" ]]; then
  echo "usage: $0 [--dry-run|--submit]" >&2
  exit 2
fi

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
submitter="${project_root}/scripts/submit_qwen_drive_b2d_closedloop_smoke.sh"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
screen_id="${SCREEN_ID:-qwen_official_input_dropout_screen_v1}"
node_list="${NODELIST:-gpu4}"

if [[ ! -x "${submitter}" ]]; then
  echo "missing executable Qwen submitter: ${submitter}" >&2
  exit 2
fi

run_one() {
  local route="$1"
  local condition="$2"
  local start_progress="$3"
  local end_progress="$4"
  local port="$5"
  local tm_port="$6"
  local run_id="${screen_id}_route${route}_${condition}"
  local job_name="qwen_${route}_${condition}"
  local output_root="${asset_root}/qwen_drive_b2d_smokes/${run_id}"
  local -a command=(bash "${submitter}" "${mode}")

  if [[ "${condition}" == "clean" ]]; then
    env -u ORION_CLOSEDLOOP_CORRUPTION \
      -u ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS \
      -u ORION_CLOSEDLOOP_CORRUPTION_END_PROGRESS \
      PROJECT_ROOT="${project_root}" ASSET_ROOT="${asset_root}" \
      TASK_ID="${route}" RUN_ID="${run_id}" OUTPUT_ROOT="${output_root}" \
      JOB_NAME="${job_name}" NODELIST="${node_list}" \
      PORT="${port}" TM_PORT="${tm_port}" "${command[@]}"
  else
    PROJECT_ROOT="${project_root}" ASSET_ROOT="${asset_root}" \
      TASK_ID="${route}" RUN_ID="${run_id}" OUTPUT_ROOT="${output_root}" \
      JOB_NAME="${job_name}" NODELIST="${node_list}" \
      PORT="${port}" TM_PORT="${tm_port}" \
      ORION_CLOSEDLOOP_CORRUPTION="camera_dropout" \
      ORION_CLOSEDLOOP_CORRUPTION_VIEWS="front" \
      ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS="${start_progress}" \
      ORION_CLOSEDLOOP_CORRUPTION_END_PROGRESS="${end_progress}" \
      "${command[@]}"
  fi
}

# Route 146 is the established clean-safe/dropout-collision pair from the old
# Orion pilot. Routes 151 and 203 add two pedestrian scenario variants while
# keeping their previously frozen route-aligned windows.
run_one 146 clean "" "" 31146 51146
run_one 146 front_dropout_p030_p055 0.30 0.55 32146 52146
run_one 151 clean "" "" 31151 51151
run_one 151 front_dropout_event 0.32162320371253544 0.4757938890155946 32151 52151
run_one 203 clean "" "" 31203 51203
run_one 203 front_dropout_full 0.0 1.0 32203 52203

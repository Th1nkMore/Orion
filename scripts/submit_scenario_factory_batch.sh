#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 BATCH_MANIFEST [--submit]" >&2
  exit 2
fi

batch_manifest="$1"
mode="${2:---dry-run}"
if [[ "${mode}" != "--dry-run" && "${mode}" != "--submit" ]]; then
  echo "[FAIL] second argument must be --submit" >&2
  exit 2
fi

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_bin="${SCENARIO_FACTORY_PYTHON:-python3}"
route_dir="$(dirname "${batch_manifest}")"

mapfile -t batch_rows < <("${python_bin}" - "${batch_manifest}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
if payload.get("schema") != "orion.scenario_factory.batch.v1":
    raise SystemExit("invalid scenario-factory batch schema")
if payload.get("status") != "prepared_no_jobs_submitted":
    raise SystemExit("batch is not in prepared_no_jobs_submitted state")
if payload.get("split") not in ("development_screen", "locked_test"):
    raise SystemExit("unsupported scenario-factory batch split")
contract = payload.get("runtime_contract", {})
required = {
    "condition": "clean_off",
    "variant": "hazard",
    "stage2_spatial_uq_source": "disabled",
    "stage1_adapter_control_influence": False,
    "legacy_density_uq": False,
    "risk_mode": "off",
    "planning_response": "off",
    "carla_quality": "Epic",
}
for key, expected in required.items():
    if contract.get(key) != expected:
        raise SystemExit("runtime contract mismatch for %s" % key)
print("RUN_ID\t%s" % payload["run_id"])
print("SPLIT\t%s" % payload["split"])
for row in payload["routes"]:
    print("ROUTE\t%d\t%s\t%s" % (
        int(row["route_index"]), row["town"], row["scenario_type"]
    ))
PY
)

run_id=""
split=""
route_rows=()
for row in "${batch_rows[@]}"; do
  if [[ "${row}" == RUN_ID$'\t'* ]]; then
    run_id="${row#*$'\t'}"
  elif [[ "${row}" == SPLIT$'\t'* ]]; then
    split="${row#*$'\t'}"
  elif [[ "${row}" == ROUTE$'\t'* ]]; then
    route_rows+=("${row}")
  fi
done
if [[ -z "${run_id}" || ${#route_rows[@]} -eq 0 ]]; then
  echo "[FAIL] batch contains no run id or routes" >&2
  exit 2
fi

result_root="${asset_root}/results/${run_id}"
if [[ -e "${result_root}" ]] && find "${result_root}" -mindepth 1 -print -quit | grep -q .; then
  echo "[FAIL] result root already contains artifacts: ${result_root}" >&2
  exit 2
fi

echo "SCENARIO_FACTORY_RUN_ID=${run_id}"
echo "SCENARIO_FACTORY_SPLIT=${split}"
echo "SCENARIO_FACTORY_ROUTE_COUNT=${#route_rows[@]}"
for row in "${route_rows[@]}"; do
  IFS=$'\t' read -r _ route_index town scenario_type <<<"${row}"
  command=(
    env
    PILOT_RUN_ID="${run_id}"
    PILOT_ROUTE_DIR="${route_dir}"
    AGENT_CONFIG_PATH="${project_root}/adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py"
    ORION_ENABLE_LEGACY_DENSITY_UQ=0
    ORION_CLOSEDLOOP_CONDITIONING=none
    ORION_OBSERVATION_UQ_CHECKPOINT=
    ORION_OBSERVATION_UQ_CHECKPOINT_SHA256=
    ORION_STAGE2_SPATIAL_UQ_SOURCE=disabled
    ORION_STAGE1_SPATIAL_UQ_CHECKPOINT=
    ORION_STAGE1_SPATIAL_UQ_CHECKPOINT_SHA256=
    ORION_STAGE2_TASK_CHECKPOINT=
    ORION_STAGE2_TASK_CHECKPOINT_SHA256=
    ORION_CLOSEDLOOP_SAFETY_TELEMETRY=1
    CARLA_QUALITY_LEVEL=Epic
    SLURM_CPUS_PER_TASK=2
    SLURM_MEM=192G
    SLURM_TIME=02:00:00
    SLURM_EXCLUDE="${SLURM_EXCLUDE:-gpu2,gpu5}"
    "${project_root}/scripts/submit_closedloop_uq_pilot.sh"
    "${route_index}" clean_off hazard
  )
  if [[ "${mode}" == "--dry-run" ]]; then
    printf 'DRY_RUN route=%s town=%s scenario=%s command=' \
      "${route_index}" "${town}" "${scenario_type}"
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    job_id=$("${command[@]}")
    echo "SUBMITTED route=${route_index} town=${town} scenario=${scenario_type} job_id=${job_id}"
  fi
done

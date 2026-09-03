#!/usr/bin/env bash
set -euo pipefail

# One non-claim Route147 engineering run.  It verifies that the terminal
# controlled-K checkpoint reaches the live ORION trajectory and PID path.

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
ASSET_ROOT="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
PYTHON_BIN="${ASSET_ROOT}/envs/orion-cl-centos7/bin/python"
RUN_ID="stage2p_v1_vertical_slice_route147_k_v1"
ROUTE_FILE="${PROJECT_ROOT}/configs/closedloop_scenario_bank/routes/route_147_hazard.xml"
ROUTE_SHA256="eb59a25dd8d72327e1c4029609e34adaab3abc4e7b0f9e00c975d2c8251e9ce1"
CHECKPOINT="${ASSET_ROOT}/scenario_factory/stage2p_smokes/v1_controlled_k_route147_80step_v1/training/stage2p_controlled_k_response.pt"
CHECKPOINT_SHA256="4a27db9104208341e04756d7dffe6f57b6b0d4d4a61a1ec4ae4c50737da890e6"
PROTOCOL="${PROJECT_ROOT}/configs/scenario_factory/stage2p_v1_route147_vertical_slice_carla_smoke_v1.json"
TERMINAL="${PROJECT_ROOT}/configs/scenario_factory/amendments/20260901_stage2p_v1_controlled_k_interface_terminal_v1.json"
TERMINAL_VALIDATION="${ASSET_ROOT}/scenario_factory/stage2p_smokes/v1_controlled_k_route147_80step_v1/terminal_validation.json"
DRY_RUN="${DRY_RUN:-0}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" - \
  "${PROTOCOL}" "${CHECKPOINT}" "${TERMINAL}" "${TERMINAL_VALIDATION}" \
  "${ROUTE_FILE}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

protocol_path, checkpoint, terminal_path, validation_path, route = map(
    Path, sys.argv[1:]
)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for path in (protocol_path, checkpoint, terminal_path, validation_path, route):
    if not path.is_file():
        raise SystemExit("Stage2-P CARLA prerequisite is missing: %s" % path)
protocol = json.loads(protocol_path.read_text())
terminal = json.loads(terminal_path.read_text())
validation = json.loads(validation_path.read_text())
if (
    protocol.get("status") != "single_route_engineering_smoke_preregistered"
    or terminal.get("status")
    != "terminal_integrity_valid_soft_specificity_failures"
    or validation.get("status")
    != "validated_integrity_pass_soft_specificity_failures"
    or protocol["checkpoint"]["sha256"] != sha256(checkpoint)
    or protocol["route"]["sha256"] != sha256(route)
    or protocol["terminal_lineage"]["validation_sha256"]
    != sha256(validation_path)
    or protocol["terminal_lineage"]["terminal_amendment_sha256"]
    != sha256(terminal_path)
):
    raise SystemExit("Stage2-P CARLA frozen lineage differs")
paths = {
    "agent_config": "adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py",
    "orion_head": "mmcv/models/dense_heads/orion_head.py",
    "orion_detector": "mmcv/models/detectors/orion.py",
    "carla_agent": "team_code/orion_b2d_agent.py",
    "stage2p_module": "uq_estimator/stage2p_task_risk_trajectory.py",
    "closed_loop_runner": "scripts/run_closedloop_uq_pilot.sh",
}
for name, relative in paths.items():
    if sha256(Path(relative)) != protocol["implementation_sha256"][name]:
        raise SystemExit("Stage2-P CARLA implementation hash differs: %s" % name)
print("STAGE2P_ROUTE147_CARLA_PREFLIGHT_OK=1")
PY

if [[ -e "${ASSET_ROOT}/results/${RUN_ID}" ]] && \
   find "${ASSET_ROOT}/results/${RUN_ID}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
  echo "[FAIL] ${RUN_ID} already has artifacts" >&2
  exit 2
fi
if squeue -h -u "${USER}" -n "uqcl_147_stage2p_controlled_k_smoke" | grep -q .; then
  echo "[FAIL] duplicate active Stage2-P CARLA smoke" >&2
  exit 2
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1; no Slurm job submitted"
  exit 0
fi

job_id=$(env \
  PILOT_RUN_ID="${RUN_ID}" \
  PILOT_ROUTE_FILE="${ROUTE_FILE}" \
  PILOT_ROUTE_FILE_SHA256="${ROUTE_SHA256}" \
  AGENT_CONFIG_PATH="${PROJECT_ROOT}/adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py" \
  ORION_ENABLE_LEGACY_DENSITY_UQ=0 \
  ORION_CLOSEDLOOP_CONDITIONING=none \
  ORION_OBSERVATION_UQ_CHECKPOINT= \
  ORION_OBSERVATION_UQ_CHECKPOINT_SHA256= \
  ORION_STAGE2_SPATIAL_UQ_SOURCE=external_oracle \
  ORION_STAGE1_SPATIAL_UQ_CHECKPOINT= \
  ORION_STAGE1_SPATIAL_UQ_CHECKPOINT_SHA256= \
  ORION_STAGE2_TASK_CHECKPOINT="${CHECKPOINT}" \
  ORION_STAGE2_TASK_CHECKPOINT_SHA256="${CHECKPOINT_SHA256}" \
  ORION_STAGE2_ENGINEERING_SMOKE=1 \
  ORION_STAGE2_EXTERNAL_K_START_PROGRESS=0.32 \
  ORION_STAGE2_EXTERNAL_K_DURATION_SECONDS=3.0 \
  ORION_STAGE2_EXTERNAL_K_CAMERA=CAM_FRONT \
  ORION_STAGE2_EXTERNAL_K_REGION=0.58,0.32,1.0,0.95 \
  ORION_STAGE2_EXTERNAL_K_STRENGTH=1.0 \
  ORION_STAGE2_EXTERNAL_K_GRID_SIZE=40 \
  ORION_STAGE2_ARTIFACT_ROOT= \
  ORION_CLOSEDLOOP_SAFETY_TELEMETRY=1 \
  ORION_CARLA_RPC_TIMEOUT_SECONDS=300 \
  ORION_SCENARIO_TRACEBACK_INTERVAL_SECONDS=75 \
  CLOSEDLOOP_PROGRESS_WATCHDOG_SECONDS=300 \
  CLOSEDLOOP_PROGRESS_STARTUP_GRACE_SECONDS=900 \
  SLURM_CPUS_PER_TASK=2 \
  SLURM_MEM=192G \
  SLURM_TIME=02:00:00 \
  SLURM_EXCLUDE=gpu5 \
  ./scripts/submit_closedloop_uq_pilot.sh \
  147 stage2p_controlled_k_smoke hazard)

echo "stage2p_v1_route147_vertical_slice_carla_smoke=${job_id}"

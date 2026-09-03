#!/usr/bin/env bash
set -euo pipefail

# One bounded data-capture replay.  The privileged Route147 expert controls the
# car exactly as in the passed mechanism run, while the frozen Stage-1 adapter
# only records spatial maps and the zero-initialized Stage-2 path remains an
# exact ORION identity.  This job is not a learned-UQ safety evaluation.

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
ASSET_ROOT="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
RUN_ID="${RUN_ID:-route147_stage2_task_context_capture_v3}"
ROUTE_DIR="${PROJECT_ROOT}/configs/closedloop_scenario_bank/routes"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/public/home/lidachuan/orion_work/observation_uq_v3/runs/counterfactual_pairwise_native_repair_seed20260828_r1/counterfactual_evidence_pairwise_native_repair.pt}"
STAGE1_SHA256="${STAGE1_SHA256:-0555f0f341c80a88e18c5864573f0be0641fb828931bea7809e2f5544665f2c8}"
MECHANISM_REPORT="${PROJECT_ROOT}/results/closedloop_scenario_bank/route147_braking_aware_v2/route147_braking_aware_v2_result.json"
DRY_RUN="${DRY_RUN:-0}"

cd "${PROJECT_ROOT}"
python_bin="${ASSET_ROOT}/envs/orion-cl-centos7/bin/python"
"${python_bin}" - "${STAGE1_CHECKPOINT}" "${STAGE1_SHA256}" "${MECHANISM_REPORT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

checkpoint = Path(sys.argv[1])
expected = sys.argv[2]
report_path = Path(sys.argv[3])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if not checkpoint.is_file() or sha256(checkpoint) != expected:
    raise SystemExit("frozen Stage-1 checkpoint missing or hash differs")
report = json.loads(report_path.read_text())
if report.get("primary_success") is not True or report.get("stage2_eligible") is not True:
    raise SystemExit("Route147 mechanism report does not unlock Stage 2")
print("ROUTE147_STAGE2_CAPTURE_PREFLIGHT_OK=1")
PY

if [[ -e "${ASSET_ROOT}/results/${RUN_ID}" ]] && \
   find "${ASSET_ROOT}/results/${RUN_ID}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
  echo "[FAIL] ${RUN_ID} already has artifacts" >&2
  exit 2
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1; no Slurm job submitted"
  exit 0
fi

job_id=$(env \
  PILOT_RUN_ID="${RUN_ID}" \
  PILOT_ROUTE_DIR="${ROUTE_DIR}" \
  AGENT_CONFIG_PATH="${PROJECT_ROOT}/adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py" \
  ORION_ENABLE_LEGACY_DENSITY_UQ=0 \
  ORION_CLOSEDLOOP_CONDITIONING=none \
  ORION_OBSERVATION_UQ_CHECKPOINT= \
  ORION_OBSERVATION_UQ_CHECKPOINT_SHA256= \
  ORION_STAGE2_SPATIAL_UQ_SOURCE=learned_adapter \
  ORION_STAGE1_SPATIAL_UQ_CHECKPOINT="${STAGE1_CHECKPOINT}" \
  ORION_STAGE1_SPATIAL_UQ_CHECKPOINT_SHA256="${STAGE1_SHA256}" \
  ORION_STAGE1_SPATIAL_UQ_WARMUP_FRAMES=60 \
  ORION_STAGE2_TASK_CHECKPOINT= \
  ORION_STAGE2_TASK_CHECKPOINT_SHA256= \
  ORION_STAGE2_ARTIFACT_ROOT=AUTO \
  ORION_STAGE2_ARTIFACT_ROUTE_GROUP=Town02/Route147/hazard/seed20260829 \
  ORION_STAGE2_ARTIFACT_STRIDE_STEPS=10 \
  ORION_CARLA_RPC_TIMEOUT_SECONDS=90 \
  ORION_SCENARIO_TRACEBACK_INTERVAL_SECONDS=75 \
  CLOSEDLOOP_PROGRESS_WATCHDOG_SECONDS=150 \
  ORION_PLANNING_ACTOR_CATEGORIES=walker \
  ORION_PLANNING_INTERPOLATION_STEP_SECONDS=0.1 \
  ORION_PLANNING_SAFETY_MARGIN_M=0.75 \
  ORION_PLANNING_IMMINENT_HORIZON_SECONDS=1.5 \
  ORION_PLANNING_CERTIFIED_DECELERATION_MPS2=3.0 \
  ORION_PLANNING_CLEARANCE_SECONDS=1.0 \
  ORION_PLANNING_RELEASE_SECONDS=0.5 \
  ORION_PLANNING_PREPARE_CREEP_SPEED_MPS=1.0 \
  ORION_PLANNING_RELEASE_CREEP_SPEED_MPS=0.5 \
  ORION_PLANNING_STOP_BUFFER_M=2.0 \
  ORION_PLANNING_RELEASE_CREEP_DISTANCE_M=1.0 \
  SLURM_NODELIST=gpu4 \
  SLURM_CPUS_PER_TASK=4 \
  SLURM_MEM=192G \
  SLURM_TIME=02:00:00 \
  ./scripts/submit_closedloop_uq_pilot.sh \
  147 native_braking_aware_crossing_oracle hazard)

echo "route147_stage2_task_context_capture_v3=${job_id}"

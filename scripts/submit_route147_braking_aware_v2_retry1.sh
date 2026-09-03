#!/usr/bin/env bash
set -euo pipefail

# One runtime-environment retry authorized by the timestamped v2 amendment.
# This wrapper does not change any model, route, response, or success setting.

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
ASSET_ROOT="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
BASE_PREREG="${PROJECT_ROOT}/configs/closedloop_scenario_bank/route147_braking_aware_v2.json"
AMENDMENT="${PROJECT_ROOT}/configs/closedloop_scenario_bank/route147_braking_aware_v2_runtime_amendment_20260829.json"
ROUTE_DIR="${PROJECT_ROOT}/configs/closedloop_scenario_bank/routes"
ORACLE_RUN_ID="${ORACLE_RUN_ID:-closedloop_route147_braking_aware_oracle_v2_retry1}"
DRY_RUN="${DRY_RUN:-0}"

cd "${PROJECT_ROOT}"

"${ASSET_ROOT}/envs/orion-cl-centos7/bin/python" - \
  "${PROJECT_ROOT}" "${BASE_PREREG}" "${AMENDMENT}" \
  "${ROUTE_DIR}/route_147_hazard.xml" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
base_path = Path(sys.argv[2])
amendment_path = Path(sys.argv[3])
route = Path(sys.argv[4])
base = json.loads(base_path.read_text())
amendment = json.loads(amendment_path.read_text())

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if sha256(base_path) != amendment["base_preregistration_sha256"]:
    raise SystemExit("base v2 preregistration changed")
for relative_path, expected in base["frozen_hashes"].items():
    path = route if relative_path == "route_147_hazard.xml" else root / relative_path
    if sha256(path) != expected:
        raise SystemExit(f"frozen v2 source changed: {relative_path}")
for artifact in amendment["invalid_run"]["frozen_artifacts"]:
    if sha256(root / artifact["path"]) != artifact["sha256"]:
        raise SystemExit(f"invalid-run evidence changed: {artifact['path']}")
watchdog = json.loads((
    root / amendment["invalid_run"]["watchdog_path"]
).read_text())
if watchdog.get("scientific_classification") != "runtime_environment_invalid":
    raise SystemExit("watchdog did not classify the run as runtime invalid")
if watchdog.get("reason") != "control_trace_stalled":
    raise SystemExit("watchdog reason differs from the amendment")
eval_payload = json.loads((
    root / amendment["invalid_run"]["eval_path"]
).read_text())
checkpoint = eval_payload.get("_checkpoint") or {}
if (
    eval_payload.get("entry_status") != "Started"
    or eval_payload.get("eligible") is not False
    or checkpoint.get("records") != []
):
    raise SystemExit("invalid run unexpectedly contains a valid route result")
rule = amendment["one_retry_rule"]
if rule.get("maximum_retry_submissions") != 1:
    raise SystemExit("amendment does not authorize exactly one retry")
if rule.get("source_or_parameter_changes_allowed") is not False:
    raise SystemExit("amendment must forbid source and parameter changes")
print("ROUTE147_BRAKING_AWARE_V2_RUNTIME_RETRY_ATTESTATION_OK=1")
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1; no Slurm jobs submitted"
  exit 0
fi

if [[ -d "${ASSET_ROOT}/results/${ORACLE_RUN_ID}" ]] && \
   find "${ASSET_ROOT}/results/${ORACLE_RUN_ID}" -mindepth 1 -print -quit | grep -q .; then
  echo "[FAIL] ${ORACLE_RUN_ID} already has artifacts; amendment forbids another retry" >&2
  exit 2
fi

retry_job_id=$(env \
  PILOT_RUN_ID="${ORACLE_RUN_ID}" \
  PILOT_ROUTE_DIR="${ROUTE_DIR}" \
  ORION_ENABLE_LEGACY_DENSITY_UQ=0 \
  ORION_OBSERVATION_UQ_CHECKPOINT= \
  ORION_OBSERVATION_UQ_CHECKPOINT_SHA256= \
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
  SLURM_CPUS_PER_TASK=2 \
  SLURM_MEM=192G \
  SLURM_TIME=02:00:00 \
  ./scripts/submit_closedloop_uq_pilot.sh \
  147 native_braking_aware_crossing_oracle hazard)

echo "route147_braking_aware_oracle_v2_retry1=${retry_job_id}"

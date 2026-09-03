#!/usr/bin/env bash
set -euo pipefail

# Submit exactly one Route147 braking-aware planning-oracle run. The clean
# comparator is the frozen current-code v1 clean artifact; it is not rerun.

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
ASSET_ROOT="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
PREREG="${PROJECT_ROOT}/configs/closedloop_scenario_bank/route147_braking_aware_v2.json"
ROUTE_DIR="${PROJECT_ROOT}/configs/closedloop_scenario_bank/routes"
ORACLE_RUN_ID="${ORACLE_RUN_ID:-closedloop_route147_braking_aware_oracle_v2}"
DRY_RUN="${DRY_RUN:-0}"

cd "${PROJECT_ROOT}"

"${ASSET_ROOT}/envs/orion-cl-centos7/bin/python" - \
  "${PROJECT_ROOT}" "${PREREG}" "${ROUTE_DIR}/route_147_hazard.xml" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
prereg = json.loads(Path(sys.argv[2]).read_text())
route = Path(sys.argv[3])

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for relative_path, expected in prereg["frozen_hashes"].items():
    path = route if relative_path == "route_147_hazard.xml" else root / relative_path
    observed = sha256(path)
    if observed != expected:
        raise SystemExit(
            f"frozen hash differs: {relative_path} {observed} != {expected}"
        )
for artifact in prereg["offline_evidence"]["frozen_artifacts"]:
    observed = sha256(root / artifact["path"])
    if observed != artifact["sha256"]:
        raise SystemExit(f"frozen evidence hash differs: {artifact['path']}")
gate = json.loads((root / prereg["offline_evidence"]["gate_path"]).read_text())
if gate.get("offline_gate_pass") is not True:
    raise SystemExit("Route147 braking-aware v2 offline gate did not pass")
if gate.get("decision") != (
    "eligible_for_one_preregistered_route147_braking_aware_v2_oracle"
):
    raise SystemExit("offline gate does not authorize this exact oracle")
rule = prereg["single_oracle_rule"]
if rule.get("maximum_oracle_submissions") != 1:
    raise SystemExit("preregistration does not authorize exactly one oracle run")
if rule.get("maximum_clean_submissions") != 0:
    raise SystemExit("v2 must use the frozen clean reference without rerunning it")
print("ROUTE147_BRAKING_AWARE_V2_PREREG_ATTESTATION_OK=1")
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1; no Slurm jobs submitted"
  exit 0
fi

if [[ -d "${ASSET_ROOT}/results/${ORACLE_RUN_ID}" ]] && \
   find "${ASSET_ROOT}/results/${ORACLE_RUN_ID}" -mindepth 1 -print -quit | grep -q .; then
  echo "[FAIL] ${ORACLE_RUN_ID} already has artifacts; single-oracle rule forbids resubmission" >&2
  exit 2
fi

oracle_job_id=$(env \
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

echo "route147_braking_aware_oracle_v2=${oracle_job_id}"

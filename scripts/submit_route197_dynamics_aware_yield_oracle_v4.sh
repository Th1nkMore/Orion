#!/usr/bin/env bash
set -euo pipefail

# Submit exactly one preregistered map- and dynamics-aware privileged oracle.
# DRY_RUN=1 attests all frozen inputs without requesting an A800.

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
ASSET_ROOT="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
PREREG="${PROJECT_ROOT}/configs/closedloop_scenario_bank/route197_dynamics_aware_yield_oracle_v4.json"
ROUTE_DIR="${PROJECT_ROOT}/configs/closedloop_native_collision_discovery/routes"
RUN_ID="${PILOT_RUN_ID:-closedloop_route197_dynamics_aware_yield_oracle_v4}"
DRY_RUN="${DRY_RUN:-0}"

cd "${PROJECT_ROOT}"

"${ASSET_ROOT}/envs/orion-cl-centos7/bin/python" - \
  "${PROJECT_ROOT}" "${PREREG}" "${ROUTE_DIR}/route_197_hazard.xml" <<'PY'
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
    path = route if relative_path == "route_197_hazard.xml" else root / relative_path
    observed = sha256(path)
    if observed != expected:
        raise SystemExit(
            f"frozen hash differs: {relative_path} {observed} != {expected}"
        )
for section in ("clean_reference", "failed_v3_reference", "offline_gate"):
    payload = prereg[section]
    for path_key, hash_key in payload["frozen_artifacts"]:
        observed = sha256(payload[path_key])
        if observed != payload[hash_key]:
            raise SystemExit(f"frozen evidence hash differs: {payload[path_key]}")
gate = json.loads(Path(prereg["offline_gate"]["report_path"]).read_text())
if gate.get("offline_gate_pass") is not True:
    raise SystemExit("offline dynamics-aware mechanism gate did not pass")
if gate.get("decision") != (
    "eligible_for_one_preregistered_dynamics_aware_privileged_oracle_run"
):
    raise SystemExit("offline gate does not authorize this single run")
if prereg["single_run_rule"]["maximum_authorized_submissions"] != 1:
    raise SystemExit("preregistration does not authorize exactly one submission")
print("DYNAMICS_AWARE_YIELD_V4_PREREG_ATTESTATION_OK=1")
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1; no Slurm job submitted"
  exit 0
fi

if [[ -d "${ASSET_ROOT}/results/${RUN_ID}" ]] && \
   find "${ASSET_ROOT}/results/${RUN_ID}" -mindepth 1 -print -quit | grep -q .; then
  echo "[FAIL] ${RUN_ID} already has artifacts; single-run rule forbids resubmission" >&2
  exit 2
fi

job_id=$(env \
  PILOT_RUN_ID="${RUN_ID}" \
  PILOT_ROUTE_DIR="${ROUTE_DIR}" \
  ORION_ENABLE_LEGACY_DENSITY_UQ=0 \
  ORION_OBSERVATION_UQ_CHECKPOINT= \
  ORION_OBSERVATION_UQ_CHECKPOINT_SHA256= \
  ORION_PLANNING_INTERPOLATION_STEP_SECONDS=0.1 \
  ORION_PLANNING_SAFETY_MARGIN_M=0.75 \
  ORION_PLANNING_CERTIFIED_DECELERATION_MPS2=3.0 \
  ORION_PLANNING_REACTION_SECONDS=0.1 \
  ORION_PLANNING_JUNCTION_FRONT_CLEARANCE_M=0.5 \
  ORION_PLANNING_MAP_RESOLUTION_M=0.1 \
  ORION_PLANNING_CLEARANCE_SECONDS=1.0 \
  ORION_PLANNING_RELEASE_SECONDS=0.5 \
  ORION_PLANNING_PREPARE_CREEP_SPEED_MPS=1.0 \
  ORION_PLANNING_RELEASE_CREEP_SPEED_MPS=0.5 \
  ORION_PLANNING_RELEASE_CREEP_DISTANCE_M=1.0 \
  SLURM_NODELIST=gpu4 \
  SLURM_CPUS_PER_TASK=2 \
  SLURM_MEM=192G \
  SLURM_TIME=03:00:00 \
  ./scripts/submit_closedloop_uq_pilot.sh \
  197 native_dynamics_aware_yield_oracle hazard)

echo "route197_dynamics_aware_yield_oracle_v4=${job_id}"

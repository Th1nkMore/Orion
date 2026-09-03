#!/usr/bin/env bash
set -euo pipefail

# Submit exactly one preregistered planning-level privileged oracle.  DRY_RUN=1
# verifies every frozen input without consuming an A800 allocation.

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
ASSET_ROOT="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
PREREG="${PROJECT_ROOT}/configs/closedloop_scenario_bank/route197_dynamic_yield_oracle_v3.json"
ROUTE_DIR="${PROJECT_ROOT}/configs/closedloop_native_collision_discovery/routes"
RUN_ID="${PILOT_RUN_ID:-closedloop_route197_dynamic_yield_oracle_v3}"
DRY_RUN="${DRY_RUN:-0}"

cd "${PROJECT_ROOT}"

/public/share/lidachuan/orion_assets/envs/orion-cl-centos7/bin/python - \
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
for section, path_key, hash_key in (
    ("clean_reference", "report_path", "report_sha256"),
    ("offline_label_audit", "report_path", "report_sha256"),
    ("offline_label_audit", "labels_path", "labels_sha256"),
):
    payload = (
        prereg[section]
        if section == "clean_reference"
        else prereg["development_evidence_used"][section]
    )
    observed = sha256(payload[path_key])
    if observed != payload[hash_key]:
        raise SystemExit(f"frozen evidence hash differs: {payload[path_key]}")
print("DYNAMIC_YIELD_V3_PREREG_ATTESTATION_OK=1")
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1; no Slurm job submitted"
  exit 0
fi

job_id=$(env \
  PILOT_RUN_ID="${RUN_ID}" \
  PILOT_ROUTE_DIR="${ROUTE_DIR}" \
  ORION_ENABLE_LEGACY_DENSITY_UQ=0 \
  ORION_OBSERVATION_UQ_CHECKPOINT= \
  ORION_OBSERVATION_UQ_CHECKPOINT_SHA256= \
  ORION_PLANNING_INTERPOLATION_STEP_SECONDS=0.1 \
  ORION_PLANNING_SAFETY_MARGIN_M=0.75 \
  ORION_PLANNING_IMMINENT_HORIZON_SECONDS=1.5 \
  ORION_PLANNING_CLEARANCE_SECONDS=1.0 \
  ORION_PLANNING_RELEASE_SECONDS=0.5 \
  ORION_PLANNING_STOP_BUFFER_M=2.0 \
  ORION_PLANNING_RELEASE_CREEP_DISTANCE_M=1.0 \
  SLURM_NODELIST=gpu4 \
  SLURM_CPUS_PER_TASK=2 \
  SLURM_MEM=192G \
  SLURM_TIME=03:00:00 \
  ./scripts/submit_closedloop_uq_pilot.sh \
  197 native_dynamic_yield_oracle hazard)

echo "route197_dynamic_yield_oracle_v3=${job_id}"

#!/usr/bin/env bash
set -euo pipefail

# Submit exactly one current-code clean/oracle pair for the finite Route147
# pedestrian crossing. DRY_RUN=1 verifies every frozen input without Slurm use.

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
ASSET_ROOT="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
PREREG="${PROJECT_ROOT}/configs/closedloop_scenario_bank/route147_bounded_crossing_pair_v1.json"
ROUTE_DIR="${PROJECT_ROOT}/configs/closedloop_scenario_bank/routes"
CLEAN_RUN_ID="${CLEAN_RUN_ID:-closedloop_route147_bounded_crossing_clean_v1}"
ORACLE_RUN_ID="${ORACLE_RUN_ID:-closedloop_route147_bounded_crossing_oracle_v1}"
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
gate = json.loads(
    (root / prereg["offline_evidence"]["gate_path"]).read_text()
)
if gate.get("offline_gate_pass") is not True:
    raise SystemExit("Route147 bounded-crossing offline gate did not pass")
if gate.get("decision") != (
    "eligible_for_one_preregistered_route147_clean_oracle_pair"
):
    raise SystemExit("offline gate does not authorize this exact pair")
rule = prereg["single_pair_rule"]
if rule.get("maximum_clean_submissions") != 1:
    raise SystemExit("preregistration does not authorize exactly one clean run")
if rule.get("maximum_oracle_submissions") != 1:
    raise SystemExit("preregistration does not authorize exactly one oracle run")
print("ROUTE147_BOUNDED_CROSSING_PREREG_ATTESTATION_OK=1")
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1; no Slurm jobs submitted"
  exit 0
fi

for run_id in "${CLEAN_RUN_ID}" "${ORACLE_RUN_ID}"; do
  if [[ -d "${ASSET_ROOT}/results/${run_id}" ]] && \
     find "${ASSET_ROOT}/results/${run_id}" -mindepth 1 -print -quit | grep -q .; then
    echo "[FAIL] ${run_id} already has artifacts; single-pair rule forbids resubmission" >&2
    exit 2
  fi
done

clean_job_id=$(env \
  PILOT_RUN_ID="${CLEAN_RUN_ID}" \
  PILOT_ROUTE_DIR="${ROUTE_DIR}" \
  ORION_ENABLE_LEGACY_DENSITY_UQ=0 \
  ORION_OBSERVATION_UQ_CHECKPOINT= \
  ORION_OBSERVATION_UQ_CHECKPOINT_SHA256= \
  SLURM_NODELIST=gpu4 \
  SLURM_CPUS_PER_TASK=2 \
  SLURM_MEM=192G \
  SLURM_TIME=02:00:00 \
  ./scripts/submit_closedloop_uq_pilot.sh \
  147 clean_off hazard)

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
  ORION_PLANNING_CLEARANCE_SECONDS=1.0 \
  ORION_PLANNING_RELEASE_SECONDS=0.5 \
  ORION_PLANNING_STOP_BUFFER_M=2.0 \
  ORION_PLANNING_RELEASE_CREEP_DISTANCE_M=1.0 \
  SLURM_NODELIST=gpu4 \
  SLURM_CPUS_PER_TASK=2 \
  SLURM_MEM=192G \
  SLURM_TIME=02:00:00 \
  ./scripts/submit_closedloop_uq_pilot.sh \
  147 native_bounded_crossing_oracle hazard)

echo "route147_bounded_crossing_clean_v1=${clean_job_id}"
echo "route147_bounded_crossing_oracle_v1=${oracle_job_id}"

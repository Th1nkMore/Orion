#!/usr/bin/env bash
set -euo pipefail

# Submit the two preregistered, scientifically separate Route197 diagnostics.
# Set DRY_RUN=1 to attest inputs without submitting Slurm jobs.

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
ASSET_ROOT="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
ROUTE_DIR="${PROJECT_ROOT}/configs/closedloop_native_collision_discovery/routes"
PAIRWISE_PREREG="${PROJECT_ROOT}/configs/closedloop_scenario_bank/clean_pairwise_trace_diagnostic_v1.json"
ORACLE_PREREG="${PROJECT_ROOT}/configs/closedloop_scenario_bank/route197_task_event_oracle_v1.json"
RUN_ID="${PILOT_RUN_ID:-closedloop_route197_sidecar_oracle_v1}"
DRY_RUN="${DRY_RUN:-0}"

cd "${PROJECT_ROOT}"

python3 - "${PROJECT_ROOT}" "${PAIRWISE_PREREG}" "${ORACLE_PREREG}" "${ROUTE_DIR}/route_197_hazard.xml" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
pairwise = json.loads(Path(sys.argv[2]).read_text())
oracle = json.loads(Path(sys.argv[3]).read_text())
route = Path(sys.argv[4])

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for relative_path, expected in pairwise["frozen_source_hashes"].items():
    observed = sha256(root / relative_path)
    if observed != expected:
        raise SystemExit(
            f"frozen source hash differs: {relative_path} {observed} != {expected}"
        )
observed_route = sha256(route)
expected_route = oracle["frozen_hashes"]["route_197_hazard.xml"]
if observed_route != expected_route:
    raise SystemExit(
        f"Route197 XML hash differs: {observed_route} != {expected_route}"
    )
checkpoint = Path(pairwise["frozen_adapter"]["path"])
observed_checkpoint = sha256(checkpoint)
expected_checkpoint = pairwise["frozen_adapter"]["sha256"]
if observed_checkpoint != expected_checkpoint:
    raise SystemExit(
        f"adapter checkpoint hash differs: {observed_checkpoint} != {expected_checkpoint}"
    )
print("ROUTE197_PREREG_ATTESTATION_OK=1")
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1; no Slurm jobs submitted"
  exit 0
fi

common_env=(
  PILOT_RUN_ID="${RUN_ID}"
  PILOT_ROUTE_DIR="${ROUTE_DIR}"
  SLURM_NODELIST=gpu4
  SLURM_CPUS_PER_TASK=2
  SLURM_MEM=192G
  SLURM_TIME=03:00:00
)

sidecar_job=$(env "${common_env[@]}" \
  ./scripts/submit_closedloop_uq_pilot.sh 197 clean_pairwise_trace hazard)

oracle_job=$(env "${common_env[@]}" \
  ORION_CLOSEDLOOP_RISK_ORACLE_START_PROGRESS=0.3690731367863814 \
  ORION_CLOSEDLOOP_RISK_ORACLE_DURATION_SECONDS=5.0 \
  ORION_CLOSEDLOOP_RISK_THRESHOLD=0.4 \
  ORION_CLOSEDLOOP_RISK_SATURATION=0.8 \
  ORION_CLOSEDLOOP_RISK_MIN_SPEED=1.5 \
  ORION_CLOSEDLOOP_RISK_MAX_SPEED=5.0 \
  ORION_CLOSEDLOOP_RISK_SLOWDOWN_MARGIN=1.0 \
  ORION_CLOSEDLOOP_RISK_BRAKE_GAIN=0.5 \
  ORION_CLOSEDLOOP_RISK_MAX_BRAKE=0.5 \
  ./scripts/submit_closedloop_uq_pilot.sh 197 native_event_oracle hazard)

echo "route197_clean_pairwise_trace=${sidecar_job}"
echo "route197_task_event_oracle=${oracle_job}"

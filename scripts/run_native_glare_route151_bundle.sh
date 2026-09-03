#!/usr/bin/env bash
set -euo pipefail

# Run the frozen Route151 clean/medium native-glare pair sequentially inside
# one A800 allocation.  This script intentionally loads original ORION twice:
# the bundle saves queue latency, not model or evaluation fidelity.

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_bin="${asset_root}/envs/orion-cl-centos7/bin/python"
bundle_run_id="${BUNDLE_RUN_ID:-native_glare_route151_pair_v1}"
bundle_root="${asset_root}/scenario_factory/native_glare_failure_induction/${bundle_run_id}"
activation="${project_root}/configs/scenario_factory/amendments/20260831_native_glare_bundle_python36_repair_v1.json"
protocol="${project_root}/configs/scenario_factory/native_glare_failure_induction_72h_v1.json"
render_protocol="${project_root}/configs/glare_visual_bakeoff_route151_v1.json"
safety_gate="${project_root}/configs/closedloop_scenario_bank/failure_induction_gate_v1.json"
source_route="${project_root}/configs/closedloop_scenario_bank/routes/route_151_hazard.xml"
route_dir="${bundle_root}/inputs/routes"
derived_route="${route_dir}/route_151_hazard.xml"
route_derivation="${bundle_root}/inputs/route_derivation.json"
clean_pilot_id="${bundle_run_id}_clean"
medium_pilot_id="${bundle_run_id}_medium"
clean_parent="${asset_root}/results/${clean_pilot_id}"
medium_parent="${asset_root}/results/${medium_pilot_id}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[FAIL] Route151 bundle must run inside a Slurm allocation" >&2
  exit 2
fi
for prerequisite in \
  "${python_bin}" "${activation}" "${protocol}" "${render_protocol}" \
  "${safety_gate}" "${source_route}" \
  "${project_root}/scripts/prepare_native_glare_route.py" \
  "${project_root}/scripts/run_closedloop_uq_pilot.sh" \
  "${project_root}/scripts/evaluate_clean_safety_gate.py" \
  "${project_root}/scripts/evaluate_native_glare_failure_induction.py"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "[FAIL] missing Route151 bundle prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
for output in "${bundle_root}" "${clean_parent}" "${medium_parent}"; do
  if [[ -e "${output}" ]]; then
    echo "[FAIL] refusing to reuse Route151 bundle output: ${output}" >&2
    exit 1
  fi
done

env "PYTHONPATH=${project_root}" "${python_bin}" - \
  "${activation}" "${project_root}" <<'PY'
import hashlib
import json
import os
import sys

activation_path, project_root = sys.argv[1:]
with open(activation_path) as handle:
    payload = json.load(handle)
if (
    payload.get("schema") != "orion.scenario_factory.amendment.v1"
    or payload.get("status") != "native_glare_orion_interface_activated"
    or payload.get("launch_locks", {}).get("route151_native_glare_pair_allowed") is not True
    or payload.get("launch_locks", {}).get("route203_native_glare_submission_allowed") is not False
):
    raise SystemExit("native-glare activation amendment does not authorize Route151")
for relative_path, expected in payload.get("activated_sources", {}).items():
    path = os.path.join(project_root, relative_path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise SystemExit("activated source hash differs: " + relative_path)
PY

mkdir -p "${route_dir}"
env "PYTHONPATH=${project_root}" "${python_bin}" \
  "${project_root}/scripts/prepare_native_glare_route.py" \
  --source-route "${source_route}" \
  --protocol "${render_protocol}" \
  --output-route "${derived_route}" \
  --manifest "${route_derivation}"

env "PYTHONPATH=${project_root}" "${python_bin}" - \
  "${bundle_root}/launch_attestation.json" "${project_root}" "${protocol}" \
  "${render_protocol}" "${safety_gate}" "${activation}" "${derived_route}" \
  "${SLURM_JOB_ID}" "${bundle_run_id}" <<'PY'
import datetime
import hashlib
import json
import os
import sys

(output, project_root, protocol, render_protocol, safety_gate, activation,
 route, job_id, run_id) = sys.argv[1:]
def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()
sources = (
    "team_code/orion_native_glare.py",
    "team_code/orion_b2d_agent.py",
    "scripts/run_closedloop_uq_pilot.sh",
    "scripts/evaluate_native_glare_failure_induction.py",
    "scripts/run_native_glare_route151_bundle.sh",
)
payload = {
    "schema": "orion.native_glare_bundle_launch_attestation.v1",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "running_frozen_clean_then_medium_pair",
    "job_id": job_id,
    "bundle_run_id": run_id,
    "resources": {"gpu": 1, "memory_gb": 192, "cpu": int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))},
    "conditions_in_order": ["clean", "medium"],
    "original_orion_only": True,
    "stage2l_checkpoint_loaded": False,
    "density_uq": False,
    "governor": False,
    "protocol_sha256": digest(protocol),
    "render_protocol_sha256": digest(render_protocol),
    "safety_gate_sha256": digest(safety_gate),
    "activation_amendment_sha256": digest(activation),
    "derived_route_sha256": digest(route),
    "source_sha256": {path: digest(os.path.join(project_root, path)) for path in sources},
}
with open(output, "x") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

run_profile() {
  local profile="$1"
  local pilot_id="$2"
  env \
    "PROJECT_ROOT=${project_root}" \
    "ASSET_ROOT=${asset_root}" \
    "PILOT_RUN_ID=${pilot_id}" \
    "PILOT_ROUTE_INDEX=151" \
    "PILOT_VARIANT=hazard" \
    "PILOT_CONDITION=clean_off" \
    "PILOT_ROUTE_DIR=${route_dir}" \
    "ORION_NATIVE_GLARE_PROFILE=${profile}" \
    "ORION_CLOSEDLOOP_SAFETY_TELEMETRY=1" \
    "CARLA_QUALITY_LEVEL=Epic" \
    bash "${project_root}/scripts/run_closedloop_uq_pilot.sh"
}

echo "[BUNDLE] starting frozen clean profile"
run_profile clean "${clean_pilot_id}"
clean_runs=("${clean_parent}"/route151_hazard_clean_off-*)
if [[ "${#clean_runs[@]}" -ne 1 || ! -d "${clean_runs[0]}" ]]; then
  echo "[FAIL] expected one clean run below ${clean_parent}" >&2
  exit 1
fi
clean_run="${clean_runs[0]}"
env "PYTHONPATH=${project_root}" "${python_bin}" \
  "${project_root}/scripts/evaluate_clean_safety_gate.py" \
  --run-dir "${clean_run}" \
  --output "${clean_run}/clean_safety_gate.json"

echo "[BUNDLE] clean gate passed; starting frozen medium profile"
run_profile medium "${medium_pilot_id}"
medium_runs=("${medium_parent}"/route151_hazard_clean_off-*)
if [[ "${#medium_runs[@]}" -ne 1 || ! -d "${medium_runs[0]}" ]]; then
  echo "[FAIL] expected one medium run below ${medium_parent}" >&2
  exit 1
fi
medium_run="${medium_runs[0]}"

decision="${bundle_root}/native_glare_failure_induction_decision.json"
env "PYTHONPATH=${project_root}" "${python_bin}" \
  "${project_root}/scripts/evaluate_native_glare_failure_induction.py" \
  --protocol "${protocol}" \
  --render-protocol "${render_protocol}" \
  --safety-gate "${safety_gate}" \
  --clean-run "${clean_run}" \
  --degraded-run "${medium_run}" \
  --output "${decision}"

env "PYTHONPATH=${project_root}" "${python_bin}" - \
  "${bundle_root}/result_index.json" "${clean_run}" "${medium_run}" \
  "${decision}" "${SLURM_JOB_ID}" <<'PY'
import datetime
import hashlib
import json
import sys

output, clean_run, medium_run, decision, job_id = sys.argv[1:]
def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()
with open(decision) as handle:
    report = json.load(handle)
payload = {
    "schema": "orion.native_glare_bundle_result_index.v1",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "pair_complete_and_evaluated",
    "job_id": job_id,
    "clean_run": clean_run,
    "medium_run": medium_run,
    "decision_path": decision,
    "decision_sha256": digest(decision),
    "valid": report["validity"]["valid"],
    "decision": report["decision"],
}
with open(output, "x") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo "[BUNDLE] Route151 clean/medium pair complete: ${decision}"

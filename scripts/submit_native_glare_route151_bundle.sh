#!/usr/bin/env bash
set -euo pipefail

submit=0
case "${1:-}" in
  --submit) submit=1 ;;
  ""|--dry-run) ;;
  *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
esac

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_bin="${asset_root}/envs/orion-cl-centos7/bin/python"
bundle_run_id="${BUNDLE_RUN_ID:-native_glare_route151_pair_v1_retry1}"
bundle_root="${asset_root}/scenario_factory/native_glare_failure_induction/${bundle_run_id}"
submission_root="${asset_root}/scenario_factory/submissions/native_glare"
submission_attestation="${submission_root}/${bundle_run_id}.json"
activation="${project_root}/configs/scenario_factory/amendments/20260831_native_glare_bundle_python36_repair_v1.json"
runner="${project_root}/scripts/run_native_glare_route151_bundle.sh"
job_name="ng151_pair_v1"
log_root="${asset_root}/scenario_factory/logs/native_glare_failure_induction"

for prerequisite in "${python_bin}" "${activation}" "${runner}"; do
  [[ -f "${prerequisite}" ]] || {
    echo "missing Route151 bundle prerequisite: ${prerequisite}" >&2
    exit 2
  }
done
for output in \
  "${bundle_root}" \
  "${asset_root}/results/${bundle_run_id}_clean" \
  "${asset_root}/results/${bundle_run_id}_medium" \
  "${submission_attestation}"; do
  [[ ! -e "${output}" ]] || {
    echo "refusing to overwrite Route151 bundle artifact: ${output}" >&2
    exit 1
  }
done
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active Route151 native-glare bundle" >&2
  exit 1
fi

wrapped=(env "PROJECT_ROOT=${project_root}" "ASSET_ROOT=${asset_root}" \
  "BUNDLE_RUN_ID=${bundle_run_id}" bash "${runner}")
printf -v wrapped_command '%q ' "${wrapped[@]}"
sbatch_args=(
  sbatch --parsable --partition=Nvidia_A800 --gres=gpu:1
  --cpus-per-task=2 --mem=192G --time=08:00:00 --exclude=gpu5
  --job-name="${job_name}"
  --output="${log_root}/${bundle_run_id}-%j.out"
  --export=ALL --wrap "${wrapped_command}"
)

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "BUNDLE_RUN_ID=${bundle_run_id}"
  echo "RESOURCE_CONTRACT=partition:Nvidia_A800,gpu:1,cpus:2,mem:192G,time:08:00:00"
  echo "PAIR_ORDER=clean,medium"
  echo "ORION_LOADS=2_full_original_model"
  echo "AUTOMATIC_RETRY=0"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${log_root}" "${submission_root}"
job_id="$("${sbatch_args[@]}")"
unattested_job_id="${job_id}"
cancel_unattested_submission() {
  if [[ -n "${unattested_job_id}" ]]; then
    scancel "${unattested_job_id}" || true
    echo "cancelled unattested Route151 bundle: ${unattested_job_id}" >&2
  fi
}
trap cancel_unattested_submission EXIT INT TERM

env "PYTHONPATH=${project_root}" "${python_bin}" - \
  "${submission_attestation}" "${activation}" "${runner}" "${job_id}" \
  "${log_root}/${bundle_run_id}-${job_id}.out" "${bundle_run_id}" <<'PY'
import datetime
import hashlib
import json
import sys

output, activation, runner, job_id, log, run_id = sys.argv[1:]
def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()
payload = {
    "schema": "orion.native_glare_bundle_submission_attestation.v1",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "single_bundle_submission_attested",
    "job_id": job_id,
    "job_name": "ng151_pair_v1",
    "bundle_run_id": run_id,
    "remote_log": log,
    "resources": {"gpu": 1, "memory_gb": 192, "cpu": 2, "time": "08:00:00"},
    "pair_order": ["clean", "medium"],
    "automatic_retry": False,
    "maximum_submissions": 1,
    "activation_amendment_sha256": digest(activation),
    "bundle_runner_sha256": digest(runner),
}
with open(output, "x") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
unattested_job_id=""
trap - EXIT INT TERM
printf '%s\n' "${job_id}"

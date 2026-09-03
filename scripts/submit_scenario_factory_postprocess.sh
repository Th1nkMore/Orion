#!/usr/bin/env bash
set -euo pipefail

submit=0
if [[ "${1:-}" == "--submit" ]]; then
  submit=1
elif [[ -n "${1:-}" && "${1:-}" != "--dry-run" ]]; then
  echo "usage: $0 [--dry-run|--submit]" >&2
  exit 2
fi

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_bin="${PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
run_id="${RUN_ID:-scenario_factory_clean_wave1_20260829}"
dependency_jobs="${DEPENDENCY_JOBS:-1091120:1091121:1091122:1091123:1091124:1091125:1091126}"
batch_manifest="${BATCH_MANIFEST:-${asset_root}/scenario_factory/batches/${run_id}/batch_manifest.json}"
results_root="${RESULTS_ROOT:-${asset_root}/results/${run_id}}"
event_root="${EVENT_ROOT:-${asset_root}/scenario_factory/event_packages/${run_id}}"
review_root="${REVIEW_ROOT:-${asset_root}/scenario_factory/review_queues/${run_id}}"
log_root="${LOG_ROOT:-${asset_root}/scenario_factory/logs}"
postprocess_partition="${SLURM_CPU_PARTITION:-Nvidia_A800}"
postprocess_gres="${POSTPROCESS_GRES-gpu:1}"
# Packaging and GIF rendering do not launch CARLA/Vulkan.  Set
# POSTPROCESS_GRES='' and choose a CPU partition to avoid consuming an A800.
exclude_nodes="${SLURM_EXCLUDE-gpu5}"

for path in "${python_bin}" "${batch_manifest}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required input: ${path}" >&2
    exit 1
  fi
done
for path in "${event_root}" "${review_root}"; do
  if [[ -e "${path}" ]]; then
    echo "refusing to overwrite postprocessing output: ${path}" >&2
    exit 1
  fi
done
if [[ ! "${dependency_jobs}" =~ ^[0-9]+(:[0-9]+)*$ ]]; then
  echo "DEPENDENCY_JOBS must be colon-separated numeric Slurm job IDs" >&2
  exit 1
fi

finalize_parts=(
  env "PYTHONPATH=${project_root}:${PYTHONPATH:-}"
  "${python_bin}" "${project_root}/scripts/finalize_scenario_factory_batch.py"
  --project-root "${project_root}"
  --batch-manifest "${batch_manifest}"
  --results-root "${results_root}"
  --output-root "${event_root}"
)
printf -v finalize_command '%q ' "${finalize_parts[@]}"

review_parts=(
  env "PYTHONPATH=${project_root}:${PYTHONPATH:-}"
  "${python_bin}" "${project_root}/scripts/build_scenario_review_queue.py"
  --batch-report "${event_root}/batch_screen_report.json"
  --output-dir "${review_root}"
)
printf -v review_command '%q ' "${review_parts[@]}"

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RUN_ID=${run_id}"
  echo "AFTERANY=${dependency_jobs}"
  echo "POSTPROCESS_PARTITION=${postprocess_partition}"
  echo "POSTPROCESS_GRES=${postprocess_gres:-none}"
  echo "EVENT_ROOT=${event_root}"
  echo "REVIEW_ROOT=${review_root}"
  exit 0
fi

mkdir -p "${log_root}"
finalize_resources=(
  "--partition=${postprocess_partition}"
  --cpus-per-task=1 --mem=16G --time=01:00:00
)
review_resources=(
  "--partition=${postprocess_partition}"
  --cpus-per-task=1 --mem=4G --time=00:15:00
)
if [[ -n "${postprocess_gres}" ]]; then
  finalize_resources+=("--gres=${postprocess_gres}")
  review_resources+=("--gres=${postprocess_gres}")
fi
if [[ -n "${exclude_nodes}" ]]; then
  finalize_resources+=("--exclude=${exclude_nodes}")
  review_resources+=("--exclude=${exclude_nodes}")
fi
finalize_job_id="$(sbatch --parsable \
  "${finalize_resources[@]}" \
  --dependency="afterany:${dependency_jobs}" \
  --job-name=scenario_finalize \
  "--output=${log_root}/${run_id}_finalize-%j.out" \
  --export=ALL --wrap "${finalize_command}")"
review_job_id="$(sbatch --parsable \
  "${review_resources[@]}" \
  --dependency="afterok:${finalize_job_id}" \
  --job-name=scenario_reviewq \
  "--output=${log_root}/${run_id}_reviewq-%j.out" \
  --export=ALL --wrap "${review_command}")"

echo "FINALIZE_JOB_ID=${finalize_job_id}"
echo "REVIEW_QUEUE_JOB_ID=${review_job_id}"
echo "EVENT_ROOT=${event_root}"
echo "REVIEW_ROOT=${review_root}"

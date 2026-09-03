#!/usr/bin/env bash
set -euo pipefail

submit=0
if [[ "${1:-}" == "--submit" ]]; then
  submit=1
elif [[ -n "${1:-}" && "${1:-}" != "--dry-run" ]]; then
  echo "usage: $0 [--dry-run|--submit]" >&2
  exit 2
fi

: "${EVENT_PACKAGE:?set EVENT_PACKAGE to a reviewed formal event package}"
: "${STAGE1_MULTIFRAME_MANIFEST:?set STAGE1_MULTIFRAME_MANIFEST to the frozen Stage1 manifest}"

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_bin="${PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
formal_plan="${FORMAL_ROUTE_PLAN:-${asset_root}/scenario_factory/formal_route_plans/stage2l_formal24_16_4_4_20260829_v1/formal_route_plan.json}"
stage1_checkpoint_sha256="${STAGE1_CHECKPOINT_SHA256:-0555f0f341c80a88e18c5864573f0be0641fb828931bea7809e2f5544665f2c8}"
qa_factory_config="${QA_FACTORY_CONFIG:-${project_root}/configs/scenario_factory/qa_factory_v5_vlm_task_fields.json}"
qa_factory_sha256="${QA_FACTORY_CONFIG_SHA256:-c942d85a3ac44b551397cbb4b47172d75594d520be8258147dee1aa4a2e9b476}"
base_qa_factory_config="${BASE_QA_FACTORY_CONFIG:-${project_root}/configs/scenario_factory/qa_factory_v2_matched_supervision.json}"
base_qa_factory_sha256="${BASE_QA_FACTORY_CONFIG_SHA256:-2236bbc84bb794abc0ce69fc3b4706b131eaa1282b2942212ef429a3b381471e}"
validator="${project_root}/scripts/validate_stage2l_formal_stage1_reuse.py"
finalizer="${project_root}/scripts/finalize_uq_relevance_multiframe_event.py"
output_parent="${OUTPUT_PARENT:-${asset_root}/scenario_factory/stage2l_formal_v5_qa}"
log_root="${LOG_ROOT:-${asset_root}/scenario_factory/logs/stage2l_formal_v5_qa}"

for path in "${python_bin}" "${formal_plan}" "${EVENT_PACKAGE}" "${STAGE1_MULTIFRAME_MANIFEST}" "${qa_factory_config}" "${base_qa_factory_config}" "${validator}" "${finalizer}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required input: ${path}" >&2
    exit 1
  fi
done
if [[ "$(sha256sum "${qa_factory_config}" | awk '{print $1}')" != "${qa_factory_sha256}" ]]; then
  echo "v5 QA factory SHA-256 mismatch" >&2
  exit 1
fi
if [[ "$(sha256sum "${base_qa_factory_config}" | awk '{print $1}')" != "${base_qa_factory_sha256}" ]]; then
  echo "v2 base QA factory SHA-256 mismatch" >&2
  exit 1
fi

validation_json="$("${python_bin}" "${validator}" \
  --formal-route-plan "${formal_plan}" \
  --event-package "${EVENT_PACKAGE}" \
  --stage1-multiframe-manifest "${STAGE1_MULTIFRAME_MANIFEST}" \
  --expected-checkpoint-sha256 "${stage1_checkpoint_sha256}")"
event_id="$(printf '%s' "${validation_json}" | "${python_bin}" -c 'import json,sys; print(json.load(sys.stdin)["event_id"])')"
stage2_split="$(printf '%s' "${validation_json}" | "${python_bin}" -c 'import json,sys; print(json.load(sys.stdin)["formal_split"])')"
output_root="${OUTPUT_ROOT:-${output_parent}/${event_id}}"
validation_output="${output_root}.stage1_reuse_validation.json"
if [[ -e "${output_root}" || -e "${validation_output}" ]]; then
  echo "refusing to overwrite existing formal v5 output: ${output_root}" >&2
  exit 1
fi

command_parts=(
  env "PYTHONPATH=${project_root}:${PYTHONPATH:-}"
  "${python_bin}" "${validator}"
  --formal-route-plan "${formal_plan}"
  --event-package "${EVENT_PACKAGE}"
  --stage1-multiframe-manifest "${STAGE1_MULTIFRAME_MANIFEST}"
  --expected-checkpoint-sha256 "${stage1_checkpoint_sha256}"
  --output "${validation_output}"
)
printf -v validate_command '%q ' "${command_parts[@]}"
finalize_parts=(
  env "PYTHONPATH=${project_root}:${PYTHONPATH:-}"
  "${python_bin}" "${finalizer}"
  --project-root "${project_root}"
  --event-package "${EVENT_PACKAGE}"
  --stage1-multiframe-manifest "${STAGE1_MULTIFRAME_MANIFEST}"
  --split "${stage2_split}"
  --output-root "${output_root}"
  --qa-factory-config "${qa_factory_config}"
  --base-qa-factory-config "${base_qa_factory_config}"
)
printf -v finalize_command '%q ' "${finalize_parts[@]}"

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "EVENT_ID=${event_id}"
  echo "STAGE2_SPLIT=${stage2_split}"
  echo "OUTPUT_ROOT=${output_root}"
  echo "QA_FACTORY_CONFIG_SHA256=${qa_factory_sha256}"
  echo "BASE_QA_FACTORY_CONFIG_SHA256=${base_qa_factory_sha256}"
  echo "STAGE1_REUSED=1"
  echo "GPU_REQUIRED_BY_WORKLOAD=0"
  echo "GPU_REQUESTED_FOR_QOS=1"
  exit 0
fi

mkdir -p "${log_root}"
job_id="$(sbatch --parsable \
  "--partition=${SLURM_PARTITION:-Nvidia_A800}" --gres=gpu:1 \
  "--cpus-per-task=${SLURM_CPUS_PER_TASK:-2}" --mem="${SLURM_MEM:-16G}" --time="${SLURM_TIME:-00:30:00}" \
  "--exclude=${SLURM_EXCLUDE:-gpu5}" \
  --job-name="v5qa_${event_id%%_*}" \
  "--output=${log_root}/${event_id}-%j.out" \
  --export=ALL --wrap "${validate_command} && ${finalize_command}")"

echo "JOB_ID=${job_id}"
echo "EVENT_ID=${event_id}"
echo "STAGE2_SPLIT=${stage2_split}"
echo "OUTPUT_ROOT=${output_root}"
echo "STAGE1_REUSE_VALIDATION=${validation_output}"
echo "GPU_REQUESTED_FOR_QOS=1"
echo "GPU_USED_BY_WORKLOAD=0"

#!/usr/bin/env bash
set -euo pipefail

submit=0
if [[ "${1:-}" == "--submit" ]]; then
  submit=1
elif [[ -n "${1:-}" && "${1:-}" != "--dry-run" ]]; then
  echo "usage: $0 [--dry-run|--submit]" >&2
  exit 2
fi

: "${EVENT_ID:?set EVENT_ID to an accepted frozen-bank event id}"
: "${EVENT_PACKAGE:?set EVENT_PACKAGE to the reviewed immutable event package}"
: "${KEYFRAME_MANIFEST:?set KEYFRAME_MANIFEST to its fixed-offset keyframe manifest}"
: "${STAGE2_SPLIT:?set STAGE2_SPLIT to train, dev, or test}"

if [[ ! "${EVENT_ID}" =~ ^route[0-9]+_step[0-9]+$ ]]; then
  echo "EVENT_ID must match route<index>_step<control_step>" >&2
  exit 1
fi
if [[ "${STAGE2_SPLIT}" != "train" && "${STAGE2_SPLIT}" != "dev" && "${STAGE2_SPLIT}" != "test" ]]; then
  echo "STAGE2_SPLIT must be train, dev, or test" >&2
  exit 1
fi

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_bin="${PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
carla_root="${CARLA_ROOT:-${asset_root}/carla/CARLA_0.9.15}"
bench2drive_root="${BENCH2DRIVE_ROOT:-${project_root}/Bench2Drive}"
bench2drive_zoo_root="${BENCH2DRIVE_ZOO_ROOT:-${project_root}/Bench2DriveZoo}"
runtime_pythonpath="${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${bench2drive_root}:${bench2drive_root}/leaderboard:${bench2drive_root}/scenario_runner:${bench2drive_zoo_root}:${PYTHONPATH:-}"
config="${ORION_CONFIG:-${project_root}/adzoo/orion/configs/orion_stage3_agent.py}"
visual_cache_config="${VISUAL_CACHE_CONFIG:-${project_root}/adzoo/orion/configs/orion_stage3_agent.py}"
checkpoint="${ORION_CHECKPOINT:-${asset_root}/checkpoints/Orion.pth}"
adapter="${ADAPTER_CHECKPOINT:-${asset_root}/checkpoints/observation_uq/counterfactual_evidence_pairwise_native_repair.pt}"
adapter_sha256="${ADAPTER_CHECKPOINT_SHA256:-0555f0f341c80a88e18c5864573f0be0641fb828931bea7809e2f5544665f2c8}"
qa_factory_config="${QA_FACTORY_CONFIG:-${project_root}/configs/scenario_factory/qa_factory_v5_vlm_task_fields.json}"
qa_factory_config_sha256="${QA_FACTORY_CONFIG_SHA256:-c942d85a3ac44b551397cbb4b47172d75594d520be8258147dee1aa4a2e9b476}"
base_qa_factory_config="${BASE_QA_FACTORY_CONFIG:-${project_root}/configs/scenario_factory/qa_factory_v2_matched_supervision.json}"
base_qa_factory_config_sha256="${BASE_QA_FACTORY_CONFIG_SHA256:-2236bbc84bb794abc0ce69fc3b4706b131eaa1282b2942212ef429a3b381471e}"
stage1_root="${STAGE1_OUTPUT_ROOT:-${asset_root}/scenario_factory/stage1_multiframe/${EVENT_ID}}"
factory_root="${FACTORY_OUTPUT_ROOT:-${asset_root}/scenario_factory/stage2l_multiframe_qa/${EVENT_ID}}"
visual_cache="${VISUAL_CACHE_OUTPUT:-${factory_root}/orion_visual_contexts.pt}"
geometry_preflight="${GEOMETRY_PREFLIGHT_OUTPUT:-${factory_root}.geometry_preflight.json}"
log_root="${LOG_ROOT:-${asset_root}/scenario_factory/logs/stage2l_multiframe_factory}"
stage2l_cpus="${STAGE2L_CPUS:-2}"

for path in "${python_bin}" "${carla_root}/PythonAPI/carla" "${EVENT_PACKAGE}" "${KEYFRAME_MANIFEST}" "${config}" "${visual_cache_config}" "${checkpoint}" "${adapter}" "${qa_factory_config}" "${base_qa_factory_config}" "${project_root}/scripts/preflight_stage2l_event_geometry.py"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required input: ${path}" >&2
    exit 1
  fi
done
actual_qa_factory_sha256="$(sha256sum "${qa_factory_config}" | awk '{print $1}')"
if [[ "${actual_qa_factory_sha256}" != "${qa_factory_config_sha256}" ]]; then
  echo "formal QA factory SHA-256 mismatch: expected ${qa_factory_config_sha256}, got ${actual_qa_factory_sha256}" >&2
  exit 1
fi
actual_base_qa_factory_sha256="$(sha256sum "${base_qa_factory_config}" | awk '{print $1}')"
if [[ "${actual_base_qa_factory_sha256}" != "${base_qa_factory_config_sha256}" ]]; then
  echo "base QA factory SHA-256 mismatch: expected ${base_qa_factory_config_sha256}, got ${actual_base_qa_factory_sha256}" >&2
  exit 1
fi
if [[ -e "${geometry_preflight}" ]]; then
  echo "refusing to overwrite geometry preflight: ${geometry_preflight}" >&2
  exit 1
fi
for path in "${stage1_root}" "${factory_root}"; do
  if [[ -e "${path}" ]]; then
    echo "refusing to overwrite event-factory output: ${path}" >&2
    exit 1
  fi
done

preflight_parts=(
  env "PYTHONPATH=${project_root}:${PYTHONPATH:-}"
  "${python_bin}" "${project_root}/scripts/preflight_stage2l_event_geometry.py"
  --event-package "${EVENT_PACKAGE}"
  --keyframe-manifest "${KEYFRAME_MANIFEST}"
  --minimum-retained 3
)
if [[ "${submit}" == "1" ]]; then
  preflight_parts+=(--output "${geometry_preflight}")
fi
"${preflight_parts[@]}" >/dev/null

extract_parts=(
  env "PYTHONPATH=${project_root}:${PYTHONPATH:-}"
  "${python_bin}" "${project_root}/scripts/extract_closedloop_stage1_uq_sequence.py"
  --event-package "${EVENT_PACKAGE}"
  --orion-config "${config}"
  --orion-checkpoint "${checkpoint}"
  --adapter-checkpoint "${adapter}"
  --adapter-checkpoint-sha256 "${adapter_sha256}"
  --keyframe-manifest "${KEYFRAME_MANIFEST}"
  --output-dir "${stage1_root}"
  --context-frames 4
  --baseline-start-frame 2
  --baseline-end-frame 8
)
printf -v extract_command '%q ' "${extract_parts[@]}"

factory_parts=(
  env "PYTHONPATH=${project_root}:${PYTHONPATH:-}"
  "${python_bin}" "${project_root}/scripts/finalize_uq_relevance_multiframe_event.py"
  --project-root "${project_root}"
  --event-package "${EVENT_PACKAGE}"
  --stage1-multiframe-manifest "${stage1_root}/stage1_observation_uq_multiframe.json"
  --split "${STAGE2_SPLIT}"
  --output-root "${factory_root}"
  --qa-factory-config "${qa_factory_config}"
  --base-qa-factory-config "${base_qa_factory_config}"
)
printf -v factory_command '%q ' "${factory_parts[@]}"
stage1_qa_command="${extract_command} && ${factory_command}"

cache_parts=(
  env "PYTHONPATH=${runtime_pythonpath}" "IS_BENCH2DRIVE=True"
  "${python_bin}" "${project_root}/scripts/cache_stage2l_multiframe_visual_contexts.py"
  --factory-report "${factory_root}/multiframe_event_factory_report.json"
  --orion-config "${visual_cache_config}"
  --orion-checkpoint "${checkpoint}"
  --output "${visual_cache}"
)
printf -v cache_command '%q ' "${cache_parts[@]}"

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "EVENT_ID=${EVENT_ID}"
  echo "STAGE2_SPLIT=${STAGE2_SPLIT}"
  echo "STAGE1_CONTROL_INFLUENCE=0"
  echo "EXPECTED_QA_PER_KEYFRAME=20"
  echo "QA_FACTORY_CONFIG=${qa_factory_config}"
  echo "QA_FACTORY_CONFIG_SHA256=${actual_qa_factory_sha256}"
  echo "BASE_QA_FACTORY_CONFIG=${base_qa_factory_config}"
  echo "BASE_QA_FACTORY_CONFIG_SHA256=${actual_base_qa_factory_sha256}"
  echo "ORION_VISUAL_CACHE_AFTER_QA=1"
  echo "GEOMETRY_PREFLIGHT_ELIGIBLE=1"
  echo "STAGE1_OUTPUT_ROOT=${stage1_root}"
  echo "FACTORY_OUTPUT_ROOT=${factory_root}"
  exit 0
fi

mkdir -p "${log_root}"
extract_job_id="$(sbatch --parsable \
  "--partition=${SLURM_PARTITION:-Nvidia_A800}" \
  --gres=gpu:1 "--cpus-per-task=${stage2l_cpus}" --mem=192G --time=02:30:00 \
  "--exclude=${SLURM_EXCLUDE:-gpu5}" \
  --job-name="s1qa_${EVENT_ID%%_*}" \
  "--output=${log_root}/${EVENT_ID}_stage1_qa-%j.out" \
  --export=ALL --wrap "${stage1_qa_command}")"
cache_job_id="$(sbatch --parsable \
  "--partition=${SLURM_PARTITION:-Nvidia_A800}" \
  --gres=gpu:1 "--cpus-per-task=${stage2l_cpus}" --mem=192G --time=01:00:00 \
  "--exclude=${VISUAL_CACHE_SLURM_EXCLUDE:-gpu5}" \
  --dependency="afterok:${extract_job_id}" \
  --job-name="vcmf_${EVENT_ID%%_*}" \
  "--output=${log_root}/${EVENT_ID}_visual_cache-%j.out" \
  --export=ALL --wrap "${cache_command}")"

echo "STAGE1_JOB_ID=${extract_job_id}"
echo "QA_FACTORY_JOB_ID=${extract_job_id}"
echo "STAGE1_QA_JOB_ID=${extract_job_id}"
echo "VISUAL_CACHE_JOB_ID=${cache_job_id}"
echo "STAGE1_OUTPUT_ROOT=${stage1_root}"
echo "FACTORY_OUTPUT_ROOT=${factory_root}"
echo "VISUAL_CACHE=${visual_cache}"
echo "GEOMETRY_PREFLIGHT=${geometry_preflight}"

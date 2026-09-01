#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]] || [[ "$1" != "lora" && "$1" != "partial_unfreeze" ]]; then
  echo "usage: $0 {lora|partial_unfreeze}" >&2
  exit 2
fi
arm="$1"

project_root="/public/home/lidachuan/project/Orion"
asset_root="/public/share/lidachuan/orion_assets"
python_bin="${asset_root}/envs/orion-cl-centos7/bin/python"
carla_root="${asset_root}/carla/CARLA_0.9.15"
bench2drive_root="${asset_root}/Bench2Drive"
bench2drive_zoo_root="${asset_root}/Bench2DriveZoo"
dataset_manifest="${asset_root}/scenario_factory/stage2l_expanded_coverage_17event_v1/manifest.json"
data_root="${asset_root}/scenario_factory/stage2l_expanded_coverage_17event_v11_1_consumer_grid_v1"
v11_records="${data_root}/records.jsonl"
dataset_audit="${data_root}/identifiability_audit.json"
view_feature_cache="${asset_root}/scenario_factory/stage2l_view_aligned_feature_cache_v1/features.pt"
u_tokenizer="${asset_root}/scenario_factory/stage1_u_tokenizer_pretraining/stage1_u_tokenizer_task_agnostic_v1_200_retry1/stage1_u_tokenizer_task_agnostic_v1.pt"
v121_root="${asset_root}/scenario_factory/stage2l_smokes/v121_factorized_r_17event_40_v1"
v121_checkpoint="${v121_root}/training/factorized_r_step040.pt"
v121_report="${v121_root}/training/report.json"
v121_validation="${v121_root}/terminal_validation.json"
config="${project_root}/adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py"
checkpoint="${asset_root}/checkpoints/Orion.pth"
protocol="${project_root}/configs/scenario_factory/stage2l_v13_1_process_qa_runtime_fix_protocol_v1.json"
launch="${project_root}/configs/scenario_factory/amendments/20260902_stage2l_v13_1_direct_u_r_gradient_anchor_launch_v1.json"
trainer="${project_root}/scripts/train_stage2l_v13_process_qa_smoke.py"
process_module="${project_root}/uq_estimator/stage2l_process_qa_v13.py"
v122_helper="${project_root}/scripts/train_stage2l_v122_vertical_slice_semantic_smoke.py"
factorized="${project_root}/uq_estimator/stage2l_factorized_relevance_v121.py"
attester="${project_root}/scripts/write_stage2l_v13_submission_attestation.py"
log_root="${asset_root}/scenario_factory/logs/stage2l_smokes"

if [[ "${arm}" == "lora" ]]; then
  run_name="v13_1_process_qa_lora_17event_200_v1"
  job_name="s2l_v131_lora"
else
  run_name="v13_1_process_qa_partial4_17event_200_v1"
  job_name="s2l_v131_p4"
fi
base_root="${asset_root}/scenario_factory/stage2l_smokes/${run_name}"
output_root="${base_root}/training"
preflight="${base_root}/trainer_preflight.json"
attestation="${base_root}/submission_attestation.json"

for prerequisite in \
  "${python_bin}" "${trainer}" "${process_module}" "${v122_helper}" \
  "${factorized}" "${attester}" "${dataset_manifest}" "${v11_records}" \
  "${dataset_audit}" "${view_feature_cache}" "${u_tokenizer}" \
  "${v121_checkpoint}" "${v121_report}" "${v121_validation}" \
  "${config}" "${checkpoint}" "${protocol}" "${preflight}" "${launch}"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "missing v13.1 prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${output_root}" ]] && find "${output_root}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty v13.1 output: ${output_root}" >&2
  exit 1
fi
if [[ -e "${attestation}" ]]; then
  echo "refusing to overwrite v13.1 submission attestation" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active v13.1 arm: ${job_name}" >&2
  exit 1
fi

runtime_pythonpath="${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${bench2drive_root}:${bench2drive_root}/leaderboard:${bench2drive_root}/scenario_runner:${bench2drive_zoo_root}:${PYTHONPATH:-}"
train_parts=(
  env "PYTHONPATH=${runtime_pythonpath}" "IS_BENCH2DRIVE=True"
  "${python_bin}" "${trainer}"
  --config "${config}"
  --checkpoint "${checkpoint}"
  --dataset-manifest "${dataset_manifest}"
  --v11-records "${v11_records}"
  --dataset-audit-report "${dataset_audit}"
  --view-feature-cache "${view_feature_cache}"
  --u-tokenizer-checkpoint "${u_tokenizer}"
  --v121-checkpoint "${v121_checkpoint}"
  --v121-report "${v121_report}"
  --v121-terminal-validation "${v121_validation}"
  --training-protocol "${protocol}"
  --training-arm "${arm}"
  --trainer-preflight "${preflight}"
  --launch-amendment "${launch}"
  --output-dir "${output_root}"
  --answer-batch-size 2
  --log-interval 10
)
printf -v train_command '%q ' "${train_parts[@]}"

mkdir -p "${log_root}"
job_id="$(sbatch --parsable \
  --partition=Nvidia_A800 \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=192G \
  --time=08:00:00 \
  --job-name="${job_name}" \
  --output="${log_root}/${run_name}-%j.out" \
  --export=ALL \
  --wrap "${train_command}")"

unattested_job_id="${job_id}"
cancel_unattested_submission() {
  if [[ -n "${unattested_job_id}" ]]; then
    scancel "${unattested_job_id}" || true
    echo "cancelled unattested v13.1 job: ${unattested_job_id}" >&2
  fi
}
trap cancel_unattested_submission EXIT INT TERM

remote_log="${log_root}/${run_name}-${job_id}.out"
env "PYTHONPATH=${project_root}" "${python_bin}" "${attester}" \
  --arm "${arm}" \
  --job-id "${job_id}" \
  --job-name "${job_name}" \
  --trainer "${trainer}" \
  --process-module "${process_module}" \
  --v122-lineage-helper "${v122_helper}" \
  --factorized-relevance "${factorized}" \
  --dataset-manifest "${dataset_manifest}" \
  --v11-records "${v11_records}" \
  --dataset-audit-report "${dataset_audit}" \
  --view-feature-cache "${view_feature_cache}" \
  --u-tokenizer-checkpoint "${u_tokenizer}" \
  --v121-checkpoint "${v121_checkpoint}" \
  --v121-report "${v121_report}" \
  --v121-terminal-validation "${v121_validation}" \
  --orion-config "${config}" \
  --orion-checkpoint "${checkpoint}" \
  --protocol "${protocol}" \
  --preflight "${preflight}" \
  --launch "${launch}" \
  --remote-log "${remote_log}" \
  --output-root "${output_root}" \
  --output "${attestation}" >/dev/null
unattested_job_id=""
trap - EXIT INT TERM
printf '%s\n' "${job_id}"

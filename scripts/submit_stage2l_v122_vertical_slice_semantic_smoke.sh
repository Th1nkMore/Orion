#!/usr/bin/env bash
set -euo pipefail

project_root="/public/home/lidachuan/project/Orion"
asset_root="/public/share/lidachuan/orion_assets"
python_bin="${asset_root}/envs/orion-cl-centos7/bin/python"
carla_root="${asset_root}/carla/CARLA_0.9.15"
bench2drive_root="${asset_root}/Bench2Drive"
bench2drive_zoo_root="${asset_root}/Bench2DriveZoo"
base_root="${asset_root}/scenario_factory/stage2l_smokes/v122_vertical_slice_semantic_17event_40_v1"
output_root="${base_root}/training"
preflight="${base_root}/trainer_preflight.json"
attestation="${base_root}/submission_attestation.json"
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
policy="${project_root}/configs/scenario_factory/amendments/20260901_vertical_slice_soft_gate_progression_v1.json"
protocol="${project_root}/configs/scenario_factory/stage2l_v12_2_vertical_slice_semantic_protocol_v1.json"
launch="${project_root}/configs/scenario_factory/amendments/20260901_stage2l_v122_vertical_slice_semantic_launch_v1.json"
trainer="${project_root}/scripts/train_stage2l_v122_vertical_slice_semantic_smoke.py"
runtime="${project_root}/uq_estimator/stage2l_factorized_runtime_v11.py"
factorized="${project_root}/uq_estimator/stage2l_factorized_relevance_v121.py"
identifiability="${project_root}/uq_estimator/stage2l_identifiability.py"
attester="${project_root}/scripts/write_stage2l_v122_submission_attestation.py"
log_root="${asset_root}/scenario_factory/logs/stage2l_smokes"
job_name="s2l_v122_semantic"

for prerequisite in \
  "${python_bin}" "${trainer}" "${runtime}" "${factorized}" \
  "${identifiability}" "${attester}" "${dataset_manifest}" \
  "${v11_records}" "${dataset_audit}" "${view_feature_cache}" \
  "${u_tokenizer}" "${v121_checkpoint}" "${v121_report}" \
  "${v121_validation}" "${config}" "${checkpoint}" "${policy}" \
  "${protocol}" "${preflight}" "${launch}"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "missing v12.2 prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${output_root}" ]] && find "${output_root}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty v12.2 output" >&2
  exit 1
fi
if [[ -e "${attestation}" ]]; then
  echo "refusing to overwrite v12.2 submission attestation" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active v12.2 submission" >&2
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
  --soft-gate-policy "${policy}"
  --training-protocol "${protocol}"
  --trainer-preflight "${preflight}"
  --launch-amendment "${launch}"
  --output-dir "${output_root}"
  --answer-batch-size 4
  --log-interval 5
)
printf -v train_command '%q ' "${train_parts[@]}"

mkdir -p "${log_root}"
job_id="$(sbatch --parsable \
  --partition=Nvidia_A800 \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=192G \
  --time=08:00:00 \
  --exclude=gpu5 \
  --job-name="${job_name}" \
  --output="${log_root}/v122_vertical_slice_semantic_17event_40_v1-%j.out" \
  --export=ALL \
  --wrap "${train_command}")"

unattested_job_id="${job_id}"
cancel_unattested_submission() {
  if [[ -n "${unattested_job_id}" ]]; then
    scancel "${unattested_job_id}" || true
    echo "cancelled unattested v12.2 job: ${unattested_job_id}" >&2
  fi
}
trap cancel_unattested_submission EXIT INT TERM

remote_log="${log_root}/v122_vertical_slice_semantic_17event_40_v1-${job_id}.out"
env "PYTHONPATH=${project_root}" "${python_bin}" "${attester}" \
  --job-id "${job_id}" \
  --trainer "${trainer}" \
  --runtime "${runtime}" \
  --factorized-relevance "${factorized}" \
  --identifiability-audit "${identifiability}" \
  --soft-gate-policy "${policy}" \
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

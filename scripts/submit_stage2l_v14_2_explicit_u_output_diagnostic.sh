#!/usr/bin/env bash
set -euo pipefail

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
config="${project_root}/adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py"
checkpoint="${asset_root}/checkpoints/Orion.pth"
trained_checkpoint="${asset_root}/scenario_factory/stage2l_smokes/v14_1_u_concept_lora_17event_200_v1/training/stage2l_v14_u_concept_lora.pt"
protocol="${project_root}/configs/scenario_factory/stage2l_v14_2_explicit_u_output_diagnostic_v1.json"
evaluator="${project_root}/scripts/evaluate_stage2l_v14_2_explicit_u_output.py"
prompt_module="${project_root}/uq_estimator/stage2l_u_concept_explicit_schema_v14_2.py"
run_name="v14_2_explicit_output_diagnostic_v1"
job_name="s2l_v142_out"
output="${asset_root}/scenario_factory/stage2l_smokes/${run_name}/report.json"
preflight="${asset_root}/scenario_factory/stage2l_smokes/${run_name}/preflight.json"
log_root="${asset_root}/scenario_factory/logs/stage2l_smokes"

for prerequisite in \
  "${python_bin}" "${evaluator}" "${prompt_module}" "${dataset_manifest}" \
  "${v11_records}" "${dataset_audit}" "${view_feature_cache}" \
  "${u_tokenizer}" "${config}" "${checkpoint}" "${trained_checkpoint}" \
  "${protocol}"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "missing v14.2 explicit-output prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${output}" || -e "${preflight}" ]]; then
  echo "refusing to overwrite an existing v14.2 diagnostic" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active v14.2 diagnostic" >&2
  exit 1
fi

runtime_pythonpath="${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${bench2drive_root}:${bench2drive_root}/leaderboard:${bench2drive_root}/scenario_runner:${bench2drive_zoo_root}:${PYTHONPATH:-}"
common=(
  env "PYTHONPATH=${runtime_pythonpath}" "IS_BENCH2DRIVE=True"
  "${python_bin}" "${evaluator}"
  --config "${config}"
  --checkpoint "${checkpoint}"
  --trained-checkpoint "${trained_checkpoint}"
  --dataset-manifest "${dataset_manifest}"
  --v11-records "${v11_records}"
  --dataset-audit-report "${dataset_audit}"
  --view-feature-cache "${view_feature_cache}"
  --u-tokenizer-checkpoint "${u_tokenizer}"
  --protocol "${protocol}"
)

mkdir -p "$(dirname "${output}")" "${log_root}"
"${common[@]}" --output "${preflight}" --preflight-only

run_parts=("${common[@]}" --output "${output}")
printf -v run_command '%q ' "${run_parts[@]}"
job_id="$(sbatch --parsable \
  --partition=Nvidia_A800 \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=192G \
  --time=02:00:00 \
  --job-name="${job_name}" \
  --output="${log_root}/${run_name}-%j.out" \
  --export=ALL \
  --wrap "${run_command}")"
printf '%s\n' "${job_id}"

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
config="${project_root}/adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py"
checkpoint="${asset_root}/checkpoints/Orion.pth"
v15_root="${asset_root}/scenario_factory/stage2l_smokes/v15_u_language_alignment_17event_720_v1/training"
v15_checkpoint="${v15_root}/stage2l_v15_u_language_alignment.pt"
v15_report="${v15_root}/report.json"
protocol="${project_root}/configs/scenario_factory/stage2l_v15_2_text_oracle_localization_v1.json"
evaluator="${project_root}/scripts/evaluate_stage2l_v15_2_text_oracle_localization.py"
prompt_module="${project_root}/uq_estimator/stage2l_u_text_oracle_v15_2.py"
run_name="v15_2_text_oracle_localization_120dev_v1"
job_name="s2l_v152_txt"
run_root="${asset_root}/scenario_factory/stage2l_smokes/${run_name}"
preflight="${run_root}/preflight.json"
output="${run_root}/report.json"
log_root="${asset_root}/scenario_factory/logs/stage2l_smokes"

for prerequisite in \
  "${python_bin}" "${evaluator}" "${prompt_module}" "${dataset_manifest}" \
  "${v11_records}" "${dataset_audit}" "${view_feature_cache}" \
  "${config}" "${checkpoint}" "${v15_checkpoint}" "${v15_report}" \
  "${protocol}"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "missing v15.2 text-oracle prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${preflight}" || -e "${output}" ]]; then
  echo "refusing to overwrite an existing v15.2 text-oracle artifact" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active v15.2 text-oracle job" >&2
  exit 1
fi

runtime_pythonpath="${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${bench2drive_root}:${bench2drive_root}/leaderboard:${bench2drive_root}/scenario_runner:${bench2drive_zoo_root}:${PYTHONPATH:-}"
common=(
  env "PYTHONPATH=${runtime_pythonpath}" "IS_BENCH2DRIVE=True"
  "${python_bin}" "${evaluator}"
  --config "${config}"
  --checkpoint "${checkpoint}"
  --v15-checkpoint "${v15_checkpoint}"
  --v15-report "${v15_report}"
  --dataset-manifest "${dataset_manifest}"
  --v11-records "${v11_records}"
  --dataset-audit-report "${dataset_audit}"
  --view-feature-cache "${view_feature_cache}"
  --protocol "${protocol}"
  --answer-batch-size 10
)

mkdir -p "${run_root}" "${log_root}"
"${common[@]}" --preflight-only --output "${preflight}"

run_parts=("${common[@]}" --preflight "${preflight}" --output "${output}")
printf -v run_command '%q ' "${run_parts[@]}"
job_id="$(sbatch --parsable \
  --partition=Nvidia_A800 \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=192G \
  --time=12:00:00 \
  --job-name="${job_name}" \
  --output="${log_root}/${run_name}-%j.out" \
  --export=ALL \
  --wrap "${run_command}")"
printf '%s\n' "${job_id}"

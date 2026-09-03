#!/usr/bin/env bash
set -euo pipefail

project_root="/public/home/lidachuan/project/Orion"
asset_root="/public/share/lidachuan/orion_assets"
python_bin="${asset_root}/envs/orion-cl-centos7/bin/python"
carla_root="${asset_root}/carla/CARLA_0.9.15"
bench2drive_root="${asset_root}/Bench2Drive"
bench2drive_zoo_root="${asset_root}/Bench2DriveZoo"
output_dir="${asset_root}/scenario_factory/stage2l_smokes/route151_v7_calibrated_v1_20/training"
log_root="${asset_root}/scenario_factory/logs/stage2l_smokes"
job_name="s2l_v7_r151_smoke"

runtime_pythonpath="${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${bench2drive_root}:${bench2drive_root}/leaderboard:${bench2drive_root}/scenario_runner:${bench2drive_zoo_root}:${PYTHONPATH:-}"

if [[ -e "${output_dir}" ]] && find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty v7 smoke output: ${output_dir}" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active v7 smoke submission: ${job_name}" >&2
  exit 1
fi

mkdir -p "${log_root}"
train_parts=(
  env "PYTHONPATH=${runtime_pythonpath}" "IS_BENCH2DRIVE=True"
  "${python_bin}" "${project_root}/scripts/train_stage2l_v7_route151_smoke.py"
  --config "${project_root}/adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py"
  --checkpoint "${asset_root}/checkpoints/Orion.pth"
  --records "${asset_root}/scenario_factory/stage2l_multiframe_qa/route151_step218/qa_dataset_v3_rebased_v1/records.jsonl"
  --visual-cache "${asset_root}/scenario_factory/stage2l_multiframe_qa/route151_step218/orion_visual_contexts.pt"
  --qa-config "${project_root}/configs/scenario_factory/qa_factory_v3_calibrated_semantics.json"
  --training-protocol "${project_root}/configs/scenario_factory/stage2l_training_v7_calibrated_matched_semantics.json"
  --objective-diagnostic "${asset_root}/scenario_factory/stage2l_smokes/route151_v7_objective_feasibility_v2/report.json"
  --launch-amendment "${project_root}/configs/scenario_factory/amendments/20260830_stage2l_v7_route151_calibrated_smoke_v1.json"
  --output-dir "${output_dir}"
  --max-optimizer-steps 20
  --answer-batch-size 2
)
printf -v train_command '%q ' "${train_parts[@]}"

job_id="$(sbatch --parsable \
    --partition=Nvidia_A800 \
    --gres=gpu:1 \
    --cpus-per-task=2 \
    --mem=192G \
    --time=04:00:00 \
    --exclude=gpu5 \
    --job-name="${job_name}" \
    --output="${log_root}/route151_v7_calibrated_v1_20-%j.out" \
    --export=ALL \
    --wrap "${train_command}")"
echo "${job_id}"

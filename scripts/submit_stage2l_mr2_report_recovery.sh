#!/usr/bin/env bash
set -euo pipefail

project_root="/public/home/lidachuan/project/Orion"
asset_root="/public/share/lidachuan/orion_assets"
python_bin="${asset_root}/envs/orion-cl-centos7/bin/python"
carla_root="${asset_root}/carla/CARLA_0.9.15"
bench2drive_root="${asset_root}/Bench2Drive"
bench2drive_zoo_root="${asset_root}/Bench2DriveZoo"
run_root="${asset_root}/scenario_factory/stage2l_smokes/mr2_17event_v1_40"
training_root="${run_root}/training"
failed_log="${asset_root}/scenario_factory/logs/stage2l_smokes/mr2_17event_v1_40-1109479.out"
output_report="${training_root}/report.recovered.json"
recovery_log_root="${asset_root}/scenario_factory/logs/stage2l_smokes"
job_name="s2l_mr2_recover"

for prerequisite in \
  "${project_root}/scripts/recover_stage2l_mr2_report.py" \
  "${project_root}/scripts/train_stage2l_mr2_coverage_smoke.py" \
  "${project_root}/scripts/train_stage2l_mr1_smoke.py" \
  "${project_root}/configs/scenario_factory/stage2l_mr2_17event_coverage_v1.json" \
  "${project_root}/configs/scenario_factory/amendments/20260831_stage2l_mr2_17event_coverage_launch_v1.json" \
  "${project_root}/configs/scenario_factory/amendments/20260831_stage2l_mr2_report_recovery_v1.json" \
  "${run_root}/trainer_preflight.json" \
  "${training_root}/stage2l_mr1_multiroute_smoke.pt" \
  "${failed_log}"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "missing MR2 recovery prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${output_report}" ]]; then
  echo "refusing to overwrite recovered MR2 report" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active MR2 recovery" >&2
  exit 1
fi

runtime_pythonpath="${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${bench2drive_root}:${bench2drive_root}/leaderboard:${bench2drive_root}/scenario_runner:${bench2drive_zoo_root}:${PYTHONPATH:-}"
command_parts=(
  env "PYTHONPATH=${runtime_pythonpath}" "IS_BENCH2DRIVE=True"
  "${python_bin}" "${project_root}/scripts/recover_stage2l_mr2_report.py"
  --config "${project_root}/adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py"
  --base-checkpoint "${asset_root}/checkpoints/Orion.pth"
  --dataset-manifest "${asset_root}/scenario_factory/stage2l_expanded_coverage_17event_v1/manifest.json"
  --training-protocol "${project_root}/configs/scenario_factory/stage2l_mr2_17event_coverage_v1.json"
  --trainer-preflight "${run_root}/trainer_preflight.json"
  --launch-amendment "${project_root}/configs/scenario_factory/amendments/20260831_stage2l_mr2_17event_coverage_launch_v1.json"
  --original-trainer "${project_root}/scripts/train_stage2l_mr2_coverage_smoke.py"
  --original-base-trainer "${project_root}/scripts/train_stage2l_mr1_smoke.py"
  --failed-log "${failed_log}"
  --trained-checkpoint "${training_root}/stage2l_mr1_multiroute_smoke.pt"
  --recovery-amendment "${project_root}/configs/scenario_factory/amendments/20260831_stage2l_mr2_report_recovery_v1.json"
  --output-report "${output_report}"
)
printf -v recovery_command '%q ' "${command_parts[@]}"
mkdir -p "${recovery_log_root}"
sbatch --parsable \
  --partition=Nvidia_A800 \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=192G \
  --time=04:00:00 \
  --exclude=gpu5 \
  --job-name="${job_name}" \
  --output="${recovery_log_root}/mr2_17event_v1_40-recovery-%j.out" \
  --export=ALL \
  --wrap "${recovery_command}"

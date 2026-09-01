#!/usr/bin/env bash
set -euo pipefail

project_root="/public/home/lidachuan/project/Orion"
asset_root="/public/share/lidachuan/orion_assets"
python_bin="${asset_root}/envs/orion-cl-centos7/bin/python"
carla_root="${asset_root}/carla/CARLA_0.9.15"
bench2drive_root="${asset_root}/Bench2Drive"
bench2drive_zoo_root="${asset_root}/Bench2DriveZoo"
dataset_manifest="${asset_root}/scenario_factory/stage2l_expanded_coverage_17event_v1/manifest.json"
protocol="${project_root}/configs/scenario_factory/stage2l_mr2_17event_coverage_v1.json"
preflight="${asset_root}/scenario_factory/stage2l_smokes/mr2_17event_v1_40/trainer_preflight.json"
amendment="${project_root}/configs/scenario_factory/amendments/20260831_stage2l_mr2_17event_coverage_launch_v1.json"
output_parent="${asset_root}/scenario_factory/stage2l_smokes/mr2_17event_v1_40"
output_dir="${output_parent}/training"
attestation="${output_parent}/submission_attestation.json"
log_root="${asset_root}/scenario_factory/logs/stage2l_smokes"
job_name="s2l_mr2_17event"

for prerequisite in \
  "${project_root}/scripts/train_stage2l_mr2_coverage_smoke.py" \
  "${project_root}/scripts/train_stage2l_mr1_smoke.py" \
  "${project_root}/scripts/write_stage2l_mr2_submission_attestation.py" \
  "${dataset_manifest}" "${protocol}" "${preflight}" "${amendment}"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "missing MR2 prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${output_dir}" ]] && find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty MR2 output: ${output_dir}" >&2
  exit 1
fi
if [[ -e "${attestation}" ]]; then
  echo "refusing to overwrite MR2 submission attestation" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active MR2 submission: ${job_name}" >&2
  exit 1
fi

runtime_pythonpath="${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${bench2drive_root}:${bench2drive_root}/leaderboard:${bench2drive_root}/scenario_runner:${bench2drive_zoo_root}:${PYTHONPATH:-}"
train_parts=(
  env "PYTHONPATH=${runtime_pythonpath}" "IS_BENCH2DRIVE=True"
  "${python_bin}" "${project_root}/scripts/train_stage2l_mr2_coverage_smoke.py"
  --config "${project_root}/adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py"
  --checkpoint "${asset_root}/checkpoints/Orion.pth"
  --dataset-manifest "${dataset_manifest}"
  --training-protocol "${protocol}"
  --trainer-preflight "${preflight}"
  --launch-amendment "${amendment}"
  --output-dir "${output_dir}"
  --max-optimizer-steps 40
  --answer-batch-size 2
  --language-anchors-per-step 6
)
printf -v train_command '%q ' "${train_parts[@]}"

mkdir -p "${log_root}"
job_id="$(sbatch --parsable \
    --partition=Nvidia_A800 \
    --gres=gpu:1 \
    --cpus-per-task=2 \
    --mem=192G \
    --time=12:00:00 \
    --exclude=gpu5 \
    --job-name="${job_name}" \
    --output="${log_root}/mr2_17event_v1_40-%j.out" \
    --export=ALL \
    --wrap "${train_command}")"

unattested_job_id="${job_id}"
cancel_unattested_submission() {
  if [[ -n "${unattested_job_id}" ]]; then
    scancel "${unattested_job_id}" || true
    echo "cancelled unattested MR2 job: ${unattested_job_id}" >&2
  fi
}
trap cancel_unattested_submission EXIT INT TERM

remote_log="${log_root}/mr2_17event_v1_40-${job_id}.out"
env "PYTHONPATH=${project_root}" "${python_bin}" \
  "${project_root}/scripts/write_stage2l_mr2_submission_attestation.py" \
  --amendment "${amendment}" \
  --training-protocol "${protocol}" \
  --dataset-manifest "${dataset_manifest}" \
  --trainer-preflight "${preflight}" \
  --trainer "${project_root}/scripts/train_stage2l_mr2_coverage_smoke.py" \
  --job-id "${job_id}" \
  --remote-log "${remote_log}" \
  --output "${attestation}" >/dev/null
unattested_job_id=""
trap - EXIT INT TERM
printf '%s\n' "${job_id}"

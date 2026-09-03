#!/usr/bin/env bash
set -euo pipefail

project_root="/public/home/lidachuan/project/Orion"
asset_root="/public/share/lidachuan/orion_assets"
python_bin="${asset_root}/envs/orion-cl-centos7/bin/python"
carla_root="${asset_root}/carla/CARLA_0.9.15"
bench2drive_root="${asset_root}/Bench2Drive"
bench2drive_zoo_root="${asset_root}/Bench2DriveZoo"
dataset_manifest="${asset_root}/scenario_factory/stage2l_expanded_coverage_17event_v1/manifest.json"
u_tokenizer="${asset_root}/scenario_factory/stage1_u_tokenizer_pretraining/stage1_u_tokenizer_task_agnostic_v1_200_retry1/stage1_u_tokenizer_task_agnostic_v1.pt"
protocol="${project_root}/configs/scenario_factory/stage2l_v10_accelerated_72h_v1.json"
preflight="${asset_root}/scenario_factory/stage2l_smokes/v10_staged_17event_40x3_v1/trainer_preflight.json"
amendment="${project_root}/configs/scenario_factory/amendments/20260831_stage2l_v10_accelerated_launch_v1.json"
trainer="${project_root}/scripts/train_stage2l_v10_staged_smoke.py"
config="${project_root}/adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py"
checkpoint="${asset_root}/checkpoints/Orion.pth"
output_parent="${asset_root}/scenario_factory/stage2l_smokes/v10_staged_17event_40x3_v1"
output_dir="${output_parent}/training"
attestation="${output_parent}/submission_attestation.json"
log_root="${asset_root}/scenario_factory/logs/stage2l_smokes"
job_name="s2l_v10_17e"

for prerequisite in \
  "${python_bin}" "${trainer}" "${project_root}/scripts/write_stage2l_v10_submission_attestation.py" \
  "${dataset_manifest}" "${u_tokenizer}" "${protocol}" "${preflight}" \
  "${amendment}" "${config}" "${checkpoint}"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "missing Stage2-L v10 prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${output_dir}" ]] && find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty v10 output: ${output_dir}" >&2
  exit 1
fi
if [[ -e "${attestation}" ]]; then
  echo "refusing to overwrite v10 submission attestation" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active v10 submission: ${job_name}" >&2
  exit 1
fi

runtime_pythonpath="${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${bench2drive_root}:${bench2drive_root}/leaderboard:${bench2drive_root}/scenario_runner:${bench2drive_zoo_root}:${PYTHONPATH:-}"
train_parts=(
  env "PYTHONPATH=${runtime_pythonpath}" "IS_BENCH2DRIVE=True"
  "${python_bin}" "${trainer}"
  --config "${config}"
  --checkpoint "${checkpoint}"
  --dataset-manifest "${dataset_manifest}"
  --u-tokenizer-checkpoint "${u_tokenizer}"
  --training-protocol "${protocol}"
  --trainer-preflight "${preflight}"
  --launch-amendment "${amendment}"
  --output-dir "${output_dir}"
  --answer-batch-size 2
  --log-interval 5
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
    --output="${log_root}/v10_staged_17event_40x3_v1-%j.out" \
    --export=ALL \
    --wrap "${train_command}")"

unattested_job_id="${job_id}"
cancel_unattested_submission() {
  if [[ -n "${unattested_job_id}" ]]; then
    scancel "${unattested_job_id}" || true
    echo "cancelled unattested v10 job: ${unattested_job_id}" >&2
  fi
}
trap cancel_unattested_submission EXIT INT TERM

remote_log="${log_root}/v10_staged_17event_40x3_v1-${job_id}.out"
env "PYTHONPATH=${project_root}" "${python_bin}" \
  "${project_root}/scripts/write_stage2l_v10_submission_attestation.py" \
  --amendment "${amendment}" \
  --protocol "${protocol}" \
  --preflight "${preflight}" \
  --dataset-manifest "${dataset_manifest}" \
  --u-tokenizer-checkpoint "${u_tokenizer}" \
  --orion-config "${config}" \
  --orion-checkpoint "${checkpoint}" \
  --trainer "${trainer}" \
  --job-id "${job_id}" \
  --remote-log "${remote_log}" \
  --output "${attestation}" >/dev/null
unattested_job_id=""
trap - EXIT INT TERM
printf '%s\n' "${job_id}"

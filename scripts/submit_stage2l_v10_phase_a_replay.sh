#!/usr/bin/env bash
set -euo pipefail

project_root="/public/home/lidachuan/project/Orion"
asset_root="/public/share/lidachuan/orion_assets"
python_bin="${asset_root}/envs/orion-cl-centos7/bin/python"
carla_root="${asset_root}/carla/CARLA_0.9.15"
bench2drive_root="${asset_root}/Bench2Drive"
bench2drive_zoo_root="${asset_root}/Bench2DriveZoo"
base_root="${asset_root}/scenario_factory/stage2l_smokes/v10_staged_17event_40x3_v1"
dataset_manifest="${asset_root}/scenario_factory/stage2l_expanded_coverage_17event_v1/manifest.json"
u_tokenizer="${asset_root}/scenario_factory/stage1_u_tokenizer_pretraining/stage1_u_tokenizer_task_agnostic_v1_200_retry1/stage1_u_tokenizer_task_agnostic_v1.pt"
protocol="${project_root}/configs/scenario_factory/stage2l_v10_phase_a_replay_v1.json"
preflight="${base_root}/phase_a_replay_v1_preflight.json"
amendment="${project_root}/configs/scenario_factory/amendments/20260831_stage2l_v10_phase_a_replay_launch_v1.json"
evaluator="${project_root}/scripts/evaluate_stage2l_v10_phase_a_checkpoint.py"
attester="${project_root}/scripts/write_stage2l_v10_phase_a_replay_attestation.py"
config="${project_root}/adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py"
checkpoint="${asset_root}/checkpoints/Orion.pth"
phase_a_checkpoint="${base_root}/training/phase_a.pt"
v10_report="${base_root}/training/report.json"
output_dir="${base_root}/phase_a_replay_v1"
attestation="${base_root}/phase_a_replay_v1_submission_attestation.json"
log_root="${asset_root}/scenario_factory/logs/stage2l_smokes"
job_name="s2l_v10_replay"

for prerequisite in \
  "${python_bin}" "${evaluator}" "${attester}" "${dataset_manifest}" \
  "${u_tokenizer}" "${protocol}" "${preflight}" "${amendment}" \
  "${config}" "${checkpoint}" "${phase_a_checkpoint}" "${v10_report}"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "missing Phase-A replay prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${output_dir}" ]] && find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty Phase-A replay output: ${output_dir}" >&2
  exit 1
fi
if [[ -e "${attestation}" ]]; then
  echo "refusing to overwrite Phase-A replay submission attestation" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active Phase-A replay submission" >&2
  exit 1
fi

runtime_pythonpath="${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${bench2drive_root}:${bench2drive_root}/leaderboard:${bench2drive_root}/scenario_runner:${bench2drive_zoo_root}:${PYTHONPATH:-}"
replay_parts=(
  env "PYTHONPATH=${runtime_pythonpath}" "IS_BENCH2DRIVE=True"
  "${python_bin}" "${evaluator}"
  --config "${config}"
  --checkpoint "${checkpoint}"
  --dataset-manifest "${dataset_manifest}"
  --u-tokenizer-checkpoint "${u_tokenizer}"
  --phase-a-checkpoint "${phase_a_checkpoint}"
  --v10-report "${v10_report}"
  --protocol "${protocol}"
  --preflight "${preflight}"
  --launch-amendment "${amendment}"
  --output-dir "${output_dir}"
)
printf -v replay_command '%q ' "${replay_parts[@]}"

mkdir -p "${log_root}"
job_id="$(sbatch --parsable \
    --partition=Nvidia_A800 \
    --gres=gpu:1 \
    --cpus-per-task=2 \
    --mem=192G \
    --time=04:00:00 \
    --exclude=gpu5 \
    --job-name="${job_name}" \
    --output="${log_root}/v10_phase_a_replay_v1-%j.out" \
    --export=ALL \
    --wrap "${replay_command}")"

unattested_job_id="${job_id}"
cancel_unattested_submission() {
  if [[ -n "${unattested_job_id}" ]]; then
    scancel "${unattested_job_id}" || true
    echo "cancelled unattested Phase-A replay job: ${unattested_job_id}" >&2
  fi
}
trap cancel_unattested_submission EXIT INT TERM

remote_log="${log_root}/v10_phase_a_replay_v1-${job_id}.out"
env "PYTHONPATH=${project_root}" "${python_bin}" "${attester}" \
  --amendment "${amendment}" \
  --protocol "${protocol}" \
  --preflight "${preflight}" \
  --evaluator "${evaluator}" \
  --dataset-manifest "${dataset_manifest}" \
  --orion-config "${config}" \
  --orion-checkpoint "${checkpoint}" \
  --u-tokenizer-checkpoint "${u_tokenizer}" \
  --phase-a-checkpoint "${phase_a_checkpoint}" \
  --v10-report "${v10_report}" \
  --job-id "${job_id}" \
  --remote-log "${remote_log}" \
  --output "${attestation}" >/dev/null
unattested_job_id=""
trap - EXIT INT TERM
printf '%s\n' "${job_id}"

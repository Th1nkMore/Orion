#!/usr/bin/env bash
set -euo pipefail

project_root="/public/home/lidachuan/project/Orion"
asset_root="/public/share/lidachuan/orion_assets"
python_bin="${asset_root}/envs/orion-cl-centos7/bin/python"
carla_root="${asset_root}/carla/CARLA_0.9.15"
bench2drive_root="${asset_root}/Bench2Drive"
bench2drive_zoo_root="${asset_root}/Bench2DriveZoo"
dataset_root="${asset_root}/scenario_factory/stage2l_multiframe_qa/route151_step218/qa_dataset_v5_task_fields_v1"
v9_preflight="${asset_root}/scenario_factory/stage2l_smokes/route151_v9_architecture_data_preflight_v4/preflight.json"
trainer_preflight="${asset_root}/scenario_factory/stage2l_smokes/route151_v9_trainer_preflight_v3/preflight.json"
amendment="${project_root}/configs/scenario_factory/amendments/20260831_stage2l_v9_route151_task_field_smoke_v1.json"
output_parent="${asset_root}/scenario_factory/stage2l_smokes/route151_v9_vlm_task_fields_v2_20"
output_dir="${output_parent}/training"
attestation="${output_parent}/submission_attestation.json"
log_root="${asset_root}/scenario_factory/logs/stage2l_smokes"
job_name="s2l_v9_r151_smoke"

if [[ ! -f "${amendment}" ]]; then
  echo "v9 smoke remains locked: immutable launch amendment is absent" >&2
  exit 2
fi
if [[ -e "${output_dir}" ]] && find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty v9 smoke output: ${output_dir}" >&2
  exit 1
fi
if [[ -e "${attestation}" ]]; then
  echo "refusing to overwrite v9 submission attestation" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active v9 smoke submission: ${job_name}" >&2
  exit 1
fi

runtime_pythonpath="${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${bench2drive_root}:${bench2drive_root}/leaderboard:${bench2drive_root}/scenario_runner:${bench2drive_zoo_root}:${PYTHONPATH:-}"
attester_parts=(
  env "PYTHONPATH=${runtime_pythonpath}"
  "${python_bin}" "${project_root}/scripts/write_stage2l_v9_submission_attestation.py"
  --project-root "${project_root}"
  --amendment "${amendment}"
  --records "${dataset_root}/records.jsonl"
  --visual-cache "${asset_root}/scenario_factory/stage2l_multiframe_qa/route151_step218/orion_visual_contexts.pt"
  --v9-preflight "${v9_preflight}"
  --trainer-preflight "${trainer_preflight}"
  --dataset-audit "${dataset_root}/audit.json"
  --reference-audit "${dataset_root}/reference_audit.json"
  --orion-config "${project_root}/adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py"
  --base-checkpoint "${asset_root}/checkpoints/Orion.pth"
  --remote-output-dir "${output_dir}"
  --output "${attestation}"
)

# Run the exact attester before sbatch.  This catches missing PYTHONPATH,
# stale hashes, bad bounds, and unwritable/occupied attestation targets before
# a job id can exist.
"${attester_parts[@]}" --validate-only >/dev/null

train_parts=(
  env "PYTHONPATH=${runtime_pythonpath}" "IS_BENCH2DRIVE=True"
  "${python_bin}" "${project_root}/scripts/train_stage2l_v9_route151_smoke.py"
  --config "${project_root}/adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py"
  --checkpoint "${asset_root}/checkpoints/Orion.pth"
  --records "${dataset_root}/records.jsonl"
  --visual-cache "${asset_root}/scenario_factory/stage2l_multiframe_qa/route151_step218/orion_visual_contexts.pt"
  --qa-config "${project_root}/configs/scenario_factory/qa_factory_v5_vlm_task_fields.json"
  --training-protocol "${project_root}/configs/scenario_factory/stage2l_training_v9_vlm_task_fields.json"
  --v9-preflight "${v9_preflight}"
  --dataset-audit "${dataset_root}/audit.json"
  --reference-audit "${dataset_root}/reference_audit.json"
  --launch-amendment "${amendment}"
  --output-dir "${output_dir}"
  --max-optimizer-steps 20
  --answer-batch-size 2
)
printf -v train_command '%q ' "${train_parts[@]}"

mkdir -p "${log_root}"
job_id="$(sbatch --parsable \
    --partition=Nvidia_A800 \
    --gres=gpu:1 \
    --cpus-per-task=2 \
    --mem=192G \
    --time=06:00:00 \
    --exclude=gpu5 \
    --job-name="${job_name}" \
    --output="${log_root}/route151_v9_vlm_task_fields_v2_20-%j.out" \
    --export=ALL \
    --wrap "${train_command}")"

# A Slurm allocation is not allowed to outlive a failed or interrupted
# attestation write.  Keep the job armed for cancellation until the immutable
# attestation has been created successfully; there is no automatic retry.
unattested_job_id="${job_id}"
cancel_unattested_submission() {
  if [[ -n "${unattested_job_id}" ]]; then
    scancel "${unattested_job_id}" || true
    echo "cancelled unattested v9 smoke job: ${unattested_job_id}" >&2
  fi
}
trap cancel_unattested_submission EXIT INT TERM

remote_log="${log_root}/route151_v9_vlm_task_fields_v2_20-${job_id}.out"
"${attester_parts[@]}" --job-id "${job_id}" --remote-log "${remote_log}" >/dev/null
unattested_job_id=""
trap - EXIT INT TERM
printf '%s\n' "${job_id}"

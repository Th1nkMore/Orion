#!/usr/bin/env bash
set -euo pipefail

project_root="/public/home/lidachuan/project/Orion"
asset_root="/public/share/lidachuan/orion_assets"
python_bin="${asset_root}/envs/orion-cl-centos7/bin/python"
carla_root="${asset_root}/carla/CARLA_0.9.15"
bench2drive_root="${asset_root}/Bench2Drive"
bench2drive_zoo_root="${asset_root}/Bench2DriveZoo"
dataset_root="${asset_root}/scenario_factory/stage2l_multiframe_qa/route151_step218/qa_dataset_v4_gradient_routed_v1"
v8_preflight="${asset_root}/scenario_factory/stage2l_smokes/route151_v8_objective_data_preflight_v5/preflight.json"
trainer_preflight="${asset_root}/scenario_factory/stage2l_smokes/route151_v8_trainer_preflight_v3/preflight.json"
amendment="${project_root}/configs/scenario_factory/amendments/20260830_stage2l_v8_route151_gradient_routed_smoke_v1.json"
output_parent="${asset_root}/scenario_factory/stage2l_smokes/route151_v8_gradient_routed_v1_60"
output_dir="${output_parent}/training"
log_root="${asset_root}/scenario_factory/logs/stage2l_smokes"
job_name="s2l_v8_r151_smoke"

if [[ ! -f "${amendment}" ]]; then
  echo "v8 smoke remains locked: immutable launch amendment is absent" >&2
  exit 2
fi
if [[ -e "${output_dir}" ]] && find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty v8 smoke output: ${output_dir}" >&2
  exit 1
fi
if [[ -e "${output_parent}/submission_attestation.json" ]]; then
  echo "refusing to overwrite v8 submission attestation" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active v8 smoke submission: ${job_name}" >&2
  exit 1
fi

runtime_pythonpath="${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${bench2drive_root}:${bench2drive_root}/leaderboard:${bench2drive_root}/scenario_runner:${bench2drive_zoo_root}:${PYTHONPATH:-}"
mkdir -p "${log_root}"
train_parts=(
  env "PYTHONPATH=${runtime_pythonpath}" "IS_BENCH2DRIVE=True"
  "${python_bin}" "${project_root}/scripts/train_stage2l_v8_route151_smoke.py"
  --config "${project_root}/adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py"
  --checkpoint "${asset_root}/checkpoints/Orion.pth"
  --records "${dataset_root}/records.jsonl"
  --visual-cache "${asset_root}/scenario_factory/stage2l_multiframe_qa/route151_step218/orion_visual_contexts.pt"
  --qa-config "${project_root}/configs/scenario_factory/qa_factory_v4_structured_semantics.json"
  --training-protocol "${project_root}/configs/scenario_factory/stage2l_training_v8_gradient_routed_structured_qa.json"
  --v8-preflight "${v8_preflight}"
  --dataset-audit "${dataset_root}/audit.json"
  --reference-audit "${dataset_root}/reference_audit.json"
  --launch-amendment "${amendment}"
  --output-dir "${output_dir}"
  --max-optimizer-steps 60
  --answer-batch-size 2
)
printf -v train_command '%q ' "${train_parts[@]}"

job_id="$(sbatch --parsable \
    --partition=Nvidia_A800 \
    --gres=gpu:1 \
    --cpus-per-task=2 \
    --mem=192G \
    --time=08:00:00 \
    --exclude=gpu5 \
    --job-name="${job_name}" \
    --output="${log_root}/route151_v8_gradient_routed_v1_60-%j.out" \
    --export=ALL \
    --wrap "${train_command}")"

remote_log="${log_root}/route151_v8_gradient_routed_v1_60-${job_id}.out"
"${python_bin}" "${project_root}/scripts/write_stage2l_v8_submission_attestation.py" \
  --job-id "${job_id}" \
  --project-root "${project_root}" \
  --records "${dataset_root}/records.jsonl" \
  --visual-cache "${asset_root}/scenario_factory/stage2l_multiframe_qa/route151_step218/orion_visual_contexts.pt" \
  --v8-preflight "${v8_preflight}" \
  --trainer-preflight "${trainer_preflight}" \
  --dataset-audit "${dataset_root}/audit.json" \
  --reference-audit "${dataset_root}/reference_audit.json" \
  --orion-config "${project_root}/adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py" \
  --base-checkpoint "${asset_root}/checkpoints/Orion.pth" \
  --remote-output-dir "${output_dir}" \
  --remote-log "${remote_log}" \
  --output "${output_parent}/submission_attestation.json" >/dev/null
printf '%s\n' "${job_id}"

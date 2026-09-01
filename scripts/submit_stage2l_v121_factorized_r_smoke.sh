#!/usr/bin/env bash
set -euo pipefail

project_root="/public/home/lidachuan/project/Orion"
asset_root="/public/share/lidachuan/orion_assets"
python_bin="${asset_root}/envs/orion-cl-centos7/bin/python"
carla_root="${asset_root}/carla/CARLA_0.9.15"
bench2drive_root="${asset_root}/Bench2Drive"
bench2drive_zoo_root="${asset_root}/Bench2DriveZoo"
base_root="${asset_root}/scenario_factory/stage2l_smokes/v121_factorized_r_17event_40_v1"
output_root="${base_root}/training"
preflight="${base_root}/trainer_preflight.json"
attestation="${base_root}/submission_attestation.json"
dataset_manifest="${asset_root}/scenario_factory/stage2l_expanded_coverage_17event_v1/manifest.json"
view_feature_cache="${asset_root}/scenario_factory/stage2l_view_aligned_feature_cache_v1/features.pt"
v101_root="${asset_root}/scenario_factory/stage2l_smokes/v101_view_aligned_phase_a_17event_120_v1/training"
v101_checkpoint="${v101_root}/phase_a_step120.pt"
v101_report="${v101_root}/report.json"
factorized_cpu_report="${asset_root}/scenario_factory/stage2l_smokes/v12_1_factorized_r_cpu_preflight_v1/report.json"
config="${project_root}/adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py"
checkpoint="${asset_root}/checkpoints/Orion.pth"
trainer="${project_root}/scripts/train_stage2l_v121_factorized_r_smoke.py"
factorized_module="${project_root}/uq_estimator/stage2l_factorized_relevance_v121.py"
attester="${project_root}/scripts/write_stage2l_v121_factorized_r_submission_attestation.py"
protocol="${project_root}/configs/scenario_factory/stage2l_v12_1_factorized_r_smoke_protocol_v1.json"
launch_amendment="${project_root}/configs/scenario_factory/amendments/20260901_stage2l_v121_factorized_r_smoke_launch_v1.json"
log_root="${asset_root}/scenario_factory/logs/stage2l_smokes"
job_name="s2l_v121_factr"

for prerequisite in \
  "${python_bin}" "${trainer}" "${factorized_module}" "${attester}" \
  "${dataset_manifest}" "${view_feature_cache}" "${v101_checkpoint}" \
  "${v101_report}" "${factorized_cpu_report}" "${config}" "${checkpoint}" \
  "${protocol}" "${preflight}" "${launch_amendment}"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "missing v12.1 factorized-R prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${output_root}" ]] && find "${output_root}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty v12.1 factorized-R output" >&2
  exit 1
fi
if [[ -e "${attestation}" ]]; then
  echo "refusing to overwrite v12.1 factorized-R submission attestation" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active v12.1 factorized-R submission" >&2
  exit 1
fi

runtime_pythonpath="${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${bench2drive_root}:${bench2drive_root}/leaderboard:${bench2drive_root}/scenario_runner:${bench2drive_zoo_root}:${PYTHONPATH:-}"
train_parts=(
  env "PYTHONPATH=${runtime_pythonpath}" "IS_BENCH2DRIVE=True"
  "${python_bin}" "${trainer}"
  --config "${config}"
  --checkpoint "${checkpoint}"
  --dataset-manifest "${dataset_manifest}"
  --view-feature-cache "${view_feature_cache}"
  --v101-checkpoint "${v101_checkpoint}"
  --v101-report "${v101_report}"
  --factorized-cpu-report "${factorized_cpu_report}"
  --training-protocol "${protocol}"
  --trainer-preflight "${preflight}"
  --launch-amendment "${launch_amendment}"
  --output-dir "${output_root}"
  --log-interval 5
)
printf -v train_command '%q ' "${train_parts[@]}"

mkdir -p "${log_root}"
job_id="$(sbatch --parsable \
  --partition=Nvidia_A800 \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=192G \
  --time=05:00:00 \
  --exclude=gpu5 \
  --job-name="${job_name}" \
  --output="${log_root}/v121_factorized_r_17event_40_v1-%j.out" \
  --export=ALL \
  --wrap "${train_command}")"

unattested_job_id="${job_id}"
cancel_unattested_submission() {
  if [[ -n "${unattested_job_id}" ]]; then
    scancel "${unattested_job_id}" || true
    echo "cancelled unattested v12.1 factorized-R job: ${unattested_job_id}" >&2
  fi
}
trap cancel_unattested_submission EXIT INT TERM

remote_log="${log_root}/v121_factorized_r_17event_40_v1-${job_id}.out"
env "PYTHONPATH=${project_root}" "${python_bin}" "${attester}" \
  --job-id "${job_id}" \
  --trainer "${trainer}" \
  --factorized-module "${factorized_module}" \
  --dataset-manifest "${dataset_manifest}" \
  --view-feature-cache "${view_feature_cache}" \
  --orion-config "${config}" \
  --orion-checkpoint "${checkpoint}" \
  --v101-checkpoint "${v101_checkpoint}" \
  --v101-report "${v101_report}" \
  --factorized-cpu-report "${factorized_cpu_report}" \
  --protocol "${protocol}" \
  --preflight "${preflight}" \
  --launch-amendment "${launch_amendment}" \
  --remote-log "${remote_log}" \
  --output-root "${output_root}" \
  --output "${attestation}" >/dev/null
unattested_job_id=""
trap - EXIT INT TERM
printf '%s\n' "${job_id}"

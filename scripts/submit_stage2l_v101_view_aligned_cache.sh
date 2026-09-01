#!/usr/bin/env bash
set -euo pipefail

project_root="/public/home/lidachuan/project/Orion"
asset_root="/public/share/lidachuan/orion_assets"
python_bin="${asset_root}/envs/orion-cl-centos7/bin/python"
carla_root="${asset_root}/carla/CARLA_0.9.15"
bench2drive_root="${asset_root}/Bench2Drive"
bench2drive_zoo_root="${asset_root}/Bench2DriveZoo"
output_root="${asset_root}/scenario_factory/stage2l_view_aligned_feature_cache_v1"
dataset_manifest="${asset_root}/scenario_factory/stage2l_expanded_coverage_17event_v1/manifest.json"
config="${project_root}/adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py"
checkpoint="${asset_root}/checkpoints/Orion.pth"
builder="${project_root}/scripts/cache_stage2l_view_aligned_features.py"
attester="${project_root}/scripts/write_stage2l_v101_view_aligned_cache_attestation.py"
protocol="${project_root}/configs/scenario_factory/stage2l_v101_view_aligned_cache_v1.json"
preflight="${output_root}/preflight.json"
amendment="${project_root}/configs/scenario_factory/amendments/20260831_stage2l_v101_view_aligned_cache_launch_v1.json"
cache_output="${output_root}/features.pt"
cache_manifest="${output_root}/features.json"
attestation="${output_root}/submission_attestation.json"
log_root="${asset_root}/scenario_factory/logs/stage2l_smokes"
job_name="s2l_v101_vcache"

for prerequisite in \
  "${python_bin}" "${dataset_manifest}" "${config}" "${checkpoint}" \
  "${builder}" "${attester}" "${protocol}" "${preflight}" "${amendment}"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "missing view-aligned cache prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
for forbidden in "${cache_output}" "${cache_manifest}" "${attestation}"; do
  if [[ -e "${forbidden}" ]]; then
    echo "refusing to overwrite view-aligned cache artifact: ${forbidden}" >&2
    exit 1
  fi
done
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active view-aligned cache submission" >&2
  exit 1
fi

runtime_pythonpath="${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${bench2drive_root}:${bench2drive_root}/leaderboard:${bench2drive_root}/scenario_runner:${bench2drive_zoo_root}:${PYTHONPATH:-}"
cache_parts=(
  env "PYTHONPATH=${runtime_pythonpath}" "IS_BENCH2DRIVE=True"
  "${python_bin}" "${builder}"
  --dataset-manifest "${dataset_manifest}"
  --orion-config "${config}"
  --orion-checkpoint "${checkpoint}"
  --protocol "${protocol}"
  --preflight "${preflight}"
  --launch-amendment "${amendment}"
  --output "${cache_output}"
)
printf -v cache_command '%q ' "${cache_parts[@]}"

mkdir -p "${log_root}"
job_id="$(sbatch --parsable \
  --partition=Nvidia_A800 \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=192G \
  --time=03:00:00 \
  --exclude=gpu5 \
  --job-name="${job_name}" \
  --output="${log_root}/v101_view_aligned_cache_v1-%j.out" \
  --export=ALL \
  --wrap "${cache_command}")"

unattested_job_id="${job_id}"
cancel_unattested_submission() {
  if [[ -n "${unattested_job_id}" ]]; then
    scancel "${unattested_job_id}" || true
    echo "cancelled unattested view-aligned cache job: ${unattested_job_id}" >&2
  fi
}
trap cancel_unattested_submission EXIT INT TERM

remote_log="${log_root}/v101_view_aligned_cache_v1-${job_id}.out"
env "PYTHONPATH=${project_root}" "${python_bin}" "${attester}" \
  --job-id "${job_id}" \
  --protocol "${protocol}" \
  --preflight "${preflight}" \
  --amendment "${amendment}" \
  --cache-builder "${builder}" \
  --dataset-manifest "${dataset_manifest}" \
  --orion-config "${config}" \
  --orion-checkpoint "${checkpoint}" \
  --remote-log "${remote_log}" \
  --cache-output "${cache_output}" \
  --output "${attestation}" >/dev/null
unattested_job_id=""
trap - EXIT INT TERM
printf '%s\n' "${job_id}"

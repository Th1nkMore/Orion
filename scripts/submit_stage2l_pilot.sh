#!/usr/bin/env bash
set -euo pipefail

submit=0
if [[ "${1:-}" == "--submit" ]]; then
  submit=1
elif [[ -n "${1:-}" && "${1:-}" != "--dry-run" ]]; then
  echo "usage: $0 [--dry-run|--submit]" >&2
  exit 2
fi

: "${PILOT_MANIFEST:?set PILOT_MANIFEST to the assembled, human-reviewed 6/2 pilot manifest}"
: "${STAGE2L_TRAINING_PROTOCOL:?set STAGE2L_TRAINING_PROTOCOL to the post-Route196 pilot protocol}"

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_bin="${PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
carla_root="${CARLA_ROOT:-${asset_root}/carla/CARLA_0.9.15}"
bench2drive_root="${BENCH2DRIVE_ROOT:-${project_root}/Bench2Drive}"
bench2drive_zoo_root="${BENCH2DRIVE_ZOO_ROOT:-${project_root}/Bench2DriveZoo}"
runtime_pythonpath="${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${bench2drive_root}:${bench2drive_root}/leaderboard:${bench2drive_root}/scenario_runner:${bench2drive_zoo_root}:${PYTHONPATH:-}"
config="${ORION_TRAIN_CONFIG:-${project_root}/adzoo/orion/configs/orion_stage3_train.py}"
checkpoint="${ORION_CHECKPOINT:-${asset_root}/checkpoints/Orion.pth}"
protocol="${STAGE2L_TRAINING_PROTOCOL}"
output_root="${OUTPUT_ROOT:-${asset_root}/scenario_factory/stage2l_pilot_runs/pilot_6_2_v1}"
log_root="${LOG_ROOT:-${asset_root}/scenario_factory/logs/stage2l_pilot_6_2_v1}"

for path in "${python_bin}" "${PILOT_MANIFEST}" "${config}" "${checkpoint}" "${protocol}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required input: ${path}" >&2
    exit 1
  fi
done
"${python_bin}" - "${protocol}" "${submit}" <<'PY'
import json
import pathlib
import sys

protocol = json.loads(pathlib.Path(sys.argv[1]).read_text())
require_active = sys.argv[2] == "1"
if protocol.get("schema") != "orion.stage2l_uq_language_grounding_protocol.v1":
    raise SystemExit("invalid Stage2-L pilot protocol schema")
losses = protocol.get("losses", {})
preference = losses.get("matched_answer_preference")
if (
    losses.get("trajectory") != 0.0
    or not isinstance(preference, dict)
    or float(preference.get("weight", 0.0)) <= 0.0
):
    raise SystemExit("pilot protocol must enable matched answer preference and disable trajectory")
allowed = protocol.get("launch_locks", {}).get(
    "stage2l_pilot_training_allowed"
)
if require_active and allowed is not True:
    raise SystemExit("Stage2-L pilot remains launch-locked by the supplied protocol")
print("STAGE2L_PILOT_LAUNCH_AUTHORIZATION=%s" % allowed)
PY
if [[ -e "${output_root}" ]] && find "${output_root}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite Stage2-L pilot output: ${output_root}" >&2
  exit 1
fi

train_parts=(
  env "PYTHONPATH=${runtime_pythonpath}" "IS_BENCH2DRIVE=True"
  "${python_bin}" "${project_root}/scripts/train_stage2l_pilot.py"
  --config "${config}"
  --checkpoint "${checkpoint}"
  --pilot-manifest "${PILOT_MANIFEST}"
  --training-protocol "${protocol}"
  --output-dir "${output_root}"
  --epochs "${EPOCHS:-1}"
)
if [[ -n "${MAX_STEPS:-}" ]]; then
  train_parts+=(--max-steps "${MAX_STEPS}")
fi
printf -v train_command '%q ' "${train_parts[@]}"

sbatch_args=(
  sbatch --parsable
  "--partition=${SLURM_PARTITION:-Nvidia_A800}"
  --gres=gpu:1 "--cpus-per-task=${STAGE2L_CPUS:-2}" --mem=192G --time=04:00:00
  "--exclude=${SLURM_EXCLUDE:-gpu5}"
  --job-name=stage2l_pilot
  "--output=${log_root}/train-%j.out"
  --export=ALL --wrap "${train_command}"
)

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "FORMAL_TRAINING_READY=0"
  echo "TRAJECTORY_TRAINING_ENABLED=0"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${log_root}"
job_id="$("${sbatch_args[@]}")"
echo "STAGE2L_PILOT_JOB_ID=${job_id}"
echo "OUTPUT_ROOT=${output_root}"

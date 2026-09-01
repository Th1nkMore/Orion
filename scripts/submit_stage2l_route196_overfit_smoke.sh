#!/usr/bin/env bash
set -euo pipefail

submit=0
if [[ "${1:-}" == "--submit" ]]; then
  submit=1
elif [[ -n "${1:-}" && "${1:-}" != "--dry-run" ]]; then
  echo "usage: $0 [--dry-run|--submit]" >&2
  exit 2
fi

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_bin="${PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
carla_root="${CARLA_ROOT:-${asset_root}/carla/CARLA_0.9.15}"
bench2drive_root="${BENCH2DRIVE_ROOT:-${project_root}/Bench2Drive}"
bench2drive_zoo_root="${BENCH2DRIVE_ZOO_ROOT:-${project_root}/Bench2DriveZoo}"
runtime_pythonpath="${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${bench2drive_root}:${bench2drive_root}/leaderboard:${bench2drive_root}/scenario_runner:${bench2drive_zoo_root}:${PYTHONPATH:-}"
qa_root="${QA_ROOT:-${asset_root}/scenario_factory/qa_factory_smokes/route196_dev_smoke_v3}"
run_root="${RUN_ROOT:-${asset_root}/scenario_factory/stage2l_smokes/route196_v1}"
log_root="${LOG_ROOT:-${asset_root}/logs/stage2l_route196_smoke_v1}"
cache_config="${ORION_CACHE_CONFIG:-${project_root}/adzoo/orion/configs/orion_stage3_agent.py}"
train_config="${ORION_TRAIN_CONFIG:-${project_root}/adzoo/orion/configs/orion_stage3_train.py}"
checkpoint="${ORION_CHECKPOINT:-${asset_root}/checkpoints/Orion.pth}"
frame_bundle="${FRAME_BUNDLE:-${qa_root}/frame_bundles/frame_bundle_observed.json}"
frame_meta="${FRAME_META:-${asset_root}/results/closedloop_native_collision_discovery_v1/route196_hazard_clean_off-1086150/records_orion_traj_0/RouteScenario_26956_rep0_Town05_SignalizedJunctionRightTurn_1_26_08_28_23_20_06/meta/0031.json}"
records="${QA_RECORDS:-${qa_root}/qa_dataset/records.jsonl}"
visual_cache="${run_root}/orion_visual_context.pt"
training_root="${run_root}/training"
existing_visual_cache="${EXISTING_VISUAL_CACHE:-}"
# This workflow is offline CUDA only; gpu2 is unsuitable for CARLA/Vulkan but
# is valid here.  Exclude only the drained node by default.
exclude_nodes="${SLURM_EXCLUDE:-gpu5}"
stage2l_cpus="${STAGE2L_CPUS:-2}"
balance_driving_stances="${BALANCE_DRIVING_STANCES:-0}"
training_protocol="${TRAINING_PROTOCOL:-}"
launch_amendment="${LAUNCH_AMENDMENT:-}"
lambda_answer_preference="${LAMBDA_ANSWER_PREFERENCE:-0}"
answer_preference_margin="${ANSWER_PREFERENCE_MARGIN:-0.2}"

required=("${python_bin}" "${carla_root}/PythonAPI/carla" "${train_config}" "${checkpoint}" "${records}")
if [[ "${balance_driving_stances}" == "1" ]]; then
  if [[ -z "${training_protocol}" ]]; then
    echo "BALANCE_DRIVING_STANCES=1 requires TRAINING_PROTOCOL" >&2
    exit 1
  fi
  if [[ -z "${launch_amendment}" ]]; then
    echo "BALANCE_DRIVING_STANCES=1 requires LAUNCH_AMENDMENT" >&2
    exit 1
  fi
  required+=("${training_protocol}" "${launch_amendment}")
elif [[ "${balance_driving_stances}" != "0" ]]; then
  echo "BALANCE_DRIVING_STANCES must be 0 or 1" >&2
  exit 1
fi
if [[ -z "${existing_visual_cache}" ]]; then
  required+=("${cache_config}" "${frame_bundle}" "${frame_meta}")
else
  required+=("${existing_visual_cache}")
  visual_cache="${existing_visual_cache}"
fi
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required input: ${path}" >&2
    exit 1
  fi
done
if [[ "${balance_driving_stances}" == "1" ]]; then
  "${python_bin}" - "${training_protocol}" "${launch_amendment}" "${submit}" <<'PY'
import json
import pathlib
import sys

protocol = json.loads(pathlib.Path(sys.argv[1]).read_text())
amendment = json.loads(pathlib.Path(sys.argv[2]).read_text())
require_active = sys.argv[3] == "1"
key = protocol.get(
    "launch_authorization_key", "route196_balanced_language_smoke_allowed"
)
locks = amendment.get("launch_locks", {})
if amendment.get("schema") != "orion.scenario_factory.amendment.v1":
    raise SystemExit("invalid Route196 launch amendment schema")
if locks.get("stage2l_pilot_training_allowed") is not False:
    raise SystemExit("Route196 amendment must keep Stage2-L pilot locked")
if require_active and locks.get(key) is not True:
    raise SystemExit("Route196 launch amendment is not active for %s" % key)
print("ROUTE196_LAUNCH_AUTHORIZATION=%s:%s" % (key, locks.get(key)))
PY
fi
if [[ -e "${training_root}" || ( -z "${existing_visual_cache}" && -e "${visual_cache}" ) ]]; then
  echo "refusing to overwrite existing Route196 smoke output under ${run_root}" >&2
  exit 1
fi

cache_parts=(
  env "PYTHONPATH=${runtime_pythonpath}" "IS_BENCH2DRIVE=True"
  "${python_bin}" "${project_root}/scripts/cache_closedloop_orion_visual_context.py"
  --frame-bundle "${frame_bundle}"
  --frame-meta "${frame_meta}"
  --orion-config "${cache_config}"
  --orion-checkpoint "${checkpoint}"
  --output "${visual_cache}"
)
printf -v cache_command '%q ' "${cache_parts[@]}"

train_parts=(
  env "PYTHONPATH=${runtime_pythonpath}" "IS_BENCH2DRIVE=True"
  "${python_bin}" "${project_root}/scripts/train_stage2l_route196_overfit_smoke.py"
  --config "${train_config}"
  --checkpoint "${checkpoint}"
  --visual-cache "${visual_cache}"
  --records "${records}"
  --output-dir "${training_root}"
  --max-steps "${MAX_STEPS:-120}"
)
if [[ "${balance_driving_stances}" == "1" ]]; then
  train_parts+=(
    --balance-driving-stances
    --training-protocol "${training_protocol}"
    --launch-amendment "${launch_amendment}"
  )
fi
if [[ "${lambda_answer_preference}" != "0" && "${lambda_answer_preference}" != "0.0" ]]; then
  train_parts+=(
    --lambda-answer-preference "${lambda_answer_preference}"
    --answer-preference-margin "${answer_preference_margin}"
  )
fi
printf -v train_command '%q ' "${train_parts[@]}"

cache_args=(
  sbatch --parsable
  "--partition=${SLURM_PARTITION:-Nvidia_A800}"
  --gres=gpu:1 "--cpus-per-task=${stage2l_cpus}" --mem=192G --time=01:00:00
  "--exclude=${exclude_nodes}"
  --job-name=s2l_r196_cache
  "--output=${log_root}/cache-%j.out"
  --export=ALL --wrap "${cache_command}"
)

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "ENGINEERING_OVERFIT_ONLY=1"
  echo "FORMAL_TRAINING_READY=0"
  if [[ -z "${existing_visual_cache}" ]]; then
    printf 'CACHE_SBATCH_COMMAND='
    printf '%q ' "${cache_args[@]}"
    printf '\n'
    printf 'TRAIN_COMMAND_AFTEROK=<cache_job_id> '
  else
    echo "REUSED_VISUAL_CACHE=${visual_cache}"
    printf 'TRAIN_COMMAND_DIRECT='
  fi
  printf '%q ' "${train_parts[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${log_root}" "${run_root}"
cache_job_id=""
if [[ -z "${existing_visual_cache}" ]]; then
  cache_job_id="$("${cache_args[@]}")"
fi
train_args=(sbatch --parsable \
  "--partition=${SLURM_PARTITION:-Nvidia_A800}" \
  --gres=gpu:1 "--cpus-per-task=${stage2l_cpus}" --mem=192G --time=02:30:00 \
  "--exclude=${exclude_nodes}" \
  --job-name=s2l_r196_train \
  "--output=${log_root}/train-%j.out" \
  --export=ALL --wrap "${train_command}")
if [[ -n "${cache_job_id}" ]]; then
  train_args+=("--dependency=afterok:${cache_job_id}")
fi
train_job_id="$("${train_args[@]}")"

if [[ -n "${cache_job_id}" ]]; then
  echo "CACHE_JOB_ID=${cache_job_id}"
else
  echo "REUSED_VISUAL_CACHE=${visual_cache}"
fi
echo "TRAIN_JOB_ID=${train_job_id}"
echo "VISUAL_CACHE=${visual_cache}"
echo "TRAINING_ROOT=${training_root}"

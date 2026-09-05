#!/usr/bin/env bash
set -euo pipefail

# Run an exact Qwen-Drive clean/front-dropout prediction pair.  This invokes the
# unmodified upstream planning runner twice; only the image resolver environment
# differs between the two invocations.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
qwen_python="${QWEN_PYTHON:-/public/share/lidachuan/orion_assets/envs/qwen-drive-py310/bin/python}"
qwen_source="${QWEN_SOURCE:-/public/share/lidachuan/orion_assets/third_party/Qwen-Drive-1.0}"
model_path="${QWEN_MODEL:-/public/share/lidachuan/orion_assets/checkpoints/Qwen-Drive-1.0-4B}"
planner_path="${QWEN_PLANNER:-${model_path}/planner-sft}"
scene_path="${SCENES:-}"
image_root="${IMAGE_ROOT:-}"
output_root="${OUTPUT_ROOT:-}"
corruption_paths="${CORRUPTION_PATHS:-}"
planning_mode="${PLANNING_MODE:-reasoning_planning}"
seed="${SEED:-42}"
limit="${LIMIT:-}"
num_workers="${NUM_WORKERS:-4}"

for required_name in SCENES IMAGE_ROOT OUTPUT_ROOT CORRUPTION_PATHS; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "${required_name} is required" >&2
    exit 2
  fi
done
for required_path in \
  "${qwen_python}" \
  "${qwen_source}/scripts/run_planning.py" \
  "${model_path}" \
  "${planner_path}" \
  "${scene_path}" \
  "${image_root}" \
  "${corruption_paths}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "missing prerequisite: ${required_path}" >&2
    exit 2
  fi
done
if [[ "${planning_mode}" != "reasoning_planning" && "${planning_mode}" != "direct_planning" ]]; then
  echo "PLANNING_MODE must be reasoning_planning or direct_planning" >&2
  exit 2
fi
if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer" >&2
  exit 2
fi

clean_output="${output_root}/clean/predictions.jsonl"
corrupt_output="${output_root}/front_dropout/predictions.jsonl"
if [[ "${ALLOW_OVERWRITE:-0}" != "1" && ( -e "${clean_output}" || -e "${corrupt_output}" ) ]]; then
  echo "pair output already exists; choose another OUTPUT_ROOT or set ALLOW_OVERWRITE=1" >&2
  exit 2
fi
mkdir -p "$(dirname "${clean_output}")" "$(dirname "${corrupt_output}")"

common_args=(
  "${qwen_source}/scripts/run_planning.py"
  --model "${model_path}"
  --planner "${planner_path}"
  --scenes "${scene_path}"
  --mode "${planning_mode}"
  --num-samples 1
  --seed "${seed}"
  --num-workers "${num_workers}"
  --dtype bfloat16
  --attn-implementation flash_attention_2
  --no-resume
)
if [[ -n "${limit}" ]]; then
  common_args+=(--limit "${limit}")
fi

export PYTHONPATH="${project_root}:${qwen_source}/src${PYTHONPATH:+:${PYTHONPATH}}"
export ORION_NAVSIM_IMAGE_ROOT="${image_root}"
export ORION_NAVSIM_CORRUPTION_CAMERAS="CAM_F0"

export ORION_NAVSIM_CORRUPTION="none"
unset ORION_NAVSIM_CORRUPTION_PATHS
"${qwen_python}" "${common_args[@]}" \
  --image-resolver uq_estimator.qwen_drive_navsim_images:make_reader \
  --output "${clean_output}"

export ORION_NAVSIM_CORRUPTION="camera_dropout"
export ORION_NAVSIM_CORRUPTION_PATHS="${corruption_paths}"
"${qwen_python}" "${common_args[@]}" \
  --image-resolver uq_estimator.qwen_drive_navsim_images:make_reader \
  --output "${corrupt_output}"

"${qwen_python}" "${project_root}/scripts/audit_qwen_navsim_pair.py" \
  --scenes "${scene_path}" \
  --clean "${clean_output}" \
  --corrupted "${corrupt_output}" \
  --output "${output_root}/pair_audit.json"

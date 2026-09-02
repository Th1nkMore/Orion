#!/usr/bin/env bash
set -euo pipefail

submit=0
case "${1:-}" in
  --submit) submit=1 ;;
  ""|--dry-run) ;;
  *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
esac

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_bin="${PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
config="${ORION_CONFIG:-${project_root}/adzoo/orion/configs/orion_stage3_agent.py}"
checkpoint="${ORION_CHECKPOINT:-${asset_root}/checkpoints/Orion.pth}"
adapter="${ADAPTER_CHECKPOINT:-/public/home/lidachuan/orion_work/observation_uq_v3/runs/counterfactual_pairwise_native_repair_seed20260828_r1/counterfactual_evidence_pairwise_native_repair.pt}"
adapter_sha256="${ADAPTER_CHECKPOINT_SHA256:-0555f0f341c80a88e18c5864573f0be0641fb828931bea7809e2f5544665f2c8}"
extractor="${project_root}/scripts/extract_closedloop_stage1_uq_visualization.py"
output_root="${OUTPUT_ROOT:-${asset_root}/results/stage1_uq_visualization_route146_20260902_v1}"
log_root="${LOG_ROOT:-${asset_root}/logs/stage1_uq_visualization}"

clean_run="${asset_root}/results/uqcl_p0/route146_hazard_clean_off-1057222"
corrupt_run="${asset_root}/results/uqcl_p3_pairwise/route146_hazard_front_corrupt_transient_pairwise_stop-1077521"

for prerequisite in \
  "${python_bin}" "${config}" "${checkpoint}" "${adapter}" "${extractor}" \
  "${clean_run}" "${corrupt_run}"; do
  if [[ ! -e "${prerequisite}" ]]; then
    echo "missing visualization replay prerequisite: ${prerequisite}" >&2
    exit 1
  fi
done

declare -a labels=(clean corrupt)
declare -a runs=("${clean_run}" "${corrupt_run}")
for label in "${labels[@]}"; do
  if [[ -e "${output_root}/${label}" ]]; then
    echo "refusing to overwrite replay output: ${output_root}/${label}" >&2
    exit 1
  fi
done

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "OUTPUT_ROOT=${output_root}"
  echo "ADAPTER_SHA256=${adapter_sha256}"
  echo "LOADS_ORION_LLM=0"
  echo "LOADS_ORION_PLANNING_HEAD=0"
  echo "CONTROL_INFLUENCE=0"
  echo "JOB_COUNT=2"
  exit 0
fi

mkdir -p "${log_root}"
for index in 0 1; do
  label="${labels[index]}"
  run_dir="${runs[index]}"
  command=(
    env "PYTHONPATH=${project_root}:${PYTHONPATH:-}"
    "${python_bin}" "${extractor}"
    --run-dir "${run_dir}"
    --orion-config "${config}"
    --orion-checkpoint "${checkpoint}"
    --adapter-checkpoint "${adapter}"
    --adapter-checkpoint-sha256 "${adapter_sha256}"
    --output-dir "${output_root}/${label}"
    --baseline-start-frame 2
    --baseline-end-frame 8
  )
  printf -v wrapped '%q ' "${command[@]}"
  job_id="$(sbatch --parsable \
    --partition=Nvidia_A800 \
    --gres=gpu:1 \
    --cpus-per-task=2 \
    --mem=192G \
    --time=01:00:00 \
    --exclude=gpu5 \
    --job-name="uqvis146_${label}" \
    --output="${log_root}/route146_${label}-%j.out" \
    --export=ALL \
    --wrap "${wrapped}")"
  echo "${label^^}_JOB_ID=${job_id}"
done
echo "OUTPUT_ROOT=${output_root}"

#!/usr/bin/env bash
set -euo pipefail

submit=0
case "${1:-}" in
  --submit) submit=1 ;;
  ""|--dry-run) ;;
  *) echo "usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
esac

project_root="${PROJECT_ROOT:-/public/home/lidachuan/orion_work/qwen_visibility_belief}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_bin="${asset_root}/envs/qwen-drive-py310/bin/python"
glibc_sysroot="${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot"
glibc_loader="${glibc_sysroot}/lib64/ld-linux-x86-64.so.2"
runtime_library_path="${glibc_sysroot}/lib64:${glibc_sysroot}/usr/lib64:${asset_root}/envs/qwen-drive-py310/lib"
config="${project_root}/configs/qwen_drive_b2d_agent_oracle_visibility_sft_v1.json"
smoke="${project_root}/scripts/smoke_qwen_visibility_vlm_insertion.py"
token_artifact="${asset_root}/qwen_visibility_token_runs/qwen_route151_o2_v2_tokens_o3_v1/step_000260.npz"
audit_root="${asset_root}/qwen_drive_b2d_smokes/qwen_oracle_visibility_route151_reasoning_sft_seed42_o2_v2/records_qwen_drive_traj_151/oracle_visibility/sensor_audit/step_000260"
front="${audit_root}/CAM_FRONT_rgb.png"
front_left="${audit_root}/CAM_FRONT_LEFT_rgb.png"
front_right="${audit_root}/CAM_FRONT_RIGHT_rgb.png"
run_id="${RUN_ID:-qwen_visibility_vlm_insertion_v0_step260_v1}"
run_root="${asset_root}/qwen_visibility_vlm_smokes/${run_id}"
output="${run_root}/report.json"
log_root="${asset_root}/qwen_visibility_vlm_smokes/logs"
job_name="${JOB_NAME:-qwen_visibility_v0}"
node_list="${NODELIST:-gpu4}"

for prerequisite in \
  "${python_bin}" "${glibc_loader}" "${config}" "${smoke}" \
  "${token_artifact}" "${front}" "${front_left}" "${front_right}"; do
  if [[ ! -f "${prerequisite}" ]]; then
    echo "missing Qwen visibility-VLM smoke prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done
if [[ -e "${run_root}" ]]; then
  echo "refusing to reuse Qwen visibility-VLM smoke output: ${run_root}" >&2
  exit 1
fi
if squeue -h -u "${USER}" -n "${job_name}" | grep -q .; then
  echo "refusing duplicate active Qwen visibility-VLM smoke" >&2
  exit 1
fi

run_parts=(
  env "PYTHONPATH=${project_root}:${asset_root}/third_party/Qwen-Drive-1.0/src:${PYTHONPATH:-}"
  "${glibc_loader}" --library-path "${runtime_library_path}"
  "${python_bin}" "${smoke}"
  --config "${config}"
  --token-artifact "${token_artifact}"
  --front "${front}"
  --front-left "${front_left}"
  --front-right "${front_right}"
  --output "${output}"
)
printf -v run_command '%q ' "${run_parts[@]}"
sbatch_args=(
  sbatch --parsable --partition=Nvidia_A800 --gres=gpu:1
  --cpus-per-task=4 --mem=64G --time=01:00:00
  --job-name="${job_name}"
  --output="${log_root}/${run_id}-%j.out"
  --export=ALL --nodelist="${node_list}"
)
sbatch_args+=(--wrap "${run_command}")

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "RUN_ID=${run_id}"
  echo "TOKEN_ARTIFACT=${token_artifact}"
  echo "OUTPUT=${output}"
  echo "NODELIST=${node_list}"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${run_root}" "${log_root}"
"${sbatch_args[@]}"

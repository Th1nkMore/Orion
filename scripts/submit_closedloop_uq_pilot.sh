#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 ROUTE_INDEX CONDITION [hazard|nohazard]" >&2
  exit 2
fi

route_index="$1"
condition="$2"
variant="${3:-hazard}"
project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
compat_python="${COMPAT_PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
approval_gate="${ORION_CORRUPTION_VISUAL_APPROVAL_GATE:-${project_root}/configs/scenario_factory/corruption_hardcase_visual_approval_gate_v2.json}"
pilot_run_id="${PILOT_RUN_ID:-uqcl_p0}"
pilot_route_dir="${PILOT_ROUTE_DIR:-}"
if [[ -z "${pilot_route_dir}" && "${pilot_run_id}" == "closedloop_native_collision_discovery_v1" ]]; then
  pilot_route_dir="${project_root}/configs/closedloop_native_collision_discovery/routes"
fi
log_root="${asset_root}/logs/${pilot_run_id}"
mkdir -p "${log_root}"

export PILOT_ROUTE_INDEX="${route_index}"
export PILOT_CONDITION="${condition}"
export PILOT_VARIANT="${variant}"
export PILOT_RUN_ID="${pilot_run_id}"
if [[ -n "${pilot_route_dir}" ]]; then
  export PILOT_ROUTE_DIR="${pilot_route_dir}"
fi
export PROJECT_ROOT="${project_root}"
export ASSET_ROOT="${asset_root}"
export ORION_CORRUPTION_VISUAL_APPROVAL_GATE="${approval_gate}"

if [[ -n "${PILOT_ROUTE_FILE:-}" ]]; then
  if [[ ! -f "${PILOT_ROUTE_FILE}" ]]; then
    echo "[FAIL] explicit route file is missing: ${PILOT_ROUTE_FILE}" >&2
    exit 1
  fi
  : "${PILOT_ROUTE_FILE_SHA256:?explicit route file requires PILOT_ROUTE_FILE_SHA256}"
  observed_route_sha256="$(sha256sum "${PILOT_ROUTE_FILE}" | awk '{print $1}')"
  if [[ "${observed_route_sha256}" != "${PILOT_ROUTE_FILE_SHA256}" ]]; then
    echo "[FAIL] explicit route file hash differs: ${PILOT_ROUTE_FILE}" >&2
    exit 1
  fi
fi

# Fail before sbatch.  The runner repeats the same check inside the allocation
# so a gate/source change between submission and launch also fails closed.
case "${condition}" in
  front_stale_transient_off|lens_waterdrop_transient_off|lens_waterdrop_paired_template_transient_off|native_motion_blur_off)
    "${compat_python}" "${project_root}/scripts/preflight_corruption_hardcase_orion_screen.py" \
      --gate "${approval_gate}" \
      --repository-root "${project_root}" \
      --pilot-condition "${condition}" \
      --corruption-severity "${ORION_CLOSEDLOOP_CORRUPTION_SEVERITY:-1}" \
      --paired-waterdrop-profile "${ORION_PAIRED_WATERDROP_PROFILE:-}" \
      --native-motion-blur-profile "${ORION_NATIVE_MOTION_BLUR_PROFILE:-none}"
    ;;
esac

submit_args=(
  --partition="${SLURM_PARTITION:-Nvidia_A800}"
  --gres=gpu:1
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-2}"
  --mem="${SLURM_MEM:-192G}"
  --time="${SLURM_TIME:-03:00:00}"
)
if [[ -n "${SLURM_NODELIST:-}" ]]; then
  submit_args+=(--nodelist="${SLURM_NODELIST}")
fi
if [[ -n "${SLURM_EXCLUDE:-}" ]]; then
  submit_args+=(--exclude="${SLURM_EXCLUDE}")
fi
submit_args+=(
  --job-name="uqcl_${route_index}_${condition}"
  --output="${log_root}/uqcl_${route_index}_${variant}_${condition}-%j.out"
  --export=ALL
  "${project_root}/scripts/run_closedloop_uq_pilot.sh"
)

sbatch --parsable "${submit_args[@]}"

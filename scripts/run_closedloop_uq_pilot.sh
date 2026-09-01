#!/usr/bin/env bash
set -euo pipefail

# Run one pre-registered route/condition pair inside an existing Slurm job.

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
ASSET_ROOT="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
export CARLA_ROOT="${CARLA_ROOT:-${ASSET_ROOT}/carla/CARLA_0.9.15}"
export BENCH2DRIVE_ROOT="${BENCH2DRIVE_ROOT:-${ASSET_ROOT}/Bench2Drive}"
export BENCH2DRIVE_ZOO_ROOT="${BENCH2DRIVE_ZOO_ROOT:-${ASSET_ROOT}/Bench2DriveZoo}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-${ASSET_ROOT}/checkpoints/Orion.pth}"
export ORION_QFORMER_PATH="${ORION_QFORMER_PATH:-${ASSET_ROOT}/checkpoints/pretrain_qformer}"
export COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-${ASSET_ROOT}/envs/orion-cl-centos7/bin/python}"
export ORION_CORRUPTION_VISUAL_APPROVAL_GATE="${ORION_CORRUPTION_VISUAL_APPROVAL_GATE:-${PROJECT_ROOT}/configs/scenario_factory/corruption_hardcase_visual_approval_gate_v2.json}"
export ORION_NATIVE_GLARE_PROFILE="${ORION_NATIVE_GLARE_PROFILE:-none}"
export ORION_NATIVE_MOTION_BLUR_PROFILE="${ORION_NATIVE_MOTION_BLUR_PROFILE:-none}"
PILOT_RUN_ID="${PILOT_RUN_ID:-uqcl_p0}"
PILOT_ROUTE_INDEX="${PILOT_ROUTE_INDEX:?set PILOT_ROUTE_INDEX}"
PILOT_VARIANT="${PILOT_VARIANT:-hazard}"
PILOT_CONDITION="${PILOT_CONDITION:?set PILOT_CONDITION}"
PILOT_ROUTE_DIR="${PILOT_ROUTE_DIR:-${PROJECT_ROOT}/configs/closedloop_uq_pilot/routes}"
export AGENT_CONFIG_PATH="${AGENT_CONFIG_PATH:-${PROJECT_ROOT}/adzoo/orion/configs/orion_stage3_agent_uq.py}"
export PILOT_RUN_ID PILOT_ROUTE_INDEX PILOT_VARIANT PILOT_CONDITION PILOT_ROUTE_DIR

echo "[Runtime] host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

case "${PILOT_VARIANT}" in
  hazard|nohazard) ;;
  *) echo "[FAIL] PILOT_VARIANT must be hazard or nohazard" >&2; exit 2 ;;
esac

export ORION_CLOSEDLOOP_UQ_MODE=none
# This variable belongs to the retired global Density-token/vision-adapter
# path.  The new frozen spatial observation adapter has its own checkpoint and
# manifest fields and must never be mislabeled as legacy vision conditioning.
export ORION_CLOSEDLOOP_CONDITIONING=none
export ORION_PLANNING_RESPONSE_MODE=off
case "${ORION_ENABLE_LEGACY_DENSITY_UQ:-0}" in
  0|false|no|off) ;;
  *)
    echo "[FAIL] legacy Density UQ is retired from the current closed-loop pipeline" >&2
    exit 2
    ;;
esac

export ORION_ENABLE_LEGACY_DENSITY_UQ=0
export ORION_CLOSEDLOOP_CORRUPTION_VIEWS="${ORION_CLOSEDLOOP_CORRUPTION_VIEWS:-front}"
export ORION_CLOSEDLOOP_CORRUPTION_SEVERITY="${ORION_CLOSEDLOOP_CORRUPTION_SEVERITY:-1}"
export ORION_CLOSEDLOOP_RISK_THRESHOLD="${ORION_CLOSEDLOOP_RISK_THRESHOLD:-0.4}"
export ORION_CLOSEDLOOP_RISK_SATURATION="${ORION_CLOSEDLOOP_RISK_SATURATION:-0.8}"
export ORION_CLOSEDLOOP_RISK_MIN_SPEED="${ORION_CLOSEDLOOP_RISK_MIN_SPEED:-1.5}"
export ORION_CLOSEDLOOP_RISK_MAX_SPEED="${ORION_CLOSEDLOOP_RISK_MAX_SPEED:-5.0}"
export ORION_SENSOR_QUEUE_DIAGNOSTICS="${ORION_SENSOR_QUEUE_DIAGNOSTICS:-1}"
export ORION_SENSOR_QUEUE_TIMEOUT_SECONDS="${ORION_SENSOR_QUEUE_TIMEOUT_SECONDS:-60}"
export ORION_EXACT_FRAME_SPEEDOMETER="${ORION_EXACT_FRAME_SPEEDOMETER:-1}"
# Bench2Drive reuses this timeout for both CARLA RPC and agent setup.  ORION's
# checkpoint load can exceed 90 seconds under concurrent shared-filesystem I/O,
# even though the node and model are healthy.  Keep the watchdog as the runtime
# liveness guard and allow setup enough time to finish.
export ORION_CARLA_RPC_TIMEOUT_SECONDS="${ORION_CARLA_RPC_TIMEOUT_SECONDS:-300}"
export ORION_SCENARIO_TRACEBACK_INTERVAL_SECONDS="${ORION_SCENARIO_TRACEBACK_INTERVAL_SECONDS:-75}"
export CLOSEDLOOP_PROGRESS_WATCHDOG_SECONDS="${CLOSEDLOOP_PROGRESS_WATCHDOG_SECONDS:-300}"
export CLOSEDLOOP_PROGRESS_STARTUP_GRACE_SECONDS="${CLOSEDLOOP_PROGRESS_STARTUP_GRACE_SECONDS:-900}"
export CLOSEDLOOP_PROGRESS_WATCHDOG_POLL_SECONDS="${CLOSEDLOOP_PROGRESS_WATCHDOG_POLL_SECONDS:-15}"

require_transient_window() {
  if [[ -z "${ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS:-}" || \
        -z "${ORION_CLOSEDLOOP_CORRUPTION_DURATION_SECONDS:-}" || \
        -n "${ORION_CLOSEDLOOP_CORRUPTION_END_PROGRESS:-}" ]]; then
    echo "[FAIL] transient corruption requires start progress and duration, without end progress" >&2
    exit 2
  fi
}

require_native_risk_oracle_window() {
  if [[ -z "${ORION_CLOSEDLOOP_RISK_ORACLE_START_PROGRESS:-}" || \
        -z "${ORION_CLOSEDLOOP_RISK_ORACLE_DURATION_SECONDS:-}" ]]; then
    echo "[FAIL] native-event oracle requires frozen risk start progress and duration" >&2
    exit 2
  fi
}

require_pairwise_observation_uq() {
  export ORION_OBSERVATION_UQ_CHECKPOINT="${ORION_OBSERVATION_UQ_CHECKPOINT:-/public/home/lidachuan/orion_work/observation_uq_v3/runs/counterfactual_pairwise_native_repair_seed20260828_r1/counterfactual_evidence_pairwise_native_repair.pt}"
  export ORION_OBSERVATION_UQ_CHECKPOINT_SHA256="${ORION_OBSERVATION_UQ_CHECKPOINT_SHA256:-0555f0f341c80a88e18c5864573f0be0641fb828931bea7809e2f5544665f2c8}"
  export ORION_OBSERVATION_UQ_FRONT_VIEW="${ORION_OBSERVATION_UQ_FRONT_VIEW:-0}"
  export ORION_OBSERVATION_UQ_BASELINE_START_SECONDS="${ORION_OBSERVATION_UQ_BASELINE_START_SECONDS:-1.0}"
  export ORION_OBSERVATION_UQ_BASELINE_END_SECONDS="${ORION_OBSERVATION_UQ_BASELINE_END_SECONDS:-4.0}"
  export ORION_OBSERVATION_UQ_MIN_BASELINE_FRAMES="${ORION_OBSERVATION_UQ_MIN_BASELINE_FRAMES:-40}"
  export ORION_OBSERVATION_UQ_RELATIVE_SCALE_FLOOR="${ORION_OBSERVATION_UQ_RELATIVE_SCALE_FLOOR:-0.05}"
  export ORION_OBSERVATION_UQ_ABSOLUTE_SCALE_FLOOR="${ORION_OBSERVATION_UQ_ABSOLUTE_SCALE_FLOOR:-0.001}"
  export ORION_OBSERVATION_UQ_Z_CENTER="${ORION_OBSERVATION_UQ_Z_CENTER:-4.0}"
  export ORION_OBSERVATION_UQ_ATTACK_ALPHA="${ORION_OBSERVATION_UQ_ATTACK_ALPHA:-0.8}"
  export ORION_OBSERVATION_UQ_RELEASE_ALPHA="${ORION_OBSERVATION_UQ_RELEASE_ALPHA:-0.2}"
  [[ -f "${ORION_OBSERVATION_UQ_CHECKPOINT}" ]] || {
    echo "[FAIL] missing frozen pairwise observation-UQ checkpoint" >&2
    exit 1
  }
  observed_sha256="$(sha256sum "${ORION_OBSERVATION_UQ_CHECKPOINT}" | awk '{print $1}')"
  [[ "${observed_sha256}" == "${ORION_OBSERVATION_UQ_CHECKPOINT_SHA256}" ]] || {
    echo "[FAIL] frozen pairwise observation-UQ checkpoint hash differs" >&2
    exit 1
  }
}

require_spatial_corruption() {
  require_transient_window
  case "${ORION_CLOSEDLOOP_CORRUPTION:-}" in
    local_blur|local_dark|local_glare|local_occlusion) ;;
    *)
      echo "[FAIL] spatial condition requires an explicit local corruption family" >&2
      exit 2
      ;;
  esac
  if [[ -z "${ORION_CLOSEDLOOP_CORRUPTION_REGION:-}" ]]; then
    echo "[FAIL] spatial condition requires a frozen normalized corruption region" >&2
    exit 2
  fi
}

case "${PILOT_CONDITION}" in
  stage2p_controlled_k_smoke)
    export ORION_CLOSEDLOOP_CORRUPTION=""
    export ORION_CLOSEDLOOP_RISK_MODE=off
    if [[ "${ORION_STAGE2_ENGINEERING_SMOKE:-0}" != "1" || \
          "${ORION_STAGE2_SPATIAL_UQ_SOURCE:-disabled}" != "external_oracle" || \
          -n "${ORION_STAGE1_SPATIAL_UQ_CHECKPOINT:-}" || \
          -n "${ORION_OBSERVATION_UQ_CHECKPOINT:-}" ]]; then
      echo "[FAIL] Stage2-P smoke requires external K without Stage-1 or sidecar UQ" >&2
      exit 2
    fi
    : "${ORION_STAGE2_TASK_CHECKPOINT:?set the hash-bound Stage2-P checkpoint}"
    : "${ORION_STAGE2_TASK_CHECKPOINT_SHA256:?set the Stage2-P checkpoint hash}"
    : "${ORION_STAGE2_EXTERNAL_K_START_PROGRESS:?set the bounded K start progress}"
    : "${ORION_STAGE2_EXTERNAL_K_DURATION_SECONDS:?set the bounded K duration}"
    : "${ORION_STAGE2_EXTERNAL_K_CAMERA:?set the controlled K camera}"
    : "${ORION_STAGE2_EXTERNAL_K_REGION:?set the controlled K region}"
    : "${ORION_STAGE2_EXTERNAL_K_STRENGTH:?set the controlled K strength}"
    : "${ORION_STAGE2_EXTERNAL_K_GRID_SIZE:?set the controlled K grid size}"
    if [[ ! -f "${ORION_STAGE2_TASK_CHECKPOINT}" ]] || \
       [[ "$(sha256sum "${ORION_STAGE2_TASK_CHECKPOINT}" | awk '{print $1}')" \
          != "${ORION_STAGE2_TASK_CHECKPOINT_SHA256}" ]]; then
      echo "[FAIL] Stage2-P checkpoint is missing or its hash differs" >&2
      exit 1
    fi
    ;;
  clean_off)
    export ORION_CLOSEDLOOP_CORRUPTION=""
    export ORION_CLOSEDLOOP_RISK_MODE=off
    ;;
  clean_pairwise_trace)
    require_pairwise_observation_uq
    export ORION_CLOSEDLOOP_CORRUPTION=""
    export ORION_CLOSEDLOOP_RISK_MODE=off
    ;;
  native_event_oracle)
    require_native_risk_oracle_window
    export ORION_ENABLE_LEGACY_DENSITY_UQ=0
    export ORION_CLOSEDLOOP_CORRUPTION=""
    export ORION_CLOSEDLOOP_RISK_MODE=oracle
    ;;
  native_dynamic_yield_oracle)
    echo "[FAIL] rejected v3 path-clamp oracle cannot be rerun; use the preregistered dynamics-aware condition" >&2
    exit 2
    ;;
  native_bounded_crossing_oracle)
    export ORION_CLOSEDLOOP_CORRUPTION=""
    export ORION_CLOSEDLOOP_RISK_MODE=off
    export ORION_PLANNING_RESPONSE_MODE=privileged_bounded_crossing
    export ORION_PLANNING_ACTOR_CATEGORIES="${ORION_PLANNING_ACTOR_CATEGORIES:-walker}"
    export ORION_PLANNING_INTERPOLATION_STEP_SECONDS="${ORION_PLANNING_INTERPOLATION_STEP_SECONDS:-0.1}"
    export ORION_PLANNING_SAFETY_MARGIN_M="${ORION_PLANNING_SAFETY_MARGIN_M:-0.75}"
    export ORION_PLANNING_IMMINENT_HORIZON_SECONDS="${ORION_PLANNING_IMMINENT_HORIZON_SECONDS:-1.5}"
    export ORION_PLANNING_CLEARANCE_SECONDS="${ORION_PLANNING_CLEARANCE_SECONDS:-1.0}"
    export ORION_PLANNING_RELEASE_SECONDS="${ORION_PLANNING_RELEASE_SECONDS:-0.5}"
    export ORION_PLANNING_STOP_BUFFER_M="${ORION_PLANNING_STOP_BUFFER_M:-2.0}"
    export ORION_PLANNING_RELEASE_CREEP_DISTANCE_M="${ORION_PLANNING_RELEASE_CREEP_DISTANCE_M:-1.0}"
    ;;
  native_braking_aware_crossing_oracle)
    export ORION_CLOSEDLOOP_CORRUPTION=""
    export ORION_CLOSEDLOOP_RISK_MODE=off
    export ORION_PLANNING_RESPONSE_MODE=privileged_braking_aware_crossing
    export ORION_PLANNING_ACTOR_CATEGORIES="${ORION_PLANNING_ACTOR_CATEGORIES:-walker}"
    export ORION_PLANNING_INTERPOLATION_STEP_SECONDS="${ORION_PLANNING_INTERPOLATION_STEP_SECONDS:-0.1}"
    export ORION_PLANNING_SAFETY_MARGIN_M="${ORION_PLANNING_SAFETY_MARGIN_M:-0.75}"
    export ORION_PLANNING_IMMINENT_HORIZON_SECONDS="${ORION_PLANNING_IMMINENT_HORIZON_SECONDS:-1.5}"
    export ORION_PLANNING_CERTIFIED_DECELERATION_MPS2="${ORION_PLANNING_CERTIFIED_DECELERATION_MPS2:-3.0}"
    export ORION_PLANNING_CLEARANCE_SECONDS="${ORION_PLANNING_CLEARANCE_SECONDS:-1.0}"
    export ORION_PLANNING_RELEASE_SECONDS="${ORION_PLANNING_RELEASE_SECONDS:-0.5}"
    export ORION_PLANNING_PREPARE_CREEP_SPEED_MPS="${ORION_PLANNING_PREPARE_CREEP_SPEED_MPS:-1.0}"
    export ORION_PLANNING_RELEASE_CREEP_SPEED_MPS="${ORION_PLANNING_RELEASE_CREEP_SPEED_MPS:-0.5}"
    export ORION_PLANNING_STOP_BUFFER_M="${ORION_PLANNING_STOP_BUFFER_M:-2.0}"
    export ORION_PLANNING_RELEASE_CREEP_DISTANCE_M="${ORION_PLANNING_RELEASE_CREEP_DISTANCE_M:-1.0}"
    ;;
  native_dynamics_aware_yield_oracle)
    export ORION_CLOSEDLOOP_CORRUPTION=""
    export ORION_CLOSEDLOOP_RISK_MODE=off
    export ORION_PLANNING_RESPONSE_MODE=privileged_dynamics_aware_yield
    export ORION_PLANNING_INTERPOLATION_STEP_SECONDS="${ORION_PLANNING_INTERPOLATION_STEP_SECONDS:-0.1}"
    export ORION_PLANNING_SAFETY_MARGIN_M="${ORION_PLANNING_SAFETY_MARGIN_M:-0.75}"
    export ORION_PLANNING_CERTIFIED_DECELERATION_MPS2="${ORION_PLANNING_CERTIFIED_DECELERATION_MPS2:-3.0}"
    export ORION_PLANNING_REACTION_SECONDS="${ORION_PLANNING_REACTION_SECONDS:-0.1}"
    export ORION_PLANNING_JUNCTION_FRONT_CLEARANCE_M="${ORION_PLANNING_JUNCTION_FRONT_CLEARANCE_M:-0.5}"
    export ORION_PLANNING_MAP_RESOLUTION_M="${ORION_PLANNING_MAP_RESOLUTION_M:-0.1}"
    export ORION_PLANNING_CLEARANCE_SECONDS="${ORION_PLANNING_CLEARANCE_SECONDS:-1.0}"
    export ORION_PLANNING_RELEASE_SECONDS="${ORION_PLANNING_RELEASE_SECONDS:-0.5}"
    export ORION_PLANNING_PREPARE_CREEP_SPEED_MPS="${ORION_PLANNING_PREPARE_CREEP_SPEED_MPS:-1.0}"
    export ORION_PLANNING_RELEASE_CREEP_SPEED_MPS="${ORION_PLANNING_RELEASE_CREEP_SPEED_MPS:-0.5}"
    export ORION_PLANNING_RELEASE_CREEP_DISTANCE_M="${ORION_PLANNING_RELEASE_CREEP_DISTANCE_M:-1.0}"
    ;;
  front_corrupt_off)
    export ORION_CLOSEDLOOP_CORRUPTION=camera_dropout
    export ORION_CLOSEDLOOP_RISK_MODE=off
    ;;
  front_corrupt_transient_off)
    require_transient_window
    export ORION_CLOSEDLOOP_CORRUPTION=camera_dropout
    export ORION_CLOSEDLOOP_RISK_MODE=off
    ;;
  front_stale_transient_off)
    require_transient_window
    export ORION_CLOSEDLOOP_CORRUPTION=front_stale
    export ORION_CLOSEDLOOP_CORRUPTION_VIEWS=front
    export ORION_CLOSEDLOOP_RISK_MODE=off
    ;;
  lens_waterdrop_transient_off)
    echo "[FAIL] lens_waterdrop_transient_off is the retired failed v1 prototype; use lens_waterdrop_paired_template_transient_off" >&2
    exit 2
    ;;
  lens_waterdrop_paired_template_transient_off)
    require_transient_window
    : "${ORION_PAIRED_WATERDROP_PROFILE:?set an explicitly approved paired waterdrop profile}"
    export ORION_CLOSEDLOOP_CORRUPTION=lens_waterdrop_paired_template
    export ORION_CLOSEDLOOP_CORRUPTION_VIEWS=front
    export ORION_PAIRED_WATERDROP_BANK="${ORION_PAIRED_WATERDROP_BANK:-${PROJECT_ROOT}/assets/waterdrop_patterns/icra2023_paired_template_v1}"
    export ORION_CLOSEDLOOP_RISK_MODE=off
    ;;
  native_motion_blur_off)
    if [[ "${ORION_NATIVE_MOTION_BLUR_PROFILE}" == "none" || \
          "${ORION_NATIVE_MOTION_BLUR_PROFILE}" == "clean" ]]; then
      echo "[FAIL] native_motion_blur_off requires an explicitly approved light, medium, or heavy profile" >&2
      exit 2
    fi
    export ORION_CLOSEDLOOP_CORRUPTION=""
    export ORION_CLOSEDLOOP_RISK_MODE=off
    ;;
  spatial_corrupt_transient_off)
    require_spatial_corruption
    export ORION_CLOSEDLOOP_RISK_MODE=off
    ;;
  spatial_corrupt_transient_oracle)
    require_spatial_corruption
    : "${ORION_CLOSEDLOOP_ORACLE_CORRUPTION_RELEVANT:?set oracle relevance to 1 or 0}"
    export ORION_CLOSEDLOOP_RISK_MODE=oracle
    ;;
  spatial_corrupt_transient_pairwise_trace)
    require_spatial_corruption
    require_pairwise_observation_uq
    export ORION_CLOSEDLOOP_RISK_MODE=off
    ;;
  front_corrupt_aligned_learned)
    echo "[FAIL] scalar learned-UQ governor is retired; use the planning-layer Stage-2 path" >&2
    exit 2
    ;;
  front_corrupt_oracle)
    export ORION_CLOSEDLOOP_CORRUPTION=camera_dropout
    export ORION_CLOSEDLOOP_RISK_MODE=oracle
    if [[ -z "${ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS:-}" || \
          -z "${ORION_CLOSEDLOOP_CORRUPTION_END_PROGRESS:-}" ]]; then
      echo "[FAIL] oracle requires a fixed route-progress event window" >&2
      exit 2
    fi
    ;;
  front_corrupt_transient_oracle)
    require_transient_window
    export ORION_CLOSEDLOOP_CORRUPTION=camera_dropout
    export ORION_CLOSEDLOOP_RISK_MODE=oracle
    ;;
  front_corrupt_transient_oracle_stop)
    require_transient_window
    export ORION_CLOSEDLOOP_CORRUPTION=camera_dropout
    export ORION_CLOSEDLOOP_RISK_MODE=oracle
    ;;
  front_corrupt_transient_pairwise_trace)
    require_transient_window
    require_pairwise_observation_uq
    export ORION_CLOSEDLOOP_CORRUPTION=camera_dropout
    export ORION_CLOSEDLOOP_RISK_MODE=off
    ;;
  front_corrupt_transient_pairwise_stop)
    echo "[FAIL] scalar learned-UQ stop governor is rejected by Route197; use Stage-2 planning" >&2
    exit 2
    ;;
  front_corrupt_constant)
    export ORION_CLOSEDLOOP_CORRUPTION=camera_dropout
    export ORION_CLOSEDLOOP_RISK_MODE=constant
    : "${ORION_CLOSEDLOOP_RISK_CONSTANT:?set matched constant score}"
    ;;
  front_corrupt_shuffled)
    export ORION_CLOSEDLOOP_CORRUPTION=camera_dropout
    export ORION_CLOSEDLOOP_RISK_MODE=trace
    : "${ORION_CLOSEDLOOP_RISK_TRACE:?set shuffled score trace}"
    ;;
  corrupt_correct)
    echo "[FAIL] corrupt_correct is scientifically ambiguous; use front_corrupt_aligned_learned" >&2
    exit 2
    ;;
  *)
    echo "[FAIL] unknown PILOT_CONDITION=${PILOT_CONDITION}" >&2
    exit 2
    ;;
esac

case "${PILOT_CONDITION}" in
  front_stale_transient_off|lens_waterdrop_transient_off|lens_waterdrop_paired_template_transient_off|native_motion_blur_off)
    "${COMPAT_PYTHON_BIN}" "${PROJECT_ROOT}/scripts/preflight_corruption_hardcase_orion_screen.py" \
      --gate "${ORION_CORRUPTION_VISUAL_APPROVAL_GATE}" \
      --repository-root "${PROJECT_ROOT}" \
      --pilot-condition "${PILOT_CONDITION}" \
      --corruption-severity "${ORION_CLOSEDLOOP_CORRUPTION_SEVERITY}" \
      --paired-waterdrop-profile "${ORION_PAIRED_WATERDROP_PROFILE:-}" \
      --native-motion-blur-profile "${ORION_NATIVE_MOTION_BLUR_PROFILE}"
    ;;
esac

if [[ -n "${ORION_OBSERVATION_UQ_CHECKPOINT:-}" && \
      "${ORION_PLANNING_RESPONSE_MODE}" != "off" ]]; then
  echo "[FAIL] do not mix the diagnostic observation-UQ sidecar with a privileged planning oracle" >&2
  exit 2
fi
if [[ "${ORION_STAGE2_ENGINEERING_SMOKE:-0}" == "1" ]]; then
  export ORION_EFFECTIVE_CONDITIONING="controlled_k_to_stage2p_trajectory_response"
elif [[ "${ORION_STAGE2_SPATIAL_UQ_SOURCE:-disabled}" != "disabled" ]]; then
  export ORION_EFFECTIVE_CONDITIONING="spatial_stage1_to_vlm_stage2:${ORION_STAGE2_SPATIAL_UQ_SOURCE}"
elif [[ -n "${ORION_OBSERVATION_UQ_CHECKPOINT:-}" ]]; then
  export ORION_EFFECTIVE_CONDITIONING=frozen_spatial_observation_uq_sidecar
elif [[ "${ORION_PLANNING_RESPONSE_MODE}" != "off" ]]; then
  export ORION_EFFECTIVE_CONDITIONING="privileged_planning_response:${ORION_PLANNING_RESPONSE_MODE}"
elif [[ "${ORION_CLOSEDLOOP_RISK_MODE}" != "off" ]]; then
  export ORION_EFFECTIVE_CONDITIONING="scalar_risk_governor:${ORION_CLOSEDLOOP_RISK_MODE}"
else
  export ORION_EFFECTIVE_CONDITIONING=none
fi

route_file="${PILOT_ROUTE_FILE:-${PILOT_ROUTE_DIR}/route_${PILOT_ROUTE_INDEX}_${PILOT_VARIANT}.xml}"
if [[ ! -f "${route_file}" ]]; then
  echo "[FAIL] missing paired route file: ${route_file}" >&2
  exit 1
fi
if [[ -n "${PILOT_ROUTE_FILE_SHA256:-}" ]]; then
  observed_route_sha256="$(sha256sum "${route_file}" | awk '{print $1}')"
  if [[ "${observed_route_sha256}" != "${PILOT_ROUTE_FILE_SHA256}" ]]; then
    echo "[FAIL] explicit route file hash differs: ${route_file}" >&2
    exit 1
  fi
fi
export PILOT_ROUTE_FILE="${route_file}"

job_tag="${SLURM_JOB_ID:-$$}"
run_name="route${PILOT_ROUTE_INDEX}_${PILOT_VARIANT}_${PILOT_CONDITION}-${job_tag}"
export OUTPUT_ROOT="${ASSET_ROOT}/results/${PILOT_RUN_ID}/${run_name}"
mkdir -p "${OUTPUT_ROOT}"
if [[ "${ORION_STAGE2_ARTIFACT_ROOT:-}" == "AUTO" ]]; then
  export ORION_STAGE2_ARTIFACT_ROOT="${OUTPUT_ROOT}/stage2_artifacts"
fi
export ORION_SENSOR_DIAGNOSTIC_PATH="${ORION_SENSOR_DIAGNOSTIC_PATH:-${OUTPUT_ROOT}/sensor_queue_diagnostics.jsonl}"

case "${ORION_NATIVE_GLARE_PROFILE}" in
  none|clean|medium|heavy) ;;
  *) echo "[FAIL] ORION_NATIVE_GLARE_PROFILE must be none, clean, medium, or heavy" >&2; exit 2 ;;
esac
if [[ "${ORION_NATIVE_GLARE_PROFILE}" != "none" && -n "${ORION_CLOSEDLOOP_CORRUPTION:-}" ]]; then
  echo "[FAIL] native glare cannot be mixed with synthetic ORION_CLOSEDLOOP_CORRUPTION" >&2
  exit 2
fi
case "${ORION_NATIVE_MOTION_BLUR_PROFILE}" in
  none|clean|light|medium|heavy) ;;
  *) echo "[FAIL] ORION_NATIVE_MOTION_BLUR_PROFILE must be none, clean, light, medium, or heavy" >&2; exit 2 ;;
esac
if [[ "${ORION_NATIVE_MOTION_BLUR_PROFILE}" != "none" && -n "${ORION_CLOSEDLOOP_CORRUPTION:-}" ]]; then
  echo "[FAIL] native motion blur cannot be mixed with synthetic ORION_CLOSEDLOOP_CORRUPTION" >&2
  exit 2
fi
if [[ "${ORION_NATIVE_GLARE_PROFILE}" != "none" && "${ORION_NATIVE_MOTION_BLUR_PROFILE}" != "none" ]]; then
  echo "[FAIL] native glare and native motion blur are mutually exclusive paired conditions" >&2
  exit 2
fi

"${COMPAT_PYTHON_BIN}" - "${OUTPUT_ROOT}/manifest.json" <<'PY'
import hashlib, json, os, sys
keys = [
    "PILOT_RUN_ID", "PILOT_ROUTE_INDEX", "PILOT_VARIANT", "PILOT_CONDITION",
    "PILOT_ROUTE_DIR", "PILOT_ROUTE_FILE", "PILOT_ROUTE_FILE_SHA256",
    "SLURM_JOB_ID", "CARLA_QUALITY_LEVEL", "ORION_CLOSEDLOOP_UQ_MODE",
    "BASE_CHECKPOINT_PATH", "ORION_QFORMER_PATH",
    "CARLA_ROOT", "BENCH2DRIVE_ROOT", "BENCH2DRIVE_ZOO_ROOT",
    "ORION_ENABLE_LEGACY_DENSITY_UQ",
    "ORION_PLANNING_RESPONSE_MODE",
    "ORION_PLANNING_ACTOR_CATEGORIES",
    "ORION_PLANNING_INTERPOLATION_STEP_SECONDS",
    "ORION_PLANNING_SAFETY_MARGIN_M",
    "ORION_PLANNING_IMMINENT_HORIZON_SECONDS",
    "ORION_PLANNING_CERTIFIED_DECELERATION_MPS2",
    "ORION_PLANNING_REACTION_SECONDS",
    "ORION_PLANNING_JUNCTION_FRONT_CLEARANCE_M",
    "ORION_PLANNING_MAP_RESOLUTION_M",
    "ORION_PLANNING_CLEARANCE_SECONDS",
    "ORION_PLANNING_RELEASE_SECONDS",
    "ORION_PLANNING_PREPARE_CREEP_SPEED_MPS",
    "ORION_PLANNING_RELEASE_CREEP_SPEED_MPS",
    "ORION_PLANNING_STOP_BUFFER_M",
    "ORION_PLANNING_RELEASE_CREEP_DISTANCE_M",
    "ORION_CLOSEDLOOP_CONDITIONING", "ORION_EFFECTIVE_CONDITIONING",
    "ORION_CLOSEDLOOP_CORRUPTION",
    "ORION_CLOSEDLOOP_CORRUPTION_VIEWS",
    "ORION_CLOSEDLOOP_CORRUPTION_SEVERITY", "ORION_CLOSEDLOOP_RISK_MODE",
    "ORION_CLOSEDLOOP_CORRUPTION_SEED",
    "ORION_CLOSEDLOOP_CORRUPTION_REGION",
    "ORION_CLOSEDLOOP_ORACLE_CORRUPTION_RELEVANT",
    "ORION_CLOSEDLOOP_RISK_ORACLE_START_PROGRESS",
    "ORION_CLOSEDLOOP_RISK_ORACLE_DURATION_SECONDS",
    "ORION_CLOSEDLOOP_CORRUPTION_START_SECONDS",
    "ORION_CLOSEDLOOP_CORRUPTION_END_SECONDS",
    "ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS",
    "ORION_CLOSEDLOOP_CORRUPTION_END_PROGRESS",
    "ORION_CLOSEDLOOP_CORRUPTION_DURATION_SECONDS",
    "ORION_CORRUPTION_VISUAL_APPROVAL_GATE",
    "ORION_PAIRED_WATERDROP_PROFILE",
    "ORION_PAIRED_WATERDROP_BANK",
    "ORION_CLOSEDLOOP_RISK_THRESHOLD", "ORION_CLOSEDLOOP_RISK_SATURATION",
    "ORION_CLOSEDLOOP_RISK_MIN_SPEED", "ORION_CLOSEDLOOP_RISK_MAX_SPEED",
    "ORION_CLOSEDLOOP_RISK_CONSTANT", "ORION_CLOSEDLOOP_RISK_TRACE",
    "ORION_CLOSEDLOOP_RISK_SLOWDOWN_MARGIN",
    "ORION_CLOSEDLOOP_RISK_BRAKE_GAIN", "ORION_CLOSEDLOOP_RISK_MAX_BRAKE",
    "ORION_CLOSEDLOOP_SAFETY_TELEMETRY",
    "ORION_CLOSEDLOOP_SAFETY_HORIZON_SECONDS",
    "ORION_CLOSEDLOOP_SAFETY_RANGE_M",
    "ORION_CLOSEDLOOP_SAFETY_ACTOR_REFRESH_STEPS",
    "ORION_CLOSEDLOOP_SAFETY_VERTICAL_MARGIN_M",
    "ORION_CLOSEDLOOP_SAFETY_MAX_ACTOR_RECORDS",
    "ORION_SENSOR_QUEUE_DIAGNOSTICS",
    "ORION_SENSOR_QUEUE_TIMEOUT_SECONDS",
    "ORION_EXACT_FRAME_SPEEDOMETER",
    "ORION_SENSOR_DIAGNOSTIC_PATH",
    "ORION_CARLA_RPC_TIMEOUT_SECONDS",
    "ORION_SCENARIO_TRACEBACK_INTERVAL_SECONDS",
    "CLOSEDLOOP_PROGRESS_WATCHDOG_SECONDS",
    "CLOSEDLOOP_PROGRESS_STARTUP_GRACE_SECONDS",
    "CLOSEDLOOP_PROGRESS_WATCHDOG_POLL_SECONDS",
    "ORION_OBSERVATION_UQ_CHECKPOINT",
    "ORION_OBSERVATION_UQ_CHECKPOINT_SHA256",
    "ORION_OBSERVATION_UQ_FRONT_VIEW",
    "ORION_OBSERVATION_UQ_BASELINE_START_SECONDS",
    "ORION_OBSERVATION_UQ_BASELINE_END_SECONDS",
    "ORION_OBSERVATION_UQ_MIN_BASELINE_FRAMES",
    "ORION_OBSERVATION_UQ_RELATIVE_SCALE_FLOOR",
    "ORION_OBSERVATION_UQ_ABSOLUTE_SCALE_FLOOR",
    "ORION_OBSERVATION_UQ_Z_CENTER",
    "ORION_OBSERVATION_UQ_ATTACK_ALPHA",
    "ORION_OBSERVATION_UQ_RELEASE_ALPHA",
    "ORION_STAGE2_SPATIAL_UQ_SOURCE",
    "ORION_STAGE1_SPATIAL_UQ_CHECKPOINT",
    "ORION_STAGE1_SPATIAL_UQ_CHECKPOINT_SHA256",
    "ORION_STAGE1_SPATIAL_UQ_WARMUP_FRAMES",
    "ORION_STAGE2_TASK_CHECKPOINT",
    "ORION_STAGE2_TASK_CHECKPOINT_SHA256",
    "ORION_STAGE2_ENGINEERING_SMOKE",
    "ORION_STAGE2_EXTERNAL_K_START_PROGRESS",
    "ORION_STAGE2_EXTERNAL_K_DURATION_SECONDS",
    "ORION_STAGE2_EXTERNAL_K_CAMERA",
    "ORION_STAGE2_EXTERNAL_K_REGION",
    "ORION_STAGE2_EXTERNAL_K_STRENGTH",
    "ORION_STAGE2_EXTERNAL_K_GRID_SIZE",
    "ORION_STAGE2_ARTIFACT_ROOT",
    "ORION_STAGE2_ARTIFACT_ROUTE_GROUP",
    "ORION_STAGE2_ARTIFACT_STRIDE_STEPS",
    "AGENT_CONFIG_PATH",
    "ORION_NATIVE_GLARE_PROFILE",
    "ORION_NATIVE_MOTION_BLUR_PROFILE",
]
payload = {key.lower(): os.environ.get(key) for key in keys}
project_root = os.environ["PROJECT_ROOT"]
sys.path.insert(0, project_root)
from team_code.orion_native_glare import requested_render_condition
from team_code.orion_native_motion_blur import requested_native_motion_blur_condition
if os.environ["ORION_NATIVE_MOTION_BLUR_PROFILE"] != "none":
    payload["render_condition"] = requested_native_motion_blur_condition(
        os.environ["ORION_NATIVE_MOTION_BLUR_PROFILE"]
    )
else:
    payload["render_condition"] = requested_render_condition(
        os.environ["ORION_NATIVE_GLARE_PROFILE"]
    )
payload["render_condition"]["actual_readback"] = {
    "status": "pending_agent_runtime",
    "path": "render_condition_readback.json",
    "schema": "orion.closedloop_render_condition_readback.v1",
}
payload["saved_image_stages"] = {
    "rgb_front": "1600x900 Q20 BGR sensor frame before ORION geometric preprocessing",
    "rgb_front_model_input": (
        "legacy 1600x900 reconstructed corruption preview; not the exact ORION tensor"
    ),
    "rgb_front_model_tensor": (
        "exact post-corruption 640x640 front tensor after reverse normalization to uint8 BGR"
    ),
}
agent_config = os.path.relpath(os.environ["AGENT_CONFIG_PATH"], project_root)
source_paths = (
    agent_config,
    "team_code/orion_b2d_agent.py",
    "team_code/orion_native_glare.py",
    "team_code/orion_native_motion_blur.py",
    "team_code/pid_controller.py",
    "mmcv/models/dense_heads/orion_head.py",
    "mmcv/models/detectors/orion.py",
    "uq_estimator/online_observation_uq.py",
    "uq_estimator/temporal_corruptions.py",
    "uq_estimator/lens_waterdrop.py",
    "uq_estimator/spatial_uq_runtime.py",
    "uq_estimator/spatial_task_fusion.py",
    "uq_estimator/stage2p_task_risk_trajectory.py",
    "uq_estimator/stage2_artifact_capture.py",
    "uq_estimator/risk_governor.py",
    "uq_estimator/closedloop_safety_metrics.py",
    "uq_estimator/closedloop_sensor_diagnostics.py",
    "uq_estimator/privileged_yield_labels.py",
    "uq_estimator/dynamic_yield_expert.py",
    "uq_estimator/bounded_crossing_expert.py",
    "scripts/audit_route147_bounded_crossing.py",
    "scripts/audit_route147_braking_aware_v2.py",
    "scripts/evaluate_route147_bounded_crossing_pair.py",
    "scripts/evaluate_route147_braking_aware_v2.py",
    "scripts/evaluate_clean_liveness_screen.py",
    "scripts/submit_route147_bounded_crossing_pair_v1.sh",
    "scripts/submit_route147_braking_aware_v2.sh",
    "scripts/submit_closedloop_uq_pilot.sh",
    "scripts/analyze_clean_pairwise_native_event.py",
    "scripts/evaluate_route197_dynamic_yield_oracle.py",
    "scripts/evaluate_route197_dynamics_aware_yield_oracle.py",
    "scripts/evaluate_native_collision_discovery.py",
    "scripts/summarize_closedloop_safety.py",
    "scripts/export_carla_junction_geometry.py",
    "scripts/audit_route197_dynamics_aware_expert.py",
    "scripts/submit_route197_dynamics_aware_yield_oracle_v4.sh",
    "scripts/render_closedloop_observation_uq_heatmap.py",
    "scripts/run_closedloop_uq_pilot.sh",
    "scripts/run_official_closedloop_smoke.sh",
    "scripts/run_bench2drive_external_server.py",
    "scripts/verify_clean_pairwise_trace_diagnostic.py",
)
source_sha256 = {}
for relative_path in source_paths:
    digest = hashlib.sha256()
    with open(os.path.join(project_root, relative_path), "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    source_sha256[relative_path] = digest.hexdigest()
payload["source_sha256"] = source_sha256
route_digest = hashlib.sha256()
with open(os.environ["PILOT_ROUTE_FILE"], "rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        route_digest.update(chunk)
payload["route_sha256"] = route_digest.hexdigest()
with open(sys.argv[1], "w") as outfile:
    json.dump(payload, outfile, indent=2, sort_keys=True)
    outfile.write("\n")
PY

# CARLA occupies the RPC port plus adjacent streaming/secondary ports.  Using
# consecutive Slurm job IDs directly therefore collides when jobs are
# backfilled together.  Reserve a ten-port lane per job for both CARLA and the
# traffic manager.
port_slot=$((job_tag % 1000))
port_offset=$((port_slot * 10))
export PORT="${PORT:-$((20000 + port_offset))}"
export TM_PORT="${TM_PORT:-$((40000 + port_offset))}"
export PROJECT_ROOT
export ROUTE_SPLIT_OVERRIDE="${route_file}"
export PYTHON_BIN="${PROJECT_ROOT}/scripts/run_compat_python.sh"
export COMPAT_GLIBC_SYSROOT="${ASSET_ROOT}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot"
export COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-${ASSET_ROOT}/envs/orion-cl/lib}"
export NVIDIA_RUNTIME_PREFIX="${ASSET_ROOT}/nvidia/nvidia_driver-linux-x86_64-535.104.05-archive"
export VULKAN_LOADER_LIBDIR="${ASSET_ROOT}/envs/vulkan-1.3.250/lib"
export VULKANINFO_BIN="${ASSET_ROOT}/envs/vulkan-1.3.250/bin/vulkaninfo"
export BENCH2DRIVE_MANAGES_CARLA=0
export BENCH2DRIVE_EXTERNAL_CARLA=1
export RESUME=False

exec "${PROJECT_ROOT}/scripts/run_official_closedloop_job.sh"

#!/usr/bin/env bash
set -euo pipefail

# Standardized server-side bundle for one round-2 experiment.
# Default behavior is train + open-loop only.
# Replay-based control checks are optional and are NOT the paper-aligned
# "closed-loop" result. Run them only when explicitly requested.
# This script does not assume local execution. Run it on the server after:
#   source /root/miniconda3/etc/profile.d/conda.sh
#   conda activate uq

EXP_ID="${EXP_ID:-R2-A}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/Orion}"
CONFIG_PATH="${CONFIG_PATH:-adzoo/orion/configs/orion_stage3_infer.py}"
ORION_CKPT="${ORION_CKPT:-ckpts/Orion.pth}"
ANN_FILE="${ANN_FILE:-data/infos/b2d_infos_val.pkl}"

RESULT_ROOT="${RESULT_ROOT:-results/round2/${EXP_ID}}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints/film_round2}"
TRAIN_OUT="${TRAIN_OUT:-${CKPT_ROOT}/${EXP_ID}.pt}"
OPEN_OUT="${OPEN_OUT:-${RESULT_ROOT}/openloop.pt}"
TRAIN_LOG="${TRAIN_LOG:-${RESULT_ROOT}/train.log}"
OPEN_LOG="${OPEN_LOG:-${RESULT_ROOT}/openloop.log}"
RUN_REPLAY_CLOSED_LOOP="${RUN_REPLAY_CLOSED_LOOP:-0}"
REPLAY_OUT="${REPLAY_OUT:-${RESULT_ROOT}/replay_closedloop.json}"
REPLAY_LOG="${REPLAY_LOG:-${RESULT_ROOT}/replay_closedloop.log}"

TRAIN_ENV_VARS="${TRAIN_ENV_VARS:-}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
OPEN_EXTRA_ARGS="${OPEN_EXTRA_ARGS:-}"
REPLAY_EXTRA_ARGS="${REPLAY_EXTRA_ARGS:-${CLOSED_EXTRA_ARGS:-}}"

mkdir -p "${PROJECT_ROOT}/${RESULT_ROOT}" "${PROJECT_ROOT}/${CKPT_ROOT}"
cd "${PROJECT_ROOT}"

echo "== Round-2 bundle =="
echo "EXP_ID=${EXP_ID}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "TRAIN_OUT=${TRAIN_OUT}"
echo "OPEN_OUT=${OPEN_OUT}"
echo "RUN_REPLAY_CLOSED_LOOP=${RUN_REPLAY_CLOSED_LOOP}"
echo "REPLAY_OUT=${REPLAY_OUT}"

cat > "${RESULT_ROOT}/manifest.json" <<EOF
{
  "exp_id": "${EXP_ID}",
  "project_root": "${PROJECT_ROOT}",
  "config_path": "${CONFIG_PATH}",
  "orion_ckpt": "${ORION_CKPT}",
  "ann_file": "${ANN_FILE}",
  "train_out": "${TRAIN_OUT}",
  "open_out": "${OPEN_OUT}",
  "run_replay_closed_loop": "${RUN_REPLAY_CLOSED_LOOP}",
  "replay_out": "${REPLAY_OUT}",
  "train_env_vars": "${TRAIN_ENV_VARS}",
  "train_extra_args": "${TRAIN_EXTRA_ARGS}",
  "open_extra_args": "${OPEN_EXTRA_ARGS}",
  "replay_extra_args": "${REPLAY_EXTRA_ARGS}"
}
EOF

echo "== Train =="
if [[ -n "${TRAIN_ENV_VARS}" ]]; then
  eval "${TRAIN_ENV_VARS} python scripts/train_film.py --config ${CONFIG_PATH} --checkpoint ${ORION_CKPT} --ann-file ${ANN_FILE} --out ${TRAIN_OUT} ${TRAIN_EXTRA_ARGS}" 2>&1 | tee "${TRAIN_LOG}"
else
  python scripts/train_film.py --config "${CONFIG_PATH}" --checkpoint "${ORION_CKPT}" --ann-file "${ANN_FILE}" --out "${TRAIN_OUT}" ${TRAIN_EXTRA_ARGS} 2>&1 | tee "${TRAIN_LOG}"
fi

echo "== Open-loop eval =="
python scripts/eval_openloop.py "${CONFIG_PATH}" "${ORION_CKPT}" --ann-file "${ANN_FILE}" --film-checkpoint "${TRAIN_OUT}" --out "${OPEN_OUT}" ${OPEN_EXTRA_ARGS} 2>&1 | tee "${OPEN_LOG}"

if [[ "${RUN_REPLAY_CLOSED_LOOP}" == "1" ]]; then
  echo "== Replay control check =="
  python scripts/eval_closedloop_replay.py "${CONFIG_PATH}" "${ORION_CKPT}" --ann-file "${ANN_FILE}" --film-checkpoint "${TRAIN_OUT}" --out "${REPLAY_OUT}" ${REPLAY_EXTRA_ARGS} 2>&1 | tee "${REPLAY_LOG}"
else
  echo "== Replay control check skipped =="
fi

echo "== Done =="
echo "Artifacts:"
echo "  ${RESULT_ROOT}/manifest.json"
echo "  ${TRAIN_LOG}"
echo "  ${OPEN_OUT}"
echo "  ${OPEN_OUT%.pt}_summary.json"
if [[ "${RUN_REPLAY_CLOSED_LOOP}" == "1" ]]; then
  echo "  ${REPLAY_OUT}"
fi
echo "  ${TRAIN_OUT}"

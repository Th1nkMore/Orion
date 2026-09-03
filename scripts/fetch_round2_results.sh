#!/usr/bin/env bash
set -euo pipefail

# Sync lightweight round-2 artifacts from the server back to the local repo.
# Example:
#   scripts/fetch_round2_results.sh autodl /root/Orion R2-A

REMOTE_HOST="${1:-autodl}"
REMOTE_PROJECT_ROOT="${2:-/root/Orion}"
EXP_ID="${3:-R2-A}"
LOCAL_RESULT_ROOT="${4:-results/round2}"
LOCAL_CKPT_ROOT="${5:-checkpoints/film_round2}"
SSH_RSH="${SSH_RSH:-ssh -o ClearAllForwardings=yes}"

mkdir -p "${LOCAL_RESULT_ROOT}" "${LOCAL_CKPT_ROOT}"

rsync -av \
  -e "${SSH_RSH}" \
  --include="*/" \
  --include="manifest.json" \
  --include="*.log" \
  --include="*_summary.json" \
  --include="*.json" \
  --include="*.pt" \
  --exclude="*" \
  "${REMOTE_HOST}:${REMOTE_PROJECT_ROOT}/results/round2/${EXP_ID}/" \
  "${LOCAL_RESULT_ROOT}/${EXP_ID}/"

rsync -av \
  -e "${SSH_RSH}" \
  "${REMOTE_HOST}:${REMOTE_PROJECT_ROOT}/checkpoints/film_round2/${EXP_ID}.pt" \
  "${LOCAL_CKPT_ROOT}/" || true

echo "Synced ${EXP_ID} artifacts into ${LOCAL_RESULT_ROOT}/${EXP_ID}"

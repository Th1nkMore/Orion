#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/Orion}"
ASSETS="${ASSETS:-/root/autodl-tmp/orion_assets}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/orion-uq/bin/python}"
STEPS="${STEPS:-30}"
EVAL_SAMPLES="${EVAL_SAMPLES:-50}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-500}"

cd "$ROOT"
mkdir -p "$ASSETS/checkpoints/uq_token/grounding_pilot"
mkdir -p "$ASSETS/logs/grounding_pilot"

export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/autodl-tmp/conda/envs/orion-uq/bin:/usr/local/cuda-11.8/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for mode in none zero shuffled correct; do
  echo "===== grounding pilot: $mode ====="
  "$PYTHON" scripts/train_uq_token.py \
    --config adzoo/orion/configs/orion_stage3_infer.py \
    --checkpoint ckpts/Orion.pth \
    --density-checkpoint checkpoints/density_uq/best.pt \
    --descriptor-cache data/density_uq/descriptors.pt \
    --ann-file data/infos/b2d_infos_val.pkl \
    --split train \
    --grounding-only \
    --uq-mode "$mode" \
    --epochs 1 \
    --max-samples "$TRAIN_SAMPLES" \
    --max-steps "$STEPS" \
    --eval-max-samples "$EVAL_SAMPLES" \
    --workers 2 \
    --grad-accum 1 \
    --lambda-vlm 0.001 \
    --lambda-ground 1.0 \
    --lambda-consistency 0.05 \
    --log-interval 10 \
    --seed 42 \
    --out "$ASSETS/checkpoints/uq_token/grounding_pilot/${mode}.pt" \
    2>&1 | tee "$ASSETS/logs/grounding_pilot/${mode}.log"
done

#!/usr/bin/env bash
set -euo pipefail

# Auxiliary Stage-2A engineering smoke only.  This job trains neither ORION's
# VLM nor its trajectory decoder, and its checkpoint is closed-loop ineligible.

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
ASSET_ROOT="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
BUILD_ROOT="${BUILD_ROOT:-${ASSET_ROOT}/stage2/route147_stage2a_optimization_smoke_v1}"
MANIFEST="${MANIFEST:-${BUILD_ROOT}/stage2_optimization_smoke_manifest.jsonl}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-52285724a9306ce5561425bc3c441447ac82cda6f54bcfc0c133c1d915307043}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ASSET_ROOT}/stage2/route147_stage2a_auxiliary_training_v1}"
DRY_RUN="${DRY_RUN:-0}"

python_bin="${ASSET_ROOT}/envs/orion-cl-centos7/bin/python"
log_root="${ASSET_ROOT}/stage2/logs"
mkdir -p "${log_root}"

"${python_bin}" - "${MANIFEST}" "${EXPECTED_MANIFEST_SHA256}" "${OUTPUT_ROOT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest = Path(sys.argv[1])
expected = sys.argv[2]
output = Path(sys.argv[3])

digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
if digest != expected:
    raise SystemExit("Stage-2A manifest hash differs")
records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
if len(records) != 189:
    raise SystemExit("Stage-2A manifest must contain exactly 189 smoke records")
for record in records:
    contract = record.get("supervision_contract") or {}
    if contract.get("uses_density_uq") is not False:
        raise SystemExit("Density UQ is forbidden in Stage-2A")
    if contract.get("uses_corruption_label") is not False:
        raise SystemExit("corruption labels are forbidden in Stage-2A")
if output.exists():
    raise SystemExit("refusing to overwrite existing Stage-2A training output")
print("ROUTE147_STAGE2A_PREFLIGHT_OK=1")
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1; no Slurm job submitted"
  exit 0
fi

job_id=$(sbatch --parsable \
  --partition=Nvidia_A800 \
  --gres=gpu:1 \
  --cpus-per-task=1 \
  --mem=8G \
  --time=00:20:00 \
  --job-name=stage2a_r147_smoke \
  --output="${log_root}/route147_stage2a_auxiliary_training_v1-%j.out" \
  --wrap="set -euo pipefail; cd '${PROJECT_ROOT}'; export LD_LIBRARY_PATH='${ASSET_ROOT}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot/usr/lib64:${ASSET_ROOT}/envs/orion-cl/lib'; '${python_bin}' scripts/train_stage2_task_response.py --manifest '${MANIFEST}' --output-dir '${OUTPUT_ROOT}' --mode optimization_smoke --epochs 30 --batch-size 16 --learning-rate 3e-4 --weight-decay 1e-4 --tokens-per-view 8 --hidden-dim 256 --num-heads 8 --seed 20260829 --device cuda")

echo "route147_stage2a_auxiliary_training_v1=${job_id}"

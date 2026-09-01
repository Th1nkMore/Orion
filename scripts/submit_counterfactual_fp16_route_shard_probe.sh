#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/observation_uq_counterfactual_fp16_route_shard_probe_v1.json}"
PYTHON_BIN="${PYTHON_BIN:-/public/share/lidachuan/orion_assets/envs/orion-cl-centos7/bin/python}"

mapfile -t CONFIG_VALUES < <(
  "${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r"))
print(payload["output"]["run_root"])
print(payload["scheduler"]["partition"])
print(payload["scheduler"]["cpus_per_task"])
print(payload["scheduler"]["memory"])
print(payload["scheduler"]["time_limit"])
print(payload["scheduler"]["gpu_count"])
PY
)

RUN_ROOT="${CONFIG_VALUES[0]}"
PARTITION="${CONFIG_VALUES[1]}"
CPUS="${CONFIG_VALUES[2]}"
MEMORY="${CONFIG_VALUES[3]}"
TIME_LIMIT="${CONFIG_VALUES[4]}"
GPU_COUNT="${CONFIG_VALUES[5]}"
mkdir -p "${RUN_ROOT}"

SBATCH_ARGS=(
  --parsable
  --partition "${PARTITION}"
  --cpus-per-task "${CPUS}"
  --mem "${MEMORY}"
  --time "${TIME_LIMIT}"
  --job-name "uq_fp16_route_probe"
  --output "${RUN_ROOT}/slurm-%j.out"
  --export "ALL,CONFIG_PATH=${CONFIG_PATH},PYTHON_BIN=${PYTHON_BIN},ORION_REPO_ROOT=${REPO_ROOT}"
)
if [[ "${GPU_COUNT}" -gt 0 ]]; then
  SBATCH_ARGS+=(--gres "gpu:${GPU_COUNT}")
fi
SBATCH_ARGS+=("${REPO_ROOT}/scripts/run_counterfactual_fp16_route_shard_probe.sh")
sbatch "${SBATCH_ARGS[@]}"

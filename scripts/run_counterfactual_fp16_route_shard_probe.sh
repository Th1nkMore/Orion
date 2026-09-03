#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${ORION_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/observation_uq_counterfactual_fp16_route_shard_probe_v1.json}"
PYTHON_BIN="${PYTHON_BIN:-/public/share/lidachuan/orion_assets/envs/orion-cl-centos7/bin/python}"

mapfile -t CONFIG_VALUES < <(
  "${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r"))
print(payload["source"]["feature_shard"])
print(payload["source"]["feature_shard_sha256"])
print(payload["output"]["run_root"])
print(payload["output"]["route_shards"])
print(payload["conversion"]["max_routes"])
print(payload["conversion"]["target_batch_size"])
PY
)

SOURCE_SHARD="${CONFIG_VALUES[0]}"
SOURCE_SHA256="${CONFIG_VALUES[1]}"
RUN_ROOT="${CONFIG_VALUES[2]}"
ROUTE_SHARDS="${CONFIG_VALUES[3]}"
MAX_ROUTES="${CONFIG_VALUES[4]}"
TARGET_BATCH_SIZE="${CONFIG_VALUES[5]}"

mkdir -p "${RUN_ROOT}"
cp "${CONFIG_PATH}" "${RUN_ROOT}/frozen_config.json"
sha256sum "${CONFIG_PATH}" > "${RUN_ROOT}/frozen_config.sha256"

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/convert_counterfactual_feature_shard_to_fp16_routes.py" \
  --input "${SOURCE_SHARD}" \
  --output-dir "${ROUTE_SHARDS}" \
  --source-sha256 "${SOURCE_SHA256}" \
  --max-routes "${MAX_ROUTES}" \
  --target-batch-size "${TARGET_BATCH_SIZE}"

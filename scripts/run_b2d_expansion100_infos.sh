#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${ORION_REPO_ROOT:-/public/home/lidachuan/project/Orion}"
ASSET_ROOT="${ORION_ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
PYTHON_BIN="${PYTHON_BIN:-${ASSET_ROOT}/envs/orion-cl-centos7/bin/python}"
PLAN_ROOT="${ASSET_ROOT}/observation_uq_v3/plans/b2d_expansion100_seed20260827"
PLAN="${PLAN_ROOT}/b2d_expansion100_plan_seed20260827.json"
EXTRACTION_RECEIPT="${PLAN_ROOT}/extraction_receipt.json"
BASELINE_MANIFEST="${REPO_ROOT}/configs/spatial_uq_route_manifests/b2d_val_exploratory_seed20260826.json"
DATA_ROOT="${ASSET_ROOT}/data/bench2drive"
OUTPUT_DIR="${ASSET_ROOT}/data/infos_expansion100_seed20260827"
INFOS="${OUTPUT_DIR}/b2d_infos_val.pkl"
FORMAL_MANIFEST="${PLAN_ROOT}/b2d_route_manifest_expansion100_seed20260827.json"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "[FAIL] refusing to reuse infos output directory: ${OUTPUT_DIR}" >&2
  exit 1
fi
if [[ -e "${FORMAL_MANIFEST}" ]]; then
  echo "[FAIL] refusing to overwrite formal route manifest: ${FORMAL_MANIFEST}" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${PLAN}" "${EXTRACTION_RECEIPT}" "${DATA_ROOT}" <<'PY'
import json
import pathlib
import sys

plan_path = pathlib.Path(sys.argv[1])
receipt_path = pathlib.Path(sys.argv[2])
data_root = pathlib.Path(sys.argv[3])
plan = json.loads(plan_path.read_text())
receipt = json.loads(receipt_path.read_text())
if receipt.get("schema_version") != "b2d-expansion-extraction-receipt/v1":
    raise RuntimeError("extraction receipt schema differs")
if receipt.get("route_count") != 50:
    raise RuntimeError("extraction receipt must contain 50 routes")
deletion = receipt.get("archive_deletion", {})
if not deletion.get("performed") or deletion.get("deleted_count") != 50:
    raise RuntimeError("verified expansion archives were not deleted as planned")
routes = sorted(
    path.name
    for path in (data_root / "v1").iterdir()
    if path.is_dir() and "Town" in path.name and "Route" in path.name and "Weather" in path.name
)
if len(routes) != 100 or len(routes) != len(set(routes)):
    raise RuntimeError("expected exactly 100 unique route directories, got %d" % len(routes))
planned = {pathlib.PurePosixPath(row["folder"]).name for row in plan["additions"]}
if not planned.issubset(routes):
    raise RuntimeError("one or more planned expansion routes are absent")
print("B2D_EXPANSION_INFO_PREFLIGHT_OK=1", flush=True)
PY

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/prepare_b2d_subset_infos.py" \
  --zoo-root "${ASSET_ROOT}/Bench2DriveZoo" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --split-name val \
  --tmp-dir tmp_expansion100_seed20260827 \
  --skip-maps

test -s "${INFOS}"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_b2d_expansion_manifest.py" \
  --infos "${INFOS}" \
  --baseline-manifest "${BASELINE_MANIFEST}" \
  --expansion-plan "${PLAN}" \
  --output "${FORMAL_MANIFEST}"

sha256sum "${INFOS}" "${FORMAL_MANIFEST}" > "${PLAN_ROOT}/expanded_infos_and_manifest.sha256"
echo "B2D_EXPANSION_INFOS_AND_MANIFEST_OK=1"

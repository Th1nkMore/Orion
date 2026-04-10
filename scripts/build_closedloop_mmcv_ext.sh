#!/usr/bin/env bash
set -euo pipefail

# Build a Python-3.8-compatible mmcv extension for the isolated official
# closed-loop env. This intentionally hides GPUs during the build so we can
# compile the CPU-only mmcv._ext without requiring nvcc.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda/envs/orion-cl/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[FAIL] missing python binary: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

echo "== Build Closed-Loop MMCV Extension =="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"

echo "[SETUP] ensure Python-side build helpers"
"${PYTHON_BIN}" -m pip install ninja psutil wheel setuptools

echo "[BUILD] compile mmcv._ext for the current interpreter"
CUDA_VISIBLE_DEVICES= ORION_SKIP_POINTCLOUD_EXTS=1 \
  "${PYTHON_BIN}" setup.py build_ext --inplace

echo "== Result =="
ls -1 "${PROJECT_ROOT}"/mmcv/_ext*.so

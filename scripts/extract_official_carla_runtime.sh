#!/usr/bin/env bash
set -euo pipefail

# Extract CARLA 0.9.15 and import AdditionalMaps into the extracted runtime.

CARLA_DOWNLOAD_ROOT="${CARLA_DOWNLOAD_ROOT:-/root/autodl-tmp/carla}"
CARLA_MAIN_ARCHIVE="${CARLA_MAIN_ARCHIVE:-${CARLA_DOWNLOAD_ROOT}/CARLA_0.9.15.tar.gz}"
CARLA_MAPS_ARCHIVE="${CARLA_MAPS_ARCHIVE:-${CARLA_DOWNLOAD_ROOT}/AdditionalMaps_0.9.15.tar.gz}"

if [[ ! -f "${CARLA_MAIN_ARCHIVE}" ]]; then
  echo "[FAIL] missing CARLA archive: ${CARLA_MAIN_ARCHIVE}" >&2
  exit 1
fi

if [[ ! -f "${CARLA_MAPS_ARCHIVE}" ]]; then
  echo "[FAIL] missing AdditionalMaps archive: ${CARLA_MAPS_ARCHIVE}" >&2
  exit 1
fi

cd "${CARLA_DOWNLOAD_ROOT}"

if [[ ! -x "${CARLA_DOWNLOAD_ROOT}/CarlaUE4.sh" ]]; then
  echo "[SETUP] extract CARLA runtime into ${CARLA_DOWNLOAD_ROOT}"
  tar -xf "${CARLA_MAIN_ARCHIVE}"
else
  echo "[OK] CARLA runtime already extracted"
fi

if [[ ! -d "${CARLA_DOWNLOAD_ROOT}/Import" ]]; then
  echo "[FAIL] Import directory missing after extraction" >&2
  exit 1
fi

ln -sfn "${CARLA_MAPS_ARCHIVE}" "${CARLA_DOWNLOAD_ROOT}/Import/AdditionalMaps_0.9.15.tar.gz"

echo "[SETUP] import AdditionalMaps"
yes | bash "${CARLA_DOWNLOAD_ROOT}/ImportAssets.sh"

echo "== Result =="
echo "CARLA_ROOT=${CARLA_DOWNLOAD_ROOT}"
ls -lah "${CARLA_DOWNLOAD_ROOT}/CarlaUE4.sh" "${CARLA_DOWNLOAD_ROOT}/ImportAssets.sh" "${CARLA_DOWNLOAD_ROOT}/PythonAPI"

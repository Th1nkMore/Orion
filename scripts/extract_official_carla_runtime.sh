#!/usr/bin/env bash
set -euo pipefail

# Extract CARLA 0.9.15 and import AdditionalMaps into the extracted runtime.

CARLA_DOWNLOAD_ROOT="${CARLA_DOWNLOAD_ROOT:-/root/autodl-tmp/carla}"
CARLA_MAIN_ARCHIVE="${CARLA_MAIN_ARCHIVE:-${CARLA_DOWNLOAD_ROOT}/CARLA_0.9.15.tar.gz}"
CARLA_MAPS_ARCHIVE="${CARLA_MAPS_ARCHIVE:-${CARLA_DOWNLOAD_ROOT}/AdditionalMaps_0.9.15.tar.gz}"
IMPORT_ADDITIONAL_MAPS="${IMPORT_ADDITIONAL_MAPS:-auto}"

if [[ ! -f "${CARLA_MAIN_ARCHIVE}" ]]; then
  echo "[FAIL] missing CARLA archive: ${CARLA_MAIN_ARCHIVE}" >&2
  exit 1
fi

mkdir -p "${CARLA_DOWNLOAD_ROOT}"

if [[ ! -x "${CARLA_DOWNLOAD_ROOT}/CarlaUE4.sh" ]]; then
  echo "[SETUP] extract CARLA runtime into ${CARLA_DOWNLOAD_ROOT}"
  tar -xf "${CARLA_MAIN_ARCHIVE}" -C "${CARLA_DOWNLOAD_ROOT}"
else
  echo "[OK] CARLA runtime already extracted"
fi

if [[ "${IMPORT_ADDITIONAL_MAPS}" == "0" ]]; then
  echo "[SKIP] AdditionalMaps import disabled"
elif [[ ! -f "${CARLA_MAPS_ARCHIVE}" ]]; then
  if [[ "${IMPORT_ADDITIONAL_MAPS}" == "1" ]]; then
    echo "[FAIL] missing AdditionalMaps archive: ${CARLA_MAPS_ARCHIVE}" >&2
    exit 1
  fi
  echo "[SKIP] AdditionalMaps archive not ready; base CARLA towns are usable"
else
  if [[ ! -d "${CARLA_DOWNLOAD_ROOT}/Import" ]]; then
    echo "[FAIL] Import directory missing after extraction" >&2
    exit 1
  fi

  ln -sfn "${CARLA_MAPS_ARCHIVE}" "${CARLA_DOWNLOAD_ROOT}/Import/AdditionalMaps_0.9.15.tar.gz"

  echo "[SETUP] import AdditionalMaps"
  set +o pipefail
  (
    cd "${CARLA_DOWNLOAD_ROOT}"
    yes | bash ./ImportAssets.sh
  )
  import_status=$?
  set -o pipefail
  if [[ "${import_status}" -ne 0 && "${import_status}" -ne 141 ]]; then
    echo "[FAIL] ImportAssets.sh exited with ${import_status}" >&2
    exit "${import_status}"
  fi
fi

echo "== Result =="
echo "CARLA_ROOT=${CARLA_DOWNLOAD_ROOT}"
ls -lah "${CARLA_DOWNLOAD_ROOT}/CarlaUE4.sh" "${CARLA_DOWNLOAD_ROOT}/ImportAssets.sh" "${CARLA_DOWNLOAD_ROOT}/PythonAPI"

#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the non-GPU parts of the official CARLA closed-loop stack on a
# remote server. This is safe to run while unrelated GPU workloads are active.
#
# The script intentionally focuses on environment/assets/bootstrap only:
# - isolated Python env for CARLA/Bench2Drive compatibility
# - Bench2Drive checkout + project symlink
# - path injection for leaderboard/scenario_runner/project imports
# - CARLA Python API installation
# - background downloads for CARLA 0.9.15 assets
#
# It does not try to launch CARLA or run evaluation.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
TMP_ROOT="${TMP_ROOT:-/root/autodl-tmp}"
BENCH2DRIVE_ROOT="${BENCH2DRIVE_ROOT:-${TMP_ROOT}/Bench2DriveZoo}"
BENCH2DRIVE_LINK="${BENCH2DRIVE_LINK:-${PROJECT_ROOT}/Bench2DriveZoo}"
CARLA_DOWNLOAD_ROOT="${CARLA_DOWNLOAD_ROOT:-${TMP_ROOT}/carla}"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CLOSEDLOOP_ENV="${CLOSEDLOOP_ENV:-orion-cl}"
START_CARLA_DOWNLOADS="${START_CARLA_DOWNLOADS:-1}"

BASE_CARLA_URL="https://carla-releases.s3.us-east-005.backblazeb2.com/Linux"
CARLA_MAIN_ARCHIVE="${CARLA_DOWNLOAD_ROOT}/CARLA_0.9.15.tar.gz"
CARLA_MAPS_ARCHIVE="${CARLA_DOWNLOAD_ROOT}/AdditionalMaps_0.9.15.tar.gz"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[FAIL] missing conda init script: ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

ensure_conda_env() {
  if ! conda env list | awk '{print $1}' | grep -qx "${CLOSEDLOOP_ENV}"; then
    echo "[SETUP] create conda env ${CLOSEDLOOP_ENV} (python=3.8)"
    conda create -n "${CLOSEDLOOP_ENV}" python=3.8 -y
  else
    echo "[OK] conda env exists: ${CLOSEDLOOP_ENV}"
  fi
}

ensure_bench2drive() {
  mkdir -p "$(dirname "${BENCH2DRIVE_ROOT}")"
  if [[ ! -d "${BENCH2DRIVE_ROOT}/.git" ]]; then
    echo "[SETUP] clone Bench2Drive -> ${BENCH2DRIVE_ROOT}"
    git clone https://github.com/Thinklab-SJTU/Bench2Drive "${BENCH2DRIVE_ROOT}"
  else
    echo "[OK] Bench2Drive checkout exists: ${BENCH2DRIVE_ROOT}"
  fi
  ln -sfn "${BENCH2DRIVE_ROOT}" "${BENCH2DRIVE_LINK}"
  echo "[OK] project link: ${BENCH2DRIVE_LINK} -> $(readlink "${BENCH2DRIVE_LINK}")"
}

install_python_side() {
  conda activate "${CLOSEDLOOP_ENV}"
  echo "[SETUP] install CARLA Python API and base closed-loop deps into ${CLOSEDLOOP_ENV}"
  pip install carla==0.9.15
  pip install \
    py-trees==0.8.3 \
    dictor \
    ephem \
    tabulate \
    networkx \
    shapely==1.8.5.post1 \
    scipy \
    Pillow \
    opencv-python \
    pyquaternion \
    xmlschema \
    six \
    matplotlib

  python - <<PY
import pathlib
import site

project_root = pathlib.Path("${PROJECT_ROOT}")
bench_root = pathlib.Path("${BENCH2DRIVE_LINK}")
site_packages = next(pathlib.Path(p) for p in site.getsitepackages() if p.endswith("site-packages"))
pth = site_packages / "orion_closedloop_paths.pth"
pth.write_text(
    f"{project_root}\n"
    f"{bench_root / 'leaderboard'}\n"
    f"{bench_root / 'scenario_runner'}\n"
)
print(f"[OK] wrote {pth}")
print(pth.read_text(), end="")
PY
}

start_carla_downloads() {
  mkdir -p "${CARLA_DOWNLOAD_ROOT}"
  if [[ "${START_CARLA_DOWNLOADS}" != "1" ]]; then
    echo "[SKIP] CARLA archive downloads disabled"
    return
  fi

  if [[ ! -f "${CARLA_MAIN_ARCHIVE}" ]]; then
    echo "[SETUP] start CARLA main archive download"
    nohup wget -c "${BASE_CARLA_URL}/CARLA_0.9.15.tar.gz" \
      > "${CARLA_DOWNLOAD_ROOT}/carla_download.log" 2>&1 < /dev/null &
    echo "[OK] main download pid: $!"
  else
    echo "[OK] main archive already present: ${CARLA_MAIN_ARCHIVE}"
  fi

  if [[ ! -f "${CARLA_MAPS_ARCHIVE}" ]]; then
    echo "[SETUP] start AdditionalMaps archive download"
    nohup wget -c "${BASE_CARLA_URL}/AdditionalMaps_0.9.15.tar.gz" \
      > "${CARLA_DOWNLOAD_ROOT}/carla_maps_download.log" 2>&1 < /dev/null &
    echo "[OK] maps download pid: $!"
  else
    echo "[OK] maps archive already present: ${CARLA_MAPS_ARCHIVE}"
  fi
}

echo "== Official Closed-Loop Bootstrap =="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "TMP_ROOT=${TMP_ROOT}"
echo "BENCH2DRIVE_ROOT=${BENCH2DRIVE_ROOT}"
echo "BENCH2DRIVE_LINK=${BENCH2DRIVE_LINK}"
echo "CARLA_DOWNLOAD_ROOT=${CARLA_DOWNLOAD_ROOT}"
echo "CLOSEDLOOP_ENV=${CLOSEDLOOP_ENV}"

ensure_conda_env
ensure_bench2drive
install_python_side
start_carla_downloads

echo "== Next =="
echo "1. Wait for CARLA archives to finish downloading."
echo "2. Extract CARLA 0.9.15 and import AdditionalMaps."
echo "3. Complete the ORION runtime layer inside ${CLOSEDLOOP_ENV}."
echo "4. Run bash scripts/check_official_closedloop_env.sh with explicit roots."

#!/usr/bin/env bash
set -euo pipefail

# Run a Python environment with an explicit glibc sysroot. This is needed on
# CentOS 7 hosts for CARLA 0.9.15's Python extension and newer binary wheels.

COMPAT_PYTHON_BIN="${COMPAT_PYTHON_BIN:-}"
COMPAT_GLIBC_SYSROOT="${COMPAT_GLIBC_SYSROOT:-}"
COMPAT_LIBRARY_PATH="${COMPAT_LIBRARY_PATH:-}"

if [[ -z "${COMPAT_PYTHON_BIN}" || ! -x "${COMPAT_PYTHON_BIN}" ]]; then
  echo "[FAIL] set COMPAT_PYTHON_BIN to an executable Python interpreter" >&2
  exit 1
fi

loader="${COMPAT_GLIBC_SYSROOT}/lib64/ld-linux-x86-64.so.2"
if [[ -z "${COMPAT_GLIBC_SYSROOT}" || ! -x "${loader}" ]]; then
  echo "[FAIL] set COMPAT_GLIBC_SYSROOT to a sysroot containing ${loader##*/}" >&2
  exit 1
fi

python_prefix="$(cd "$(dirname "${COMPAT_PYTHON_BIN}")/.." && pwd)"
library_path="${COMPAT_GLIBC_SYSROOT}/lib64:${COMPAT_GLIBC_SYSROOT}/usr/lib64:${python_prefix}/lib"
if [[ -n "${COMPAT_LIBRARY_PATH}" ]]; then
  library_path="${library_path}:${COMPAT_LIBRARY_PATH}"
fi
library_path="${library_path}:/lib64:/usr/lib64"

exec "${loader}" --library-path "${library_path}" "${COMPAT_PYTHON_BIN}" "$@"

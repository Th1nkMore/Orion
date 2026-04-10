#!/usr/bin/env bash
set -euo pipefail

# Validate whether the current machine exposes a usable GPU Vulkan runtime for
# CARLA. A successful check should create a Vulkan instance and enumerate a
# non-CPU device rather than falling back to Mesa llvmpipe.

VULKANINFO_BIN="${VULKANINFO_BIN:-vulkaninfo}"
NVIDIA_RUNTIME_PREFIX="${NVIDIA_RUNTIME_PREFIX:-}"
VK_ICD_JSON="${VK_ICD_JSON:-}"
VK_LOADER_LAYERS_DISABLE="${VK_LOADER_LAYERS_DISABLE:-~all}"
XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/vulkan-runtime-${USER:-unknown}}"

if ! command -v "${VULKANINFO_BIN}" >/dev/null 2>&1; then
  echo "[MISS] vulkaninfo not found: ${VULKANINFO_BIN}" >&2
  exit 1
fi

if [[ -n "${NVIDIA_RUNTIME_PREFIX}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_RUNTIME_PREFIX}/usr/lib/x86_64-linux-gnu:${NVIDIA_RUNTIME_PREFIX}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  if [[ -z "${VK_ICD_JSON}" && -f "${NVIDIA_RUNTIME_PREFIX}/usr/share/vulkan/icd.d/nvidia_icd.json" ]]; then
    VK_ICD_JSON="${NVIDIA_RUNTIME_PREFIX}/usr/share/vulkan/icd.d/nvidia_icd.json"
  fi
fi

mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

export XDG_RUNTIME_DIR
export VK_LOADER_LAYERS_DISABLE
if [[ -n "${VK_ICD_JSON}" ]]; then
  export VK_ICD_FILENAMES="${VK_ICD_JSON}"
fi

summary_file="$(mktemp)"
trap 'rm -f "${summary_file}"' EXIT

if ! "${VULKANINFO_BIN}" --summary >"${summary_file}" 2>&1; then
  echo "[MISS] Vulkan instance creation failed." >&2
  sed -n '1,80p' "${summary_file}" >&2
  exit 1
fi

if grep -qi 'llvmpipe' "${summary_file}" || grep -qi 'PHYSICAL_DEVICE_TYPE_CPU' "${summary_file}"; then
  echo "[MISS] Vulkan fell back to CPU rendering." >&2
  sed -n '1,120p' "${summary_file}" >&2
  exit 1
fi

echo "[OK] GPU Vulkan runtime looks usable for CARLA."
sed -n '1,80p' "${summary_file}"

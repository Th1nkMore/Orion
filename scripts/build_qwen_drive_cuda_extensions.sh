#!/usr/bin/env bash
set -euo pipefail

# Build the two CUDA extensions required by Qwen-Drive against the cluster's
# glibc-compatible toolchain.  The upstream binary wheels require glibc 2.32,
# while this deployment deliberately runs through a glibc 2.28 loader.

asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_env="${asset_root}/envs/qwen-drive-py310"
python_bin="${python_env}/bin/python"
glibc_sysroot="${asset_root}/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot"
glibc_loader="${glibc_sysroot}/lib64/ld-linux-x86-64.so.2"
runtime_library_path="${glibc_sysroot}/lib64:${glibc_sysroot}/usr/lib64:${python_env}/lib:/lib64:/usr/lib64"
source_root="${asset_root}/sources"
wheel_root="${asset_root}/wheels/glibc228"
flash_source="${source_root}/flash_attn-2.8.3.sdist.tar.gz"
causal_source="${source_root}/causal_conv1d-1.6.2.post1.sdist.tar.gz"

for prerequisite in \
  "${python_bin}" "${glibc_loader}" "${flash_source}" "${causal_source}" \
  "${python_env}/bin/x86_64-conda-linux-gnu-gcc" \
  "${python_env}/bin/x86_64-conda-linux-gnu-g++" \
  "/usr/local/cuda/bin/nvcc"; do
  if [[ ! -x "${prerequisite}" && ! -f "${prerequisite}" ]]; then
    echo "missing Qwen-Drive CUDA build prerequisite: ${prerequisite}" >&2
    exit 2
  fi
done

printf '%s  %s\n' \
  "1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d" \
  "${flash_source}" \
  "245e314ea21064ded7a5bf6b3b842b644aa6f92e45cecfe3e935629744c35ff4" \
  "${causal_source}" | sha256sum -c -

build_root="$(mktemp -d "${TMPDIR:-/tmp}/qwen-drive-cuda-build.XXXXXX")"
trap 'rm -rf "${build_root}"' EXIT
mkdir -p "${wheel_root}"

export PATH="${python_env}/bin:/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="/usr/local/cuda"
export CC="${python_env}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${python_env}/bin/x86_64-conda-linux-gnu-g++"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE
export FLASH_ATTN_CUDA_ARCHS=80
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE

run_python=(
  "${glibc_loader}" --library-path "${runtime_library_path}" "${python_bin}"
)

mkdir -p "${build_root}/wheels"
tar -xzf "${causal_source}" -C "${build_root}"
tar -xzf "${flash_source}" -C "${build_root}"

# Invoke setup.py through the explicit loader.  pip's PEP 517 helper respawns
# sys.executable without that loader and falls back to the host's glibc 2.17.
# Persist each successful wheel immediately so a later retry does not rebuild it.
if ! compgen -G "${wheel_root}/causal_conv1d-1.6.2.post1-*.whl" >/dev/null; then
  (
    cd "${build_root}/causal_conv1d-1.6.2.post1"
    "${run_python[@]}" setup.py bdist_wheel --dist-dir "${build_root}/wheels"
  )
  cp "${build_root}"/wheels/causal_conv1d-*.whl "${wheel_root}/"
else
  echo "reusing completed causal-conv1d cluster wheel"
fi
if ! compgen -G "${wheel_root}/flash_attn-2.8.3-*.whl" >/dev/null; then
  (
    cd "${build_root}/flash_attn-2.8.3"
    "${run_python[@]}" setup.py bdist_wheel --dist-dir "${build_root}/wheels"
  )
  cp "${build_root}"/wheels/flash_attn-*.whl "${wheel_root}/"
else
  echo "reusing completed FlashAttention cluster wheel"
fi

sha256sum "${wheel_root}"/causal_conv1d-*.whl "${wheel_root}"/flash_attn-*.whl

"${run_python[@]}" -m pip install --no-deps --force-reinstall \
  "${wheel_root}"/causal_conv1d-*.whl \
  "${wheel_root}"/flash_attn-*.whl

echo "Qwen-Drive CUDA extensions built and installed for the cluster runtime"

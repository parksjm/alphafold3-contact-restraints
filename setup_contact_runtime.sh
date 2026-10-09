#!/usr/bin/env bash
# Make this source clone runnable with an existing binary AF3 installation.
# This creates only ignored symlinks inside this clone; it does not modify the
# existing AF3 environment or installation.

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
af3_python="${AF3_PYTHON:-/data/jpc/code/.conda/envs/af3/bin/python}"

if [[ ! -x "${af3_python}" ]]; then
  echo "AF3 Python is not executable: ${af3_python}" >&2
  echo "Set AF3_PYTHON to the Python executable of an installed AF3 environment." >&2
  exit 1
fi

installed_package="$(env -u PYTHONPATH "${af3_python}" -c \
  'import pathlib, alphafold3; print(pathlib.Path(alphafold3.__file__).parent)')"

mapfile -t cpp_extensions < <(find "${installed_package}" -maxdepth 1 \
  -type f -name 'cpp*.so' -print)
if [[ "${#cpp_extensions[@]}" -ne 1 ]]; then
  echo "Expected one AF3 C++ extension in ${installed_package}; found ${#cpp_extensions[@]}." >&2
  exit 1
fi

link_runtime_file() {
  local source_path="$1"
  local destination_path="$2"
  if [[ ! -f "${source_path}" ]]; then
    echo "Required AF3 runtime file is missing: ${source_path}" >&2
    exit 1
  fi
  if [[ -e "${destination_path}" && ! -L "${destination_path}" ]]; then
    echo "Refusing to replace non-symlink runtime file: ${destination_path}" >&2
    exit 1
  fi
  ln -sfn "${source_path}" "${destination_path}"
}

link_runtime_file \
  "${cpp_extensions[0]}" \
  "${repo_dir}/src/alphafold3/$(basename "${cpp_extensions[0]}")"
link_runtime_file \
  "${installed_package}/constants/converters/ccd.pickle" \
  "${repo_dir}/src/alphafold3/constants/converters/ccd.pickle"
link_runtime_file \
  "${installed_package}/constants/converters/chemical_component_sets.pickle" \
  "${repo_dir}/src/alphafold3/constants/converters/chemical_component_sets.pickle"

echo "Contact-restraint runtime overlay is ready."
echo "Use: PYTHONPATH=${repo_dir}/src ${af3_python} <AF3 command and arguments>"

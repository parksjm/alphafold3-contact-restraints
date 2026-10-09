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
purelib="$("${af3_python}" -S -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
wheel_package="${purelib}/alphafold3"

mapfile -t cpp_extensions < <(
  find "${installed_package}" "${wheel_package}" -maxdepth 1 \
    -type f -name 'cpp*.so' -print 2>/dev/null | sort -u
)
if [[ "${#cpp_extensions[@]}" -ne 1 ]]; then
  echo "Expected one AF3 C++ extension; found ${#cpp_extensions[@]}." >&2
  echo "Searched ${installed_package} and ${wheel_package}." >&2
  exit 1
fi

find_runtime_file() {
  local relative_path="$1"
  local candidate
  for candidate in \
    "${installed_package}/${relative_path}" \
    "${wheel_package}/${relative_path}"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  echo "Required AF3 runtime file was not found: ${relative_path}" >&2
  return 1
}

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
  "$(find_runtime_file constants/converters/ccd.pickle)" \
  "${repo_dir}/src/alphafold3/constants/converters/ccd.pickle"
link_runtime_file \
  "$(find_runtime_file constants/converters/chemical_component_sets.pickle)" \
  "${repo_dir}/src/alphafold3/constants/converters/chemical_component_sets.pickle"
link_runtime_file \
  "${repo_dir}/OUTPUT_TERMS_OF_USE.md" \
  "${repo_dir}/src/alphafold3/OUTPUT_TERMS_OF_USE.md"

echo "Contact-restraint runtime overlay is ready."
echo "Use: PYTHONPATH=${repo_dir}/src:${purelib} ${af3_python} -S <AF3 command and arguments>"

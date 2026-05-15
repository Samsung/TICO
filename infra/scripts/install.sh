#!/bin/bash

# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is called by `ccex`
#
# [EXPORTED VARIABLES]
# - CCEX_PROJECT_PATH

###############################################################################
# Helpers & constants
###############################################################################
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SUPPORTED_FAMILIES=("2.5" "2.6" "2.7" "2.8" "2.9" "2.10")
DEFAULT_FAMILY="2.7"

show_help() {
cat <<EOF_HELP
Usage: ./ccex install [OPTIONS]

--dist                 Install from wheel in ./dist instead of editable mode
--torch_ver VER        Torch version or family to install.
                       Accepts:
                         • 2.5 ~ 2.10                 (family, installs latest)
                         • 2.6.3, 2.7.0+cu118 ...   (exact)
                         • nightly
                       Default: ${DEFAULT_FAMILY}
--cuda_ver MAJ.MIN     Override detected CUDA version (e.g. 12.1)
--cpu_only             Force CPU-only Torch installation
                       (disables CUDA detection / --cuda_ver)
-h | --help            Show this help
EOF_HELP
}

version_le() {
  local a="$1" b="$2"
  [[ "$(printf '%s\n%s\n' "$a" "$b" | sort -V | head -n1)" == "$a" ]]
}

add_unique_index_url() {
  local candidate="$1" existing
  for existing in "${INDEX_URLS[@]}"; do
    [[ "$existing" == "$candidate" ]] && return 0
  done
  INDEX_URLS+=("$candidate")
}

###############################################################################
# Option parsing
###############################################################################
_DIST=0
_TORCH_VER="${DEFAULT_FAMILY}"
_USER_CUDA=""
_CPU_ONLY=""

options=$(getopt -o h --long dist,torch_ver:,cuda_ver:,cpu_only,help -- "$@") || {
  echo "[ERROR] Invalid command-line options" >&2; exit 1; }
eval set -- "$options"

while true; do
  case "$1" in
      --dist)        _DIST=1 ;;
      --torch_ver)   _TORCH_VER="$2"; shift ;;
      --cuda_ver)    _USER_CUDA="$2"; shift ;;
      --cpu_only)    _CPU_ONLY=1 ;;
      -h|--help)     show_help; exit 0 ;;
      --)            shift; break ;;
      *)             echo "[ERROR] Unknown option $1"; exit 1 ;;
  esac
  shift
done

###############################################################################
# Detect (and maybe keep) any existing torch installation
###############################################################################
INSTALLED_TORCH_FULL=""
INSTALLED_TORCH_FAMILY=""
read -r INSTALLED_TORCH_FULL < <(
  python3 - <<'PY'
import importlib.util, re, sys
spec = importlib.util.find_spec("torch")
if spec is None:                       # Torch not found → just print blanks
    print()
    sys.exit(0)
import torch
print(torch.__version__)
PY
)

if [[ -n "$INSTALLED_TORCH_FULL" ]]; then
  INSTALLED_TORCH_FAMILY=$(echo "$INSTALLED_TORCH_FULL" | cut -d. -f1,2)
fi

# Normalise requested spec to family / exact
REQUEST_IS_NIGHTLY=""
REQUEST_IS_EXACT=""
if [[ "$_TORCH_VER" == "nightly" ]]; then
  REQUEST_IS_NIGHTLY=1
elif [[ "$_TORCH_VER" =~ ^[0-9]+\.[0-9]+$ ]]; then
  : # family only
elif [[ "$_TORCH_VER" =~ ^[0-9]+\.[0-9]+\.[0-9]+ ]]; then
  REQUEST_IS_EXACT=1
else
  echo "[ERROR] Unsupported --torch_ver value '${_TORCH_VER}'"; exit 1
fi

# Respect pre-installed Torch if allowed
SKIP_TORCH_INSTALL=""
if [[ -n "$INSTALLED_TORCH_FULL" ]]; then
  if [[ " ${SUPPORTED_FAMILIES[*]} " =~ " ${INSTALLED_TORCH_FAMILY} " ]]; then
    if [[ -z "$REQUEST_IS_NIGHTLY" && -z "$REQUEST_IS_EXACT" ]]; then
      echo "[INFO] Supported torch ${INSTALLED_TORCH_FULL} already present — keeping it"
      SKIP_TORCH_INSTALL=1
      _TORCH_VER="$INSTALLED_TORCH_FAMILY"   # for later requirements file pick
    else
      echo "[INFO] '--torch_ver' explicitly requests ${_TORCH_VER}; will override existing ${INSTALLED_TORCH_FULL}"
    fi
  else
    echo "[WARN] Found unsupported torch ${INSTALLED_TORCH_FULL}; will install supported default"
  fi
fi

###############################################################################
# CUDA index-URL logic
###############################################################################
get_index_url_for_cuda_version() {
  local cuda_ver="$1" nightly="$2"
  local maj=${cuda_ver%.*} min=${cuda_ver#*.}
  echo "https://download.pytorch.org/whl${nightly:+/nightly}/cu${maj}${min}"
}

INDEX_URL="https://download.pytorch.org/whl${REQUEST_IS_NIGHTLY:+/nightly}/cpu"
CUDA_TO_USE=""

if [[ -n "$_CPU_ONLY" ]]; then
  echo "[INFO] Forcing CPU-only Torch installation"
else
  if [[ -n "$_USER_CUDA" ]]; then
    CUDA_TO_USE="$_USER_CUDA"
    echo "[INFO] Using CUDA ${CUDA_TO_USE} specified with --cuda_ver"
  elif command -v nvcc &>/dev/null; then
    CUDA_TO_USE=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
    echo "[INFO] Detected CUDA ${CUDA_TO_USE}"
  elif command -v nvidia-smi &>/dev/null; then
    CUDA_TO_USE=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
    echo "[INFO] Detected CUDA ${CUDA_TO_USE}"
  fi

  if [[ -n "$CUDA_TO_USE" ]]; then
    INDEX_URL=$(get_index_url_for_cuda_version "$CUDA_TO_USE" "$REQUEST_IS_NIGHTLY")
  fi
fi

INDEX_URLS=()
add_unique_index_url "$INDEX_URL"

if [[ -z "$_CPU_ONLY" && -n "$CUDA_TO_USE" ]]; then
  # PyTorch does not publish wheels for every CUDA minor version.
  # Example: CUDA 13.1 maps to cu131, but cu131 may not exist.
  # Try the detected/requested CUDA first, then fall back to known PyTorch
  # CUDA wheel indices that are <= the detected/requested CUDA version.
  PYTORCH_CUDA_FALLBACKS=("13.0" "12.8" "12.6" "12.4" "12.1" "11.8")
  for cuda_fb in "${PYTORCH_CUDA_FALLBACKS[@]}"; do
    if version_le "$cuda_fb" "$CUDA_TO_USE"; then
      add_unique_index_url "$(get_index_url_for_cuda_version "$cuda_fb" "$REQUEST_IS_NIGHTLY")"
    fi
  done

  # Last resort: keep install working on machines without a matching CUDA wheel.
  add_unique_index_url "https://download.pytorch.org/whl${REQUEST_IS_NIGHTLY:+/nightly}/cpu"
fi

###############################################################################
# Torch installation (may be skipped)
###############################################################################
install_torch() {
  local spec="$1"
  local index_url

  for index_url in "${INDEX_URLS[@]}"; do
    echo "[INFO] Installing torch (${spec}) from ${index_url}"
    if python3 -m pip install ${spec} --index-url "${index_url}"; then
      INDEX_URL="${index_url}"
      echo "[INFO] Successfully installed torch from ${index_url}"
      return 0
    fi

    echo "[WARN] Failed to install torch (${spec}) from ${index_url}; trying next candidate..." >&2
  done

  echo "[ERROR] Could not install torch (${spec}) from any candidate PyTorch index." >&2
  return 1
}

if [[ -z "$SKIP_TORCH_INSTALL" ]]; then
  if [[ -n "$REQUEST_IS_NIGHTLY" ]]; then
    install_torch "-r ${SCRIPTS_DIR}/../dependency/torch_dev.txt" || exit 1
  else
    if [[ -n "$REQUEST_IS_EXACT" ]]; then
      install_torch "torch==${_TORCH_VER}" || exit 1
    else
      # family only → pip’s ~= spec picks the newest patch in the family
      install_torch "torch==${_TORCH_VER}.*" || exit 1
    fi
  fi
fi

###############################################################################
# Install the auxiliary Python requirements
###############################################################################
REQ_FILE="${SCRIPTS_DIR}/install_requirements.txt"
echo "[INFO] Installing auxiliary requirements from ${REQ_FILE##*/}"
python3 -m pip install -r "$REQ_FILE"

###############################################################################
# TICO itself
###############################################################################
if [[ $_DIST -eq 1 ]]; then
  echo "[INFO] Installing TICO wheel from ./dist"
  python3 -m pip install --force-reinstall --no-deps "${CCEX_PROJECT_PATH}"/dist/tico*.whl
else
  echo "[INFO] Installing TICO in editable mode"
  python3 -m pip install --editable "${CCEX_PROJECT_PATH}"
fi

echo "[SUCCESS] ./ccex install completed"

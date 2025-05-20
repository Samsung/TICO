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

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_help() {
# `cat << EOF` This means that cat should stop reading when EOF is detected
cat << EOF
Usage: ./ccex install [--dist|--torch_ver|--help|-h]
--dist          Install from whl file
                (default: Install from source, editable mode)
--torch_ver     [2.5|2.6|nightly]
                Specify torch version to be installed.
                (default: 2.6)
--cuda_ver      [11.8|12.6|12.8]
                Specify the target CUDA version. This overrides automatic 
                 detection.
--cpu_only      Forces installation of the CPU-only version of PyTorch.
                Disables both CUDA version detection and use of --cuda-version.
-h | --help     Show help message and exit
EOF
}

options=$(getopt -o h --long dist,torch_ver:,cuda_ver:,cpu_only,help -- "$@")

[ $? -eq 0 ] || {
    echo "Incorrect options provided"
    exit 1
}

_DIST=0
_TORCH_VER="2.6"
_USER_SPECIFIED_CUDA=""
_CPU_ONLY=false
_NIGHTLY=""

eval set -- "$options"

while true; do
    case "$1" in
        --dist)
            _DIST=1
            ;;
        --torch_ver)
            _TORCH_VER="$2"
            shift
            ;;
        --cuda_ver)
            _USER_SPECIFIED_CUDA="$2"
            shift
            ;;
        --cpu_only)
            _CPU_ONLY=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0;
            ;;
        --)
            shift
            break;;
        *)  break;;
    esac
    shift
done

# Check torch version
if [ "$_TORCH_VER" != "2.5" ] && [ "$_TORCH_VER" != "2.6" ] && [ "$_TORCH_VER" != "nightly" ]; then
    echo "Invalid torch version '$_TORCH_VER'"
    echo "(Use --help to see available torch versions)"
    exit 1
fi
# Check if torch version is nightly
if [[ "$_TORCH_VER" == "nightly" ]]; then
  _NIGHTLY=true
fi
# Check for conflicting options
if [[ -n "$_USER_SPECIFIED_CUDA" && "$_CPU_ONLY" = true ]]; then
  echo "[ERROR] Cannot use --cpu_only and --cuda_ver together"
  exit 1
fi

# set index-url
get_index_url_for_cuda_version() {
  local version="$1"
  local is_nightly="$2"

  local major=${version%.*}
  local minor=${version#*.}
  local cuda_suffix="cu${major}${minor}"

  echo "https://download.pytorch.org/whl${is_nightly:+/nightly}/${cuda_suffix}"
}
INDEX_URL="https://download.pytorch.org/whl${_NIGHTLY:+/nightly}/cpu"
if [[ "$_CPU_ONLY" = true ]]; then
  echo "[INFO] Installing CPU-only version of PyTorch"
else
  if [[ -n "$_USER_SPECIFIED_CUDA" ]]; then
    echo "[INFO] Using user-specified CUDA version: $_USER_SPECIFIED_CUDA"
    INDEX_URL=$(get_index_url_for_cuda_version "$_USER_SPECIFIED_CUDA" "$_NIGHTLY")
  else
    if command -v nvcc &> /dev/null; then
      DETECTED_CUDA=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
    elif command -v nvidia-smi &> /dev/null; then
      DETECTED_CUDA=$(nvcc --version | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
    else
      DETECTED_CUDA=""
    fi
  fi
  if [[ -n "$DETECTED_CUDA" ]]; then
    echo "[INFO] Detected CUDA version: $DETECTED_CUDA"
    INDEX_URL=$(get_index_url_for_cuda_version "$DETECTED_CUDA" "$_NIGHTLY")
  fi
fi

SCRIPTS_DIR="${CCEX_PROJECT_PATH}/infra/scripts"

if [ "$_TORCH_VER" == "nightly" ]; then
  echo "Install package dependencies from torch nightly version"
  REQ_FILE="${SCRIPTS_DIR}/../dependency/torch_dev.txt"
  python3 -m pip install -r ${REQ_FILE} --index-url ${INDEX_URL}
  python3 -m pip install -r "${SCRIPTS_DIR}/install_requirements_dev.txt"
elif [ "$_TORCH_VER" == "2.6" ]; then
  echo "Install package dependencies from torch stable version"
  python3 -m pip install torch==2.6.0 --index-url ${INDEX_URL}
  python3 -m pip install -r "${SCRIPTS_DIR}/install_requirements_2_6.txt"
elif [ "$_TORCH_VER" == "2.5" ]; then
  echo "Install package dependencies from torch stable version"
  python3 -m pip install torch==2.5.0 --index-url ${INDEX_URL}
  python3 -m pip install -r "${SCRIPTS_DIR}/install_requirements_2_5.txt"
else
  echo "Assertion: Cannot reach here"
  exit 1
fi

if [ $_DIST -eq 1 ]; then
  echo "Install from whl file"
  python3 -m pip install --force-reinstall --no-deps dist/tico*.whl
else
  echo "Install as editable mode"
  pip install -e .
fi

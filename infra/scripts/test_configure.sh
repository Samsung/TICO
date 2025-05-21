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
SCRIPTS_DIR="${CCEX_PROJECT_PATH}/infra/scripts"
TEST_DIR="${CCEX_PROJECT_PATH}/test"

SUPPORTED_FAMILIES=("2.5" "2.6" "2.7")
DEFAULT_FAMILY="2.6"

show_help() {
cat <<EOF
Usage: ./ccex configure test [OPTIONS]

--torch_ver VER       Torch version or family to install.
                      Accepts:
                        • 2.5  | 2.6  | 2.7      (family, installs latest)
                        • 2.6.3, 2.7.0+cu118 ... (exact)
                        • nightly
                      Default: ${DEFAULT_FAMILY}
--cuda_ver MAJ.MIN    Override detected CUDA version (e.g. 12.1)
--cpu_only            Force CPU-only Torch installation 
                      (disables CUDA detection / --cuda_ver)
-h, --help            Show this help
EOF
}

###############################################################################
# Option parsing
###############################################################################
_TORCH_VER="${DEFAULT_FAMILY}"
_USER_CUDA=""
_CPU_ONLY=""

options=$(getopt -o h --long torch_ver:,cuda_ver:,cpu_only,help -- "$@") || {
  echo "[ERROR] Invalid command-line options" >&2; exit 1; }
eval set -- "$options"

while true; do
  case "$1" in
      --torch_ver) _TORCH_VER="$2"; shift 2;;
      --cuda_ver)  _USER_CUDA="$2"; shift 2;;
      --cpu_only)  _CPU_ONLY=1; shift;;
      -h|--help)   show_help; exit 0;;
      --) shift; break;;
      *)  echo "[ERROR] Unknown option $1"; exit 1;;
  esac
done

###############################################################################
# Torch version analysis
###############################################################################
REQUEST_IS_NIGHTLY=""
REQUEST_IS_EXACT=""
if [[ "${_TORCH_VER}" == "nightly" ]]; then
  REQUEST_IS_NIGHTLY=1
fi
FAMILY="$(echo "${_TORCH_VER}" | grep -oE '^[0-9]+\.[0-9]+' || echo "${DEFAULT_FAMILY}")"
if [[ -z "${REQUEST_IS_NIGHTLY}" && " ${SUPPORTED_FAMILIES[*]} " != *" ${FAMILY} "* ]]; then
  echo "[ERROR] Unsupported --torch_ver ${_TORCH_VER}"; exit 1
fi

###############################################################################
# Index-url  (same rules as install.sh)
###############################################################################
get_index_url() {
  local cuda="$1" nightly="$2"
  local maj=${cuda%.*} min=${cuda#*.}
  echo "https://download.pytorch.org/whl${nightly:+/nightly}/cu${maj}${min}"
}

INDEX_URL="https://download.pytorch.org/whl${REQUEST_IS_NIGHTLY:+/nightly}/cpu"
if [[ -z "${_CPU_ONLY}" ]]; then
  if [[ -n "${_USER_CUDA}" ]]; then
    CUDA="${_USER_CUDA}"
    echo "[INFO] Using user-specified CUDA ${CUDA}"
  else
    if command -v nvcc &>/dev/null;   then CUDA="$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')" ; fi
    if [[ -z "${CUDA:-}" ]] && command -v nvidia-smi &>/dev/null; then
      CUDA="$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')"
    fi
  fi
  if [[ -n "${CUDA:-}" ]]; then
    echo "[INFO] Detected CUDA ${CUDA}"
    INDEX_URL="$(get_index_url "${CUDA}" "${REQUEST_IS_NIGHTLY}")"
  fi
else
  echo "[INFO] Forcing CPU-only test deps"
fi

###############################################################################
# pip helper  (handles spec + arbitrary flags)
###############################################################################
pip_install() {
  local spec="$1"
  python3 -m pip install ${spec} --index-url "${INDEX_URL}"
}

###############################################################################
# torchvision version mapping per torch family
###############################################################################
declare -A TORCHVISION_FAMILY=(
  ["2.5"]="0.20.*"
  ["2.6"]="0.21.*"
  ["2.7"]="0.22.*"
)

###############################################################################
# 1) Install torchvision
###############################################################################
if [[ -n "${REQUEST_IS_NIGHTLY}" ]]; then
  echo "[INFO] Installing torchvision (nightly) from ${INDEX_URL}"
  pip_install "-r ${SCRIPTS_DIR}/../dependency/torchvision_dev.txt"
else
  VISION_VER="${TORCHVISION_FAMILY[${FAMILY}]}"
  echo "[INFO] Installing torchvision==${VISION_VER} from ${INDEX_URL}"
  pip_install "torchvision==${VISION_VER}"
fi


###############################################################################
# 2) Install additional test-only requirements
###############################################################################
if [[ -n "${REQUEST_IS_NIGHTLY}" ]]; then
  EXTRA_REQ_FILES=(
    "${TEST_DIR}/requirements_dev.txt"
  )
else
  DEP_FILE="${SCRIPTS_DIR}/../dependency/torchvision_${FAMILY/./_}.txt"
  TEST_FILE="${TEST_DIR}/requirements_${FAMILY/./_}.txt"
  EXTRA_REQ_FILES=("${DEP_FILE}" "${TEST_FILE}")
fi

for req in "${EXTRA_REQ_FILES[@]}"; do
  if [[ -f "${req}" ]]; then
    echo "[INFO] Installing auxiliary test deps from ${req##*/}"
    pip_install "-r ${req}"
  fi
done

echo "[SUCCESS] ./ccex configure test completed"

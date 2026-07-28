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
source "${SCRIPTS_DIR}/pytorch_package_utils.sh" || exit 1

show_help() {
cat <<EOF_HELP
Usage: ./ccex install [OPTIONS]

--dist                 Install from wheel in ./dist instead of editable mode
--torch_ver VER        Torch version or family to install.
                       Accepts:
                         • 2.5 ~ 2.10                 (family, installs latest supported patch)
                         • 2.6.0, 2.7.1+cu126 ...   (exact)
                         • nightly
                       Default: ${PYTORCH_DEFAULT_FAMILY}
--cuda_ver MAJ.MIN     Override detected host CUDA capability (e.g. 12.1)
--cpu_only             Force CPU-only Torch installation
                       (disables CUDA detection / --cuda_ver)
-h | --help            Show this help
EOF_HELP
}

install_torch() {
  local index_url
  local -a pip_args=("$@")

  for index_url in "${PYTORCH_INDEX_URLS[@]}"; do
    echo "[INFO] Installing ${pip_args[*]} from ${index_url}"
    if python3 -m pip install "${pip_args[@]}" --index-url "${index_url}"; then
      echo "[INFO] Successfully installed torch from ${index_url}"
      return 0
    fi

    echo "[WARN] Failed to install ${pip_args[*]} from ${index_url}; trying next candidate..." >&2
  done

  echo "[ERROR] Could not install torch from any candidate PyTorch index." >&2
  return 1
}

installed_compute_matches_request() {
  local installed_wheel_tag="$1"
  local installed_cuda=""

  if [[ "${_CPU_ONLY}" == "1" ]]; then
    [[ "${installed_wheel_tag}" == "cpu" ]]
    return
  fi

  if [[ -n "${_USER_CUDA}" ]]; then
    if [[ "${installed_wheel_tag}" == "cpu" ]]; then
      return 1
    fi

    installed_cuda="$(pytorch_cuda_tag_to_version "${installed_wheel_tag}")" || return 1
    pytorch_version_le "${installed_cuda}" "${_USER_CUDA}"
    return
  fi

  return 0
}

###############################################################################
# Option parsing
###############################################################################
_DIST=0
_TORCH_VER="${PYTORCH_DEFAULT_FAMILY}"
_TORCH_VER_WAS_SET=0
_USER_CUDA=""
_CPU_ONLY=0

options=$(getopt -o h --long dist,torch_ver:,cuda_ver:,cpu_only,help -- "$@") || {
  echo "[ERROR] Invalid command-line options" >&2; exit 1; }
eval set -- "${options}"

while true; do
  case "$1" in
      --dist)        _DIST=1 ;;
      --torch_ver)   _TORCH_VER="$2"; _TORCH_VER_WAS_SET=1; shift ;;
      --cuda_ver)    _USER_CUDA="$2"; shift ;;
      --cpu_only)    _CPU_ONLY=1 ;;
      -h|--help)     show_help; exit 0 ;;
      --)            shift; break ;;
      *)             echo "[ERROR] Unknown option $1"; exit 1 ;;
  esac
  shift
done

if [[ "${_CPU_ONLY}" == "1" && -n "${_USER_CUDA}" ]]; then
  echo "[ERROR] --cpu_only and --cuda_ver cannot be used together." >&2
  exit 1
fi

if [[ -n "${_USER_CUDA}" && ! "${_USER_CUDA}" =~ ^[0-9]+\.[0-9]+$ ]]; then
  echo "[ERROR] Invalid --cuda_ver value '${_USER_CUDA}'. Expected MAJ.MIN." >&2
  exit 1
fi

###############################################################################
# Normalize the requested Torch version
###############################################################################
REQUEST_IS_NIGHTLY=0
REQUEST_IS_EXACT=0
REQUESTED_FAMILY=""
REQUESTED_EXACT_VERSION=""
REQUESTED_BUILD_TAG=""
RESOLVED_TORCH_VERSION=""

if [[ "${_TORCH_VER}" == "nightly" ]]; then
  REQUEST_IS_NIGHTLY=1
elif [[ "${_TORCH_VER}" =~ ^[0-9]+\.[0-9]+$ ]]; then
  REQUESTED_FAMILY="${_TORCH_VER}"
  if ! pytorch_is_supported_family "${REQUESTED_FAMILY}"; then
    echo "[ERROR] Unsupported --torch_ver family '${_TORCH_VER}'" >&2
    exit 1
  fi
  RESOLVED_TORCH_VERSION="$(pytorch_resolve_latest_stable_version "${REQUESTED_FAMILY}")" || exit 1
elif [[ "${_TORCH_VER}" =~ ^[0-9]+\.[0-9]+\.[0-9]+(\+[A-Za-z0-9._-]+)?$ ]]; then
  REQUEST_IS_EXACT=1
  REQUESTED_EXACT_VERSION="${_TORCH_VER}"
  REQUESTED_FAMILY="$(pytorch_version_family "${_TORCH_VER}")" || exit 1
  REQUESTED_BUILD_TAG="$(pytorch_local_build_tag "${_TORCH_VER}")"
  if ! pytorch_is_supported_family "${REQUESTED_FAMILY}"; then
    echo "[ERROR] Unsupported --torch_ver value '${_TORCH_VER}'" >&2
    exit 1
  fi
  if [[ -n "${REQUESTED_BUILD_TAG}" ]] && ! pytorch_is_supported_wheel_tag "${REQUESTED_BUILD_TAG}"; then
    echo "[ERROR] Unsupported Torch build tag '+${REQUESTED_BUILD_TAG}'" >&2
    exit 1
  fi
  RESOLVED_TORCH_VERSION="${REQUESTED_EXACT_VERSION}"
else
  echo "[ERROR] Unsupported --torch_ver value '${_TORCH_VER}'" >&2
  exit 1
fi

###############################################################################
# Detect and possibly keep an existing Torch installation
###############################################################################
INSTALLED_TORCH_FULL=""
INSTALLED_TORCH_BASE=""
INSTALLED_TORCH_FAMILY=""
INSTALLED_TORCH_WHEEL_TAG=""
INSTALLED_TORCH_IS_NIGHTLY="0"

if TORCH_INFO="$(pytorch_get_installed_torch_info 2>/dev/null)"; then
  IFS=$'\t' read -r \
    INSTALLED_TORCH_FULL \
    INSTALLED_TORCH_BASE \
    INSTALLED_TORCH_FAMILY \
    INSTALLED_TORCH_WHEEL_TAG \
    INSTALLED_TORCH_IS_NIGHTLY <<< "${TORCH_INFO}"
fi

SKIP_TORCH_INSTALL=0
if [[ -n "${INSTALLED_TORCH_FULL}" ]]; then
  # With no explicit Torch version, preserve any supported stable family. If a
  # compute-platform option requires reinstalling the wheel, reinstall the same
  # family at its latest supported patch rather than falling back to the default.
  if [[ "${_TORCH_VER_WAS_SET}" == "0" && \
        "${INSTALLED_TORCH_IS_NIGHTLY}" == "0" ]] && \
     pytorch_is_supported_family "${INSTALLED_TORCH_FAMILY}"; then
    REQUESTED_FAMILY="${INSTALLED_TORCH_FAMILY}"
    RESOLVED_TORCH_VERSION="$(pytorch_resolve_latest_stable_version \
      "${REQUESTED_FAMILY}")" || exit 1
  fi

  if installed_compute_matches_request "${INSTALLED_TORCH_WHEEL_TAG}"; then
    if [[ "${_TORCH_VER_WAS_SET}" == "0" ]]; then
      if [[ "${INSTALLED_TORCH_IS_NIGHTLY}" == "0" ]] && \
         pytorch_is_supported_family "${INSTALLED_TORCH_FAMILY}"; then
        echo "[INFO] Supported torch ${INSTALLED_TORCH_FULL} already present — keeping it"
        SKIP_TORCH_INSTALL=1
        RESOLVED_TORCH_VERSION="${INSTALLED_TORCH_BASE}"
      fi
    elif [[ "${REQUEST_IS_NIGHTLY}" == "1" && "${INSTALLED_TORCH_IS_NIGHTLY}" == "1" ]]; then
      echo "[INFO] Requested nightly torch ${INSTALLED_TORCH_FULL} already present — keeping it"
      SKIP_TORCH_INSTALL=1
    elif [[ "${REQUEST_IS_EXACT}" == "1" ]]; then
      REQUESTED_BASE="$(pytorch_strip_local_version "${REQUESTED_EXACT_VERSION}")"
      if [[ "${INSTALLED_TORCH_BASE}" == "${REQUESTED_BASE}" ]]; then
        if [[ -z "${REQUESTED_BUILD_TAG}" || \
              "${INSTALLED_TORCH_WHEEL_TAG}" == "${REQUESTED_BUILD_TAG}" ]]; then
          echo "[INFO] Requested torch ${INSTALLED_TORCH_FULL} already present — keeping it"
          SKIP_TORCH_INSTALL=1
        fi
      fi
    elif [[ "${INSTALLED_TORCH_IS_NIGHTLY}" == "0" && \
            "${INSTALLED_TORCH_BASE}" == "${RESOLVED_TORCH_VERSION}" ]]; then
      echo "[INFO] Requested torch ${INSTALLED_TORCH_FULL} already present — keeping it"
      SKIP_TORCH_INSTALL=1
    fi
  fi

  if [[ "${SKIP_TORCH_INSTALL}" == "0" ]]; then
    echo "[INFO] Requested torch '${_TORCH_VER}' will replace existing ${INSTALLED_TORCH_FULL}"
  fi
fi

###############################################################################
# Resolve candidate wheel indices and install Torch
###############################################################################
if [[ "${SKIP_TORCH_INSTALL}" == "0" ]]; then
  HOST_CUDA=""
  PINNED_BUILD_TAG=""

  if [[ "${REQUEST_IS_NIGHTLY}" == "1" ]]; then
    TORCH_DEV_FILE="${SCRIPTS_DIR}/../dependency/torch_dev.txt"
    PINNED_TORCH_VERSION="$(pytorch_get_pinned_requirement_version "${TORCH_DEV_FILE}" torch)"
    if [[ -z "${PINNED_TORCH_VERSION}" ]]; then
      echo "[ERROR] ${TORCH_DEV_FILE} must pin torch with 'torch==VERSION'." >&2
      exit 1
    fi
    PINNED_BUILD_TAG="$(pytorch_local_build_tag "${PINNED_TORCH_VERSION}")"
  fi

  if [[ "${_CPU_ONLY}" == "1" ]]; then
    if [[ -n "${REQUESTED_BUILD_TAG}" && "${REQUESTED_BUILD_TAG}" != "cpu" ]]; then
      echo "[ERROR] --cpu_only conflicts with torch build '+${REQUESTED_BUILD_TAG}'." >&2
      exit 1
    fi
    if [[ -n "${PINNED_BUILD_TAG}" && "${PINNED_BUILD_TAG}" != "cpu" ]]; then
      echo "[ERROR] --cpu_only conflicts with nightly torch build '+${PINNED_BUILD_TAG}'." >&2
      exit 1
    fi
    REQUESTED_BUILD_TAG="cpu"
    echo "[INFO] Forcing CPU-only Torch installation"
  elif [[ -n "${REQUESTED_BUILD_TAG}" ]]; then
    echo "[INFO] Using explicitly requested Torch wheel ${REQUESTED_BUILD_TAG}"
  elif [[ -n "${PINNED_BUILD_TAG}" ]]; then
    REQUESTED_BUILD_TAG="${PINNED_BUILD_TAG}"
    echo "[INFO] Using Torch wheel ${REQUESTED_BUILD_TAG} pinned by torch_dev.txt"
  elif [[ -n "${_USER_CUDA}" ]]; then
    HOST_CUDA="${_USER_CUDA}"
    echo "[INFO] Using CUDA ${HOST_CUDA} specified with --cuda_ver"
  elif HOST_CUDA="$(pytorch_detect_host_cuda_version)"; then
    echo "[INFO] Detected CUDA ${HOST_CUDA}"
  else
    HOST_CUDA=""
    echo "[INFO] CUDA was not detected; using a CPU Torch wheel"
  fi

  if [[ "${REQUEST_IS_NIGHTLY}" == "1" ]]; then
    pytorch_build_index_urls "" "${HOST_CUDA}" 1 "${REQUESTED_BUILD_TAG}" || exit 1
    install_torch -r "${TORCH_DEV_FILE}" || exit 1
  else
    pytorch_build_index_urls \
      "${REQUESTED_FAMILY}" "${HOST_CUDA}" 0 "${REQUESTED_BUILD_TAG}" || exit 1
    echo "[INFO] Resolved torch ${_TORCH_VER} to ${RESOLVED_TORCH_VERSION}"
    install_torch "torch==${RESOLVED_TORCH_VERSION}" || exit 1
  fi
fi

###############################################################################
# Verify the installed Torch package
###############################################################################
TORCH_INFO="$(pytorch_get_installed_torch_info)" || {
  echo "[ERROR] Torch is not importable after installation." >&2
  exit 1
}
IFS=$'\t' read -r \
  INSTALLED_TORCH_FULL \
  INSTALLED_TORCH_BASE \
  INSTALLED_TORCH_FAMILY \
  INSTALLED_TORCH_WHEEL_TAG \
  INSTALLED_TORCH_IS_NIGHTLY <<< "${TORCH_INFO}"

echo "[INFO] Installed torch: ${INSTALLED_TORCH_FULL}"
echo "[INFO] Torch wheel build: ${INSTALLED_TORCH_WHEEL_TAG}"

###############################################################################
# Install the auxiliary Python requirements
###############################################################################
REQ_FILE="${SCRIPTS_DIR}/install_requirements.txt"
echo "[INFO] Installing auxiliary requirements from ${REQ_FILE##*/}"
python3 -m pip install -r "${REQ_FILE}" || exit 1

###############################################################################
# TICO itself
###############################################################################
if [[ "${_DIST}" -eq 1 ]]; then
  echo "[INFO] Installing TICO wheel from ./dist"
  python3 -m pip install --force-reinstall --no-deps \
    "${CCEX_PROJECT_PATH}"/dist/tico*.whl || exit 1
else
  echo "[INFO] Installing TICO in editable mode"
  python3 -m pip install --editable "${CCEX_PROJECT_PATH}" || exit 1
fi

# TorchVision is installed by `./ccex configure test`, so a global `pip check`
# belongs at the end of that command rather than between the two setup stages.
echo "[SUCCESS] ./ccex install completed"

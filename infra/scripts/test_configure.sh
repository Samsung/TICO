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
source "${SCRIPTS_DIR}/pytorch_package_utils.sh" || exit 1

show_help() {
cat <<EOF_HELP
Usage: ./ccex configure test [OPTIONS]

--torch_ver VER       Validate the installed Torch version before installing
                      its matching TorchVision package. Accepts:
                        • 2.5 ~ 2.10
                        • 2.6.0, 2.7.1+cu126 ...
                        • nightly
                      If omitted, the installed Torch version is used.
--cuda_ver MAJ.MIN    Validate that this CUDA capability can run the installed
                      Torch wheel. TorchVision still follows the Torch wheel.
--cpu_only            Require an installed CPU-only Torch package
-h, --help            Show this help
EOF_HELP
}

###############################################################################
# Option parsing
###############################################################################
_TORCH_VER=""
_TORCH_VER_WAS_SET=0
_USER_CUDA=""
_CPU_ONLY=0

options=$(getopt -o h --long torch_ver:,cuda_ver:,cpu_only,help -- "$@") || {
  echo "[ERROR] Invalid command-line options" >&2; exit 1; }
eval set -- "${options}"

while true; do
  case "$1" in
      --torch_ver) _TORCH_VER="$2"; _TORCH_VER_WAS_SET=1; shift ;;
      --cuda_ver)  _USER_CUDA="$2"; shift ;;
      --cpu_only)  _CPU_ONLY=1 ;;
      -h|--help)   show_help; exit 0 ;;
      --)          shift; break ;;
      *)           echo "[ERROR] Unknown option $1"; exit 1 ;;
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
# Inspect the installed Torch package
###############################################################################
TORCH_INFO="$(pytorch_get_installed_torch_info)" || {
  echo "[ERROR] Torch is not installed or cannot be imported." >&2
  echo "[ERROR] Run './ccex install' before './ccex configure test'." >&2
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

if [[ "${INSTALLED_TORCH_WHEEL_TAG}" != "cpu" && \
      ! "${INSTALLED_TORCH_WHEEL_TAG}" =~ ^cu[0-9]+$ ]]; then
  echo "[ERROR] Unsupported installed Torch wheel '${INSTALLED_TORCH_WHEEL_TAG}'." >&2
  echo "[ERROR] ccex currently supports CPU and CUDA PyTorch wheels." >&2
  exit 1
fi

if [[ "${INSTALLED_TORCH_IS_NIGHTLY}" == "0" ]] && \
   ! pytorch_is_supported_family "${INSTALLED_TORCH_FAMILY}"; then
  echo "[ERROR] Installed torch ${INSTALLED_TORCH_FULL} is not supported." >&2
  exit 1
fi

###############################################################################
# Validate optional version and compute-platform requests
###############################################################################
if [[ "${_TORCH_VER_WAS_SET}" == "1" ]]; then
  if [[ "${_TORCH_VER}" == "nightly" ]]; then
    if [[ "${INSTALLED_TORCH_IS_NIGHTLY}" != "1" ]]; then
      echo "[ERROR] --torch_ver nightly was requested, but ${INSTALLED_TORCH_FULL} is installed." >&2
      exit 1
    fi
  elif [[ "${_TORCH_VER}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    if ! pytorch_is_supported_family "${_TORCH_VER}"; then
      echo "[ERROR] Unsupported --torch_ver family '${_TORCH_VER}'" >&2
      exit 1
    fi
    if [[ "${INSTALLED_TORCH_IS_NIGHTLY}" == "1" || \
          "${INSTALLED_TORCH_FAMILY}" != "${_TORCH_VER}" ]]; then
      echo "[ERROR] Requested torch family ${_TORCH_VER}, but ${INSTALLED_TORCH_FULL} is installed." >&2
      echo "[ERROR] Re-run './ccex install --torch_ver ${_TORCH_VER}' first." >&2
      exit 1
    fi
  elif [[ "${_TORCH_VER}" =~ ^[0-9]+\.[0-9]+\.[0-9]+(\+[A-Za-z0-9._-]+)?$ ]]; then
    REQUESTED_BASE="$(pytorch_strip_local_version "${_TORCH_VER}")"
    REQUESTED_BUILD_TAG="$(pytorch_local_build_tag "${_TORCH_VER}")"
    REQUESTED_FAMILY="$(pytorch_version_family "${_TORCH_VER}")" || exit 1

    if ! pytorch_is_supported_family "${REQUESTED_FAMILY}"; then
      echo "[ERROR] Unsupported --torch_ver value '${_TORCH_VER}'" >&2
      exit 1
    fi
    if [[ "${INSTALLED_TORCH_BASE}" != "${REQUESTED_BASE}" ]]; then
      echo "[ERROR] Requested torch ${_TORCH_VER}, but ${INSTALLED_TORCH_FULL} is installed." >&2
      exit 1
    fi
    if [[ -n "${REQUESTED_BUILD_TAG}" && \
          "${INSTALLED_TORCH_WHEEL_TAG}" != "${REQUESTED_BUILD_TAG}" ]]; then
      echo "[ERROR] Requested Torch wheel ${REQUESTED_BUILD_TAG}, but ${INSTALLED_TORCH_WHEEL_TAG} is installed." >&2
      exit 1
    fi
  else
    echo "[ERROR] Unsupported --torch_ver value '${_TORCH_VER}'" >&2
    exit 1
  fi
fi

if [[ "${_CPU_ONLY}" == "1" && "${INSTALLED_TORCH_WHEEL_TAG}" != "cpu" ]]; then
  echo "[ERROR] --cpu_only was requested, but ${INSTALLED_TORCH_WHEEL_TAG} Torch is installed." >&2
  echo "[ERROR] Re-run './ccex install --cpu_only' first." >&2
  exit 1
fi

if [[ -n "${_USER_CUDA}" ]]; then
  if [[ "${INSTALLED_TORCH_WHEEL_TAG}" == "cpu" ]]; then
    echo "[INFO] Installed Torch is CPU-only; --cuda_ver does not change TorchVision selection."
  else
    INSTALLED_WHEEL_CUDA="$(pytorch_cuda_tag_to_version "${INSTALLED_TORCH_WHEEL_TAG}")" || exit 1
    if ! pytorch_version_le "${INSTALLED_WHEEL_CUDA}" "${_USER_CUDA}"; then
      echo "[ERROR] CUDA ${_USER_CUDA} cannot run the installed ${INSTALLED_TORCH_WHEEL_TAG} Torch wheel." >&2
      exit 1
    fi
    echo "[INFO] CUDA ${_USER_CUDA} is compatible with ${INSTALLED_TORCH_WHEEL_TAG}."
  fi
fi

###############################################################################
# Install the TorchVision build matching the installed Torch package
###############################################################################
INDEX_URL="$(pytorch_index_url \
  "${INSTALLED_TORCH_WHEEL_TAG}" "${INSTALLED_TORCH_IS_NIGHTLY}")"
EXPECTED_VISION_VERSION=""

if [[ "${INSTALLED_TORCH_IS_NIGHTLY}" == "1" ]]; then
  TORCHVISION_DEV_FILE="${SCRIPTS_DIR}/../dependency/torchvision_dev.txt"
  EXPECTED_VISION_FULL="$(pytorch_get_pinned_requirement_version \
    "${TORCHVISION_DEV_FILE}" torchvision)"
  if [[ -z "${EXPECTED_VISION_FULL}" ]]; then
    echo "[ERROR] ${TORCHVISION_DEV_FILE} must pin torchvision with 'torchvision==VERSION'." >&2
    exit 1
  fi

  PINNED_VISION_BUILD_TAG="$(pytorch_local_build_tag "${EXPECTED_VISION_FULL}")"
  if [[ -n "${PINNED_VISION_BUILD_TAG}" && \
        "${PINNED_VISION_BUILD_TAG}" != "${INSTALLED_TORCH_WHEEL_TAG}" ]]; then
    echo "[ERROR] torchvision_dev.txt pins +${PINNED_VISION_BUILD_TAG}, but Torch uses ${INSTALLED_TORCH_WHEEL_TAG}." >&2
    exit 1
  fi

  EXPECTED_VISION_VERSION="$(pytorch_strip_local_version "${EXPECTED_VISION_FULL}")"
  echo "[INFO] Installing torchvision ${EXPECTED_VISION_FULL} from ${INDEX_URL}"
  python3 -m pip install -r "${TORCHVISION_DEV_FILE}" \
    --index-url "${INDEX_URL}" || exit 1
else
  EXPECTED_VISION_VERSION="$(pytorch_resolve_torchvision_version \
    "${INSTALLED_TORCH_BASE}")" || {
      echo "[ERROR] No TorchVision mapping for torch ${INSTALLED_TORCH_BASE}." >&2
      exit 1
    }

  echo "[INFO] Installing torchvision==${EXPECTED_VISION_VERSION} from ${INDEX_URL}"
  python3 -m pip install "torchvision==${EXPECTED_VISION_VERSION}" \
    --index-url "${INDEX_URL}" || exit 1
fi

###############################################################################
# Install additional test-only requirements
###############################################################################
EXTRA_REQ_FILE="${TEST_DIR}/requirements.txt"
EXTRA_REQ_PRE_FILE="${TEST_DIR}/requirements_pre.txt"
python3 -m pip install -r "${EXTRA_REQ_FILE}" || exit 1
python3 -m pip install -r "${EXTRA_REQ_PRE_FILE}" --pre || exit 1

###############################################################################
# Validate the final Torch and TorchVision pair
###############################################################################
EXPECTED_TORCH_BASE="${INSTALLED_TORCH_BASE}" \
EXPECTED_VISION_BASE="${EXPECTED_VISION_VERSION}" \
EXPECTED_WHEEL_TAG="${INSTALLED_TORCH_WHEEL_TAG}" \
python3 - <<'PY'
import os
import sys

try:
    import torch
    import torchvision
except Exception as exc:
    print(f"[ERROR] Failed to import torch/torchvision: {exc}", file=sys.stderr)
    sys.exit(1)

torch_full = str(torch.__version__)
vision_full = str(torchvision.__version__)
torch_base = torch_full.split("+", 1)[0]
vision_base = vision_full.split("+", 1)[0]
expected_torch = os.environ["EXPECTED_TORCH_BASE"]
expected_vision = os.environ["EXPECTED_VISION_BASE"]
expected_tag = os.environ["EXPECTED_WHEEL_TAG"]

if torch_base != expected_torch:
    print(
        f"[ERROR] Torch changed during configure: expected {expected_torch}, got {torch.__version__}",
        file=sys.stderr,
    )
    sys.exit(1)

if vision_base != expected_vision:
    print(
        f"[ERROR] Unexpected TorchVision: expected {expected_vision}, got {vision_full}",
        file=sys.stderr,
    )
    sys.exit(1)

vision_tag = vision_full.split("+", 1)[1] if "+" in vision_full else ""
# x86_64 PyTorch wheel indices encode +cpu/+cuXXX in TorchVision's version,
# while some other architectures publish a tagless but index-specific wheel.
if vision_tag and vision_tag != expected_tag:
    print(
        f"[ERROR] TorchVision wheel mismatch: expected {expected_tag}, got {vision_full}",
        file=sys.stderr,
    )
    sys.exit(1)

actual_tag = "cpu" if torch.version.cuda is None else "cu" + torch.version.cuda.replace(".", "")
if actual_tag != expected_tag:
    print(
        f"[ERROR] Torch wheel changed during configure: expected {expected_tag}, got {actual_tag}",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"[INFO] Verified torch {torch_full} with torchvision {vision_full}")
PY

python3 -m pip check || {
  echo "[ERROR] Installed Python packages have incompatible dependencies." >&2
  exit 1
}

echo "[SUCCESS] ./ccex configure test completed"

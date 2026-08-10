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

# Shared PyTorch package metadata and helper functions for ccex scripts.

if [[ -n "${TICO_PYTORCH_PACKAGE_UTILS_SOURCED:-}" ]]; then
  return 0
fi
TICO_PYTORCH_PACKAGE_UTILS_SOURCED=1

PYTORCH_PACKAGE_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTORCH_POLICY_FILE="${PYTORCH_PACKAGE_UTILS_DIR}/../../tico/utils/compat/torch_version_policy.py"

if ! PYTORCH_POLICY_SHELL="$(python3 "${PYTORCH_POLICY_FILE}" shell)"; then
  echo "[ERROR] Failed to load PyTorch policy from ${PYTORCH_POLICY_FILE}" >&2
  return 1
fi
eval "${PYTORCH_POLICY_SHELL}"
unset PYTORCH_POLICY_SHELL

pytorch_is_supported_family() {
  local family="$1"
  local supported

  for supported in "${PYTORCH_SUPPORTED_FAMILIES[@]}"; do
    [[ "${supported}" == "${family}" ]] && return 0
  done
  return 1
}

pytorch_is_installable_family() {
  local family="$1"
  local installable

  for installable in "${PYTORCH_INSTALLABLE_FAMILIES[@]}"; do
    [[ "${installable}" == "${family}" ]] && return 0
  done
  return 1
}

pytorch_is_nightly_selector() {
  local selector="$1"
  local nightly_selector

  for nightly_selector in "${PYTORCH_NIGHTLY_SELECTORS[@]}"; do
    [[ "${nightly_selector}" == "${selector}" ]] && return 0
  done
  return 1
}

pytorch_version_le() {
  local lhs="$1"
  local rhs="$2"

  [[ "$(printf '%s\n%s\n' "${lhs}" "${rhs}" | sort -V | head -n1)" == "${lhs}" ]]
}

pytorch_strip_local_version() {
  local version="$1"
  echo "${version%%+*}"
}

pytorch_version_family() {
  local version
  version="$(pytorch_strip_local_version "$1")"

  if [[ "${version}" =~ ^([0-9]+\.[0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi

  return 1
}

pytorch_is_nightly_version() {
  local version
  version="$(pytorch_strip_local_version "$1")"
  [[ "${version}" == *".dev"* ]]
}

pytorch_local_build_tag() {
  local version="$1"

  if [[ "${version}" == *+* ]]; then
    echo "${version#*+}"
  fi
}

pytorch_resolve_latest_stable_version() {
  local family="$1"

  if ! pytorch_is_installable_family "${family}"; then
    return 1
  fi

  echo "${PYTORCH_LATEST_STABLE_VERSION[${family}]}"
}

pytorch_resolve_torchvision_version() {
  local torch_version
  local major
  local minor
  local patch
  local family
  local vision_minor

  torch_version="$(pytorch_strip_local_version "$1")"
  if [[ ! "${torch_version}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    return 1
  fi

  major="${BASH_REMATCH[1]}"
  minor="${BASH_REMATCH[2]}"
  patch="${BASH_REMATCH[3]}"
  family="${major}.${minor}"

  if [[ "${major}" != "2" ]] || ! pytorch_is_installable_family "${family}"; then
    return 1
  fi

  # Torch 2.N.P is paired with TorchVision 0.(N+15).P for the stable
  # release families configured by the central policy.
  vision_minor=$((10#${minor} + 15))
  echo "0.${vision_minor}.${patch}"
}

pytorch_cuda_version_to_tag() {
  local cuda_version="$1"
  local major
  local minor

  if [[ ! "${cuda_version}" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    return 1
  fi

  major="${BASH_REMATCH[1]}"
  minor="${BASH_REMATCH[2]}"
  echo "cu${major}${minor}"
}

pytorch_cuda_tag_to_version() {
  local tag="$1"
  local digits
  local length

  if [[ ! "${tag}" =~ ^cu([0-9]+)$ ]]; then
    return 1
  fi

  digits="${BASH_REMATCH[1]}"
  length=${#digits}
  if (( length < 3 )); then
    return 1
  fi

  # Supported CUDA releases use a two-digit major version (11, 12, 13, ...).
  # Keep the remaining digits as the minor version so a future cu1310 tag maps
  # to 13.10 instead of 131.0.
  echo "${digits:0:2}.${digits:2}"
}

pytorch_is_supported_wheel_tag() {
  local tag="$1"
  [[ "${tag}" == "cpu" || "${tag}" =~ ^cu[0-9]+$ ]]
}

pytorch_index_url() {
  local wheel_tag="$1"
  local is_nightly="${2:-0}"
  local channel=""

  if [[ "${is_nightly}" == "1" ]]; then
    channel="/nightly"
  fi

  echo "https://download.pytorch.org/whl${channel}/${wheel_tag}"
}

pytorch_detect_host_cuda_version() {
  local cuda_version=""

  if command -v nvcc &>/dev/null; then
    cuda_version="$(nvcc --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | head -n1)"
  fi

  if [[ -z "${cuda_version}" ]] && command -v nvidia-smi &>/dev/null; then
    cuda_version="$(nvidia-smi | sed -nE 's/.*CUDA Version: ([0-9]+\.[0-9]+).*/\1/p' | head -n1)"
  fi

  [[ -n "${cuda_version}" ]] || return 1
  echo "${cuda_version}"
}

pytorch_add_unique_index_url() {
  local candidate="$1"
  local existing

  for existing in "${PYTORCH_INDEX_URLS[@]}"; do
    [[ "${existing}" == "${candidate}" ]] && return 0
  done

  PYTORCH_INDEX_URLS+=("${candidate}")
}

pytorch_build_index_urls() {
  local family="$1"
  local host_cuda="$2"
  local is_nightly="${3:-0}"
  local requested_wheel_tag="${4:-}"
  local cuda_version
  local cuda_tag
  local -a cuda_candidates=()

  PYTORCH_INDEX_URLS=()

  if [[ -n "${requested_wheel_tag}" ]]; then
    if ! pytorch_is_supported_wheel_tag "${requested_wheel_tag}"; then
      echo "[ERROR] Unsupported PyTorch wheel tag '${requested_wheel_tag}'" >&2
      return 1
    fi

    pytorch_add_unique_index_url "$(pytorch_index_url "${requested_wheel_tag}" "${is_nightly}")"
    return 0
  fi

  if [[ -n "${host_cuda}" ]]; then
    if [[ "${is_nightly}" == "1" ]]; then
      # Try the detected CUDA index first because nightly availability changes
      # independently of stable release families.
      cuda_tag="$(pytorch_cuda_version_to_tag "${host_cuda}")" || return 1
      pytorch_add_unique_index_url "$(pytorch_index_url "${cuda_tag}" 1)"
      cuda_candidates=("${PYTORCH_NIGHTLY_CUDA_FALLBACKS[@]}")
    else
      if ! pytorch_is_installable_family "${family}"; then
        echo "[ERROR] Unsupported Torch family '${family}'" >&2
        return 1
      fi
      read -r -a cuda_candidates <<< "${PYTORCH_STABLE_CUDA_WHEELS[${family}]}"
    fi

    for cuda_version in "${cuda_candidates[@]}"; do
      if pytorch_version_le "${cuda_version}" "${host_cuda}"; then
        cuda_tag="$(pytorch_cuda_version_to_tag "${cuda_version}")" || return 1
        pytorch_add_unique_index_url "$(pytorch_index_url "${cuda_tag}" "${is_nightly}")"
      fi
    done
  fi

  # A CPU wheel keeps installation usable on machines without a compatible
  # published CUDA wheel. An explicit +cuXXX request does not reach this path.
  pytorch_add_unique_index_url "$(pytorch_index_url "cpu" "${is_nightly}")"
}

pytorch_get_pinned_requirement_version() {
  local requirement_file="$1"
  local package_name="$2"

  [[ -f "${requirement_file}" ]] || return 1

  sed -nE \
    "s/^[[:space:]]*${package_name}==([^[:space:]#]+).*$/\\1/p" \
    "${requirement_file}" | head -n1
}

pytorch_get_installed_torch_info() {
  python3 - <<'PY'
import importlib.util
import re
import sys

if importlib.util.find_spec("torch") is None:
    sys.exit(1)

try:
    import torch
except Exception as exc:
    print(f"Failed to import torch: {exc}", file=sys.stderr)
    sys.exit(2)

full_version = str(torch.__version__)
base_version = full_version.split("+", 1)[0]
family_match = re.match(r"^(\d+\.\d+)", base_version)
if family_match is None:
    print(f"Unrecognized torch version: {full_version}", file=sys.stderr)
    sys.exit(3)

family = family_match.group(1)
is_nightly = "1" if ".dev" in base_version else "0"
cuda_version = getattr(torch.version, "cuda", None)
hip_version = getattr(torch.version, "hip", None)

if cuda_version:
    wheel_tag = "cu" + str(cuda_version).replace(".", "")
elif hip_version:
    wheel_tag = "rocm" + str(hip_version)
else:
    wheel_tag = "cpu"

print("\t".join((full_version, base_version, family, wheel_tag, is_nightly)))
PY
}

pytorch_get_installed_torchvision_info() {
  python3 - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("torchvision") is None:
    sys.exit(1)

try:
    import torchvision
except Exception as exc:
    print(f"Failed to import torchvision: {exc}", file=sys.stderr)
    sys.exit(2)

full_version = str(torchvision.__version__)
base_version = full_version.split("+", 1)[0]
wheel_tag = full_version.split("+", 1)[1] if "+" in full_version else "none"
is_nightly = "1" if ".dev" in base_version else "0"
print("\t".join((full_version, base_version, wheel_tag, is_nightly)))
PY
}

pytorch_pip_check() {
  local allow_nightly_mismatch="${1:-0}"
  local output
  local status

  if output="$(python3 -m pip check 2>&1)"; then
    [[ -n "${output}" ]] && printf '%s\n' "${output}"
    return 0
  else
    status=$?
  fi

  if [[ "${allow_nightly_mismatch}" == "1" ]]; then
    if printf '%s\n' "${output}" | \
       python3 "${PYTORCH_POLICY_FILE}" filter-nightly-pip-check; then
      return 0
    fi
    return "${status}"
  fi

  printf '%s\n' "${output}" >&2
  return "${status}"
}

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

PYTORCH_DEFAULT_FAMILY="2.7"
PYTORCH_SUPPORTED_FAMILIES=("2.5" "2.6" "2.7" "2.8" "2.9" "2.10")

# Keep family requests deterministic. A family such as 2.7 resolves to the
# latest stable patch explicitly supported by TICO instead of asking pip to
# backtrack over every 2.7.x release.
declare -A PYTORCH_LATEST_STABLE_VERSION=(
  ["2.5"]="2.5.1"
  ["2.6"]="2.6.0"
  ["2.7"]="2.7.1"
  ["2.8"]="2.8.0"
  ["2.9"]="2.9.1"
  ["2.10"]="2.10.0"
)

# Official CUDA wheel variants for each supported stable Torch family. Values
# are ordered from newest to oldest so that the best compatible wheel is tried
# first. The CPU index is appended separately as the final fallback.
declare -A PYTORCH_STABLE_CUDA_WHEELS=(
  ["2.5"]="12.4 12.1 11.8"
  ["2.6"]="12.6 12.4 11.8"
  ["2.7"]="12.8 12.6 11.8"
  ["2.8"]="12.9 12.8 12.6"
  ["2.9"]="13.0 12.8 12.6"
  ["2.10"]="13.0 12.8 12.6"
)

# Used only when a nightly requirement does not contain an explicit local
# build tag such as +cpu or +cu130.
PYTORCH_NIGHTLY_CUDA_FALLBACKS=("13.0" "12.8" "12.6")

pytorch_is_supported_family() {
  local family="$1"
  local supported

  for supported in "${PYTORCH_SUPPORTED_FAMILIES[@]}"; do
    [[ "${supported}" == "${family}" ]] && return 0
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

  if ! pytorch_is_supported_family "${family}"; then
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

  if [[ "${major}" != "2" ]] || ! pytorch_is_supported_family "${family}"; then
    return 1
  fi

  # Torch 2.N.P is paired with TorchVision 0.(N+15).P for the supported
  # release families (2.5 through 2.10).
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
      if ! pytorch_is_supported_family "${family}"; then
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

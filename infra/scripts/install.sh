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

show_help() {
# `cat << EOF` This means that cat should stop reading when EOF is detected
cat << EOF
Usage: ./ccex install [--dist|--torch_ver|--help|-h]
--dist          Install from whl file
                (default: Install from source, editable mode)
--torch_ver     [2.5|2.6|nightly]
                Specify torch version to be installed.
                (default: 2.6)
-h | --help     Show help message and exit
EOF
}

options=$(getopt -o h --long dist,torch_ver:,help -- "$@")

[ $? -eq 0 ] || {
    echo "Incorrect options provided"
    exit 1
}

_DIST=0
_TORCH_VER="2.6"

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

if [ "$_TORCH_VER" != "2.5" ] && [ "$_TORCH_VER" != "2.6" ] && [ "$_TORCH_VER" != "nightly" ]; then
    echo "Invalid torch version '$_TORCH_VER'"
    echo "(Use --help to see available torch versions)"
    exit 1
fi

SCRIPTS_DIR="${CCEX_PROJECT_PATH}/infra/scripts"

if [ "$_TORCH_VER" == "nightly" ]; then
  echo "Install package dependencies from torch nightly version"
  python3 -m pip install -r "${SCRIPTS_DIR}/install_requirements_dev.txt"
elif [ "$_TORCH_VER" == "2.6" ]; then
  echo "Install package dependencies from torch stable version"
  python3 -m pip install -r "${SCRIPTS_DIR}/install_requirements_2_6.txt"
elif [ "$_TORCH_VER" == "2.5" ]; then
  echo "Install package dependencies from torch stable version"
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

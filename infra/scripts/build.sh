#!/bin/bash

# Install Required Package
#
# NOTE To add additional build dependencies, append to `requires` field of [build-system] in pyproject.toml

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

###############################################################################
# Install the auxiliary Python requirements
###############################################################################
REQ_FILE="${SCRIPTS_DIR}/build_requirements.txt"
echo "[INFO] Installing auxiliary requirements from ${REQ_FILE##*/}"
python3 -m pip install -r "$REQ_FILE"
python3 -m pip install build

# Build
python3 -m build

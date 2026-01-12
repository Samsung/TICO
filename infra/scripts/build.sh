#!/bin/bash

# Install Required Package
#
# NOTE To add additional build dependencies, append to `requires` field of [build-system] in pyproject.toml 
python3 -m pip install build

# Build
python3 -m build

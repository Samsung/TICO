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

TEST_DIR="${CCEX_PROJECT_PATH}/test"
COVERAGE_REPORT_DIR="${TEST_DIR}/reports/cov"

# The test should be run at the project root or test directory
pushd ${CCEX_PROJECT_PATH} > /dev/null

command_args="$@"

COVERAGE_EXIST=$(pip list | grep -w coverage)
if [ -z "${COVERAGE_EXIST}" ] > /dev/null; then
  echo "'coverage' does not exist."
  echo "Run python3 -m pip install coverage==7.6.1"
  exit 1
fi

coverage run -m unittest discover -s ${TEST_DIR} -v 2>&1
COVERAGE_TEST_STATUS=$?

if [ $# -eq 0 ]; then
  coverage report -i -m
else
  OPTION=$1; shift
  if [[ "${OPTION}" != '-f' ]]; then
    echo "${OPTION} is not supported"
  else
    if [ ! -d ${COVERAGE_REPORT_DIR} ] ; then
      mkdir -p ${COVERAGE_REPORT_DIR}
    fi

    FORMAT=$1; shift
    if [[ "${FORMAT}" == 'txt' ]]; then
      coverage report -i -m > ${COVERAGE_REPORT_DIR}/coverage.txt
    elif [[ "${FORMAT}" == 'xml' ]]; then
      coverage xml -i -o ${COVERAGE_REPORT_DIR}/coverage.xml
    else
      echo "Unknown format: ${FORMAT}"
      echo "Following formats are supported: txt, xml"
    fi
  fi
fi
COVERAGE_REPORT_STATUS=$?

if [ ${COVERAGE_TEST_STATUS} -ne 0 ]; then
  exit ${COVERAGE_TEST_STATUS}
fi
exit ${COVERAGE_REPORT_STATUS}

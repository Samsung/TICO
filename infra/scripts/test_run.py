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

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run unit tests")
    parser.add_argument("-a", "--all", action="store_true", help="Run all unit tests.")
    parser.add_argument(
        "-k", "--keyword", type=str, help="Specify keywords to filter tests."
    )
    parser.add_argument(
        "-m", "--model", type=str, help="Specify model to run tests on."
    )
    parser.add_argument(
        "-i",
        "--include-internal",
        action="store_true",
        help="Include internal-only tests",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    args = parser.parse_args()

    project_path = os.getenv("CCEX_PROJECT_PATH")
    if not project_path:
        print("Error: Environment variable CCEX_PROJECT_PATH is not set.")
        sys.exit(1)

    test_dir = os.path.join(project_path, "test")
    if not os.path.isdir(test_dir):
        print(f"Error: Test directory '{test_dir}' does not exist.")
        sys.exit(1)

    if args.all and args.keyword:
        print("Error: Cannot specify both --all and --keyword at the same time.")
        sys.exit(1)

    if args.verbose is True:
        os.environ["TICO_LOG"] = "4"

    if args.model:
        if args.keyword or args.all:
            print(
                "Error: Cannot specify both --model and --keyword/--all at the same time."
            )
            sys.exit(1)

    if args.include_internal:
        os.environ["RUN_INTERNAL_TESTS"] = "1"

    ##############
    # Run commands
    ##############

    if args.all or (not args.keyword and not args.model):
        print("RUN ALL unit tests ...")

        cmd = f"python3 -m unittest discover -s {test_dir} -v".split(" ")

        subprocess.run(cmd, check=True)
    elif args.keyword:
        if args.keyword in ["net", "op"]:  # TODO Add "model"
            args.keyword = "test.modules." + args.keyword

        print(f"RUN unit tests with -k {args.keyword} ...")

        cmd = f"python3 -m unittest discover -s {test_dir} -k {args.keyword} -v".split(
            " "
        )
        subprocess.run(cmd, check=True)
    elif args.model:
        print(f"RUN unit tests for model {args.model} ...")

        cmd = f"python3 -m unittest test.pt2_to_circle_test.test_model -k {args.model} -v".split(
            " "
        )
        subprocess.run(cmd, check=True)
    else:
        assert False, "Cannot reach here"


if __name__ == "__main__":
    main()

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

import os


# NOTE load_tests is `unittest` protocol to find tests dynamically
# https://docs.python.org/ko/3.13/library/unittest.html#load-tests-protocol
def load_tests(loader, standard_tests, pattern):
    # top level directory cached on loader instance
    this_dir = os.path.dirname(__file__)

    # Add test files to be found by `unittest`
    # WHY? Not to include other files by mistake and to make it clear which files are being tested
    for testfile in ["test_net.py", "test_op.py", "test_dyn_op.py"]:
        package_tests = loader.discover(start_dir=this_dir, pattern=testfile)
        standard_tests.addTests(package_tests)

    return standard_tests

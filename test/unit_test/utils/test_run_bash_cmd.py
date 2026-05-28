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

import unittest

from tico.utils.utils import run_bash_cmd


class RunBashCmdTest(unittest.TestCase):
    def test_simple_bash_cmd(self):
        completed_process = run_bash_cmd(["echo", "Hello World"])
        self.assertEqual(completed_process.stdout, "Hello World\n")

    def test_invalid_cmd_neg(self):
        with self.assertRaises(RuntimeError):
            run_bash_cmd(["ls", "for_invalid_test"])

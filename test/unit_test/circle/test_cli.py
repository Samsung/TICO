# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

from tico.circle.cli.main import _build_parser, _parse_passes
from tico.circle.passes.cleanup import CompactIndicesPass, DeadCodeEliminationPass


class CircleCLITest(unittest.TestCase):
    def test_extract_accepts_tensor_patterns_without_marker_flag(self):
        parser = _build_parser()

        args = parser.parse_args(
            [
                "extract",
                "input.circle",
                "-o",
                "output.circle",
                "--from-tensor",
                "input",
                "--to-tensor",
                "output",
            ]
        )

        self.assertEqual(args.from_tensor, ["input"])
        self.assertEqual(args.to_tensor, ["output"])

    def test_cleanup_pass_names_are_resolved(self):
        passes = _parse_passes("dce,compact")

        self.assertIsInstance(passes[0], DeadCodeEliminationPass)
        self.assertIsInstance(passes[1], CompactIndicesPass)

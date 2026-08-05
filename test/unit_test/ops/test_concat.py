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

"""Unit tests for the public concatenation module facade."""

import unittest

import torch

from tico.ops import Concat


class ConcatTest(unittest.TestCase):
    """Verify eager and exported concatenation behavior."""

    def test_eager_matches_torch_cat(self) -> None:
        """Match torch.cat for a fixed concatenation dimension."""
        values = (torch.randn(2, 3, 4), torch.randn(2, 5, 4))
        actual = Concat(dim=1)(values)
        expected = torch.cat(values, dim=1)
        torch.testing.assert_close(actual, expected)

    def test_torch_export_keeps_cat(self) -> None:
        """Export the facade as one aten.cat operation."""
        module = Concat(dim=1).eval()
        exported = torch.export.export(
            module,
            ((torch.randn(1, 2, 3), torch.randn(1, 4, 3)),),
            strict=True,
        )
        targets = [
            node.target for node in exported.graph.nodes if node.op == "call_function"
        ]
        self.assertEqual(targets.count(torch.ops.aten.cat.default), 1)

    def test_rejects_empty_input(self) -> None:
        """Reject an empty tensor sequence."""
        with self.assertRaisesRegex(ValueError, "at least one"):
            Concat(dim=1)(())


if __name__ == "__main__":
    unittest.main()

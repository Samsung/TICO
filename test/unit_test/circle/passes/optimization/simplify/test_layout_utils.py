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

from __future__ import annotations

import importlib
import importlib.util
import struct
import unittest
from types import SimpleNamespace

import numpy as np

from tico.circle.passes.optimization.simplify import _layout_utils


class LayoutUtilsTest(unittest.TestCase):
    """Test helpers retained after removing the legacy layout pass module."""

    def test_retained_region_modules_import_without_legacy_layout_ops(self) -> None:
        """Import both retained modules after deleting the legacy pass package."""

        importlib.import_module(
            "tico.circle.passes.optimization.simplify.transpose_region"
        )
        importlib.import_module(
            "tico.circle.passes.optimization.simplify.transpose_region_rules"
        )

        self.assertIsNone(
            importlib.util.find_spec("tico.circle.passes.optimization.remove")
        )

    def test_inverse_permutations_are_recognized(self) -> None:
        """Accept valid inverses and reject malformed permutations."""

        self.assertTrue(_layout_utils._check_perm([2, 0, 1], [1, 2, 0]))
        self.assertFalse(_layout_utils._check_perm([1, 1], [0, 1]))
        self.assertFalse(_layout_utils._check_perm([0, 1], [-1, 0]))

    def test_inline_i32_constants_are_decoded(self) -> None:
        """Decode bytes and generated NumPy vectors without accepting externals."""

        payload = struct.pack("<ii", 1, -1)
        graph = SimpleNamespace(
            subgraph=SimpleNamespace(
                tensors=[SimpleNamespace(buffer=1)],
            ),
            model=SimpleNamespace(
                buffers=[
                    SimpleNamespace(data=None, offset=0, size=0),
                    SimpleNamespace(
                        data=np.frombuffer(payload, dtype=np.uint8).copy(),
                        offset=0,
                        size=0,
                    ),
                ],
            ),
        )

        self.assertEqual(_layout_utils._get_const_data(graph, 0), [1, -1])
        graph.model.buffers[1].offset = 4
        self.assertIsNone(_layout_utils._get_const_data(graph, 0))

    def test_transpose_detection_validates_opcode_index(self) -> None:
        """Match only operators that reference the TRANSPOSE builtin code."""

        operator_codes = [
            SimpleNamespace(builtinCode=_layout_utils._TRANSPOSE_BUILTIN_CODE)
        ]
        self.assertTrue(
            _layout_utils._is_transpose_op(
                SimpleNamespace(opcodeIndex=0),
                operator_codes,
            )
        )
        self.assertFalse(
            _layout_utils._is_transpose_op(
                SimpleNamespace(opcodeIndex=1),
                operator_codes,
            )
        )


if __name__ == "__main__":
    unittest.main()

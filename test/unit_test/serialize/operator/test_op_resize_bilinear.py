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

"""Serialization tests for Circle RESIZE_BILINEAR."""

from __future__ import annotations

import unittest

import tico
import torch

from circle_schema import circle
from tico.circle.io import model_from_bytes
from tico.ops import ResizeBilinear2d


def _single_resize_options(module: ResizeBilinear2d):
    """Export one module and return its single ResizeBilinear options object."""
    result = tico.convert(module.eval(), (torch.randn(1, 2, 3, 4),))
    model = model_from_bytes(result.circle_binary)
    resize_opcode_indices = {
        index
        for index, opcode in enumerate(model.operatorCodes)
        if opcode.builtinCode == circle.BuiltinOperator.BuiltinOperator.RESIZE_BILINEAR
    }
    operators = [
        operator
        for operator in model.subgraphs[0].operators
        if operator.opcodeIndex in resize_opcode_indices
    ]
    if len(operators) != 1:
        raise RuntimeError(
            f"Expected one Circle RESIZE_BILINEAR, but found {len(operators)}."
        )
    return operators[0].builtinOptions


class ResizeBilinearVisitorTest(unittest.TestCase):
    """Verify that the serializer preserves both coordinate options."""

    def test_serializes_asymmetric_options(self) -> None:
        """Serialize align-corners false and half-pixel-centers false."""
        options = _single_resize_options(
            ResizeBilinear2d(
                (6, 8),
                align_corners=False,
                half_pixel_centers=False,
            )
        )
        self.assertFalse(options.alignCorners)
        self.assertFalse(options.halfPixelCenters)

    def test_serializes_half_pixel_options(self) -> None:
        """Serialize the half-pixel option used by the hand detector."""
        options = _single_resize_options(
            ResizeBilinear2d(
                (6, 8),
                align_corners=False,
                half_pixel_centers=True,
            )
        )
        self.assertFalse(options.alignCorners)
        self.assertTrue(options.halfPixelCenters)


if __name__ == "__main__":
    unittest.main()

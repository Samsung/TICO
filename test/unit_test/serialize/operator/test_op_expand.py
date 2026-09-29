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

"""Serialization tests for Circle BROADCAST_TO with static and dynamic sizes."""

from __future__ import annotations

import unittest

import numpy as np
import tico
import torch
from circle_schema import circle
from tico.circle.io import model_from_bytes
from torch.export import Dim


def _builtin_names(circle_binary: bytes) -> list[str]:
    model = model_from_bytes(circle_binary)
    names = {
        value: name
        for name, value in vars(circle.BuiltinOperator.BuiltinOperator).items()
        if isinstance(value, int)
    }
    return [
        names[model.operatorCodes[operator.opcodeIndex].builtinCode]
        for operator in model.subgraphs[0].operators
    ]


class StaticExpand(torch.nn.Module):
    def forward(self, x):
        return x.expand(3, -1)


class DynamicExpand(torch.nn.Module):
    """Expand a [1, dim] input to [2, dim]; `dim` is only known at runtime."""

    def forward(self, x):
        return x.expand(2, -1)


class ExpandSerializationTest(unittest.TestCase):
    def test_static_expand_uses_constant_shape(self):
        model = tico.convert(StaticExpand().eval(), (torch.randn(1, 4),))
        self.assertEqual(_builtin_names(model.circle_binary), ["BROADCAST_TO"])
        x = torch.randn(1, 4)
        np.testing.assert_array_equal(model(x), x.expand(3, -1).numpy())

    def test_dynamic_expand_reads_the_size_from_the_runtime_shape(self):
        """A kept dynamic dim is serialized as SHAPE -> STRIDED_SLICE -> CONCATENATION."""

        batch = Dim("dim", min=1, max=64)
        model = tico.convert(
            DynamicExpand().eval(),
            (torch.randn(1, 4),),
            dynamic_shapes=({1: batch},),  # type: ignore[arg-type]
        )
        self.assertEqual(
            _builtin_names(model.circle_binary),
            ["SHAPE", "STRIDED_SLICE", "CONCATENATION", "BROADCAST_TO"],
        )
        for size in (4, 1, 9):
            x = torch.randn(1, size)
            result = model(x)
            self.assertEqual(result.shape, (2, size))
            np.testing.assert_array_equal(result, x.expand(2, -1).numpy())


if __name__ == "__main__":
    unittest.main()

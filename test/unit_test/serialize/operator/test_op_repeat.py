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

"""Validate the serialized tensor contracts of multi-axis repeat chains."""

import unittest

import numpy as np
import tico
import torch
from circle_schema import circle
from tico.circle import CircleDocument
from torch.export import Dim


class _Repeat(torch.nn.Module):
    def __init__(self, repeats):
        super().__init__()
        self.repeats = repeats

    def forward(self, x):
        return x.repeat(self.repeats)


class RepeatSerializationTest(unittest.TestCase):
    def test_multi_axis_repeat_preserves_cumulative_shapes(self):
        cases = (
            ((2, 2, 2), (2, 3, 4), ((4, 2, 2), (4, 6, 2), (4, 6, 8))),
            (
                (2, 2, 2, 2),
                (2, 1, 3, 4),
                ((4, 2, 2, 2), (4, 2, 6, 2), (4, 2, 6, 8)),
            ),
            (
                (2, 2, 2, 2),
                (2, 3, 2, 3),
                ((4, 2, 2, 2), (4, 6, 2, 2), (4, 6, 4, 2), (4, 6, 4, 6)),
            ),
        )
        for shape, repeats, expected_shapes in cases:
            for dtype in (torch.float32, torch.int32):
                for optimize in (False, True):
                    with self.subTest(
                        shape=shape, repeats=repeats, dtype=dtype, optimize=optimize
                    ):
                        x = torch.arange(np.prod(shape).item(), dtype=dtype).reshape(
                            shape
                        )
                        module = _Repeat(repeats).eval()
                        model = tico.convert(
                            module,
                            (x,),
                            config=tico.CompileConfigV1(circle_optimize=optimize),
                        )
                        document = CircleDocument.from_bytes(model.circle_binary)
                        self.assertTrue(document.verify().ok)
                        graph = document.model.subgraphs[0]
                        self.assertEqual(len(graph.operators), len(expected_shapes))
                        for operator, expected_shape in zip(
                            graph.operators, expected_shapes
                        ):
                            self.assertEqual(
                                document.model.operatorCodes[
                                    operator.opcodeIndex
                                ].builtinCode,
                                circle.BuiltinOperator.BuiltinOperator.CONCATENATION,
                            )
                            output = graph.tensors[operator.outputs[0]]
                            self.assertEqual(tuple(output.shape), expected_shape)
                            self.assertEqual(
                                output.type, graph.tensors[graph.inputs[0]].type
                            )
                        actual = model(x)
                        self.assertEqual(actual.dtype, x.numpy().dtype)
                        np.testing.assert_array_equal(actual, module(x).numpy())

    def test_multi_axis_repeat_preserves_dynamic_shape_signatures(self):
        module = _Repeat((1, 2, 3, 4)).eval()
        x = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2)
        model = tico.convert(
            module,
            (x,),
            dynamic_shapes=({0: Dim("batch", min=1, max=8)},),  # type: ignore[arg-type]
        )
        document = CircleDocument.from_bytes(model.circle_binary)
        self.assertTrue(document.verify().ok)
        graph = document.model.subgraphs[0]
        expected_signatures = ((-1, 4, 2, 2), (-1, 4, 6, 2), (-1, 4, 6, 8))
        self.assertEqual(len(graph.operators), len(expected_signatures))
        for operator, signature in zip(graph.operators, expected_signatures):
            output = graph.tensors[operator.outputs[0]]
            self.assertEqual(tuple(output.shapeSignature), signature)
            self.assertEqual(tuple(output.shape), (1, *signature[1:]))
        for batch in (1, 2, 5):
            with self.subTest(batch=batch):
                inputs = torch.arange(batch * 8, dtype=torch.float32).reshape(
                    batch, 2, 2, 2
                )
                np.testing.assert_array_equal(model(inputs), module(inputs).numpy())


if __name__ == "__main__":
    unittest.main()

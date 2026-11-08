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

import tico
import torch
from circle_schema import circle
from tico.circle import CircleDocument
from tico.passes.convert_to_relu6 import ConvertToReLU6

from test.support.helper import num_of_ops


class _Clamp(torch.nn.Module):
    def __init__(self, minimum, maximum, reuse_input=False):
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum
        self.reuse_input = reuse_input

    def forward(self, x):
        result = torch.clamp(x, self.minimum, self.maximum)
        return (result, x) if self.reuse_input else result


class _DoubleClamp(torch.nn.Module):
    def __init__(self, inner_bounds, outer_bounds, shared=False):
        super().__init__()
        self.inner_bounds = inner_bounds
        self.outer_bounds = outer_bounds
        self.shared = shared

    def forward(self, x):
        inner = torch.clamp(x, *self.inner_bounds)
        result = torch.clamp(inner, *self.outer_bounds)
        return (result, inner, x) if self.shared else result


class _HardTanh(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.hardtanh(x, 0.0, 6.0)


class ConvertToReLU6Test(unittest.TestCase):
    def _lower(self, model, x, casts, dynamic_shapes=None):
        ep = torch.export.export(model.eval(), (x,), dynamic_shapes=dynamic_shapes)
        expected = ep.module()(x)
        with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
            self.assertTrue(ConvertToReLU6().call(ep).modified)
        ep.validate()
        torch.testing.assert_close(ep.module()(x), expected, rtol=0, atol=0)
        self.assertEqual(num_of_ops(ep, [torch.ops.aten.relu6.default]), 1)
        self.assertEqual(num_of_ops(ep, [torch.ops.aten._to_copy.default]), casts)
        relu = next(
            node
            for node in ep.graph.nodes
            if node.target == torch.ops.aten.relu6.default
        )
        self.assertEqual(
            relu.meta["val"].dtype,
            expected[0].dtype if isinstance(expected, tuple) else expected.dtype,
        )
        graph = str(ep.graph)
        self.assertFalse(ConvertToReLU6().call(ep).modified)
        self.assertEqual(str(ep.graph), graph)
        return ep

    def test_float_bounds_preserve_integer_and_boolean_promotion(self):
        for dtype in (
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            for bounds in ((0.0, 6.0), (0, 6.0), (0.0, 6)):
                with self.subTest(dtype=dtype, bounds=bounds):
                    x = torch.tensor([0, 1, 5, 6, 100], dtype=dtype)
                    ep = self._lower(_Clamp(*bounds), x, casts=1)
                    self.assertEqual(ep.module()(x).dtype, torch.get_default_dtype())

    def test_double_clamps_preserve_promotion_from_either_bound(self):
        x = torch.tensor([-100, -1, 0, 1, 5, 6, 100], dtype=torch.int32)
        cases = (
            ((0.0, None), (None, 6)),
            ((0, None), (None, 6.0)),
            ((-2.0, 6), (0, None)),
            ((0, 100.0), (None, 6)),
            ((None, 6.0), (0, None)),
        )
        for inner, outer in cases:
            with self.subTest(inner=inner, outer=outer):
                ep = self._lower(_DoubleClamp(inner, outer), x, casts=1)
                self.assertEqual(num_of_ops(ep, [torch.ops.aten.clamp.default]), 0)

    def test_integer_bounds_preserve_integer_dtype(self):
        for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            x = torch.tensor([0, 1, 6, 100], dtype=dtype)
            for model in (_Clamp(0, 6), _DoubleClamp((0, None), (None, 6))):
                with self.subTest(dtype=dtype, model=type(model).__name__):
                    self._lower(model, x, casts=0)

    def test_boolean_input_with_integer_bounds_promotes_to_int64(self):
        x = torch.tensor([False, True])
        ep = self._lower(_Clamp(0, 6), x, casts=1)
        self.assertEqual(ep.module()(x).dtype, torch.int64)

    def test_float_inputs_and_hardtanh_do_not_need_a_cast(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            x = torch.tensor([-10, -1, 0, 1, 5, 6, 10], dtype=dtype)
            for model in (_Clamp(0.0, 6.0), _HardTanh()):
                with self.subTest(dtype=dtype, model=type(model).__name__):
                    self._lower(model, x, casts=0)
        self._lower(_HardTanh(), torch.tensor([-10, 1, 10], dtype=torch.int32), casts=0)

    def test_scalar_empty_and_shared_inputs_keep_the_original_input(self):
        for x in (
            torch.tensor(-1, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.tensor([[[-1, 0, 5, 100]]], dtype=torch.int32),
        ):
            with self.subTest(shape=x.shape):
                ep = self._lower(_Clamp(0.0, 6.0, reuse_input=True), x, casts=1)
                original = next(
                    node for node in ep.graph.nodes if node.op == "placeholder"
                )
                output = next(node for node in ep.graph.nodes if node.op == "output")
                self.assertIs(output.args[0][1], original)

    def test_double_clamp_keeps_shared_inner_result(self):
        x = torch.tensor([-1, 0, 1, 100], dtype=torch.int32)
        ep = self._lower(_DoubleClamp((0.0, None), (None, 6), shared=True), x, casts=1)
        self.assertEqual(num_of_ops(ep, [torch.ops.aten.clamp.default]), 1)
        actual, inner, original = ep.module()(x)
        self.assertEqual(actual.dtype, torch.get_default_dtype())
        torch.testing.assert_close(
            inner, torch.tensor([0.0, 0.0, 1.0, 100.0]), rtol=0, atol=0
        )
        torch.testing.assert_close(original, x, rtol=0, atol=0)

    def test_symbolic_batch_preserves_shape_and_dtype(self):
        model = _Clamp(0.0, 6.0)
        x = torch.tensor([[-1, 0, 1], [5, 6, 100]], dtype=torch.int32)
        ep = self._lower(
            model,
            x,
            casts=1,
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=8)}},
        )
        other = torch.cat((x, x[:1]))
        torch.testing.assert_close(ep.module()(other), model(other), rtol=0, atol=0)

    def test_uses_exported_dtype_after_default_dtype_changes(self):
        original_default = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            x = torch.tensor([-1, 0, 1, 100], dtype=torch.int32)
            for model in (_Clamp(0.0, 6), _DoubleClamp((0.0, None), (None, 6))):
                torch.set_default_dtype(torch.float64)
                ep = torch.export.export(model, (x,))
                expected = ep.module()(x)
                torch.set_default_dtype(torch.float32)
                with ep.graph_module._set_replace_hook(
                    ep.graph_signature.get_replace_hook()
                ):
                    self.assertTrue(ConvertToReLU6().call(ep).modified)
                ep.validate()
                torch.testing.assert_close(ep.module()(x), expected, rtol=0, atol=0)
                self.assertEqual(ep.module()(x).dtype, torch.float64)
        finally:
            torch.set_default_dtype(original_default)

    def test_similar_clamps_with_other_bounds_do_not_match(self):
        x = torch.tensor([-1, 0, 1, 100], dtype=torch.int32)
        for model in (
            _Clamp(0.0, 5.0),
            _DoubleClamp((0.0, None), (None, 5)),
            _Clamp(torch.tensor(0.0), torch.tensor(6.0)),
        ):
            with self.subTest(model=type(model).__name__):
                ep = torch.export.export(model, (x,))
                graph = str(ep.graph)
                self.assertFalse(ConvertToReLU6().call(ep).modified)
                self.assertEqual(str(ep.graph), graph)
                torch.testing.assert_close(ep.module()(x), model(x), rtol=0, atol=0)

    def test_default_circle_conversion_preserves_dtypes_and_values(self):
        x = torch.tensor([-100, -1, 0, 1, 5, 6, 100], dtype=torch.int32)
        cases = (
            (
                _Clamp(0.0, 6, reuse_input=True),
                x,
                circle.TensorType.TensorType.INT32,
                circle.TensorType.TensorType.FLOAT32,
            ),
            (
                _DoubleClamp((0.0, None), (None, 6), shared=True),
                x,
                circle.TensorType.TensorType.INT32,
                circle.TensorType.TensorType.FLOAT32,
            ),
            (
                _Clamp(0, 6, reuse_input=True),
                x,
                circle.TensorType.TensorType.INT32,
                circle.TensorType.TensorType.INT32,
            ),
            (
                _Clamp(0.0, 6.0, reuse_input=True),
                x.float(),
                circle.TensorType.TensorType.FLOAT32,
                circle.TensorType.TensorType.FLOAT32,
            ),
            (
                _Clamp(0, 6, reuse_input=True),
                torch.tensor([False, True]),
                circle.TensorType.TensorType.BOOL,
                circle.TensorType.TensorType.INT64,
            ),
        )
        for model, input_, input_type, output_type in cases:
            with self.subTest(
                model=type(model).__name__, dtype=input_.dtype, output_type=output_type
            ):
                model.eval()
                expected = model(input_)
                original = input_.clone()
                converted = tico.convert(model, (input_,))
                self.assertEqual(converted.runtime, "reference")
                document = CircleDocument.from_bytes(converted.circle_binary)
                self.assertTrue(document.verify().ok)
                subgraph = document.model.subgraphs[0]
                self.assertEqual(len(subgraph.inputs), 1)
                self.assertEqual(len(subgraph.outputs), len(expected))
                self.assertEqual(subgraph.tensors[subgraph.inputs[0]].type, input_type)
                self.assertEqual(
                    subgraph.tensors[subgraph.outputs[0]].type, output_type
                )
                self.assertEqual(
                    subgraph.tensors[subgraph.outputs[-1]].type, input_type
                )
                opcodes = [
                    document.model.operatorCodes[op.opcodeIndex].builtinCode
                    for op in subgraph.operators
                ]
                self.assertIn(circle.BuiltinOperator.BuiltinOperator.RELU6, opcodes)
                self.assertEqual(
                    opcodes.count(circle.BuiltinOperator.BuiltinOperator.CAST),
                    int(input_type != output_type),
                )
                for actual, reference in zip(converted(input_), expected):
                    torch.testing.assert_close(
                        torch.from_numpy(actual), reference, rtol=0, atol=0
                    )
                torch.testing.assert_close(input_, original, rtol=0, atol=0)

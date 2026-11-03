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

import tico
import torch
from circle_schema import circle
from tico.circle import CircleDocument
from tico.passes.lower_pow2_to_mul import LowerPow2ToMul

from test.support.helper import num_of_ops
from test.support.pass_value_test import SinglePassValueTest


class Pow2Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = x.pow(2)
        return z

    def get_example_inputs(self):
        return (torch.randn(3, 4),), {}


class LowerPow2ToMulTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(Pow2Net())
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.pow.Tensor_Scalar]), 1
        )

        self.run_value_test(LowerPow2ToMul())
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.pow.Tensor_Scalar]), 0
        )


class _Power(torch.nn.Module):
    def __init__(self, exponent, reuse_input=False):
        super().__init__()
        self.exponent = exponent
        self.reuse_input = reuse_input

    def forward(self, x):
        square = x.pow(self.exponent)
        return (square, x) if self.reuse_input else square


class LowerPow2PromotionTest(unittest.TestCase):
    def _lower(self, model, x):
        expected = model(x)
        ep = torch.export.export(model.eval(), (x,))
        self.assertEqual(num_of_ops(ep, [torch.ops.aten.pow.Tensor_Scalar]), 1)
        with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
            self.assertTrue(LowerPow2ToMul().call(ep).modified)
        self.assertEqual(num_of_ops(ep, [torch.ops.aten.pow.Tensor_Scalar]), 0)
        self.assertEqual(num_of_ops(ep, [torch.ops.aten.mul.Tensor]), 1)
        ep.validate()
        actual = ep.module()(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        graph = str(ep.graph)
        self.assertFalse(LowerPow2ToMul().call(ep).modified)
        self.assertEqual(str(ep.graph), graph)
        return ep

    def test_floating_exponent_promotes_before_integer_multiplication(self):
        for dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                values = [0, 12, 100] if dtype == torch.uint8 else [-100, -12, 0, 100]
                if dtype in (torch.int32, torch.int64):
                    values = [-50000, -1, 0, 50000]
                x = torch.tensor(values, dtype=dtype)
                ep = self._lower(_Power(2.0), x)
                casts = [
                    n
                    for n in ep.graph.nodes
                    if n.target == torch.ops.aten._to_copy.default
                ]
                self.assertEqual(len(casts), 1)
                cast = casts[0]
                self.assertEqual(cast.kwargs, {"dtype": torch.get_default_dtype()})
                self.assertEqual(cast.meta["val"].dtype, torch.get_default_dtype())
                mul = next(
                    n for n in ep.graph.nodes if n.target == torch.ops.aten.mul.Tensor
                )
                self.assertEqual(mul.args, (cast, cast))
                self.assertEqual(mul.meta["val"].dtype, torch.get_default_dtype())

    def test_boolean_square_preserves_integer_and_floating_promotion(self):
        x = torch.tensor([False, True])
        for exponent, dtype in ((2, torch.int64), (2.0, torch.get_default_dtype())):
            with self.subTest(exponent=exponent):
                ep = self._lower(_Power(exponent), x)
                output = ep.module()(x)
                self.assertEqual(output.dtype, dtype)

    def test_integer_exponent_preserves_native_overflow(self):
        cases = (
            (torch.int8, [-100, -12, 0, 100]),
            (torch.uint8, [0, 12, 100]),
            (torch.int16, [-300, -1, 0, 300]),
            (torch.int32, [-50000, -1, 0, 50000]),
            (torch.int64, [-3037000500, -1, 0, 3037000500]),
        )
        for dtype, values in cases:
            with self.subTest(dtype=dtype):
                ep = self._lower(_Power(2), torch.tensor(values, dtype=dtype))
                self.assertEqual(num_of_ops(ep, [torch.ops.aten._to_copy.default]), 0)

    def test_floating_inputs_do_not_need_a_cast(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for exponent in (2, 2.0):
                with self.subTest(dtype=dtype, exponent=exponent):
                    ep = self._lower(
                        _Power(exponent), torch.tensor([-3, 0, 4], dtype=dtype)
                    )
                    self.assertEqual(
                        num_of_ops(ep, [torch.ops.aten._to_copy.default]), 0
                    )

    def test_scalar_empty_and_shared_inputs(self):
        for x in (
            torch.tensor(-50000, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.tensor([[-50000, 50000]], dtype=torch.int32),
        ):
            with self.subTest(shape=x.shape):
                ep = self._lower(_Power(2.0, reuse_input=True), x)
                original_input = next(
                    n for n in ep.graph.nodes if n.op == "placeholder"
                )
                output = next(n for n in ep.graph.nodes if n.op == "output")
                self.assertIs(output.args[0][1], original_input)

    def test_uses_exported_result_dtype_after_default_dtype_changes(self):
        original_default = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            x = torch.tensor([-50000, 50000], dtype=torch.int32)
            ep = torch.export.export(_Power(2.0), (x,))
            expected = ep.module()(x)
            torch.set_default_dtype(torch.float32)
            with ep.graph_module._set_replace_hook(
                ep.graph_signature.get_replace_hook()
            ):
                self.assertTrue(LowerPow2ToMul().call(ep).modified)
            actual = ep.module()(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.dtype, torch.float64)
        finally:
            torch.set_default_dtype(original_default)

    def test_other_scalar_powers_and_tensor_exponents_do_not_match(self):
        x = torch.tensor([1, 3], dtype=torch.int32)
        for exponent in (1, 3, 2.5, torch.tensor(2)):
            with self.subTest(exponent=exponent):
                model = _Power(exponent)
                ep = torch.export.export(model, (x,))
                graph = str(ep.graph)
                self.assertFalse(LowerPow2ToMul().call(ep).modified)
                self.assertEqual(str(ep.graph), graph)
                torch.testing.assert_close(ep.module()(x), model(x), rtol=0, atol=0)

    def test_default_circle_conversion_preserves_input_and_output_dtypes(self):
        cases = (
            (
                torch.tensor([-50000, -1, 0, 50000], dtype=torch.int32),
                2.0,
                circle.TensorType.TensorType.INT32,
                circle.TensorType.TensorType.FLOAT32,
            ),
            (
                torch.tensor([-50000, -1, 0, 50000], dtype=torch.int32),
                2,
                circle.TensorType.TensorType.INT32,
                circle.TensorType.TensorType.INT32,
            ),
            (
                torch.tensor([False, True]),
                2,
                circle.TensorType.TensorType.BOOL,
                circle.TensorType.TensorType.INT64,
            ),
        )
        for x, exponent, input_type, output_type in cases:
            with self.subTest(dtype=x.dtype, exponent=exponent):
                model = _Power(exponent, reuse_input=True).eval()
                converted = tico.convert(model, (x,))
                self.assertEqual(converted.runtime, "reference")
                document = CircleDocument.from_bytes(converted.circle_binary)
                self.assertTrue(document.verify().ok)
                subgraph = document.model.subgraphs[0]
                self.assertEqual(len(subgraph.inputs), 1)
                self.assertEqual(len(subgraph.outputs), 2)
                self.assertEqual(subgraph.tensors[subgraph.inputs[0]].type, input_type)
                self.assertEqual(
                    subgraph.tensors[subgraph.outputs[0]].type, output_type
                )
                self.assertEqual(subgraph.tensors[subgraph.outputs[1]].type, input_type)
                opcodes = [
                    document.model.operatorCodes[op.opcodeIndex].builtinCode
                    for op in subgraph.operators
                ]
                self.assertIn(circle.BuiltinOperator.BuiltinOperator.MUL, opcodes)
                self.assertNotIn(circle.BuiltinOperator.BuiltinOperator.POW, opcodes)
                self.assertEqual(
                    opcodes.count(circle.BuiltinOperator.BuiltinOperator.CAST),
                    int(input_type != output_type),
                )
                for actual, expected in zip(converted(x), model(x)):
                    torch.testing.assert_close(
                        torch.from_numpy(actual), expected, rtol=0, atol=0
                    )

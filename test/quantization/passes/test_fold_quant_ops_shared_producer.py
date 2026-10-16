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

"""Do not round an unquantized residual when only its other branch requests Q."""

import unittest

import numpy as np

import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.quantization.passes.remove_weight_dequant_op import RemoveWeightDequantOp
from tico.serialize.quant_param import QPARAM_KEY


class _SharedFloatResidual(torch.nn.Module):
    def __init__(self, *, quantized_weight: bool = False, per_channel: bool = False):
        super().__init__()
        self.quantized_weight = quantized_weight
        self.per_channel = per_channel
        self.register_buffer("weight", torch.tensor([1, 3], dtype=torch.int8))
        self.register_buffer("scales", torch.tensor([0.0625, 0.1875]))
        self.register_buffer("zero_points", torch.tensor([0, 0], dtype=torch.int32))

    def forward(self, value):
        value = torch.fake_quantize_per_tensor_affine(value, 0.125, 128, 0, 255)
        if self.quantized_weight:
            if self.per_channel:
                other = torch.ops.quantized_decomposed.dequantize_per_channel.default(
                    self.weight, self.scales, self.zero_points, 0, -128, 127, torch.int8
                )
            else:
                other = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                    self.weight, 0.0625, 0, -128, 127, torch.int8
                )
        else:
            other = 0.0625
        residual = value + other
        branch = torch.fake_quantize_per_tensor_affine(residual, 0.25, 128, 0, 255)
        return residual, branch


class _StackRequantize(torch.nn.Module):
    def forward(self, value):
        value = torch.fake_quantize_per_tensor_affine(value, 0.125, 128, 0, 255)
        stacked = torch.stack((value, value), dim=1).reshape(1, 4)
        return torch.fake_quantize_per_tensor_affine(stacked, 0.3, 128, 0, 255)


class _DistinctConcatDomains(torch.nn.Module):
    def forward(self, left, right):
        left = torch.fake_quantize_per_tensor_affine(left, 0.125, 128, 0, 255)
        right = torch.fake_quantize_per_tensor_affine(right, 0.25, 128, 0, 255)
        combined = torch.cat((left, right), dim=0).reshape(1, 4)
        return torch.fake_quantize_per_tensor_affine(combined, 0.3, 128, 0, 255)


class FoldQuantOpsSharedProducerTest(unittest.TestCase):
    def test_preserves_raw_branch_and_explicit_float_input_bridge(self):
        program = torch.export.export(
            _SharedFloatResidual().eval(), (torch.tensor([0.25, 0.75]),)
        )
        DecomposeFakeQuantize().call(program)
        FoldQuantOps().call(program)
        residual = next(
            node
            for node in program.graph.nodes
            if node.target == torch.ops.aten.add.Tensor
        )
        self.assertNotIn(QPARAM_KEY, residual.meta)
        dequantize = residual.args[0]
        self.assertEqual(
            dequantize.target,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        )
        self.assertNotIn(QPARAM_KEY, dequantize.meta)
        self.assertEqual(dequantize.meta["val"].dtype, torch.float32)
        self.assertEqual(dequantize.args[0].meta[QPARAM_KEY].scale, [0.125])
        quantize = next(
            node
            for node in program.graph.nodes
            if node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
        )
        self.assertIs(quantize.args[0], residual)
        self.assertEqual(quantize.meta[QPARAM_KEY].scale, [0.25])
        output = next(node for node in program.graph.nodes if node.op == "output")
        self.assertEqual(output.args[0], (residual, quantize))
        FoldQuantOps().call(program)
        self.assertIs(residual.args[0], dequantize)

    def test_public_conversion_preserves_float_and_quantized_branch_values(self):
        inputs = np.array([0.25, 0.75], dtype=np.float32)
        for quantized_weight, per_channel in (
            (False, False),
            (True, False),
            (True, True),
        ):
            with self.subTest(
                quantized_weight=quantized_weight, per_channel=per_channel
            ):
                module = _SharedFloatResidual(
                    quantized_weight=quantized_weight, per_channel=per_channel
                ).eval()
                converted = tico.convert(module, (torch.from_numpy(inputs),))
                runtime = CircleReferenceRuntime(
                    CircleDocument.from_bytes(converted.circle_binary)
                )
                expected = [value.numpy() for value in module(torch.from_numpy(inputs))]
                actual = runtime.run([inputs], mode=ExecutionMode.FAKE_QUANTIZE).outputs
                for got, want in zip(actual, expected):
                    np.testing.assert_array_equal(got, want)
                add = next(op for op in runtime.program.operators if op.name == "ADD")
                for index in (*add.inputs, *add.outputs):
                    self.assertEqual(runtime.program.tensor(index).dtype, np.float32)
                native_input = (inputs / np.float32(0.125) + 128).astype(np.uint8)
                native = runtime.run([native_input]).outputs
                np.testing.assert_array_equal(native[0], expected[0])
                np.testing.assert_array_equal(
                    native[1], (expected[1] / 0.25 + 128).astype(np.uint8)
                )

    def test_weight_dequantize_survives_for_protected_raw_producer(self):
        program = torch.export.export(
            _SharedFloatResidual(quantized_weight=True).eval(),
            (torch.tensor([0.25, 0.75]),),
        )
        DecomposeFakeQuantize().call(program)
        FoldQuantOps().call(program)
        RemoveWeightDequantOp().call(program)
        residual = next(
            node
            for node in program.graph.nodes
            if node.target == torch.ops.aten.add.Tensor
        )
        weight_dequantize = residual.args[1]
        self.assertEqual(
            weight_dequantize.target,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        )
        self.assertEqual(weight_dequantize.args[0].meta[QPARAM_KEY].scale, [0.0625])
        RemoveWeightDequantOp().call(program)
        self.assertIs(residual.args[1], weight_dequantize)

    def test_stack_view_preserves_independent_output_quantization(self):
        inputs = np.array([0.25, 0.875], dtype=np.float32)
        module = _StackRequantize().eval()
        converted = tico.convert(module, (torch.from_numpy(inputs),))
        runtime = CircleReferenceRuntime(
            CircleDocument.from_bytes(converted.circle_binary)
        )
        final = runtime.program.operators[-1]
        self.assertEqual(final.name, "QUANTIZE")
        source = runtime.program.tensor(final.inputs[0])
        target = runtime.program.tensor(final.outputs[0])
        assert source.quantization is not None
        assert target.quantization is not None
        self.assertEqual(source.quantization.scale, (0.125,))
        self.assertEqual(target.quantization.scale, (float(np.float32(0.3)),))
        actual = runtime.run([inputs], mode=ExecutionMode.FAKE_QUANTIZE).outputs[0]
        np.testing.assert_array_equal(actual, module(torch.from_numpy(inputs)).numpy())
        native_input = np.asarray(inputs / np.float32(0.125) + 128, dtype=np.uint8)
        native = runtime.run([native_input]).outputs[0]
        np.testing.assert_array_equal(
            native, np.asarray(actual / np.float32(0.3) + 128, dtype=np.uint8)
        )

    def test_distinct_concat_input_domains_are_not_inherited(self):
        inputs = (torch.tensor([0.25, 0.875]), torch.tensor([0.25, 0.75]))
        program = torch.export.export(_DistinctConcatDomains().eval(), inputs)
        DecomposeFakeQuantize().call(program)
        FoldQuantOps().call(program)
        concat = next(
            node
            for node in program.graph.nodes
            if node.target == torch.ops.aten.cat.default
        )
        left, right = concat.args[0]
        self.assertEqual(left.meta[QPARAM_KEY].scale, [0.125])
        self.assertEqual(right.meta[QPARAM_KEY].scale, [0.25])
        self.assertNotIn(QPARAM_KEY, concat.meta)


if __name__ == "__main__":
    unittest.main()

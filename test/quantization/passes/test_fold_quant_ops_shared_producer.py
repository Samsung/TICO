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

"""Shared producers own a domain; distinct consumer domains need requantization."""

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


class _SharedResidual(torch.nn.Module):
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


class _SharedQuantizedDomains(_SharedResidual):
    def __init__(self):
        super().__init__(quantized_weight=True)

    def forward(self, value):
        residual, first = super().forward(value)
        second = torch.fake_quantize_per_tensor_affine(residual, 0.5, 128, 0, 255)
        return first, second


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
    def test_first_domain_is_folded_into_shared_producer(self):
        program = torch.export.export(
            _SharedResidual().eval(), (torch.tensor([0.25, 0.75]),)
        )
        DecomposeFakeQuantize().call(program)
        FoldQuantOps().call(program)
        residual = next(
            node
            for node in program.graph.nodes
            if node.target == torch.ops.aten.add.Tensor
        )
        self.assertEqual(residual.meta[QPARAM_KEY].scale, [0.25])
        self.assertEqual(residual.meta[QPARAM_KEY].zero_point, [128])
        self.assertEqual(residual.meta[QPARAM_KEY].dtype, "uint8")
        self.assertEqual(residual.args[0].meta[QPARAM_KEY].scale, [0.125])
        output = next(node for node in program.graph.nodes if node.op == "output")
        self.assertEqual(output.args[0], (residual, residual))
        self.assertFalse(
            any(
                node.target
                in {
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                }
                for node in program.graph.nodes
            )
        )
        FoldQuantOps().call(program)
        self.assertEqual(residual.meta[QPARAM_KEY].scale, [0.25])

    def test_shared_quantized_domains_keep_requantization(self):
        module = _SharedQuantizedDomains().eval()
        inputs = (torch.tensor([0.25, 0.75]),)
        program = torch.export.export(module, inputs)
        DecomposeFakeQuantize().call(program)
        FoldQuantOps().call(program)
        RemoveWeightDequantOp().call(program)
        residual = next(
            node
            for node in program.graph.nodes
            if node.target == torch.ops.aten.add.Tensor
        )
        quantizers = [
            node
            for node in program.graph.nodes
            if node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
        ]
        self.assertEqual(len(quantizers), 1)
        quantize = quantizers[0]
        self.assertIs(quantize.args[0], residual)
        self.assertEqual(residual.meta[QPARAM_KEY].scale, [0.25])
        self.assertEqual(quantize.meta[QPARAM_KEY].scale, [0.5])
        output = next(node for node in program.graph.nodes if node.op == "output")
        self.assertEqual(output.args[0], (residual, quantize))
        self.assertFalse(
            any(
                node.target
                == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                for node in program.graph.nodes
            )
        )

        converted = tico.convert(module, inputs)
        document = CircleDocument.from_bytes(converted.circle_binary)
        self.assertTrue(document.verify().ok)
        runtime = CircleReferenceRuntime(document)
        self.assertEqual(
            [op.name for op in runtime.program.operators], ["ADD", "QUANTIZE"]
        )
        add, requantize = runtime.program.operators
        self.assertEqual(requantize.inputs, add.outputs)
        source = runtime.program.tensor(requantize.inputs[0])
        target = runtime.program.tensor(requantize.outputs[0])
        assert source.quantization is not None
        assert target.quantization is not None
        self.assertEqual(source.quantization.scale, (0.25,))
        self.assertEqual(target.quantization.scale, (0.5,))
        for index in (*add.inputs, *add.outputs, *requantize.outputs):
            tensor = runtime.program.tensor(index)
            self.assertIsNotNone(tensor.quantization)
            self.assertNotEqual(tensor.dtype, np.float32)

    def test_weight_domain_is_kept_without_dequantize_for_shared_producer(self):
        for per_channel in (False, True):
            with self.subTest(per_channel=per_channel):
                program = torch.export.export(
                    _SharedResidual(
                        quantized_weight=True, per_channel=per_channel
                    ).eval(),
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
                weight = residual.args[1]
                self.assertEqual(weight.op, "placeholder")
                self.assertEqual(weight.meta[QPARAM_KEY].dtype, "int8")
                self.assertEqual(
                    weight.meta[QPARAM_KEY].scale,
                    [0.0625, 0.1875] if per_channel else [0.0625],
                )
                self.assertEqual(
                    weight.meta[QPARAM_KEY].quantized_dimension,
                    0 if per_channel else None,
                )
                RemoveWeightDequantOp().call(program)
                self.assertIs(residual.args[1], weight)

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

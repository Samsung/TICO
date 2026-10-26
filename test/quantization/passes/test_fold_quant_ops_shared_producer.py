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

import io
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


class _ExternalFloatInput(torch.nn.Module):
    def __init__(self, *, preserve_float: bool):
        super().__init__()
        self.preserve_float = preserve_float

    def forward(self, value):
        if self.preserve_float:
            with (
                torch.fx.traceback.preserve_node_meta(),
                torch.fx.traceback.annotate({"tico_preserve_float_input": True}),
            ):
                return torch.fake_quantize_per_tensor_affine(
                    value, 0.125, 0, -32768, 32767
                )
        return torch.fake_quantize_per_tensor_affine(value, 0.125, 0, -32768, 32767)


class FoldQuantOpsSharedProducerTest(unittest.TestCase):
    def test_marked_single_user_float_input_retains_explicit_quantize(self):
        for strict in (True, False):
            for preserve_float in (True, False):
                with self.subTest(strict=strict, preserve_float=preserve_float):
                    program = torch.export.export(
                        _ExternalFloatInput(preserve_float=preserve_float).eval(),
                        (torch.tensor([0.22, -0.44]),),
                        strict=strict,
                    )
                    DecomposeFakeQuantize().call(program)
                    dequantize = next(
                        node
                        for node in program.graph.nodes
                        if node.target
                        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                    )
                    self.assertEqual(
                        dequantize.meta.get("custom", {}).get(
                            "tico_preserve_float_input", False
                        ),
                        preserve_float,
                    )
                    FoldQuantOps().call(program)
                    source = next(
                        node for node in program.graph.nodes if node.op == "placeholder"
                    )
                    quantize = [
                        node
                        for node in program.graph.nodes
                        if node.target
                        == torch.ops.quantized_decomposed.quantize_per_tensor.default
                    ]
                    if preserve_float:
                        self.assertNotIn(QPARAM_KEY, source.meta)
                        self.assertEqual(source.meta["val"].dtype, torch.float32)
                        self.assertEqual(len(quantize), 1)
                        self.assertIs(quantize[0].args[0], source)
                        self.assertEqual(quantize[0].meta[QPARAM_KEY].scale, [0.125])
                        FoldQuantOps().call(program)
                        self.assertIs(quantize[0].args[0], source)
                    else:
                        self.assertEqual(source.meta[QPARAM_KEY].scale, [0.125])
                        self.assertEqual(source.meta[QPARAM_KEY].dtype, "int16")
                        self.assertEqual(quantize, [])

    def test_public_conversion_preserves_declared_float_input_and_cpu_values(self):
        inputs = np.array([0.22, -0.44], dtype=np.float32)
        expected_float = np.array([0.25, -0.5], dtype=np.float32)
        expected_integer = np.array([2, -4], dtype=np.int16)
        for strict in (True, False):
            for preserve_float in (True, False):
                with self.subTest(strict=strict, preserve_float=preserve_float):
                    module = _ExternalFloatInput(preserve_float=preserve_float).eval()
                    exported = torch.export.export(
                        module, (torch.from_numpy(inputs),), strict=strict
                    )
                    checkpoint = io.BytesIO()
                    torch.export.save(exported, checkpoint)
                    checkpoint.seek(0)
                    restored = torch.export.load(checkpoint)
                    marked = [
                        node
                        for node in restored.graph.nodes
                        if node.meta.get("custom", {}).get("tico_preserve_float_input")
                    ]
                    self.assertEqual(bool(marked), preserve_float)
                    converted = tico.convert_from_exported_program(restored)
                    document = CircleDocument.from_bytes(converted.circle_binary)
                    self.assertTrue(document.verify().ok)
                    runtime = CircleReferenceRuntime(document)
                    source = runtime.program.tensor(runtime.program.inputs[0])
                    target = runtime.program.tensor(runtime.program.outputs[0])
                    self.assertEqual(target.dtype, np.int16)
                    self.assertIsNotNone(target.quantization)
                    assert target.quantization is not None
                    self.assertEqual(target.quantization.scale, (0.125,))
                    self.assertEqual(target.quantization.zero_point, (0,))
                    if preserve_float:
                        self.assertEqual(source.dtype, np.float32)
                        self.assertIsNone(source.quantization)
                        self.assertEqual(
                            [op.name for op in runtime.program.operators], ["QUANTIZE"]
                        )
                        bridge = runtime.program.operators[0]
                        self.assertEqual(bridge.inputs, runtime.program.inputs)
                        self.assertEqual(bridge.outputs, runtime.program.outputs)
                    else:
                        self.assertEqual(source.dtype, np.int16)
                        self.assertIsNotNone(source.quantization)
                        self.assertEqual(runtime.program.operators, ())
                    actual = runtime.run(
                        [inputs], mode=ExecutionMode.FAKE_QUANTIZE
                    ).outputs[0]
                    np.testing.assert_array_equal(actual, expected_float)
                    np.testing.assert_array_equal(
                        actual, module(torch.from_numpy(inputs)).numpy()
                    )
                    native_input = inputs if preserve_float else expected_integer
                    native = runtime.run([native_input]).outputs[0]
                    np.testing.assert_array_equal(native, expected_integer)

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

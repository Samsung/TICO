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

"""Public conversion and serialized boundaries, separate from MX runtime support."""

import unittest

import numpy as np

import tico
import torch
from circle_schema import circle
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, UnsupportedCircleOperatorError
from tico.utils.errors import NotYetSupportedError
from tico.utils.mx.mx_ops import quantize_mx

_TYPES = circle.TensorType.TensorType
_OPS = circle.BuiltinOperator.BuiltinOperator


def _affine(value, scale=0.25, zero_point=3):
    return torch.fake_quantize_per_tensor_affine(
        value, scale, zero_point, -32768, 32767
    )


class _MixedCat(torch.nn.Module):
    def __init__(self, output_mx=False):
        super().__init__()
        self.output_mx = output_mx

    def forward(self, left, right):
        left = quantize_mx(left, "int8", -1)
        right = _affine(right)
        result = torch.cat((left, right), dim=-1)
        if self.output_mx:
            result = quantize_mx(result, "int8", -1)
        else:
            result = _affine(result, 0.03125, -5)
        return result, left, right


class _AffineMxSigmoid(torch.nn.Module):
    def forward(self, value):
        original = torch.fake_quantize_per_tensor_affine(value, 0.3, 127, 0, 255)
        source = quantize_mx(original, "int8", -1)
        result = torch.fake_quantize_per_tensor_affine(
            torch.sigmoid(source), 1 / 256, 0, 0, 255
        )
        return result, source, original


class _SplitBoundaries(torch.nn.Module):
    def forward(self, value):
        source = quantize_mx(value, "int8", -1)
        parts = torch.split(source, [32, 32, 32], dim=-1)
        # The middle result is unused but must retain its native split domain.
        return _affine(parts[0], 0.125, 5), parts[-1], parts[0], source


class _MixedMul(torch.nn.Module):
    def forward(self, left, right):
        left = _affine(left, 0.25, 7)
        right = quantize_mx(right, "int8", -1)
        return _affine(left * right, 0.125, -3), left, right


class _MixedMulLayout(torch.nn.Module):
    def forward(self, left, right):
        left = _affine(left, 0.25, 7)
        right = quantize_mx(right, "int8", -1)
        native = quantize_mx(left * right, "int8", -1)
        return _affine(native.reshape(1, 2, 16), 0.125, -3), left, right


class _MxLinear(torch.nn.Module):
    weight: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor

    def __init__(self, with_bias):
        super().__init__()
        self.register_buffer("weight", torch.linspace(-2, 3, 128).reshape(4, 32))
        self.register_buffer("scales", torch.tensor([0.03125, 0.0625, 0.125, 0.25]))
        self.register_buffer("zero_points", torch.tensor([127] * 4, dtype=torch.int32))
        self.bias = (
            torch.nn.Parameter(torch.tensor([0.125, -0.25, 0.5, 1.0]))
            if with_bias
            else None
        )

    def forward(self, value):
        source = quantize_mx(value, "int8", -1)
        weight = torch.fake_quantize_per_channel_affine(
            self.weight, self.scales, self.zero_points, 0, 0, 255
        )
        return (
            _affine(torch.nn.functional.linear(source, weight, self.bias), 0.125, -3),
            source,
        )


class _MixedAddSub(torch.nn.Module):
    def __init__(self, operation, output_mx):
        super().__init__()
        self.operation = operation
        self.output_mx = output_mx

    def forward(self, left, right):
        left = _affine(left, 0.25, 7)
        right = quantize_mx(right, "int8", -1)
        if self.operation == "add":
            result = torch.add(left, right, alpha=1)
        else:
            result = torch.sub(left, right, alpha=1)
        result = (
            quantize_mx(result, "int8", -1)
            if self.output_mx
            else _affine(result, 0.125, -3)
        )
        return result, left, right


class _RepeatedBmm(torch.nn.Module):
    def __init__(self, output_mx):
        super().__init__()
        self.output_mx = output_mx

    def forward(self, value):
        source = quantize_mx(value, "int8", -1)
        result = torch.bmm(source, source)
        result = (
            quantize_mx(result, "int8", -1)
            if self.output_mx
            else _affine(result, 0.125, -3)
        )
        return result, source


class _CrossRelu(torch.nn.Module):
    def __init__(self, output_mx):
        super().__init__()
        self.output_mx = output_mx

    def forward(self, value):
        source = (
            _affine(value, 0.25, 7)
            if self.output_mx
            else quantize_mx(value, "int8", -1)
        )
        result = torch.relu(source)
        result = (
            quantize_mx(result, "int8", -1)
            if self.output_mx
            else _affine(result, 0.125, -3)
        )
        return result, source


class _CrossRMSNorm(torch.nn.Module):
    gamma: torch.Tensor

    def __init__(self, custom, output_mx):
        super().__init__()
        self.custom = custom
        self.output_mx = output_mx
        self.register_buffer("gamma", torch.linspace(0.25, 2, 32))

    def forward(self, value):
        source = (
            _affine(value, 0.25, 7)
            if self.output_mx
            else quantize_mx(value, "int8", -1)
        )
        gamma = _affine(self.gamma, 0.03125, 3)
        if self.custom:
            result = torch.ops.circle_custom.rms_norm(source, gamma, 0.0001)
        else:
            result = torch.nn.functional.rms_norm(source, (32,), gamma, 0.0001)
        result = (
            quantize_mx(result, "int8", -1)
            if self.output_mx
            else _affine(result, 0.125, -3)
        )
        return result, source, gamma


class _NarrowOutput(torch.nn.Module):
    def forward(self, value):
        source = quantize_mx(value, "int8", -1)
        result = torch.add(source, source, alpha=1)
        return (
            torch.fake_quantize_per_tensor_affine(result, 0.125, 0, -32767, 32767),
            source,
        )


class _WithIndependentClamp(torch.nn.Module):
    def __init__(self, include_clamp):
        super().__init__()
        self.include_clamp = include_clamp

    def forward(self, x, y):
        affine_source = _affine(x, 0.125, 0)
        mx_source = quantize_mx(y, "int8", -1)
        first = quantize_mx(affine_source * mx_source, "int8", -1)
        second = quantize_mx(first.relu(), "int8", -1)
        if self.include_clamp:
            other = _affine(torch.clamp(affine_source, -0.22, 0.88), 0.125, 0)
            return second, other
        return second


class _MultipleOutputBoundaries(torch.nn.Module):
    def __init__(self, mx_first, include_mx=True):
        super().__init__()
        self.mx_first = mx_first
        self.include_mx = include_mx

    def forward(self, x):
        source = _affine(x, 0.125, 0)
        raw = source.sigmoid()
        if self.mx_first:
            first = (
                quantize_mx(raw, "int8", -1)
                if self.include_mx
                else _affine(raw, 0.007, 0)
            )
            affine_result = _affine(raw, 0.003, 0)
        else:
            affine_result = _affine(raw, 0.003, 0)
            first = (
                quantize_mx(raw, "int8", -1)
                if self.include_mx
                else _affine(raw, 0.007, 0)
            )
        return first, affine_result


class _SingleMxOutputBoundary(torch.nn.Module):
    def __init__(self, shared_raw=False):
        super().__init__()
        self.shared_raw = shared_raw

    def forward(self, x):
        raw = _affine(x, 0.125, 0).sigmoid()
        result = quantize_mx(raw, "int8", -1)
        return (result, raw) if self.shared_raw else result


def _operators(document, code):
    return [
        (index, op)
        for index, op in enumerate(document.model.subgraphs[0].operators)
        if document.model.operatorCodes[op.opcodeIndex].builtinCode == code
    ]


def _export(module, values, *, optimize, move_layout, enabled=True):
    converted = tico.convert(
        module.eval(),
        values,
        config=tico.CompileConfigV1(
            legalize_mx_affine_domains=enabled,
            move_affine_quantize_before_layout=move_layout,
            circle_optimize=optimize,
        ),
    )
    document = CircleDocument.from_bytes(converted.circle_binary)
    document.verify(raise_on_error=True)
    return document


class LegalizeMxAffineDomainsExportTest(unittest.TestCase):
    def _assert_affine(self, tensor, scale, zero_point, dtype=_TYPES.INT16):
        self.assertEqual(tensor.type, dtype)
        np.testing.assert_array_equal(tensor.quantization.scale, [scale])
        np.testing.assert_array_equal(tensor.quantization.zeroPoint, [zero_point])

    def _assert_mx(self, tensor, axis=1):
        self.assertEqual(tensor.type, _TYPES.MXINT8)
        self.assertEqual(tensor.quantization.quantizedDimension, axis)
        self.assertIsNone(tensor.quantization.scale)
        self.assertIsNone(tensor.quantization.zeroPoint)

    def _assert_output(self, graph, output_mx, *, axis=1):
        if output_mx:
            self._assert_mx(graph.tensors[graph.outputs[0]], axis)
        else:
            self._assert_affine(graph.tensors[graph.outputs[0]], 0.125, -3)

    def test_add_sub_broadcast_preserves_all_operand_domains(self):
        left = torch.linspace(-17, 13, 64).reshape(2, 32)
        right = torch.linspace(-3, 5, 32).reshape(1, 32)
        for operation, code in (("add", _OPS.ADD), ("sub", _OPS.SUB)):
            for output_mx in (False, True):
                for optimize in (False, True):
                    with self.subTest(
                        operation=operation, output_mx=output_mx, optimize=optimize
                    ):
                        document = _export(
                            _MixedAddSub(operation, output_mx),
                            (left, right),
                            optimize=optimize,
                            move_layout=True,
                        )
                        graph = document.model.subgraphs[0]
                        operations = _operators(document, code)
                        self.assertEqual(len(operations), 1)
                        operation_op = operations[0][1]
                        for index in (*operation_op.inputs, *operation_op.outputs):
                            self._assert_mx(graph.tensors[index])
                        self.assertEqual(
                            list(graph.tensors[operation_op.inputs[0]].shape), [2, 32]
                        )
                        self.assertEqual(
                            list(graph.tensors[operation_op.inputs[1]].shape), [1, 32]
                        )
                        self._assert_affine(graph.tensors[graph.outputs[1]], 0.25, 7)
                        self._assert_mx(graph.tensors[graph.outputs[2]])
                        self._assert_output(graph, output_mx)
                        self.assertEqual(
                            len(_operators(document, _OPS.QUANTIZE)),
                            1 if output_mx else 2,
                        )

    def test_bmm_same_source_gets_independent_operand_axes(self):
        value = torch.linspace(-17, 13, 1024).reshape(1, 32, 32)
        for output_mx in (False, True):
            for optimize in (False, True):
                with self.subTest(output_mx=output_mx, optimize=optimize):
                    document = _export(
                        _RepeatedBmm(output_mx),
                        (value,),
                        optimize=optimize,
                        move_layout=True,
                    )
                    graph = document.model.subgraphs[0]
                    operations = _operators(document, _OPS.BATCH_MATMUL)
                    self.assertEqual(len(operations), 1)
                    bmm = operations[0][1]
                    self._assert_mx(graph.tensors[bmm.inputs[0]], 2)
                    self._assert_mx(graph.tensors[bmm.inputs[1]], 1)
                    self._assert_mx(graph.tensors[bmm.outputs[0]], 2)
                    self._assert_mx(graph.tensors[graph.outputs[1]], 2)
                    self.assertEqual(bmm.inputs[0], graph.outputs[1])
                    self.assertNotEqual(bmm.inputs[1], graph.outputs[1])
                    conversions = [op for _, op in _operators(document, _OPS.QUANTIZE)]
                    axis_adapter = next(
                        op for op in conversions if op.outputs[0] == bmm.inputs[1]
                    )
                    self.assertEqual(axis_adapter.inputs[0], graph.outputs[1])
                    self.assertEqual(len(conversions), 1 if output_mx else 2)
                    self._assert_output(graph, output_mx, axis=2)
                    self.assertFalse(bmm.builtinOptions.adjointLhs)
                    self.assertFalse(bmm.builtinOptions.adjointRhs)

    def test_relu_and_both_rms_norm_paths_keep_native_and_gamma_roles(self):
        value = torch.linspace(-131, 137, 64).reshape(2, 32)
        for kind in ("relu", "aten_rms_norm", "custom_rms_norm"):
            for output_mx in (False, True):
                for optimize in (False, True):
                    with self.subTest(
                        kind=kind, output_mx=output_mx, optimize=optimize
                    ):
                        rms = kind != "relu"
                        module = (
                            _CrossRMSNorm(kind == "custom_rms_norm", output_mx)
                            if rms
                            else _CrossRelu(output_mx)
                        )
                        document = _export(
                            module, (value,), optimize=optimize, move_layout=True
                        )
                        graph = document.model.subgraphs[0]
                        operations = _operators(
                            document, _OPS.RMS_NORM if rms else _OPS.RELU
                        )
                        self.assertEqual(len(operations), 1)
                        operation_op = operations[0][1]
                        self._assert_mx(graph.tensors[operation_op.inputs[0]])
                        self._assert_mx(graph.tensors[operation_op.outputs[0]])
                        self._assert_output(graph, output_mx)
                        if output_mx:
                            self._assert_affine(
                                graph.tensors[graph.outputs[1]], 0.25, 7
                            )
                        else:
                            self._assert_mx(graph.tensors[graph.outputs[1]])
                        if rms:
                            gamma = graph.tensors[operation_op.inputs[1]]
                            self._assert_affine(gamma, 0.03125, 3)
                            self.assertEqual(list(gamma.shape), [32])
                            self.assertEqual(operation_op.inputs[1], graph.outputs[2])
                            self.assertEqual(
                                operation_op.builtinOptions.epsilon, np.float32(0.0001)
                            )
                        self.assertEqual(len(_operators(document, _OPS.QUANTIZE)), 1)

    def test_narrow_affine_output_keeps_explicit_code_clamp(self):
        value = torch.linspace(-10000, 3, 32).reshape(1, 32)
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                document = _export(
                    _NarrowOutput(),
                    (value,),
                    optimize=optimize,
                    move_layout=True,
                )
                graph = document.model.subgraphs[0]
                additions = _operators(document, _OPS.ADD)
                self.assertEqual(len(additions), 1)
                native_output = additions[0][1].outputs[0]
                self._assert_mx(graph.tensors[native_output])
                quantizers = _operators(document, _OPS.QUANTIZE)
                self.assertEqual(len(quantizers), 1)
                conversion = quantizers[0][1]
                self.assertEqual(conversion.inputs[0], native_output)
                maximums = _operators(document, _OPS.MAXIMUM)
                self.assertEqual(len(maximums), 1)
                maximum = maximums[0][1]
                self.assertEqual(maximum.inputs[0], conversion.outputs[0])
                self.assertEqual(maximum.outputs[0], graph.outputs[0])
                for index in (*maximum.inputs, *maximum.outputs):
                    self._assert_affine(graph.tensors[index], 0.125, 0)
                bound = graph.tensors[maximum.inputs[1]]
                codes = np.frombuffer(
                    document.model.buffers[bound.buffer].data, dtype=np.int16
                )
                np.testing.assert_array_equal(codes, [-32767])

    def test_mixed_cat_preserves_source_branches_and_destination_contract(self):
        left = torch.linspace(-17, 13, 32).reshape(1, 32)
        right = torch.linspace(-3, 5, 32).reshape(1, 32)
        for output_mx in (False, True):
            for optimize in (False, True):
                for move_layout in (False, True):
                    with self.subTest(
                        output_mx=output_mx,
                        optimize=optimize,
                        move_layout=move_layout,
                    ):
                        document = _export(
                            _MixedCat(output_mx),
                            (left, right),
                            optimize=optimize,
                            move_layout=move_layout,
                        )
                        graph = document.model.subgraphs[0]
                        self.assertEqual(len(graph.inputs), 2)
                        self.assertEqual(len(graph.outputs), 3)
                        self._assert_mx(graph.tensors[graph.outputs[1]])
                        self._assert_affine(graph.tensors[graph.outputs[2]], 0.25, 3)
                        cats = _operators(document, _OPS.CONCATENATION)
                        self.assertEqual(len(cats), 1)
                        cat = cats[0][1]
                        for index in (*cat.inputs, *cat.outputs):
                            tensor = graph.tensors[index]
                            if output_mx:
                                self._assert_mx(tensor)
                            else:
                                self._assert_affine(tensor, 0.03125, -5)
                        self.assertEqual(cat.outputs[0], graph.outputs[0])
                        conversions = [
                            op for _, op in _operators(document, _OPS.QUANTIZE)
                        ]
                        self.assertEqual(len(conversions), 1 if output_mx else 2)
                        # The existing source boundaries remain visible at graph outputs.
                        # Only CAT's operand edges use the destination adapters.
                        for conversion in conversions:
                            self.assertIn(conversion.inputs[0], graph.outputs[1:])
                            self.assertIn(conversion.outputs[0], cat.inputs)

    def test_sigmoid_keeps_original_input_and_native_output_quantization(self):
        value = torch.linspace(-3, 3, 32).reshape(1, 32)
        value[0, 0] = 38.1
        source = quantize_mx(
            torch.fake_quantize_per_tensor_affine(value, 0.3, 127, 0, 255),
            "int8",
            -1,
        )
        expected = torch.fake_quantize_per_tensor_affine(
            quantize_mx(torch.sigmoid(source), "int8", -1), 1 / 256, 0, 0, 255
        )
        eager = _AffineMxSigmoid().eval()(value)[0]
        self.assertFalse(torch.equal(expected, eager))
        for optimize in (False, True):
            for move_layout in (False, True):
                with self.subTest(optimize=optimize, move_layout=move_layout):
                    document = _export(
                        _AffineMxSigmoid(),
                        (value,),
                        optimize=optimize,
                        move_layout=move_layout,
                    )
                    graph = document.model.subgraphs[0]
                    self._assert_affine(
                        graph.tensors[graph.outputs[0]], 1 / 256, 0, _TYPES.UINT8
                    )
                    self._assert_mx(graph.tensors[graph.outputs[1]])
                    self._assert_affine(
                        graph.tensors[graph.outputs[2]],
                        np.float32(0.3),
                        127,
                        _TYPES.UINT8,
                    )
                    quantizers = [op for _, op in _operators(document, _OPS.QUANTIZE)]
                    self.assertEqual(len(quantizers), 2)
                    from_mx = next(
                        op
                        for op in quantizers
                        if graph.tensors[op.inputs[0]].type == _TYPES.MXINT8
                    )
                    self._assert_affine(
                        graph.tensors[from_mx.outputs[0]],
                        1 / 256,
                        0,
                        _TYPES.UINT8,
                    )
                    logistic = _operators(document, _OPS.LOGISTIC)
                    self.assertEqual(len(logistic), 1)
                    self.assertEqual(logistic[0][1].inputs[0], graph.outputs[1])
                    self._assert_mx(graph.tensors[logistic[0][1].outputs[0]])
                    self.assertEqual(logistic[0][1].outputs[0], from_mx.inputs[0])
                    self.assertEqual(from_mx.outputs[0], graph.outputs[0])
                    with self.assertRaisesRegex(
                        UnsupportedCircleOperatorError, "tensor type -7.*no value codec"
                    ):
                        CircleReferenceRuntime(document)

    def test_mixed_mul_keeps_native_mx_output_rounding_and_affine_output_grid(self):
        left = torch.linspace(-11, 17, 32).reshape(1, 32)
        right = torch.linspace(-5, 13, 32).reshape(1, 32)
        for optimize in (False, True):
            for move_layout in (False, True):
                with self.subTest(optimize=optimize, move_layout=move_layout):
                    document = _export(
                        _MixedMul(),
                        (left, right),
                        optimize=optimize,
                        move_layout=move_layout,
                    )
                    graph = document.model.subgraphs[0]
                    self._assert_affine(graph.tensors[graph.outputs[0]], 0.125, -3)
                    self._assert_affine(graph.tensors[graph.outputs[1]], 0.25, 7)
                    self._assert_mx(graph.tensors[graph.outputs[2]])
                    multiplies = _operators(document, _OPS.MUL)
                    self.assertEqual(len(multiplies), 1)
                    multiply = multiplies[0][1]
                    for index in (*multiply.inputs, *multiply.outputs):
                        self._assert_mx(graph.tensors[index])
                    quantizers = [op for _, op in _operators(document, _OPS.QUANTIZE)]
                    self.assertEqual(len(quantizers), 2)
                    into_mx = next(
                        op
                        for op in quantizers
                        if graph.tensors[op.outputs[0]].type == _TYPES.MXINT8
                    )
                    self.assertEqual(into_mx.inputs[0], graph.outputs[1])
                    self.assertIn(into_mx.outputs[0], multiply.inputs)
                    from_mx = next(op for op in quantizers if op is not into_mx)
                    self.assertEqual(from_mx.inputs[0], multiply.outputs[0])
                    self.assertEqual(from_mx.outputs[0], graph.outputs[0])

    def test_disabled_policy_keeps_existing_cross_domain_sigmoid(self):
        value = torch.linspace(-3, 3, 32).reshape(1, 32)
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                document = _export(
                    _AffineMxSigmoid(),
                    (value,),
                    optimize=optimize,
                    move_layout=True,
                    enabled=False,
                )
                graph = document.model.subgraphs[0]
                logistic = _operators(document, _OPS.LOGISTIC)
                self.assertEqual(len(logistic), 1)
                self._assert_mx(graph.tensors[logistic[0][1].inputs[0]])
                self._assert_affine(
                    graph.tensors[logistic[0][1].outputs[0]],
                    1 / 256,
                    0,
                    _TYPES.UINT8,
                )
                self.assertEqual(len(_operators(document, _OPS.QUANTIZE)), 1)

    def test_existing_layout_policy_precedes_domain_legalization(self):
        left = torch.linspace(-11, 17, 32).reshape(1, 32)
        right = torch.linspace(-5, 13, 32).reshape(1, 32)
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                document = _export(
                    _MixedMulLayout(),
                    (left, right),
                    optimize=optimize,
                    move_layout=True,
                )
                graph = document.model.subgraphs[0]
                reshape = _operators(document, _OPS.RESHAPE)
                self.assertEqual(len(reshape), 1)
                for index in (reshape[0][1].inputs[0], reshape[0][1].outputs[0]):
                    self._assert_affine(graph.tensors[index], 0.125, -3)
                multiply = _operators(document, _OPS.MUL)
                self.assertEqual(len(multiply), 1)
                for index in (*multiply[0][1].inputs, *multiply[0][1].outputs):
                    self._assert_mx(graph.tensors[index])
                with self.assertRaisesRegex(
                    NotYetSupportedError, "MX.*reshape|reshape.*MX"
                ):
                    _export(
                        _MixedMulLayout(),
                        (left, right),
                        optimize=optimize,
                        move_layout=False,
                    )

    def test_linear_output_adapter_survives_o1_and_preserves_weight_and_bias_roles(
        self,
    ):
        value = torch.linspace(-3, 3, 32).reshape(1, 32)
        for with_bias in (False, True):
            for optimize in (False, True):
                with self.subTest(with_bias=with_bias, optimize=optimize):
                    document = _export(
                        _MxLinear(with_bias),
                        (value,),
                        optimize=optimize,
                        move_layout=True,
                    )
                    graph = document.model.subgraphs[0]
                    linear = _operators(document, _OPS.FULLY_CONNECTED)
                    self.assertEqual(len(linear), 1)
                    linear = linear[0][1]
                    self._assert_mx(graph.tensors[linear.inputs[0]])
                    self._assert_mx(graph.tensors[linear.outputs[0]])
                    weight = graph.tensors[linear.inputs[1]]
                    self.assertEqual(weight.type, _TYPES.UINT8)
                    self.assertEqual(weight.quantization.quantizedDimension, 0)
                    np.testing.assert_array_equal(
                        weight.quantization.scale, [0.03125, 0.0625, 0.125, 0.25]
                    )
                    np.testing.assert_array_equal(
                        weight.quantization.zeroPoint, [127] * 4
                    )
                    if with_bias:
                        bias = graph.tensors[linear.inputs[2]]
                        self.assertEqual(bias.type, _TYPES.FLOAT32)
                        self.assertIsNone(bias.quantization)
                    else:
                        self.assertTrue(
                            len(linear.inputs) == 2 or linear.inputs[2] == -1
                        )
                    quantizers = _operators(document, _OPS.QUANTIZE)
                    self.assertEqual(len(quantizers), 1)
                    self.assertEqual(quantizers[0][1].inputs[0], linear.outputs[0])
                    self.assertEqual(quantizers[0][1].outputs[0], graph.outputs[0])
                    self._assert_affine(graph.tensors[graph.outputs[0]], 0.125, -3)

    def test_split_preserves_each_native_slot_and_only_converts_selected_result(self):
        value = torch.linspace(-31, 17, 96).reshape(1, 96)
        for optimize in (False, True):
            for move_layout in (False, True):
                with self.subTest(optimize=optimize, move_layout=move_layout):
                    document = _export(
                        _SplitBoundaries(),
                        (value,),
                        optimize=optimize,
                        move_layout=move_layout,
                    )
                    graph = document.model.subgraphs[0]
                    splits = _operators(document, _OPS.SPLIT_V) + _operators(
                        document, _OPS.SPLIT
                    )
                    self.assertEqual(len(splits), 1)
                    split = splits[0][1]
                    self.assertEqual(len(split.outputs), 3)
                    for index in split.outputs:
                        self._assert_mx(graph.tensors[index])
                    self.assertEqual(split.outputs[0], graph.outputs[2])
                    self.assertEqual(split.outputs[2], graph.outputs[1])
                    quantizers = _operators(document, _OPS.QUANTIZE)
                    self.assertEqual(len(quantizers), 1)
                    self.assertEqual(quantizers[0][1].inputs[0], split.outputs[0])
                    self.assertEqual(quantizers[0][1].outputs[0], graph.outputs[0])
                    self._assert_affine(graph.tensors[graph.outputs[0]], 0.125, 5)

    def test_independent_clamp_restart_preserves_mx_path_and_affine_output(self):
        x = torch.linspace(-4, 4, 64).reshape(1, 64)
        y = torch.linspace(0.01, 2, 64).reshape(1, 64)
        for include_clamp in (False, True):
            for optimize in (False, True):
                with self.subTest(include_clamp=include_clamp, optimize=optimize):
                    document = _export(
                        _WithIndependentClamp(include_clamp),
                        (x, y),
                        optimize=optimize,
                        move_layout=True,
                    )
                    graph = document.model.subgraphs[0]
                    self.assertEqual(len(graph.inputs), 2)
                    self.assertEqual(len(graph.outputs), 2 if include_clamp else 1)
                    self._assert_affine(graph.tensors[graph.inputs[0]], 0.125, 0)
                    self._assert_mx(graph.tensors[graph.inputs[1]])
                    self._assert_mx(graph.tensors[graph.outputs[0]])
                    multiplies = _operators(document, _OPS.MUL)
                    relus = _operators(document, _OPS.RELU)
                    self.assertEqual(len(multiplies), 1)
                    self.assertEqual(len(relus), 1)
                    multiply, relu = multiplies[0][1], relus[0][1]
                    for operation in (multiply, relu):
                        for index in (*operation.inputs, *operation.outputs):
                            self._assert_mx(graph.tensors[index])
                    self.assertEqual(multiply.outputs[0], relu.inputs[0])
                    self.assertEqual(relu.outputs[0], graph.outputs[0])
                    quantizers = _operators(document, _OPS.QUANTIZE)
                    self.assertEqual(len(quantizers), 1)
                    adapter = quantizers[0][1]
                    self.assertEqual(adapter.inputs[0], graph.inputs[0])
                    self.assertEqual(adapter.outputs[0], multiply.inputs[0])
                    self.assertEqual(multiply.inputs[1], graph.inputs[1])
                    minimums = _operators(document, _OPS.MINIMUM)
                    maximums = _operators(document, _OPS.MAXIMUM)
                    self.assertEqual(len(minimums), 1 if include_clamp else 0)
                    self.assertEqual(len(maximums), 1 if include_clamp else 0)
                    if include_clamp:
                        self._assert_affine(graph.tensors[graph.outputs[1]], 0.125, 0)
                        minimum, maximum = minimums[0][1], maximums[0][1]
                        self.assertEqual(minimum.inputs[0], graph.inputs[0])
                        self.assertEqual(minimum.outputs[0], maximum.inputs[0])
                        self.assertEqual(maximum.outputs[0], graph.outputs[1])
                        for operation in (minimum, maximum):
                            for index in (*operation.inputs, *operation.outputs):
                                self._assert_affine(graph.tensors[index], 0.125, 0)
                        # Bounds are rounded independently in the preserved .125 grid:
                        # -.22 -> code -2, .88 -> code 7.
                        for operation, expected_code in ((minimum, 7), (maximum, -2)):
                            bound = graph.tensors[operation.inputs[1]]
                            codes = np.frombuffer(
                                document.model.buffers[bound.buffer].data,
                                dtype=np.int16,
                            )
                            np.testing.assert_array_equal(codes, [expected_code])

    def test_multiple_output_boundaries_detect_mx_with_affine_only_input(self):
        value = torch.linspace(-4, 4, 64).reshape(1, 64)
        for mx_first in (False, True):
            for optimize in (False, True):
                with self.subTest(mx_first=mx_first, optimize=optimize):
                    with self.assertRaises(NotYetSupportedError) as caught:
                        _export(
                            _MultipleOutputBoundaries(mx_first),
                            (value,),
                            optimize=optimize,
                            move_layout=True,
                        )
                    message = str(caught.exception).lower()
                    self.assertIn("sigmoid", message)
                    self.assertIn("aten.sigmoid.default", message)
                    self.assertIn("multiple", message)
                    self.assertIn("output", message)
                    self.assertIn("boundar", message)
                    self.assertIn("native", message)

    def test_single_private_mx_output_boundary_still_legalizes_affine_input(self):
        value = torch.linspace(-4, 4, 64).reshape(1, 64)
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                document = _export(
                    _SingleMxOutputBoundary(),
                    (value,),
                    optimize=optimize,
                    move_layout=True,
                )
                graph = document.model.subgraphs[0]
                self.assertEqual(len(graph.inputs), 1)
                self.assertEqual(len(graph.outputs), 1)
                self._assert_affine(graph.tensors[graph.inputs[0]], 0.125, 0)
                logistic = _operators(document, _OPS.LOGISTIC)
                self.assertEqual(len(logistic), 1)
                operation = logistic[0][1]
                for index in (*operation.inputs, *operation.outputs):
                    self._assert_mx(graph.tensors[index])
                self.assertEqual(operation.outputs[0], graph.outputs[0])
                quantizers = _operators(document, _OPS.QUANTIZE)
                self.assertEqual(len(quantizers), 1)
                adapter = quantizers[0][1]
                self.assertEqual(adapter.inputs[0], graph.inputs[0])
                self.assertEqual(adapter.outputs[0], operation.inputs[0])

    def test_private_boundary_requirement_still_rejects_shared_raw_output(self):
        value = torch.linspace(-4, 4, 64).reshape(1, 64)
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                with self.assertRaises(NotYetSupportedError) as caught:
                    _export(
                        _SingleMxOutputBoundary(shared_raw=True),
                        (value,),
                        optimize=optimize,
                        move_layout=True,
                    )
                message = str(caught.exception).lower()
                self.assertIn("sigmoid", message)
                self.assertIn("shared", message)
                self.assertIn("output", message)
                self.assertIn("native", message)

    def test_affine_only_multiple_output_boundaries_remain_supported(self):
        value = torch.linspace(-4, 4, 64).reshape(1, 64)
        for mx_first in (False, True):
            for optimize in (False, True):
                with self.subTest(first_grid_first=mx_first, optimize=optimize):
                    document = _export(
                        _MultipleOutputBoundaries(mx_first, include_mx=False),
                        (value,),
                        optimize=optimize,
                        move_layout=True,
                    )
                    graph = document.model.subgraphs[0]
                    self.assertEqual(len(graph.inputs), 1)
                    self.assertEqual(len(graph.outputs), 2)
                    self._assert_affine(graph.tensors[graph.inputs[0]], 0.125, 0)
                    self._assert_affine(
                        graph.tensors[graph.outputs[0]], np.float32(0.007), 0
                    )
                    self._assert_affine(
                        graph.tensors[graph.outputs[1]], np.float32(0.003), 0
                    )
                    logistic = _operators(document, _OPS.LOGISTIC)
                    self.assertEqual(len(logistic), 1)
                    for index in (*logistic[0][1].inputs, *logistic[0][1].outputs):
                        self.assertEqual(graph.tensors[index].type, _TYPES.INT16)
                    self.assertFalse(
                        any(tensor.type == _TYPES.MXINT8 for tensor in graph.tensors)
                    )


if __name__ == "__main__":
    unittest.main()

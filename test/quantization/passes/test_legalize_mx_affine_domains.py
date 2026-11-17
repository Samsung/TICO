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

"""CPU Q/DQ policy tests; these do not assert native MX backend execution."""

import copy
import unittest

import numpy as np

import tico
import torch
from circle_schema import circle
from tico.circle import CircleDocument, extract_by_operator_indices
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode

from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.passes.fill_meta_val import FillMetaVal
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.quantization.passes.legalize_mx_affine_domains import (
    LegalizeMxAffineDomains,
    run_target_reference,
)
from tico.quantization.passes.quantize_bias import QuantizeBias
from tico.quantization.passes.remove_weight_dequant_op import (
    get_constant,
    RemoveWeightDequantOp,
)
from tico.serialize.quant_param import QPARAM_KEY
from tico.utils.errors import NotYetSupportedError
from tico.utils.mx.mx_ops import quantize_mx
from tico.utils.passes import PassManager

_Q = torch.ops.quantized_decomposed.quantize_per_tensor.default
_DQ = torch.ops.quantized_decomposed.dequantize_per_tensor.default
_MQ = torch.ops.circle_custom.quantize_mx.default
_MDQ = torch.ops.circle_custom.dequantize_mx.default


def _affine(x, scale=0.13, zero=7, qmin=-32768, qmax=32767):
    return torch.fake_quantize_per_tensor_affine(x, scale, zero, qmin, qmax)


def _grid(node):
    schema = node.target._schema
    return tuple(
        (
            node.args[i]
            if i < len(node.args)
            else node.kwargs.get(arg.name, arg.default_value)
        )
        for i, arg in enumerate(schema.arguments)
    )[1:]


def _nodes(ep, target):
    return [n for n in ep.graph.nodes if n.target == target]


def _export(module, *inputs):
    ep = torch.export.export(module.eval(), inputs)
    PassManager([DecomposeFakeQuantize(), FillMetaVal()]).run(ep)
    return ep


def _legalize(ep):
    # Match PassManager's graph-signature contract while preserving modified.
    with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
        return LegalizeMxAffineDomains().call(ep)


def _execute(ep, *inputs):
    # module() un-lifts parameters, retaining CPU execution for LINEAR/RMS weights.
    return run_target_reference(ep, *inputs)


class _Unary(torch.nn.Module):
    def __init__(self, kind="sigmoid", grid=True, shared=False):
        super().__init__()
        self.kind, self.grid, self.shared = kind, grid, shared

    def forward(self, x):
        source = _affine(x, 0.19, 127, 0, 255) if self.grid else x
        source = quantize_mx(source, "int8", -1)
        result = source.sigmoid() if self.kind == "sigmoid" else source.relu()
        result = _affine(result, 1 / 255, 0, 0, 255)
        if self.shared:
            return result, source, source.sin()
        return result


class _MultipleOutputBoundaries(torch.nn.Module):
    def __init__(self, mx_first, include_mx=True):
        super().__init__()
        self.mx_first, self.include_mx = mx_first, include_mx

    def forward(self, x):
        source = _affine(x, 0.125, 0)
        raw = source.sigmoid()
        if self.mx_first:
            first = (
                quantize_mx(raw, "int8", -1)
                if self.include_mx
                else _affine(raw, 0.005, 1)
            )
            second = _affine(raw, 0.003, 0)
        else:
            second = _affine(raw, 0.003, 0)
            first = (
                quantize_mx(raw, "int8", -1)
                if self.include_mx
                else _affine(raw, 0.005, 1)
            )
        return first, second


class _Binary(torch.nn.Module):
    def __init__(self, kind="mul", mx_input=True, mx_output=False, alpha=1):
        super().__init__()
        self.kind, self.mx_input, self.mx_output, self.alpha = (
            kind,
            mx_input,
            mx_output,
            alpha,
        )

    def forward(self, x, y):
        x = _affine(x)
        if self.mx_input:
            x = quantize_mx(x, "int8", -1)
        y = _affine(y, 0.07, -3)
        if self.kind == "mul":
            out = x * y
        elif self.kind == "add":
            out = torch.add(x, y, alpha=self.alpha)
        elif self.kind == "sub":
            out = torch.sub(x, y, alpha=self.alpha)
        else:
            out = torch.bmm(x, y)
        if self.mx_output:
            return quantize_mx(out, "int8", -1)
        return _affine(out, 0.007, 19)


class _RMS(torch.nn.Module):
    weight: torch.Tensor

    def __init__(self, custom=False):
        super().__init__()
        self.register_buffer("weight", torch.linspace(0.5, 1.7, 64))
        self.custom = custom

    def forward(self, x):
        x = quantize_mx(_affine(x), "int8", -1)
        gamma = _affine(self.weight, 0.01, 2)
        if self.custom:
            result = torch.ops.circle_custom.rms_norm.default(x, gamma, 1e-4)
        else:
            result = torch.nn.functional.rms_norm(x, [64], gamma, 1e-4)
        return _affine(result, 0.003, -9)


class _RepeatedBmm(torch.nn.Module):
    def __init__(self, output_axis=None):
        super().__init__()
        self.output_axis = output_axis

    def forward(self, x):
        source = quantize_mx(_affine(x), "int8", 2)
        result = torch.bmm(source, source)
        result = (
            _affine(result, 0.017, -3)
            if self.output_axis is None
            else quantize_mx(result, "int8", self.output_axis)
        )
        return result, source


class _Linear(torch.nn.Module):
    weight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bias: torch.Tensor | None

    def __init__(self, bias=True):
        super().__init__()
        self.register_buffer("weight", torch.linspace(-1.1, 1.3, 64 * 4).reshape(4, 64))
        self.register_buffer("scales", torch.tensor([0.01, 0.02, 0.03, 0.04]))
        self.register_buffer("zeros", torch.tensor([0, 1, 2, 3], dtype=torch.int32))
        self.register_buffer("bias", torch.linspace(-0.3, 0.7, 4) if bias else None)

    def forward(self, x):
        source = quantize_mx(_affine(x), "int8", -1)
        weight = torch.fake_quantize_per_channel_affine(
            self.weight, self.scales, self.zeros, 0, 0, 255
        )
        result = torch.nn.functional.linear(source, weight, self.bias)
        return _affine(result, 0.017, -3), weight, self.bias


class _Cat(torch.nn.Module):
    def __init__(self, mx_output=False, shared=False, axis=-1):
        super().__init__()
        self.mx_output, self.shared, self.axis = mx_output, shared, axis

    def forward(self, x, y):
        left = quantize_mx(_affine(x), "int8", self.axis)
        right = _affine(y, 0.07, -3)
        raw = torch.cat([left, right], dim=-1)
        result = (
            quantize_mx(raw, "int8", self.axis)
            if self.mx_output
            else _affine(raw, 0.017, 13)
        )
        return (result, raw) if self.shared else (result, left, right)


class _NativeAffineSharedCat(torch.nn.Module):
    def forward(self, x, y, z):
        source = torch.cat((_affine(x), _affine(y)), dim=-1)
        a = _affine(source * quantize_mx(z, "int8", -1), 0.017, -3)
        b = _affine(source + quantize_mx(z, "int8", -1), 0.031, 13)
        return a, b


class _RewrittenMixedSharedCat(torch.nn.Module):
    def forward(self, x, y, z):
        source = _affine(
            torch.cat((quantize_mx(x, "int8", -1), _affine(y)), dim=-1),
            0.017,
            -3,
        )
        a = quantize_mx(source * quantize_mx(z, "int8", -1), "int8", -1)
        b = quantize_mx(source + quantize_mx(z, "int8", -1), "int8", -1)
        return a, b


class _Split(torch.nn.Module):
    def forward(self, x):
        source = quantize_mx(x, "int8", -1)
        parts = torch.split(source, [32, 32, 32], dim=-1)
        return (
            _affine(parts[0], 0.01, 1),
            _affine(parts[-1], 0.02, -2),
            parts[0],
            source,
        )


class LegalizeMxAffineDomainsTest(unittest.TestCase):
    def _assert_idempotent(self, ep):
        graph = str(ep.graph)
        self.assertFalse(_legalize(ep).modified)
        self.assertEqual(str(ep.graph), graph)
        ep.graph.lint()
        ep.validate()

    def test_sigmoid_keeps_wide_input_and_rounds_native_output_before_affine(self):
        x = torch.linspace(-22.13, 22.17, 128).reshape(2, 64)
        ep = _export(_Unary(shared=True), x)
        original_quantizers = {n: (_grid(n), n.args[0]) for n in _nodes(ep, _Q)}
        source = _nodes(ep, _MDQ)[0]
        self.assertTrue(_legalize(ep).modified)
        sigmoid = _nodes(ep, torch.ops.aten.sigmoid.default)[0]
        self.assertIs(sigmoid.args[0], source)
        output_q = _nodes(ep, _Q)[-1]
        self.assertEqual(output_q.args[0].target, _MDQ)
        self.assertIs(output_q.args[0].args[0].args[0], sigmoid)
        for node, (grid, original_source) in original_quantizers.items():
            self.assertEqual(_grid(node), grid)
            if original_source is not sigmoid:
                self.assertIs(node.args[0], original_source)
        rounded = quantize_mx(_affine(x, 0.19, 127, 0, 255), "int8", -1)
        expected = _affine(
            quantize_mx(rounded.sigmoid(), "int8", -1), 1 / 255, 0, 0, 255
        )
        actual, actual_source, other = _execute(ep, x)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_source, rounded))
        self.assertTrue(torch.equal(other, rounded.sin()))
        # The sigmoid output grid would clip negative values and all values >1
        # if it were incorrectly reused at the input.
        incorrect = _affine(
            _affine(rounded, 1 / 255, 0, 0, 255).sigmoid(), 1 / 255, 0, 0, 255
        )
        self.assertFalse(torch.equal(actual, incorrect))
        self._assert_idempotent(ep)

    def test_sigmoid_does_not_invent_an_affine_input_grid(self):
        x = torch.linspace(-80, 80, 64).reshape(1, 64)
        ep = _export(_Unary(grid=False), x)
        source = _nodes(ep, _MDQ)[0]
        self.assertTrue(_legalize(ep).modified)
        sigmoid = _nodes(ep, torch.ops.aten.sigmoid.default)[0]
        self.assertIs(sigmoid.args[0], source)
        self.assertEqual(len(_nodes(ep, _Q)), 1)
        expected = _affine(
            quantize_mx(quantize_mx(x, "int8", -1).sigmoid(), "int8", -1),
            1 / 255,
            0,
            0,
            255,
        )
        self.assertTrue(torch.equal(_execute(ep, x), expected))
        self._assert_idempotent(ep)

    def test_arithmetic_preserves_distinct_grids_alpha_and_broadcast(self):
        x = torch.linspace(-120.31, 120.29, 128).reshape(2, 64)
        y = torch.linspace(-80.11, 80.13, 64).reshape(1, 64)
        for kind, alpha in (("mul", 1), ("add", 3), ("sub", -2)):
            with self.subTest(kind=kind):
                ep = _export(_Binary(kind, alpha=alpha), x, y)
                before = copy.deepcopy(ep)
                old_mx = _nodes(ep, _MQ)[0]
                self.assertTrue(_legalize(ep).modified)
                target = getattr(torch.ops.aten, kind).Tensor
                op = _nodes(ep, target)[0]
                self.assertEqual(op.args[0].target, _MDQ)
                self.assertIs(op.args[0].args[0], old_mx)
                self.assertEqual(op.args[1].target, _MDQ)
                affine_right = op.args[1].args[0].args[0]
                self.assertEqual(affine_right.target, _DQ)
                self.assertEqual(
                    _grid(affine_right.args[0]), (0.07, -3, -32768, 32767, torch.int16)
                )
                self.assertIs(_nodes(ep, _MQ)[0], old_mx)
                left = quantize_mx(_affine(x), "int8", -1)
                right = quantize_mx(_affine(y, 0.07, -3), "int8", -1)
                if kind == "mul":
                    raw = left * right
                    self.assertGreater(raw.abs().max().item(), 0.13 * 32767)
                elif kind == "add":
                    raw = torch.add(left, right, alpha=alpha)
                else:
                    raw = torch.sub(left, right, alpha=alpha)
                expected = _affine(quantize_mx(raw, "int8", -1), 0.007, 19)
                self.assertTrue(torch.equal(_execute(ep, x, y), expected))
                # The original reference is deliberately evaluated separately;
                # no equivalence claim is made when the new adapter rounds.
                original = _execute(before, x, y)
                self.assertEqual(original.shape, expected.shape)
                self.assertFalse(torch.equal(original, expected))
                self._assert_idempotent(ep)

    def test_relu_is_an_arithmetic_boundary_not_a_layout_move(self):
        x = torch.linspace(-22.13, 22.17, 128).reshape(2, 64)
        ep = _export(_Unary(kind="relu"), x)
        output_q = _nodes(ep, _Q)[-1]
        self.assertTrue(_legalize(ep).modified)
        relu = _nodes(ep, torch.ops.aten.relu.default)[0]
        self.assertEqual(output_q.args[0].target, _MDQ)
        self.assertIs(output_q.args[0].args[0].args[0], relu)
        self.assertEqual(relu.args[0].target, _MDQ)
        expected = _affine(
            quantize_mx(
                quantize_mx(_affine(x, 0.19, 127, 0, 255), "int8", -1).relu(),
                "int8",
                -1,
            ),
            1 / 255,
            0,
            0,
            255,
        )
        self.assertTrue(torch.equal(_execute(ep, x), expected))

    def test_rms_input_output_and_gamma_grids_are_independent(self):
        x = torch.linspace(-300.1, 300.3, 128).reshape(2, 64)
        ep = _export(_RMS(), x)
        rms = _nodes(ep, torch.ops.aten.rms_norm.default)[0]
        gamma, shape, eps = rms.args[2], rms.args[1], rms.args[3]
        self.assertTrue(_legalize(ep).modified)
        self.assertIs(rms.args[2], gamma)
        self.assertEqual(rms.args[1], shape)
        self.assertEqual(rms.args[3], eps)
        rounded = quantize_mx(_affine(x), "int8", -1)
        weight = _affine(torch.linspace(0.5, 1.7, 64), 0.01, 2)
        expected = _affine(
            quantize_mx(
                torch.nn.functional.rms_norm(rounded, [64], weight, 1e-4), "int8", -1
            ),
            0.003,
            -9,
        )
        self.assertTrue(torch.equal(_execute(ep, x), expected))
        self._assert_idempotent(ep)

    def test_custom_rms_preserves_gamma_and_eps(self):
        x = torch.linspace(-200.1, 200.3, 128).reshape(2, 64)
        module = _RMS(custom=True)
        ep = _export(module, x)
        rms = _nodes(ep, torch.ops.circle_custom.rms_norm.default)[0]
        gamma, eps = rms.args[1:]
        self.assertTrue(_legalize(ep).modified)
        self.assertIs(rms.args[1], gamma)
        self.assertEqual(rms.args[2], eps)
        source = quantize_mx(_affine(x), "int8", -1)
        weight = _affine(module.weight, 0.01, 2)
        raw = torch.ops.circle_custom.rms_norm.default(source, weight, 1e-4)
        expected = _affine(quantize_mx(raw, "int8", -1), 0.003, -9)
        self.assertTrue(torch.equal(_execute(ep, x), expected))
        self._assert_idempotent(ep)

    def test_bmm_repeated_source_uses_operand_specific_axes(self):
        x = torch.linspace(-20.1, 20.3, 1024).reshape(1, 32, 32)
        for output_axis in (None, 1):
            with self.subTest(output_axis=output_axis):
                ep = _export(_RepeatedBmm(output_axis), x)
                original_source = _nodes(ep, _MDQ)[0]
                self.assertTrue(_legalize(ep).modified)
                bmm = _nodes(ep, torch.ops.aten.bmm.default)[0]
                self.assertIs(bmm.args[0], original_source)
                rhs = bmm.args[1]
                self.assertEqual(rhs.target, _MDQ)
                self.assertEqual(_grid(rhs.args[0])[1], 1)
                self.assertIs(rhs.args[0].args[0], original_source)
                source = quantize_mx(_affine(x), "int8", 2)
                native = quantize_mx(
                    torch.bmm(source, quantize_mx(source, "int8", 1)), "int8", 2
                )
                expected = (
                    _affine(native, 0.017, -3)
                    if output_axis is None
                    else quantize_mx(native, "int8", 1)
                )
                result, source_result = _execute(ep, x)
                self.assertTrue(torch.equal(result, expected))
                self.assertTrue(torch.equal(source_result, source))
                self._assert_idempotent(ep)

    def test_linear_preserves_per_channel_weight_and_float_bias(self):
        x = torch.linspace(-20.1, 20.3, 128).reshape(2, 64)
        for has_bias in (False, True):
            with self.subTest(bias=has_bias):
                module = _Linear(has_bias)
                ep = _export(module, x)
                linear = _nodes(ep, torch.ops.aten.linear.default)[0]
                weight = linear.args[1]
                bias = linear.args[2] if len(linear.args) > 2 else None
                state = {k: v.clone() for k, v in ep.state_dict.items()}
                self.assertTrue(_legalize(ep).modified)
                self.assertIs(linear.args[1], weight)
                self.assertIs(linear.args[2] if len(linear.args) > 2 else None, bias)
                for key, value in state.items():
                    self.assertTrue(torch.equal(ep.state_dict[key], value))
                w = torch.fake_quantize_per_channel_affine(
                    module.weight, module.scales, module.zeros, 0, 0, 255
                )
                native = quantize_mx(
                    torch.nn.functional.linear(
                        quantize_mx(_affine(x), "int8", -1), w, module.bias
                    ),
                    "int8",
                    -1,
                )
                result, actual_weight, actual_bias = _execute(ep, x)
                self.assertTrue(torch.equal(result, _affine(native, 0.017, -3)))
                self.assertTrue(torch.equal(actual_weight, w))
                if module.bias is not None:
                    self.assertTrue(torch.equal(actual_bias, module.bias))
                else:
                    self.assertIsNone(actual_bias)
                self._assert_idempotent(ep)

    def test_shared_linear_weight_and_bias_survive_role_specific_bias_quantization(
        self,
    ):
        class SharedLinear(torch.nn.Module):
            weight: torch.Tensor
            scales: torch.Tensor
            zeros: torch.Tensor
            bias: torch.nn.Parameter

            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "weight", torch.arange(4 * 64, dtype=torch.uint8).reshape(4, 64)
                )
                self.register_buffer("scales", torch.tensor([0.01, 0.02, 0.03, 0.04]))
                self.register_buffer(
                    "zeros", torch.tensor([120, 121, 122, 123], dtype=torch.int64)
                )
                self.bias = torch.nn.Parameter(torch.tensor([-0.17, 0.11, 0.23, 0.41]))

            def forward(self, x):
                affine = _affine(x)
                mx = quantize_mx(affine, "int8", -1)
                weight = torch.ops.quantized_decomposed.dequantize_per_channel.default(
                    self.weight, self.scales, self.zeros, 0, 0, 255, torch.uint8
                )
                mx_result = _affine(
                    torch.nn.functional.linear(mx, weight, self.bias), 0.017, -3
                )
                affine_result = _affine(
                    torch.nn.functional.linear(affine, weight, self.bias), 0.031, 13
                )
                return mx_result, affine_result, self.bias

        x = torch.linspace(-20.1, 20.3, 128).reshape(2, 64)
        module = SharedLinear()
        ep = _export(module, x)
        native, affine = _nodes(ep, torch.ops.aten.linear.default)
        original_weight = native.args[1]
        original_bias = native.args[2]
        self.assertIs(affine.args[1], original_weight)
        self.assertIs(affine.args[2], original_bias)
        weight_data = ep.state_dict["weight"].clone()
        bias_data = ep.state_dict["bias"].clone()
        self.assertTrue(_legalize(ep).modified)
        self.assertIs(native.args[1], affine.args[1])
        self.assertIs(native.args[2], affine.args[2])
        weight = (module.weight.float() - module.zeros[:, None]) * module.scales[
            :, None
        ]
        source = _affine(x)
        expected_native = _affine(
            quantize_mx(
                torch.nn.functional.linear(
                    quantize_mx(source, "int8", -1), weight, module.bias
                ),
                "int8",
                -1,
            ),
            0.017,
            -3,
        )
        expected_affine = _affine(
            torch.nn.functional.linear(source, weight, module.bias), 0.031, 13
        )
        actual_native, actual_affine, actual_bias = _execute(ep, x)
        self.assertTrue(torch.equal(actual_native, expected_native))
        self.assertTrue(torch.equal(actual_affine, expected_affine))
        self.assertTrue(torch.equal(actual_bias, module.bias))
        self._assert_idempotent(ep)
        PassManager([FoldQuantOps(), RemoveWeightDequantOp(), QuantizeBias()]).run(ep)
        self.assertIs(native.args[1], affine.args[1])
        weight_node = native.args[1]
        self.assertEqual(weight_node.meta[QPARAM_KEY].dtype, "uint8")
        self.assertEqual(weight_node.meta[QPARAM_KEY].quantized_dimension, 0)
        self.assertEqual(weight_node.meta[QPARAM_KEY].scale, module.scales.tolist())
        self.assertEqual(weight_node.meta[QPARAM_KEY].zero_point, module.zeros.tolist())
        self.assertIs(native.args[2], original_bias)
        self.assertNotIn(QPARAM_KEY, original_bias.meta)
        self.assertIsNot(affine.args[2], original_bias)
        quantized_bias = affine.args[2]
        self.assertEqual(quantized_bias.meta[QPARAM_KEY].dtype, "int64")
        expected_scales = 0.13 * module.scales
        expected_bias = torch.round(module.bias / expected_scales).to(torch.int64)
        self.assertTrue(torch.equal(get_constant(ep, quantized_bias), expected_bias))
        self.assertTrue(torch.equal(ep.state_dict["weight"], weight_data))
        self.assertTrue(torch.equal(ep.state_dict["bias"], bias_data))
        ep.graph.lint()
        ep.validate()

    def test_same_domain_mx_cat_without_output_quantizer_is_noop_and_checks_blocks(
        self,
    ):
        class NativeCat(torch.nn.Module):
            def forward(self, x, y):
                x = quantize_mx(x, "int8", -1)
                y = quantize_mx(y, "int8", -1)
                return torch.cat((x, y), -1)

        y = torch.linspace(-0.03, 0.05, 32).reshape(1, 32)
        for first_length in (32, 16):
            with self.subTest(first_length=first_length):
                x = torch.linspace(-40.1, 40.3, first_length).reshape(1, first_length)
                ep = _export(NativeCat(), x, y)
                graph = str(ep.graph)
                if first_length == 16:
                    with self.assertRaisesRegex(
                        NotYetSupportedError, "incomplete MX blocks"
                    ):
                        _legalize(ep)
                    continue
                self.assertFalse(_legalize(ep).modified)
                self.assertEqual(str(ep.graph), graph)
                expected = torch.cat(
                    (quantize_mx(x, "int8", -1), quantize_mx(y, "int8", -1)), -1
                )
                self.assertTrue(torch.equal(_execute(ep, x, y), expected))
                self._assert_idempotent(ep)

    def test_affine_cat_without_output_quantizer_feeds_mixed_mx_arithmetic(self):
        class AffineCat(torch.nn.Module):
            def forward(self, x, y, z):
                left = _affine(x, qmin=-32767)
                right = _affine(y, qmin=-32767)
                source = torch.cat((left, right), -1)
                other = quantize_mx(z, "int8", -1)
                result = _affine(source * other, 0.017, -3)
                return result, source

        x = torch.linspace(-40.1, 40.3, 32).reshape(1, 32)
        y = torch.linspace(-0.03, 0.05, 32).reshape(1, 32)
        z = torch.linspace(-3.1, 3.3, 64).reshape(1, 64)
        ep = _export(AffineCat(), x, y, z)
        original_affine = {n: (_grid(n), n.args[0]) for n in _nodes(ep, _Q)}
        cat = _nodes(ep, torch.ops.aten.cat.default)[0]
        self.assertTrue(_legalize(ep).modified)
        mul = _nodes(ep, torch.ops.aten.mul.Tensor)[0]
        self.assertEqual(mul.args[0].target, _MDQ)
        self.assertIs(mul.args[0].args[0].args[0], cat)
        for node, (grid, source) in original_affine.items():
            self.assertEqual(_grid(node), grid)
            if source is not mul:
                self.assertIs(node.args[0], source)
        source = torch.cat((_affine(x, qmin=-32767), _affine(y, qmin=-32767)), -1)
        expected = _affine(
            quantize_mx(
                quantize_mx(source, "int8", -1) * quantize_mx(z, "int8", -1), "int8", -1
            ),
            0.017,
            -3,
        )
        actual, actual_source = _execute(ep, x, y, z)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_source, source))
        self._assert_idempotent(ep)
        original_inputs = [
            node for node, (_, source) in original_affine.items() if source is not mul
        ]
        self.assertEqual(len(original_inputs), 2)
        for node in original_inputs:
            self.assertTrue(node.meta.get("_preserve_affine_clipping"))
        PassManager([FoldQuantOps()]).run(ep)
        for node in original_inputs:
            self.assertIn(node, ep.graph.nodes)
            self.assertTrue(node.meta.get("_preserve_affine_clipping"))
        self.assertEqual(list(cat.args[0]), original_inputs)
        ep.graph.lint()
        ep.validate()

    def _assert_shared_cat_stable(self, module, rewritten):
        x = torch.linspace(-40.1, 40.3, 32).reshape(1, 32)
        y = torch.linspace(-0.03, 0.05, 32).reshape(1, 32)
        z = torch.linspace(-3.1, 3.3, 64).reshape(1, 64)
        ep = _export(module, x, y, z)
        cat = _nodes(ep, torch.ops.aten.cat.default)[0]
        original_quantizers = {
            n: (_grid(n), n.args[0]) for n in _nodes(ep, _Q) + _nodes(ep, _MQ)
        }
        moved_output = [n for n in _nodes(ep, _Q) if n.args[0] is cat]
        self.assertEqual(len(moved_output), int(rewritten))
        self.assertTrue(_legalize(ep).modified)
        self.assertEqual(bool(cat.meta.get("_mx_affine_domains_legalized")), rewritten)
        cat_grid = (0.017, -3) if rewritten else (0.13, 7)
        full_grid = (*cat_grid, -32768, 32767, torch.int16)
        for source in cat.args[0]:
            self.assertEqual(source.target, _DQ)
            self.assertEqual(_grid(source.args[0]), full_grid)
        for node, (grid, source) in original_quantizers.items():
            if node in moved_output:
                self.assertNotIn(node, ep.graph.nodes)
                continue
            self.assertIn(node, ep.graph.nodes)
            self.assertEqual(_grid(node), grid)
            if source.target not in {
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.add.Tensor,
            }:
                self.assertIs(node.args[0], source)
        adapters = [n for n in cat.users if n.target == _MQ]
        self.assertEqual(len(adapters), 2)
        for adapter in adapters:
            self.assertEqual(_grid(adapter), ("int8", 1, "max", "nearest"))
            self.assertEqual(len(adapter.users), 1)
            dq = next(iter(adapter.users))
            self.assertEqual(dq.target, _MDQ)
            self.assertEqual(len(dq.users), 1)
            consumer = next(iter(dq.users))
            self.assertIn(
                consumer.target, {torch.ops.aten.mul.Tensor, torch.ops.aten.add.Tensor}
            )
            self.assertIs(consumer.args[0], dq)
        output = next(n for n in ep.graph.nodes if n.op == "output")
        self.assertNotIn(cat, output.args[0])
        cat_value = (
            _affine(torch.cat((quantize_mx(x, "int8", -1), _affine(y)), -1), 0.017, -3)
            if rewritten
            else torch.cat((_affine(x), _affine(y)), -1)
        )
        mx_cat = quantize_mx(cat_value, "int8", -1)
        mx_other = quantize_mx(z, "int8", -1)
        native_a = quantize_mx(mx_cat * mx_other, "int8", -1)
        native_b = quantize_mx(mx_cat + mx_other, "int8", -1)
        expected = (
            (native_a, native_b)
            if rewritten
            else (_affine(native_a, 0.017, -3), _affine(native_b, 0.031, 13))
        )
        first = _execute(ep, x, y, z)
        for actual, reference in zip(first, expected):
            self.assertTrue(torch.equal(actual, reference))
        counts = (len(_nodes(ep, _Q)), len(_nodes(ep, _MQ)))
        for _ in range(2):
            self._assert_idempotent(ep)
            self.assertEqual((len(_nodes(ep, _Q)), len(_nodes(ep, _MQ))), counts)
            self.assertEqual([n for n in cat.users if n.target == _MQ], adapters)
            for actual, previous in zip(_execute(ep, x, y, z), first):
                self.assertTrue(torch.equal(actual, previous))
            for source in cat.args[0]:
                self.assertEqual(_grid(source.args[0]), full_grid)

    def test_native_affine_cat_keeps_two_consumer_adapters_on_reexecution(self):
        self._assert_shared_cat_stable(_NativeAffineSharedCat(), rewritten=False)

    def test_rewritten_mixed_cat_keeps_two_consumer_adapters_on_reexecution(self):
        self._assert_shared_cat_stable(_RewrittenMixedSharedCat(), rewritten=True)

    def test_mixed_cat_with_ambiguous_output_domains_is_still_rejected(self):
        class AmbiguousCat(torch.nn.Module):
            def __init__(self, mx_first):
                super().__init__()
                self.mx_first = mx_first

            def forward(self, x, y):
                source = torch.cat((quantize_mx(x, "int8", -1), _affine(y)), -1)
                if self.mx_first:
                    mx_result = quantize_mx(source, "int8", -1)
                    affine_result = _affine(source, 0.017, -3)
                else:
                    affine_result = _affine(source, 0.017, -3)
                    mx_result = quantize_mx(source, "int8", -1)
                return mx_result, affine_result

        x = torch.linspace(-40.1, 40.3, 32).reshape(1, 32)
        y = torch.linspace(-0.03, 0.05, 32).reshape(1, 32)
        for mx_first in (False, True):
            with self.subTest(mx_first=mx_first):
                ep = _export(AmbiguousCat(mx_first), x, y)
                original = str(ep.graph)
                with self.assertRaisesRegex(
                    NotYetSupportedError, "cat.*multiple output"
                ):
                    _legalize(ep)
                self.assertEqual(str(ep.graph), original)
                ep.validate()

    def test_native_mx_cat_multiple_boundaries_require_intact_blocks(self):
        class MultipleCatOutputs(torch.nn.Module):
            def forward(self, x, y):
                source = torch.cat(
                    (quantize_mx(x, "int8", -1), quantize_mx(y, "int8", -1)), -1
                )
                return _affine(source, 0.017, -3), quantize_mx(source, "int8", -1)

        y = torch.linspace(-0.03, 0.05, 32).reshape(1, 32)
        for first_length in (16, 32):
            with self.subTest(first_length=first_length):
                x = torch.linspace(-40.1, 40.3, first_length).reshape(1, first_length)
                ep = _export(MultipleCatOutputs(), x, y)
                original = str(ep.graph)
                if first_length == 16:
                    with self.assertRaisesRegex(NotYetSupportedError, "block|multiple"):
                        _legalize(ep)
                    self.assertEqual(str(ep.graph), original)
                    continue
                self.assertFalse(_legalize(ep).modified)
                self.assertEqual(str(ep.graph), original)
                cat = _nodes(ep, torch.ops.aten.cat.default)[0]
                self.assertEqual(len(cat.users), 2)
                self.assertFalse(cat.meta.get("_mx_affine_domains_legalized"))
                source = torch.cat(
                    (quantize_mx(x, "int8", -1), quantize_mx(y, "int8", -1)), -1
                )
                expected = (_affine(source, 0.017, -3), quantize_mx(source, "int8", -1))
                for actual, reference in zip(_execute(ep, x, y), expected):
                    self.assertTrue(torch.equal(actual, reference))
                self._assert_idempotent(ep)

    def test_private_affine_cat_fallback_preserves_incomplete_mx_inputs(self):
        class AffineFallback(torch.nn.Module):
            def forward(self, x, y):
                source = torch.cat(
                    (quantize_mx(x, "int8", -1), quantize_mx(y, "int8", -1)), -1
                )
                return _affine(source, 0.017, -3)

        x = torch.linspace(-40.1, 40.3, 16).reshape(1, 16)
        y = torch.linspace(-0.03, 0.05, 32).reshape(1, 32)
        ep = _export(AffineFallback(), x, y)
        original_mx = {n: (_grid(n), n.args[0]) for n in _nodes(ep, _MQ)}
        expected = _affine(
            torch.cat((quantize_mx(x, "int8", -1), quantize_mx(y, "int8", -1)), -1),
            0.017,
            -3,
        )
        self.assertTrue(_legalize(ep).modified)
        cat = _nodes(ep, torch.ops.aten.cat.default)[0]
        self.assertTrue(cat.meta.get("_mx_affine_domains_legalized"))
        for source in cat.args[0]:
            self.assertEqual(source.target, _DQ)
            self.assertEqual(
                _grid(source.args[0]), (0.017, -3, -32768, 32767, torch.int16)
            )
            self.assertEqual(source.args[0].args[0].target, _MDQ)
        for node, (grid, source) in original_mx.items():
            self.assertIn(node, ep.graph.nodes)
            self.assertEqual(_grid(node), grid)
            self.assertIs(node.args[0], source)
        self.assertTrue(torch.equal(_execute(ep, x, y), expected))
        self._assert_idempotent(ep)

    def test_uint4_activation_is_rejected_before_mx_adaptation(self):
        class UInt4Activation(torch.nn.Module):
            def forward(self, x, y):
                source = quantize_mx(x, "int8", -1)
                other = _affine(y, 0.125, 0, 0, 15)
                return _affine(source * other, 0.017, -3)

        x = torch.linspace(-3.1, 3.3, 64).reshape(1, 64)
        ep = _export(UInt4Activation(), x, x)
        with self.assertRaises(NotYetSupportedError) as caught:
            _legalize(ep)
        message = str(caught.exception).lower()
        self.assertIn("mul", message)
        self.assertIn("operand[1]", message)
        self.assertIn("unsupported affine clipping range", message)

    def test_linear_rejects_prequantized_bias_in_both_activation_policies(self):
        class QuantizedBias(_Linear):
            bias_codes: torch.Tensor

            def __init__(self, integer, output_mx):
                super().__init__(True)
                self.integer, self.output_mx = integer, output_mx
                self.register_buffer(
                    "bias_codes", torch.tensor([-17, 11, 23, 41], dtype=torch.int32)
                )

            def forward(self, x):
                source = quantize_mx(_affine(x), "int8", -1)
                weight = torch.fake_quantize_per_channel_affine(
                    self.weight, self.scales, self.zeros, 0, 0, 255
                )
                if self.integer:
                    bias = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                        self.bias_codes, 0.01, 0, -2147483648, 2147483647, torch.int32
                    )
                else:
                    assert self.bias is not None
                    bias = _affine(self.bias, 0.01, 0)
                raw = torch.nn.functional.linear(source, weight, bias)
                return (
                    quantize_mx(raw, "int8", -1)
                    if self.output_mx
                    else _affine(raw, 0.017, -3)
                )

        x = torch.linspace(-20.1, 20.3, 128).reshape(2, 64)
        for integer in (False, True):
            for output_mx in (False, True):
                with self.subTest(integer=integer, output_mx=output_mx):
                    ep = _export(QuantizedBias(integer, output_mx), x)
                    with self.assertRaises(NotYetSupportedError) as caught:
                        _legalize(ep)
                    message = str(caught.exception).lower()
                    self.assertIn("linear", message)
                    self.assertIn("bias", message)
                    self.assertIn("cannot be reinterpreted", message)

    def test_cat_commutes_output_grid_without_erasing_input_rounding(self):
        x = torch.linspace(-45.1, 45.3, 64).reshape(1, 64)
        y = torch.linspace(-11.3, 11.7, 32).reshape(1, 32)
        ep = _export(_Cat(), x, y)
        original_mx = _nodes(ep, _MQ)[0]
        original_affine = _nodes(ep, _Q)[:2]
        before = _execute(ep, x, y)
        self.assertTrue(_legalize(ep).modified)
        cat = _nodes(ep, torch.ops.aten.cat.default)[0]
        for inp in cat.args[0]:
            self.assertEqual(inp.target, _DQ)
            self.assertEqual(
                _grid(inp.args[0]), (0.017, 13, -32768, 32767, torch.int16)
            )
        self.assertIs(_nodes(ep, _MQ)[0], original_mx)
        for node in original_affine:
            self.assertIn(node, ep.graph.nodes)
        after = _execute(ep, x, y)
        for old, new in zip(before, after):
            self.assertTrue(torch.equal(old, new))
        self._assert_idempotent(ep)

    def test_cat_mx_destination_checks_block_membership(self):
        for size, legal in ((32, True), (16, False)):
            with self.subTest(first_length=size):
                x = torch.linspace(-40.1, 40.3, size).reshape(1, size)
                y = torch.linspace(-0.03, 0.05, 32).reshape(1, 32)
                ep = _export(_Cat(mx_output=True), x, y)
                if not legal:
                    with self.assertRaises(NotYetSupportedError) as caught:
                        _legalize(ep)
                    self.assertIn("block", str(caught.exception).lower())
                    continue
                self.assertTrue(_legalize(ep).modified)
                cat = _nodes(ep, torch.ops.aten.cat.default)[0]
                self.assertTrue(all(inp.target == _MDQ for inp in cat.args[0]))
                expected = torch.cat(
                    (
                        quantize_mx(quantize_mx(_affine(x), "int8", -1), "int8", -1),
                        quantize_mx(_affine(y, 0.07, -3), "int8", -1),
                    ),
                    -1,
                )
                self.assertTrue(torch.equal(_execute(ep, x, y)[0], expected))
                self._assert_idempotent(ep)

    def test_cat_rejects_shared_raw_output(self):
        x = torch.ones(1, 32)
        ep = _export(_Cat(shared=True), x, x)
        with self.assertRaises(NotYetSupportedError) as caught:
            _legalize(ep)
        self.assertIn("cat", str(caught.exception))
        self.assertTrue(
            "shared" in str(caught.exception).lower()
            or "user" in str(caught.exception).lower()
        )

    def test_split_preserves_independent_results_and_unused_middle_slot(self):
        x = torch.linspace(-60.1, 60.3, 96).reshape(1, 96)
        ep = _export(_Split(), x)
        before = _execute(ep, x)
        quantizers = {n: (_grid(n), n.args[0]) for n in _nodes(ep, _Q)}
        _legalize(ep)
        for node, (grid, source) in quantizers.items():
            self.assertEqual(_grid(node), grid)
            self.assertIs(node.args[0], source)
        after = _execute(ep, x)
        for old, new in zip(before, after):
            self.assertTrue(torch.equal(old, new))
        split = _nodes(ep, torch.ops.aten.split_with_sizes.default)[0]
        self.assertEqual(split.args[1], [32, 32, 32])
        self._assert_idempotent(ep)

    def test_affine_unary_input_to_mx_output_keeps_original_rounding(self):
        class Unary(torch.nn.Module):
            def __init__(self, kind):
                super().__init__()
                self.kind = kind

            def forward(self, x):
                source = _affine(x, 0.19, 127, 0, 255)
                result = source.sigmoid() if self.kind == "sigmoid" else source.relu()
                return quantize_mx(result, "int8", -1)

        x = torch.linspace(-22.13, 22.17, 128).reshape(2, 64)
        for kind in ("sigmoid", "relu"):
            with self.subTest(kind=kind):
                ep = _export(Unary(kind), x)
                affine = _nodes(ep, _Q)[0]
                grid, original_input = _grid(affine), affine.args[0]
                self.assertTrue(_legalize(ep).modified)
                op = _nodes(ep, getattr(torch.ops.aten, kind).default)[0]
                self.assertEqual(op.args[0].target, _MDQ)
                self.assertEqual(_grid(affine), grid)
                self.assertIs(affine.args[0], original_input)
                source = quantize_mx(_affine(x, 0.19, 127, 0, 255), "int8", -1)
                expected = quantize_mx(
                    source.sigmoid() if kind == "sigmoid" else source.relu(), "int8", -1
                )
                self.assertTrue(torch.equal(_execute(ep, x), expected))
                self._assert_idempotent(ep)

    def test_affine_bmm_inputs_to_mx_output_have_independent_roles(self):
        x = torch.linspace(-2.1, 2.3, 1024).reshape(1, 32, 32)
        y = torch.linspace(-8.1, 8.3, 1024).reshape(1, 32, 32)
        ep = _export(_Binary("bmm", mx_input=False, mx_output=True), x, y)
        original_quantizers = {n: _grid(n) for n in _nodes(ep, _Q)}
        self.assertTrue(_legalize(ep).modified)
        bmm = _nodes(ep, torch.ops.aten.bmm.default)[0]
        self.assertEqual(_grid(bmm.args[0].args[0])[1], 2)
        self.assertEqual(_grid(bmm.args[1].args[0])[1], 1)
        for node, grid in original_quantizers.items():
            self.assertEqual(_grid(node), grid)
        expected = quantize_mx(
            torch.bmm(
                quantize_mx(_affine(x), "int8", 2),
                quantize_mx(_affine(y, 0.07, -3), "int8", 1),
            ),
            "int8",
            2,
        )
        self.assertTrue(torch.equal(_execute(ep, x, y), expected))
        self._assert_idempotent(ep)

    def test_affine_only_and_float_only_graphs_are_unchanged(self):
        x = torch.linspace(-2, 2, 64).reshape(1, 64)
        ep = _export(_Binary(mx_input=False), x, x)
        graph = str(ep.graph)
        self.assertFalse(_legalize(ep).modified)
        self.assertEqual(str(ep.graph), graph)

        class Float(torch.nn.Module):
            def forward(self, x):
                return torch.sigmoid(x * x)

        ep = _export(Float(), x)
        graph = str(ep.graph)
        self.assertFalse(_legalize(ep).modified)
        self.assertEqual(str(ep.graph), graph)

    def test_narrow_input_quantizer_survives_before_new_mx_adapter(self):
        class NarrowInput(torch.nn.Module):
            def forward(self, x, y):
                source = quantize_mx(x, "int8", -1)
                other = _affine(y, 0.13, 7, -32767, 32767)
                return _affine(source * other, 0.017, -3)

        x = torch.linspace(-3.1, 3.3, 64).reshape(1, 64)
        y = torch.linspace(-5000.3, 5000.7, 64).reshape(1, 64)
        ep = _export(NarrowInput(), x, y)
        original = _nodes(ep, _Q)[0]
        grid = _grid(original)
        self.assertEqual(grid[2:4], (-32767, 32767))
        self.assertTrue(_legalize(ep).modified)
        self.assertEqual(_grid(original), grid)
        mul = _nodes(ep, torch.ops.aten.mul.Tensor)[0]
        self.assertIs(mul.args[1].args[0].args[0].args[0], original)
        expected = _affine(
            quantize_mx(
                quantize_mx(x, "int8", -1)
                * quantize_mx(_affine(y, 0.13, 7, -32767, 32767), "int8", -1),
                "int8",
                -1,
            ),
            0.017,
            -3,
        )
        self.assertTrue(torch.equal(_execute(ep, x, y), expected))
        self._assert_idempotent(ep)

    def test_roundtrips_keep_intermediate_rounding_and_saturation_with_o1(self):
        class RoundTrip(torch.nn.Module):
            def __init__(self, kind):
                super().__init__()
                self.kind = kind

            def forward(self, x):
                if self.kind == "affine_mx_affine":
                    first = _affine(x, 0.125, 7)
                    intermediate = quantize_mx(first, "int8", -1)
                    return _affine(intermediate, 0.125, 7)
                first = quantize_mx(x, "int8", -1)
                intermediate = _affine(first, 0.25, 127, 0, 255)
                return quantize_mx(intermediate, "int8", -1)

        # One large value sets a shared exponent for the same block's small
        # values. The affine intermediate in the reverse trip also saturates.
        x = torch.tensor([1024.0, 0.625, 0.875, 31.375] + [0.625] * 28).reshape(1, 32)
        types = circle.TensorType.TensorType
        quantize_code = circle.BuiltinOperator.BuiltinOperator.QUANTIZE
        for kind in ("affine_mx_affine", "mx_affine_mx"):
            with self.subTest(kind=kind):
                if kind == "affine_mx_affine":
                    endpoint = _affine(x, 0.125, 7)
                    expected = _affine(quantize_mx(endpoint, "int8", -1), 0.125, 7)
                    expected_types = [
                        (types.INT16, types.MXINT8),
                        (types.MXINT8, types.INT16),
                    ]
                    self.assertEqual(endpoint[0, 1], 0.625)
                    self.assertEqual(expected[0, 1], 0)
                else:
                    endpoint = quantize_mx(x, "int8", -1)
                    expected = quantize_mx(
                        _affine(endpoint, 0.25, 127, 0, 255), "int8", -1
                    )
                    expected_types = [
                        (types.MXINT8, types.UINT8),
                        (types.UINT8, types.MXINT8),
                    ]
                    self.assertEqual(endpoint[0, 0], 1024)
                    self.assertEqual(expected[0, 0], 32)
                self.assertFalse(torch.equal(endpoint, expected))
                ep = _export(RoundTrip(kind), x)
                self.assertFalse(_legalize(ep).modified)
                self.assertTrue(torch.equal(_execute(ep, x), expected))
                PassManager([FoldQuantOps()]).run(ep)
                self.assertEqual(len(_nodes(ep, _MQ)), 1)
                self.assertEqual(len(_nodes(ep, _Q)), 1)
                ep.graph.lint()
                ep.validate()
                for optimize in (False, True):
                    with self.subTest(circle_optimize=optimize):
                        converted = tico.convert(
                            RoundTrip(kind).eval(),
                            (x,),
                            config=tico.CompileConfigV1(
                                legalize_mx_affine_domains=True,
                                circle_optimize=optimize,
                            ),
                        )
                        document = CircleDocument.from_bytes(converted.circle_binary)
                        document.verify(raise_on_error=True)
                        graph = document.model.subgraphs[0]
                        conversions = [
                            op
                            for op in graph.operators
                            if document.model.operatorCodes[op.opcodeIndex].builtinCode
                            == quantize_code
                        ]
                        actual_types = [
                            (
                                graph.tensors[op.inputs[0]].type,
                                graph.tensors[op.outputs[0]].type,
                            )
                            for op in conversions
                        ]
                        self.assertEqual(actual_types, expected_types)

    def test_add_cancellation_keeps_alpha_and_nonzero_zero_points(self):
        x = (1.17 + 0.03 * torch.arange(32)).reshape(1, 32)
        y = -x / 3 + 0.01
        ep = _export(_Binary("add", alpha=3), x, y)
        original_output = _nodes(ep, _Q)[-1]
        output_grid = _grid(original_output)
        self.assertTrue(_legalize(ep).modified)
        add = _nodes(ep, torch.ops.aten.add.Tensor)[0]
        self.assertEqual(add.kwargs["alpha"], 3)
        self.assertEqual(_grid(original_output), output_grid)
        left = quantize_mx(_affine(x, 0.13, 7), "int8", -1)
        right = quantize_mx(_affine(y, 0.07, -3), "int8", -1)
        self.assertGreater(left.abs().max().item(), 1)
        raw = torch.add(left, right, alpha=3)
        self.assertLess(raw.abs().max().item(), 0.3)
        expected = _affine(quantize_mx(raw, "int8", -1), 0.007, 19)
        self.assertTrue(torch.equal(_execute(ep, x, y), expected))
        self._assert_idempotent(ep)

    def test_original_narrow_input_survives_folding_and_serialized_affine_prefix(self):
        class NarrowInput(torch.nn.Module):
            def forward(self, x, y):
                source = quantize_mx(x, "int8", -1)
                other = _affine(y, 0.125, 7, -32767, 32767)
                return _affine(source * other, 0.017, -3)

        x = torch.linspace(-3.1, 3.3, 64).reshape(1, 64)
        y = torch.linspace(-5000.3, 5000.7, 64).reshape(1, 64)
        ep = _export(NarrowInput(), x, y)
        original = _nodes(ep, _Q)[0]
        grid = _grid(original)
        self.assertTrue(_legalize(ep).modified)
        self.assertTrue(original.meta.get("_preserve_affine_clipping"))
        PassManager([FoldQuantOps()]).run(ep)
        self.assertIn(original, ep.graph.nodes)
        self.assertEqual(_grid(original), grid)
        source = original.args[0]
        self.assertEqual(source.op, "placeholder")
        self.assertEqual(source.meta[QPARAM_KEY].dtype, "int16")
        self.assertEqual(source.meta[QPARAM_KEY].scale, [0.125])
        self.assertEqual(source.meta[QPARAM_KEY].zero_point, [7])
        mul = _nodes(ep, torch.ops.aten.mul.Tensor)[0]
        self.assertEqual(mul.args[1].target, _MQ)
        self.assertIs(mul.args[1].args[0], original)
        ep.graph.lint()
        ep.validate()
        for optimize in (False, True):
            with self.subTest(circle_optimize=optimize):
                model = tico.convert(
                    NarrowInput().eval(),
                    (x, y),
                    config=tico.CompileConfigV1(
                        legalize_mx_affine_domains=True, circle_optimize=optimize
                    ),
                )
                document = CircleDocument.from_bytes(model.circle_binary)
                document.verify(raise_on_error=True)
                graph = document.model.subgraphs[0]
                ops = circle.BuiltinOperator.BuiltinOperator
                codes = [
                    document.model.operatorCodes[op.opcodeIndex].builtinCode
                    for op in graph.operators
                ]
                clamp_index = codes.index(ops.MAXIMUM)
                selected = [clamp_index]
                clamp = graph.operators[clamp_index]
                preceding = [
                    i
                    for i, op in enumerate(graph.operators)
                    if clamp.inputs[0] in op.outputs
                ]
                if preceding:
                    self.assertEqual(codes[preceding[0]], ops.QUANTIZE)
                    selected.insert(0, preceding[0])
                prefix = extract_by_operator_indices(document, selected).document
                prefix.verify(raise_on_error=True)
                self.assertEqual(len(prefix.model.subgraphs[0].inputs), 1)
                result = (
                    CircleReferenceRuntime(prefix)
                    .run([y.numpy()], mode=ExecutionMode.FAKE_QUANTIZE)
                    .outputs[0]
                )
                expected = _affine(y, 0.125, 7, -32767, 32767)
                np.testing.assert_array_equal(result, expected.numpy())
                self.assertEqual(result.dtype, np.float32)
                self.assertEqual(result.min(), (-32767 - 7) * 0.125)

    def test_narrow_output_preserves_original_range_after_native_mx_rounding(self):
        class NarrowOutput(torch.nn.Module):
            def forward(self, x):
                source = quantize_mx(x, "int8", -1)
                return _affine(source.sigmoid(), 0.01, 0, -32767, 32767)

        x = torch.linspace(-3.1, 3.3, 64).reshape(1, 64)
        ep = _export(NarrowOutput(), x)
        original = _nodes(ep, _Q)[0]
        grid = _grid(original)
        self.assertTrue(_legalize(ep).modified)
        self.assertEqual(_grid(original), grid)
        sigmoid = _nodes(ep, torch.ops.aten.sigmoid.default)[0]
        self.assertEqual(original.args[0].target, _MDQ)
        self.assertIs(original.args[0].args[0].args[0], sigmoid)
        self.assertTrue(original.meta.get("_preserve_affine_clipping"))
        expected = _affine(
            quantize_mx(quantize_mx(x, "int8", -1).sigmoid(), "int8", -1),
            0.01,
            0,
            -32767,
            32767,
        )
        self.assertTrue(torch.equal(_execute(ep, x), expected))
        self._assert_idempotent(ep)

    def test_existing_mx_native_arithmetic_is_unchanged(self):
        class Native(torch.nn.Module):
            def __init__(self, kind):
                super().__init__()
                self.kind = kind

            def forward(self, x, y):
                x = quantize_mx(x, "int8", -1)
                if self.kind == "relu":
                    result = x.relu()
                else:
                    y = quantize_mx(y, "int8", 1 if self.kind == "bmm" else -1)
                    result = torch.bmm(x, y) if self.kind == "bmm" else x * y
                return quantize_mx(result, "int8", -1)

        for kind in ("relu", "mul", "bmm"):
            with self.subTest(kind=kind):
                shape = (1, 32, 32) if kind == "bmm" else (1, 64)
                x = torch.linspace(-3.1, 3.3, 1024 if kind == "bmm" else 64).reshape(
                    shape
                )
                ep = _export(Native(kind), x, x)
                graph = str(ep.graph)
                self.assertFalse(_legalize(ep).modified)
                self.assertEqual(str(ep.graph), graph)
                self._assert_idempotent(ep)

    def test_missing_source_boundary_reports_node_role_and_domains(self):
        class Missing(torch.nn.Module):
            def forward(self, x, y):
                source = quantize_mx(x, "int8", -1)
                return _affine(source * y, 0.017, -3)

        x = torch.linspace(-3.1, 3.3, 64).reshape(1, 64)
        ep = _export(Missing(), x, x)
        op = _nodes(ep, torch.ops.aten.mul.Tensor)[0]
        with self.assertRaises(NotYetSupportedError) as caught:
            _legalize(ep)
        message = str(caught.exception)
        self.assertIn(op.name, message)
        self.assertIn(str(op.target), message)
        self.assertIn("source=", message)
        self.assertIn("required=", message)
        self.assertIn("input", message.lower())
        self.assertIn("boundary", message.lower())

    def test_chained_cat_legalization_preserves_domains_on_second_run(self):
        class Chain(torch.nn.Module):
            def forward(self, x, y, z):
                x = quantize_mx(x, "int8", -1)
                y = _affine(y, 0.13, 7)
                first = _affine(torch.cat((x, y), -1), 0.017, -3)
                z = quantize_mx(z, "int8", -1)
                return _affine(torch.cat((first, z), -1), 0.031, 13), x, y, z

        x = torch.linspace(-7.1, 7.3, 32).reshape(1, 32)
        y = torch.linspace(-0.71, 0.73, 32).reshape(1, 32)
        z = torch.linspace(-71.1, 71.3, 32).reshape(1, 32)
        ep = _export(Chain(), x, y, z)
        before = _execute(ep, x, y, z)
        self.assertTrue(_legalize(ep).modified)
        after = _execute(ep, x, y, z)
        for original, transformed in zip(before, after):
            self.assertTrue(torch.equal(original, transformed))
        cats = _nodes(ep, torch.ops.aten.cat.default)
        self.assertEqual(len(cats), 2)
        self.assertEqual(
            _grid(cats[0].args[0][0].args[0]), (0.017, -3, -32768, 32767, torch.int16)
        )
        self.assertEqual(
            _grid(cats[1].args[0][0].args[0]), (0.031, 13, -32768, 32767, torch.int16)
        )
        self._assert_idempotent(ep)

    def test_shared_raw_arithmetic_output_is_rejected(self):
        class Shared(torch.nn.Module):
            def __init__(self, mx_output):
                super().__init__()
                self.mx_output = mx_output

            def forward(self, x):
                source = quantize_mx(x, "int8", -1)
                raw = source.sigmoid()
                result = (
                    quantize_mx(raw, "int8", -1)
                    if self.mx_output
                    else _affine(raw, 1 / 255, 0, 0, 255)
                )
                return result, raw

        x = torch.linspace(-3.1, 3.3, 64).reshape(1, 64)
        for mx_output in (False, True):
            with self.subTest(mx_output=mx_output):
                ep = _export(Shared(mx_output), x)
                with self.assertRaises(NotYetSupportedError) as caught:
                    _legalize(ep)
                message = str(caught.exception).lower()
                self.assertIn("sigmoid", message)
                self.assertTrue("shared" in message or "user" in message)

    def test_multiple_output_boundaries_find_mx_after_affine_only_input(self):
        x = torch.linspace(-4, 4, 64).reshape(1, 64)
        for mx_first in (False, True):
            with self.subTest(mx_first=mx_first):
                ep = _export(_MultipleOutputBoundaries(mx_first), x)
                sigmoid = _nodes(ep, torch.ops.aten.sigmoid.default)[0]
                self.assertEqual(sigmoid.args[0].target, _DQ)
                quantizers = _nodes(ep, _Q) + _nodes(ep, _MQ)
                original_arguments = {
                    node: (node.args, dict(node.kwargs)) for node in quantizers
                }
                original_graph = str(ep.graph)
                original_signature = copy.deepcopy(ep.graph_signature)
                original_state = {
                    name: value.clone() for name, value in ep.state_dict.items()
                }
                with self.assertRaises(NotYetSupportedError) as caught:
                    _legalize(ep)
                message = str(caught.exception)
                self.assertIn(sigmoid.name, message)
                self.assertIn(str(sigmoid.target), message)
                self.assertIn("multiple", message.lower())
                self.assertIn("output", message.lower())
                self.assertIn("boundar", message.lower())
                self.assertIn("native", message.lower())
                self.assertNotIn("missing quantization boundary", message.lower())
                self.assertEqual(str(ep.graph), original_graph)
                self.assertEqual(ep.graph_signature, original_signature)
                for node, arguments in original_arguments.items():
                    self.assertIn(node, ep.graph.nodes)
                    self.assertEqual((node.args, dict(node.kwargs)), arguments)
                self.assertEqual(set(ep.state_dict), set(original_state))
                for name, value in original_state.items():
                    self.assertTrue(torch.equal(ep.state_dict[name], value))
                ep.graph.lint()
                ep.validate()

    def test_affine_only_multiple_output_boundaries_remain_a_noop(self):
        x = torch.linspace(-4, 4, 64).reshape(1, 64)
        for first_boundary_first in (False, True):
            with self.subTest(first_boundary_first=first_boundary_first):
                ep = _export(
                    _MultipleOutputBoundaries(first_boundary_first, include_mx=False), x
                )
                original_graph = str(ep.graph)
                original = _execute(ep, x)
                self.assertFalse(_legalize(ep).modified)
                self.assertEqual(str(ep.graph), original_graph)
                for actual, expected in zip(_execute(ep, x), original):
                    self.assertTrue(torch.equal(actual, expected))
                self._assert_idempotent(ep)

    def test_mx_gamma_is_not_reinterpreted_as_affine_weight(self):
        class Gamma(torch.nn.Module):
            def __init__(self, output_mx):
                super().__init__()
                self.output_mx = output_mx

            def forward(self, x, gamma):
                source = quantize_mx(x, "int8", -1)
                gamma = quantize_mx(gamma, "int8", -1)
                raw = torch.nn.functional.rms_norm(source, [64], gamma, 1e-4)
                return (
                    quantize_mx(raw, "int8", -1)
                    if self.output_mx
                    else _affine(raw, 0.01, 0)
                )

        x = torch.linspace(-3.1, 3.3, 64).reshape(1, 64)
        for output_mx in (False, True):
            with self.subTest(output_mx=output_mx):
                ep = _export(Gamma(output_mx), x, torch.ones(64))
                with self.assertRaises(NotYetSupportedError) as caught:
                    _legalize(ep)
                message = str(caught.exception).lower()
                self.assertIn("rms_norm", message)
                self.assertTrue("weight" in message or "gamma" in message)
                self.assertIn("affine", message)

    def test_affine_operands_with_mx_output_keep_both_original_input_grids(self):
        x = torch.linspace(-3.1, 3.3, 64).reshape(1, 64)
        y = torch.linspace(1.1, 8.7, 64).reshape(1, 64)
        ep = _export(_Binary(mx_input=False, mx_output=True), x, y)
        original = {n: (_grid(n), n.args[0]) for n in _nodes(ep, _Q)}
        output_mx = _nodes(ep, _MQ)[0]
        self.assertTrue(_legalize(ep).modified)
        op = _nodes(ep, torch.ops.aten.mul.Tensor)[0]
        self.assertTrue(all(inp.target == _MDQ for inp in op.args))
        for node, (grid, source) in original.items():
            self.assertEqual(_grid(node), grid)
            self.assertIs(node.args[0], source)
        self.assertIs(_nodes(ep, _MQ)[-1], output_mx)
        expected = quantize_mx(
            quantize_mx(_affine(x), "int8", -1)
            * quantize_mx(_affine(y, 0.07, -3), "int8", -1),
            "int8",
            -1,
        )
        self.assertTrue(torch.equal(_execute(ep, x, y), expected))
        self._assert_idempotent(ep)


if __name__ == "__main__":
    unittest.main()

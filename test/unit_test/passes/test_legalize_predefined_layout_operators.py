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

import math
import unittest
from typing import cast

import torch

from tico.passes.legalize_predefined_layout_operators import (
    get_permute_weight_input,
    LegalizePreDefinedLayoutOperators,
)
from tico.utils.errors import NotYetSupportedError
from tico.utils.validate_args_kwargs import Conv2DArgs, ConvTranspose2DArgs

from test.support.helper import num_of_ops
from test.support.pass_value_test import SinglePassValueTest


class ChannelWisePReLUNet(torch.nn.Module):
    """Apply one PReLU slope per channel to an NCHW input."""

    def __init__(self) -> None:
        super().__init__()
        self.prelu = torch.nn.PReLU(num_parameters=4)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Return the channel-wise PReLU output."""
        return self.prelu(input_)

    def get_example_inputs(self):
        """Return representative inputs for export and value comparison."""
        return (torch.randn(1, 4, 5, 7),), {}


class SharedPReLUNet(torch.nn.Module):
    """Apply one PReLU slope shared by every input element."""

    def __init__(self) -> None:
        super().__init__()
        self.prelu = torch.nn.PReLU(num_parameters=1)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Return the shared-slope PReLU output."""
        return self.prelu(input_)

    def get_example_inputs(self):
        """Return representative inputs for export and value comparison."""
        return (torch.randn(1, 4, 5, 7),), {}


class ConvChannelWisePReLUNet(torch.nn.Module):
    """Apply channel-wise PReLU directly after an NCHW convolution."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.prelu = torch.nn.PReLU(num_parameters=4)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Return the convolution followed by channel-wise PReLU."""
        return self.prelu(self.conv(input_))

    def get_example_inputs(self):
        """Return representative inputs for export and value comparison."""
        return (torch.randn(1, 3, 5, 7),), {}


class LegalizeChannelWisePReLUTest(SinglePassValueTest):
    """Verify channel-wise PReLU layout legalization and numerical parity."""

    def test_pass(self):
        """Replace aten.prelu with the channel-last Circle custom operator."""
        self.setup(ChannelWisePReLUNet())
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.prelu.default]), 1
        )

        self.run_value_test(LegalizePreDefinedLayoutOperators())

        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.prelu.default]), 0
        )
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.circle_custom.prelu]), 1
        )
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.permute.default]), 2
        )

    def test_shared_slope_is_not_rewritten(self):
        """Keep a broadcast-safe shared PReLU slope in native aten form."""
        self.setup(SharedPReLUNet())
        self.run_value_test(LegalizePreDefinedLayoutOperators())

        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.prelu.default]), 1
        )
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.circle_custom.prelu]), 0
        )

    def test_reuses_channel_last_convolution_output(self):
        """Avoid an inverse permutation pair between Conv2D and PReLU."""
        self.setup(ConvChannelWisePReLUNet())
        self.run_value_test(LegalizePreDefinedLayoutOperators())

        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.circle_custom.conv2d]), 1
        )
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.circle_custom.prelu]), 1
        )

        graph_nodes = list(self.exported_program().graph.nodes)
        circle_conv = next(
            node
            for node in graph_nodes
            if node.target == torch.ops.circle_custom.conv2d
        )
        circle_prelu = next(
            node for node in graph_nodes if node.target == torch.ops.circle_custom.prelu
        )

        self.assertIs(circle_prelu.args[0], circle_conv)


class _ConvWeightLayoutNet(torch.nn.Module):
    """Share a deterministic convolution weight with a permutation-invariant sum."""

    def __init__(self, kind, quantization, dtype=torch.uint8, negative_axis=False):
        super().__init__()
        self.kind = kind
        self.quantization = quantization
        shape = {
            "conv": (3, 2, 2, 2),
            "depthwise": (2, 1, 2, 2),
            "transpose": (2, 3, 2, 2),
            "convolution": (3, 2, 2, 2),
        }[kind]
        self.axis = 1 if kind == "transpose" else 0
        channels = shape[self.axis]
        if negative_axis:
            self.axis -= len(shape)
        values = torch.arange(math.prod(shape)).reshape(shape)
        weight = values.float() / 16 if quantization == "float" else values.to(dtype)
        self.register_buffer("weight", weight)
        self.register_buffer("scales", torch.arange(1, channels + 1).float() / 8)
        self.register_buffer("zero_points", torch.arange(channels, dtype=torch.int64))
        self.quant_min, self.quant_max = (
            (0, 255) if dtype == torch.uint8 else (-32768, 32767)
        )

    def forward(self, input_):
        weight = cast(torch.Tensor, self.weight)
        if self.quantization == "tensor":
            weight = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                weight, 0.125, 2, self.quant_min, self.quant_max, weight.dtype
            )
        elif self.quantization == "channel":
            weight = torch.ops.quantized_decomposed.dequantize_per_channel.default(
                weight,
                self.scales,
                self.zero_points,
                self.axis,
                self.quant_min,
                self.quant_max,
                weight.dtype,
            )
        if self.kind == "transpose":
            output = torch.ops.aten.conv_transpose2d.input(
                input_, weight, None, [1, 1], [0, 0], [0, 0], 1, [1, 1]
            )
        elif self.kind == "convolution":
            output = torch.ops.aten.convolution.default(
                input_, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
            )
        else:
            groups = 2 if self.kind == "depthwise" else 1
            output = torch.ops.aten.conv2d.default(
                input_, weight, None, [1, 1], [0, 0], [1, 1], groups
            )
        return output, weight.sum()


class LegalizeConvWeightLayoutTest(unittest.TestCase):
    """Characterize the existing floating-point and quantized weight rewrites."""

    _CASES = (
        ("conv", [0, 2, 3, 1], torch.ops.circle_custom.conv2d, 0),
        ("depthwise", [1, 2, 3, 0], torch.ops.circle_custom.depthwise_conv2d, 3),
        ("transpose", [1, 2, 3, 0], torch.ops.circle_custom.transpose_conv, 0),
    )

    def _check_layout(self, kind, permutation, target, axis, quantization, **kwargs):
        model = _ConvWeightLayoutNet(kind, quantization, **kwargs).eval()
        input_ = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) / 16
        original_input = input_.clone()
        original_buffers = {
            name: value.clone() for name, value in model.named_buffers()
        }
        ep = torch.export.export(model, (input_,))
        before = ep.module()(input_)
        original_target = (
            torch.ops.aten.conv_transpose2d.input
            if kind == "transpose"
            else torch.ops.aten.conv2d.default
        )
        conv = next(node for node in ep.graph.nodes if node.target == original_target)
        args = (ConvTranspose2DArgs if kind == "transpose" else Conv2DArgs)(
            *conv.args, **conv.kwargs  # type: ignore[arg-type]
        )
        weight = args.weight
        integer_weight = weight if quantization == "float" else weight.args[0]
        assert isinstance(integer_weight, torch.fx.Node)
        self.assertIs(get_permute_weight_input(args), integer_weight)
        original_weight_args = weight.args
        original_weight_kwargs = dict(weight.kwargs)
        shared_user = next(node for node in weight.users if node is not conv)
        signature = str(ep.graph_signature)

        test_pass = LegalizePreDefinedLayoutOperators()
        self.assertTrue(test_pass.call(ep).modified)

        circle_conv = next(node for node in ep.graph.nodes if node.target == target)
        if quantization == "float":
            permute = circle_conv.args[1]
            self.assertIn("val", weight.meta)
        else:
            self.assertIs(circle_conv.args[1], weight)
            permute = weight.args[0]
            expected_args = list(original_weight_args)
            expected_args[0] = permute
            if quantization == "channel":
                expected_args[3] = axis
                self.assertIs(weight.args[1], original_weight_args[1])
                self.assertIs(weight.args[2], original_weight_args[2])
            self.assertEqual(weight.args, tuple(expected_args))
            self.assertEqual(weight.kwargs, original_weight_kwargs)
            self.assertNotIn("val", weight.meta)

        assert isinstance(permute, torch.fx.Node)
        self.assertEqual(permute.target, torch.ops.aten.permute.default)
        self.assertEqual(permute.args, (integer_weight, permutation))
        self.assertIs(shared_user.args[0], weight)
        nodes = list(ep.graph.nodes)
        self.assertLess(nodes.index(integer_weight), nodes.index(permute))
        self.assertLess(nodes.index(permute), nodes.index(circle_conv))
        if quantization != "float":
            self.assertLess(nodes.index(permute), nodes.index(weight))
        self.assertNotIn(conv, nodes)
        self.assertEqual(str(ep.graph_signature), signature)

        after = ep.module()(input_)
        for expected, actual in zip(before, after):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, expected.dtype)
            # Binary-exact inputs, scales, and small reductions avoid rounding drift.
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(input_, original_input))
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(value, original_buffers[name]))
        graph = str(ep.graph)
        self.assertFalse(test_pass.call(ep).modified)
        self.assertEqual(str(ep.graph), graph)

    def test_float_weight_layout_and_shared_user(self):
        for case in self._CASES:
            with self.subTest(kind=case[0]):
                self._check_layout(*case, "float")

    def test_per_tensor_weight_preserves_dtype_and_qparams(self):
        for case in self._CASES:
            for dtype in (torch.uint8, torch.int16):
                with self.subTest(kind=case[0], dtype=dtype):
                    self._check_layout(*case, "tensor", dtype=dtype)

    def test_per_channel_weight_remaps_positive_and_negative_axis(self):
        for case in self._CASES:
            for dtype in (torch.uint8, torch.int16):
                for negative_axis in (False, True):
                    with self.subTest(
                        kind=case[0], dtype=dtype, negative=negative_axis
                    ):
                        self._check_layout(
                            *case, "channel", dtype=dtype, negative_axis=negative_axis
                        )

    def test_invalid_per_channel_axis_keeps_exact_error(self):
        for kind, permutation, _target, _axis in self._CASES:
            for invalid_axis in (-5, 4):
                with self.subTest(kind=kind, axis=invalid_axis):
                    model = _ConvWeightLayoutNet(kind, "channel").eval()
                    ep = torch.export.export(model, (torch.zeros(1, 2, 3, 4),))
                    dq = next(
                        node
                        for node in ep.graph.nodes
                        if node.target
                        == torch.ops.quantized_decomposed.dequantize_per_channel.default
                    )
                    dq.update_arg(3, invalid_axis)
                    with self.assertRaises(NotYetSupportedError) as caught:
                        LegalizePreDefinedLayoutOperators().call(ep)
                    self.assertEqual(
                        str(caught.exception),
                        f"Invalid per-channel axis {invalid_axis} for weight permutation {permutation}.",
                    )

    def test_generic_convolution_is_not_matched(self):
        model = _ConvWeightLayoutNet("convolution", "channel").eval()
        ep = torch.export.export(model, (torch.zeros(1, 2, 3, 4),))
        graph = str(ep.graph)
        self.assertFalse(LegalizePreDefinedLayoutOperators().call(ep).modified)
        self.assertEqual(str(ep.graph), graph)

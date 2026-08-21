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

"""End-to-end UINT8 and INT16 Circle export tests for CNN WrapQ."""

from __future__ import annotations

import unittest

import tico

import torch
import torch.nn.functional as F
from circle_schema import circle
from tico.circle.io import model_from_bytes
from tico.ops import ResizeBilinear2d
from tico.quantization import convert, prepare, QuantStub
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme
from torch import nn


class ExportableCNN(nn.Module):
    """Exercise regular Conv2d, PReLU, depthwise Conv2d, and ResizeBilinear."""

    def __init__(self) -> None:
        """Create a small static CNN with bias on both convolution types."""
        super().__init__()
        self.input_quantizer = QuantStub()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True)
        self.prelu = nn.PReLU(num_parameters=4)
        self.depthwise = nn.Conv2d(
            4,
            4,
            kernel_size=3,
            padding=1,
            groups=4,
            bias=True,
        )
        self.resize = ResizeBilinear2d(
            (12, 12),
            align_corners=False,
            half_pixel_centers=True,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Run the static CNN."""
        hidden = self.input_quantizer(input_)
        hidden = self.prelu(self.conv(hidden))
        hidden = self.depthwise(hidden)
        return self.resize(hidden)


class ResidualDownsampleCNN(nn.Module):
    """Exercise quantized MaxPool2D and zero-padding on a residual shortcut."""

    def __init__(self) -> None:
        """Create one stride-two main path and a padded shortcut path."""
        super().__init__()
        self.input_quantizer = QuantStub()
        self.main = nn.Conv2d(3, 4, kernel_size=1, stride=2, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.output = nn.PReLU(num_parameters=4)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Add a pooled channel-padded shortcut to the main convolution."""
        input_q = self.input_quantizer(input_)
        main = self.main(input_q)
        shortcut = F.pad(self.pool(input_q), (0, 0, 0, 0, 0, 1))
        return self.output(main + shortcut)


class ConcatenatedHeadCNN(nn.Module):
    """Produce a quantized output by concatenating two Conv2d heads."""

    def __init__(self) -> None:
        """Create a shared trunk and two output heads."""
        super().__init__()
        self.input_quantizer = QuantStub()
        self.trunk = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True)
        self.head0 = nn.Conv2d(4, 2, kernel_size=1, bias=True)
        self.head1 = nn.Conv2d(4, 3, kernel_size=1, bias=True)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Concatenate the two head tensors along the channel dimension."""
        hidden = self.trunk(self.input_quantizer(input_))
        return torch.cat((self.head0(hidden), self.head1(hidden)), dim=1)


def _single_quant_conv(module: nn.Module):
    """Return the single QuantConv2d descendant of one wrapped module."""
    from tico.quantization.wrapq.wrappers.nn.quant_conv2d import QuantConv2d

    matches = [child for child in module.modules() if isinstance(child, QuantConv2d)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one QuantConv2d, but found {len(matches)}.")
    return matches[0]


@torch.inference_mode()
def _synchronize_head_ranges(model: nn.Module) -> None:
    """Assign one collected activation range to both concatenated heads."""
    observers = [
        _single_quant_conv(model.head0).obs_act_out,
        _single_quant_conv(model.head1).obs_act_out,
    ]
    common_min = torch.stack(
        [observer.min_val.reshape(()) for observer in observers]
    ).min()
    common_max = torch.stack(
        [observer.max_val.reshape(()) for observer in observers]
    ).max()
    for observer in observers:
        observer.min_val.copy_(common_min)
        observer.max_val.copy_(common_max)


def _quant_config(bit_width: int) -> PTQConfig:
    """Create the UINT8 or INT16 policy used by the test."""
    if bit_width == 8:
        dtype = DType.uint(8)
        activation_qscheme = QScheme.PER_TENSOR_ASYMM
        weight_qscheme = QScheme.PER_CHANNEL_ASYMM
    elif bit_width == 16:
        dtype = DType.int(16)
        activation_qscheme = QScheme.PER_TENSOR_SYMM
        weight_qscheme = QScheme.PER_CHANNEL_SYMM
    else:
        raise ValueError(f"Unsupported bit width: {bit_width}")

    return PTQConfig(
        activation=affine(dtype, qscheme=activation_qscheme),
        weight=affine(dtype, qscheme=weight_qscheme),
        strict_wrap=False,
    )


def _builtin_code(model, operator) -> int:
    """Return the builtin code referenced by one Circle operator."""
    return model.operatorCodes[operator.opcodeIndex].builtinCode


def _affine_qparams(tensor) -> tuple[tuple[float, ...], tuple[int, ...], int]:
    """Return one serialized tensor's affine quantization tuple."""
    quantization = tensor.quantization
    if quantization is None:
        raise AssertionError("Expected tensor quantization metadata.")
    return (
        tuple(quantization.scale),
        tuple(quantization.zeroPoint),
        int(quantization.quantizedDimension),
    )


class CNNQuantizedExportTest(unittest.TestCase):
    """Verify quantized CNN Circle types, qparams, and resize options."""

    def _export(self, bit_width: int):
        """Calibrate, freeze, export, and deserialize one test model."""
        torch.manual_seed(20260728)
        model = ExportableCNN().eval()
        sample = torch.randn(1, 3, 6, 6)
        prepared = prepare(model, _quant_config(bit_width), inplace=True)
        with torch.inference_mode():
            prepared(sample)
        quantized = convert(prepared, inplace=True).eval()
        circle_model = tico.convert(quantized, (sample,))
        return model_from_bytes(circle_model.circle_binary)

    def _assert_export(self, bit_width: int) -> None:
        """Validate one deserialized UINT8 or INT16 Circle graph."""
        model = self._export(bit_width)
        graph = model.subgraphs[0]
        expected_type = (
            circle.TensorType.TensorType.UINT8
            if bit_width == 8
            else circle.TensorType.TensorType.INT16
        )
        expected_bias_type = (
            circle.TensorType.TensorType.INT32
            if bit_width == 8
            else circle.TensorType.TensorType.INT64
        )

        self.assertEqual(graph.tensors[graph.inputs[0]].type, expected_type)
        self.assertEqual(graph.tensors[graph.outputs[0]].type, expected_type)

        resize_count = 0
        conv_count = 0
        depthwise_count = 0
        for operator in graph.operators:
            builtin = _builtin_code(model, operator)
            if builtin in {
                circle.BuiltinOperator.BuiltinOperator.CONV_2D,
                circle.BuiltinOperator.BuiltinOperator.DEPTHWISE_CONV_2D,
            }:
                input_tensor = graph.tensors[operator.inputs[0]]
                weight_tensor = graph.tensors[operator.inputs[1]]
                bias_tensor = graph.tensors[operator.inputs[2]]
                output_tensor = graph.tensors[operator.outputs[0]]
                self.assertEqual(input_tensor.type, expected_type)
                self.assertEqual(weight_tensor.type, expected_type)
                self.assertEqual(bias_tensor.type, expected_bias_type)
                self.assertEqual(output_tensor.type, expected_type)
                self.assertIsNotNone(input_tensor.quantization)
                self.assertIsNotNone(weight_tensor.quantization)
                self.assertIsNotNone(bias_tensor.quantization)
                self.assertIsNotNone(output_tensor.quantization)

                if builtin == circle.BuiltinOperator.BuiltinOperator.CONV_2D:
                    conv_count += 1
                    self.assertEqual(
                        weight_tensor.quantization.quantizedDimension,
                        0,
                    )
                else:
                    depthwise_count += 1
                    self.assertEqual(
                        weight_tensor.quantization.quantizedDimension,
                        3,
                    )
            elif builtin == circle.BuiltinOperator.BuiltinOperator.RESIZE_BILINEAR:
                resize_count += 1
                input_tensor = graph.tensors[operator.inputs[0]]
                output_tensor = graph.tensors[operator.outputs[0]]
                self.assertEqual(input_tensor.type, expected_type)
                self.assertEqual(output_tensor.type, expected_type)
                self.assertFalse(operator.builtinOptions.alignCorners)
                self.assertTrue(operator.builtinOptions.halfPixelCenters)

        self.assertEqual(conv_count, 1)
        self.assertEqual(depthwise_count, 1)
        self.assertEqual(resize_count, 1)

    def test_identical_qparams_propagate_through_concat(self) -> None:
        """Keep UINT8 and INT16 concatenation outputs quantized."""
        for bit_width in (8, 16):
            with self.subTest(bit_width=bit_width):
                torch.manual_seed(20260729)
                model = ConcatenatedHeadCNN().eval()
                sample = torch.randn(1, 3, 6, 6)
                prepared = prepare(model, _quant_config(bit_width), inplace=True)
                with torch.inference_mode():
                    prepared(sample)
                _synchronize_head_ranges(prepared)
                quantized = convert(prepared, inplace=True).eval()
                circle_model = tico.convert(quantized, (sample,))
                exported = model_from_bytes(circle_model.circle_binary)
                graph = exported.subgraphs[0]
                expected_type = (
                    circle.TensorType.TensorType.UINT8
                    if bit_width == 8
                    else circle.TensorType.TensorType.INT16
                )
                self.assertEqual(
                    graph.tensors[graph.outputs[0]].type,
                    expected_type,
                )
                concat_ops = [
                    operator
                    for operator in graph.operators
                    if _builtin_code(exported, operator)
                    == circle.BuiltinOperator.BuiltinOperator.CONCATENATION
                ]
                self.assertEqual(len(concat_ops), 1)
                concat = concat_ops[0]
                for tensor_index in [*concat.inputs, *concat.outputs]:
                    tensor = graph.tensors[tensor_index]
                    self.assertEqual(tensor.type, expected_type)
                    self.assertIsNotNone(tensor.quantization)

    def test_pool_and_pad_shortcut_remains_quantized(self) -> None:
        """Preserve independent MaxPool qparams and quantized padding."""
        pattern = torch.tensor(
            [[-10.0, -9.0], [-8.0, 1.0]],
            dtype=torch.float32,
        )
        tiled = pattern.repeat(4, 4)
        sample = tiled.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        for bit_width in (8, 16):
            with self.subTest(bit_width=bit_width):
                torch.manual_seed(20260730)
                model = ResidualDownsampleCNN().eval()
                prepared = prepare(model, _quant_config(bit_width), inplace=True)
                with torch.inference_mode():
                    prepared(sample)
                quantized = convert(prepared, inplace=True).eval()
                circle_model = tico.convert(quantized, (sample,))
                exported = model_from_bytes(circle_model.circle_binary)
                graph = exported.subgraphs[0]
                expected_type = (
                    circle.TensorType.TensorType.UINT8
                    if bit_width == 8
                    else circle.TensorType.TensorType.INT16
                )
                producers = {
                    output: operator
                    for operator in graph.operators
                    for output in operator.outputs
                }
                checked = 0
                max_pool_checked = 0
                for operator in graph.operators:
                    builtin = _builtin_code(exported, operator)
                    if builtin not in {
                        circle.BuiltinOperator.BuiltinOperator.MAX_POOL_2D,
                        circle.BuiltinOperator.BuiltinOperator.PAD,
                    }:
                        continue
                    data_input = graph.tensors[operator.inputs[0]]
                    data_output = graph.tensors[operator.outputs[0]]
                    self.assertEqual(data_input.type, expected_type)
                    self.assertEqual(data_output.type, expected_type)
                    self.assertIsNotNone(data_input.quantization)
                    self.assertIsNotNone(data_output.quantization)
                    if builtin == circle.BuiltinOperator.BuiltinOperator.MAX_POOL_2D:
                        self.assertNotEqual(
                            _affine_qparams(data_input),
                            _affine_qparams(data_output),
                        )
                        producer = producers.get(operator.inputs[0])
                        if producer is not None:
                            self.assertNotEqual(
                                _builtin_code(exported, producer),
                                circle.BuiltinOperator.BuiltinOperator.QUANTIZE,
                            )
                        max_pool_checked += 1
                    checked += 1
                self.assertEqual(checked, 2)
                self.assertEqual(max_pool_checked, 1)

    def test_uint8_circle_export(self) -> None:
        """Export the CNN as a fully quantized UINT8 Circle model."""
        self._assert_export(8)

    def test_int16_circle_export(self) -> None:
        """Export the CNN as a fully quantized INT16 Circle model."""
        self._assert_export(16)


if __name__ == "__main__":
    unittest.main()

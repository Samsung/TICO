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

"""Quantized Circle export tests for SAME Conv2d and Concat boundaries."""

from __future__ import annotations

import unittest

import tico

import torch
from circle_schema import circle
from tico.circle.io import model_from_bytes
from tico.ops import Concat, SamePaddingConv2d
from tico.quantization import convert, prepare, QuantStub
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme
from torch import nn


class SamePaddingExportCNN(nn.Module):
    """Exercise regular and depthwise Circle SAME-padding convolutions."""

    def __init__(self) -> None:
        """Create a compact CNN whose convolutions must not emit PAD operators."""
        super().__init__()
        self.input_quantizer = QuantStub()
        self.conv = SamePaddingConv2d(
            3,
            4,
            kernel_size=5,
            stride=2,
            bias=True,
        )
        self.prelu = nn.PReLU(num_parameters=4)
        self.depthwise = SamePaddingConv2d(
            4,
            4,
            kernel_size=5,
            groups=4,
            bias=True,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Run the two SAME-padding convolution paths."""
        hidden = self.input_quantizer(input_)
        hidden = self.prelu(self.conv(hidden))
        return self.depthwise(hidden)


class ConcatHeadCNN(nn.Module):
    """Concatenate two independently scaled output heads through TICO Concat."""

    def __init__(self) -> None:
        """Create a trunk and two heads with a quantizable concatenation boundary."""
        super().__init__()
        self.input_quantizer = QuantStub()
        self.trunk = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True)
        self.head0 = nn.Conv2d(4, 2, kernel_size=1, bias=True)
        self.head1 = nn.Conv2d(4, 3, kernel_size=1, bias=True)
        self.concat = Concat(dim=1)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Run both heads and concatenate their outputs along the channel axis."""
        hidden = self.trunk(self.input_quantizer(input_))
        return self.concat((self.head0(hidden), self.head1(hidden)))


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
    """Return the builtin operator code referenced by one Circle operator."""
    return model.operatorCodes[operator.opcodeIndex].builtinCode


def _export(model: nn.Module, sample: torch.Tensor, bit_width: int):
    """Calibrate, freeze, export, and deserialize one quantized model."""
    prepared = prepare(model.eval(), _quant_config(bit_width), inplace=True)
    with torch.inference_mode():
        prepared(sample)
    quantized = convert(prepared, inplace=True).eval()
    circle_model = tico.convert(quantized, (sample,))
    return model_from_bytes(circle_model.circle_binary)


class CNNBoundaryQuantizedExportTest(unittest.TestCase):
    """Verify quantized export semantics introduced for the hand detector."""

    def test_same_padding_convolutions_do_not_emit_pad(self) -> None:
        """Keep SAME padding in Conv options for UINT8 and INT16 models."""
        for bit_width in (8, 16):
            with self.subTest(bit_width=bit_width):
                torch.manual_seed(20260731)
                exported = _export(
                    SamePaddingExportCNN(),
                    torch.randn(1, 3, 7, 9),
                    bit_width,
                )
                graph = exported.subgraphs[0]
                padding_values = []
                pad_count = 0
                for operator in graph.operators:
                    builtin = _builtin_code(exported, operator)
                    if builtin == circle.BuiltinOperator.BuiltinOperator.PAD:
                        pad_count += 1
                    if builtin in {
                        circle.BuiltinOperator.BuiltinOperator.CONV_2D,
                        circle.BuiltinOperator.BuiltinOperator.DEPTHWISE_CONV_2D,
                    }:
                        padding_values.append(operator.builtinOptions.padding)

                self.assertEqual(pad_count, 0)
                self.assertEqual(padding_values, [0, 0])

    def test_concat_wrapper_exports_one_shared_quantization_domain(self) -> None:
        """Use one observed qparam domain for all Concat inputs and its output."""
        for bit_width in (8, 16):
            with self.subTest(bit_width=bit_width):
                torch.manual_seed(20260801)
                model = ConcatHeadCNN().eval()
                with torch.no_grad():
                    model.head0.weight.mul_(0.01)
                    model.head1.weight.mul_(10.0)
                exported = _export(
                    model,
                    torch.randn(1, 3, 6, 6),
                    bit_width,
                )
                graph = exported.subgraphs[0]
                concat_ops = [
                    operator
                    for operator in graph.operators
                    if _builtin_code(exported, operator)
                    == circle.BuiltinOperator.BuiltinOperator.CONCATENATION
                ]
                self.assertEqual(len(concat_ops), 1)

                concat = concat_ops[0]
                tensors = [
                    graph.tensors[index] for index in [*concat.inputs, *concat.outputs]
                ]
                self.assertTrue(
                    all(tensor.quantization is not None for tensor in tensors)
                )
                qparams = [
                    (
                        tuple(tensor.quantization.scale),
                        tuple(tensor.quantization.zeroPoint),
                    )
                    for tensor in tensors
                ]
                self.assertTrue(all(qparam == qparams[0] for qparam in qparams[1:]))


if __name__ == "__main__":
    unittest.main()

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

"""Unit tests for ConvTranspose2d WrapQ support."""

from __future__ import annotations

import unittest

import tico
import torch

from circle_schema import circle
from tico.circle import CircleDocument
from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.nn.quant_conv_transpose2d import (
    QuantConvTranspose2d,
)
from tico.quantization.wrapq.wrappers.registry import lookup
from torch import nn


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


class _OutputSizedTranspose(nn.Module):
    def __init__(self, wrapped=False):
        super().__init__()
        conv = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, bias=False)
        conv.weight.data.fill_(1.0)
        self.conv = QuantConvTranspose2d(conv) if wrapped else conv

    def forward(self, input_):
        return self.conv(input_, output_size=(6, 6))


class QuantConvTranspose2dTest(unittest.TestCase):
    """Verify ConvTranspose2d registration, calibration, and fake quantization."""

    def test_registry_maps_torch_conv_transpose2d(self) -> None:
        """Resolve the native wrapper for torch.nn.ConvTranspose2d."""
        self.assertIs(lookup(nn.ConvTranspose2d), QuantConvTranspose2d)

    def test_weight_observer_uses_output_channel_axis(self) -> None:
        """Use ConvTranspose2d weight dimension one for per-channel weights."""
        wrapper = QuantConvTranspose2d(
            nn.ConvTranspose2d(3, 5, kernel_size=4, stride=2),
            qcfg=_quant_config(8),
        )
        self.assertEqual(wrapper.obs_weight.channel_axis, 1)

    def test_int16_uses_symmetric_activation_and_weight_qschemes(self) -> None:
        """Use per-tensor activation and per-channel weight symmetry for INT16."""
        wrapper = QuantConvTranspose2d(
            nn.ConvTranspose2d(3, 5, kernel_size=4, stride=2),
            qcfg=_quant_config(16),
        )
        self.assertIs(wrapper.obs_act_in.qscheme, QScheme.PER_TENSOR_SYMM)
        self.assertIs(wrapper.obs_act_out.qscheme, QScheme.PER_TENSOR_SYMM)
        self.assertIs(wrapper.obs_weight.qscheme, QScheme.PER_CHANNEL_SYMM)

    def test_upsampling_uint8_int16(self) -> None:
        """Calibrate and execute a stride-2 upsampling transposed convolution."""
        for bit_width in (8, 16):
            with self.subTest(bit_width=bit_width):
                module = nn.ConvTranspose2d(4, 4, kernel_size=4, stride=2, padding=3)
                wrapper = QuantConvTranspose2d(module, qcfg=_quant_config(bit_width))
                input_ = torch.randn(2, 4, 8, 8)
                wrapper.enable_calibration()
                wrapper(input_)
                wrapper.freeze_qparams()
                with torch.inference_mode():
                    output = wrapper(input_)
                self.assertEqual(output.shape, (2, 4, 12, 12))
                self.assertTrue(torch.isfinite(output).all())

    def test_quantized_output_tracks_float_reference(self) -> None:
        """Keep INT16 fake-quantized output close to the float reference."""
        module = nn.ConvTranspose2d(4, 4, kernel_size=4, stride=2, padding=1)
        input_ = torch.randn(1, 4, 8, 8)
        with torch.inference_mode():
            reference = module(input_)
        wrapper = QuantConvTranspose2d(module, qcfg=_quant_config(16))
        wrapper.enable_calibration()
        wrapper(input_)
        wrapper.freeze_qparams()
        with torch.inference_mode():
            output = wrapper(input_)
        self.assertLess((output - reference).abs().max().item(), 1e-2)

    def test_output_size_matches_native_for_batched_and_unbatched_inputs(self) -> None:
        """Accept spatial/full targets as positional or keyword arguments."""
        module = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, bias=False)
        module.weight.data.fill_(1.0)
        for batched in (False, True):
            input_ = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
            if batched:
                input_ = input_.unsqueeze(0)
            full_size = (1, 1, 6, 6) if batched else (1, 6, 6)
            for target in ([6, 6], full_size, torch.Size(full_size)):
                with self.subTest(batched=batched, target=target):
                    wrapper = QuantConvTranspose2d(module)
                    expected = module(input_, output_size=target)
                    torch.testing.assert_close(
                        wrapper(input_, target), expected, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        wrapper(input_, output_size=target), expected, rtol=0, atol=0
                    )

    def test_output_size_overrides_configured_padding_with_dilation(self) -> None:
        """Resolve each ambiguous axis using the native dilation-aware bounds."""
        module = nn.ConvTranspose2d(
            2,
            2,
            3,
            stride=(2, 3),
            padding=1,
            output_padding=(0, 2),
            dilation=(2, 1),
            groups=2,
        )
        module.weight.data.fill_(0.25)
        assert module.bias is not None
        module.bias.data.fill_(0.5)
        input_ = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3)
        wrapper = QuantConvTranspose2d(module)
        for target in (None, (7, 7), (8, 8), (8, 9)):
            with self.subTest(target=target):
                torch.testing.assert_close(
                    wrapper(input_, output_size=target),
                    module(input_, output_size=target),
                    rtol=0,
                    atol=0,
                )
        self.assertEqual(wrapper(input_).shape, (1, 2, 7, 9))

    def test_invalid_output_sizes_preserve_native_errors(self) -> None:
        """Reject unreachable sizes and invalid target ranks like ConvTranspose2d."""
        module = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1)
        wrapper = QuantConvTranspose2d(module)
        for batched in (False, True):
            input_ = torch.ones((1, 1, 3, 3) if batched else (1, 3, 3))
            targets: list[tuple[int, ...]] = [(4, 6), (7, 6), (6,), (1, 1, 1, 6, 6)]
            if not batched:
                targets.append((1, 1, 6, 6))
            for target in targets:
                with self.subTest(batched=batched, target=target):
                    with self.assertRaises(ValueError) as expected:
                        module(input_, output_size=target)
                    with self.assertRaises(ValueError) as actual:
                        wrapper(input_, output_size=target)
                    self.assertEqual(str(actual.exception), str(expected.exception))

    def test_calibration_and_quantization_honor_output_size(self) -> None:
        """Observe the requested output and keep fake-quantized dispatch equivalent."""
        input_ = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3) / 8
        for bit_width in (8, 16):
            with self.subTest(bit_width=bit_width):
                model = _OutputSizedTranspose().eval()
                expected = model(input_)
                prepared = prepare(model, _quant_config(bit_width), inplace=True)
                torch.testing.assert_close(prepared(input_), expected, rtol=0, atol=0)
                quantized = convert(prepared, inplace=True).eval()
                wrapper = next(
                    child
                    for child in quantized.modules()
                    if isinstance(child, QuantConvTranspose2d)
                )
                native = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, bias=False)
                native.weight.data.copy_(
                    wrapper.obs_weight.fake_quant(wrapper.module.weight)
                )
                input_q = wrapper.obs_act_in.fake_quant(input_)
                for target in ((5, 5), (5, 6), (6, 5), (6, 6)):
                    with self.subTest(target=target):
                        reference = wrapper.obs_act_out.fake_quant(
                            native(input_q, output_size=target)
                        )
                        torch.testing.assert_close(
                            wrapper(input_, output_size=target),
                            reference,
                            rtol=0,
                            atol=0,
                        )

    def test_output_size_exports_to_default_circle_with_matching_contract(self) -> None:
        """Serialize and execute the resolved target shape with default O1."""
        input_ = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
        model = _OutputSizedTranspose(wrapped=True).eval()
        expected = model(input_)
        converted = tico.convert(model, (input_,))
        document = CircleDocument.from_bytes(converted.circle_binary)
        self.assertTrue(document.verify().ok)
        subgraph = document.model.subgraphs[0]
        output = subgraph.tensors[subgraph.outputs[0]]
        self.assertEqual(tuple(output.shape), (1, 1, 6, 6))
        self.assertEqual(output.type, circle.TensorType.TensorType.FLOAT32)
        torch.testing.assert_close(
            torch.from_numpy(converted(input_)), expected, rtol=0, atol=0
        )
        prepared = prepare(
            _OutputSizedTranspose().eval(), _quant_config(16), inplace=True
        )
        prepared(input_)
        quantized = convert(prepared, inplace=True).eval()
        artifact = tico.convert(quantized, (input_,))
        document = CircleDocument.from_bytes(artifact.circle_binary)
        self.assertTrue(document.verify().ok)
        subgraph = document.model.subgraphs[0]
        output = subgraph.tensors[subgraph.outputs[0]]
        self.assertEqual(tuple(output.shape), (1, 1, 6, 6))
        self.assertEqual(output.type, circle.TensorType.TensorType.INT16)
        self.assertIsNotNone(output.quantization)


if __name__ == "__main__":
    unittest.main()

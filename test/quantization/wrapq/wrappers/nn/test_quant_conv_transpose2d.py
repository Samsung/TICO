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

import torch

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


if __name__ == "__main__":
    unittest.main()

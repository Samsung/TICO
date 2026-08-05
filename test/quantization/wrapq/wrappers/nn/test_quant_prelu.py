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

"""Unit tests for torch.nn.PReLU WrapQ support."""

from __future__ import annotations

import unittest

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.nn.quant_prelu import QuantPReLU
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


class QuantPReLUTest(unittest.TestCase):
    """Verify PReLU registration and fake-quantized execution."""

    def test_registry_maps_torch_prelu(self) -> None:
        """Resolve the native wrapper for torch.nn.PReLU."""
        self.assertIs(lookup(nn.PReLU), QuantPReLU)

    def test_slope_uses_per_channel_quantization(self) -> None:
        """Quantize the PReLU slope per channel along axis 0."""
        wrapper = QuantPReLU(
            nn.PReLU(num_parameters=4),
            qcfg=_quant_config(8),
        )
        self.assertEqual(wrapper.obs_weight.channel_axis, 0)
        self.assertIs(wrapper.obs_weight.qscheme, QScheme.PER_CHANNEL_ASYMM)

    def test_int16_slope_uses_per_channel_symmetric_quantization(self) -> None:
        """Keep the INT16 PReLU slope symmetric with per-channel scales."""
        wrapper = QuantPReLU(
            nn.PReLU(num_parameters=4),
            qcfg=_quant_config(16),
        )
        self.assertIs(wrapper.obs_weight.qscheme, QScheme.PER_CHANNEL_SYMM)

    def test_uint8_and_int16(self) -> None:
        """Calibrate and execute channel-wise PReLU at both bit widths."""
        input_ = torch.randn(2, 4, 8, 8)
        for bit_width in (8, 16):
            with self.subTest(bit_width=bit_width):
                wrapper = QuantPReLU(
                    nn.PReLU(num_parameters=4),
                    qcfg=_quant_config(bit_width),
                )
                wrapper.enable_calibration()
                wrapper(input_)
                wrapper.freeze_qparams()
                with torch.inference_mode():
                    output = wrapper(input_)
                self.assertEqual(tuple(output.shape), tuple(input_.shape))
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()

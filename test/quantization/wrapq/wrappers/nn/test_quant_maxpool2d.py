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

"""Unit tests for MaxPool2d WrapQ support."""

import unittest

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.nn.quant_maxpool2d import QuantMaxPool2d
from tico.quantization.wrapq.wrappers.registry import lookup
from torch import nn


def _quant_config(bit_width: int) -> PTQConfig:
    """Create the UINT8 or INT16 policy under test."""
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


class QuantMaxPool2dTest(unittest.TestCase):
    """Verify registration and shared MaxPool2d qparams."""

    def test_registry_maps_maxpool2d(self) -> None:
        """Resolve the native MaxPool2d wrapper."""
        self.assertIs(lookup(nn.MaxPool2d), QuantMaxPool2d)

    def test_input_and_output_share_one_observer(self) -> None:
        """Expose the exact same observer for both sides of MaxPool2d."""
        wrapper = QuantMaxPool2d(
            nn.MaxPool2d(kernel_size=2, stride=2),
            qcfg=_quant_config(8),
        )
        self.assertIs(wrapper.obs_act_in, wrapper.obs_act_out)
        self.assertEqual(len(tuple(wrapper._all_observers())), 1)

    def test_uint8_int16_execution(self) -> None:
        """Calibrate and execute UINT8 and INT16 quantization."""
        input_ = torch.randn(2, 4, 8, 8)
        for bit_width in (8, 16):
            with self.subTest(bit_width=bit_width):
                wrapper = QuantMaxPool2d(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    qcfg=_quant_config(bit_width),
                )
                wrapper.enable_calibration()
                wrapper(input_)
                wrapper.freeze_qparams()
                output = wrapper(input_)
                self.assertEqual(tuple(output.shape), (2, 4, 4, 4))
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()

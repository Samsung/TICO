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

"""Unit tests for the public TICO quantization boundary module."""

from __future__ import annotations

import unittest

import torch

from tico.quantization import QuantStub
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.quant_stub import QuantStubWrapper
from tico.quantization.wrapq.wrappers.registry import lookup


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


class QuantStubTest(unittest.TestCase):
    """Verify identity behavior and explicit activation fake quantization."""

    def test_float_module_is_stateless_identity(self) -> None:
        """Keep the floating-point boundary transparent and state-free."""
        module = QuantStub().eval()
        input_ = torch.randn(1, 3, 6, 6)
        self.assertIs(module(input_), input_)
        self.assertEqual(module.state_dict(), {})

    def test_registry_maps_quant_stub(self) -> None:
        """Resolve the native wrapper for the public boundary module."""
        self.assertIs(lookup(QuantStub), QuantStubWrapper)

    def test_uint8_and_int16(self) -> None:
        """Calibrate and execute the explicit input boundary at both widths."""
        input_ = torch.randn(1, 3, 6, 6)
        for bit_width in (8, 16):
            with self.subTest(bit_width=bit_width):
                wrapper = QuantStubWrapper(
                    QuantStub(),
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

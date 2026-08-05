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

"""Unit tests for the quantized concatenation boundary."""

import unittest

import torch

from tico.ops import Concat
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.ops.quant_concat import QuantConcat
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


class QuantConcatTest(unittest.TestCase):
    """Verify registration and one shared concatenation observer."""

    def test_registry_maps_concat(self) -> None:
        """Resolve the TICO Concat wrapper."""
        self.assertIs(lookup(Concat), QuantConcat)

    def test_shared_observer_collects_all_inputs(self) -> None:
        """Collect the union range of every concatenation input."""
        wrapper = QuantConcat(Concat(dim=1), qcfg=_quant_config(8))
        left = torch.full((1, 2, 3), -4.0)
        right = torch.full((1, 3, 3), 7.0)
        wrapper.enable_calibration()
        wrapper((left, right))
        self.assertEqual(wrapper.obs_act_out.min_val.item(), -4.0)
        self.assertEqual(wrapper.obs_act_out.max_val.item(), 7.0)

    def test_uint8_int16_execution(self) -> None:
        """Calibrate and execute UINT8 and INT16 quantization."""
        values = (torch.randn(2, 3, 4), torch.randn(2, 5, 4))
        for bit_width in (8, 16):
            with self.subTest(bit_width=bit_width):
                wrapper = QuantConcat(Concat(dim=1), qcfg=_quant_config(bit_width))
                wrapper.enable_calibration()
                wrapper(values)
                wrapper.freeze_qparams()
                output = wrapper(values)
                self.assertEqual(tuple(output.shape), (2, 8, 4))
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()

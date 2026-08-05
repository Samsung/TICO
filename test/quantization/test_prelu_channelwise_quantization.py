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

"""Tests for channel-wise PReLU slope quantization."""

from __future__ import annotations

import unittest

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.utils.reduce_utils import channelwise_minmax
from tico.quantization.wrapq.wrappers.nn.quant_prelu import QuantPReLU
from torch import nn


class ChannelwiseMinMaxTest(unittest.TestCase):
    """Verify rank-one per-channel statistics used by PReLU slopes."""

    def test_rank_one_tensor_keeps_one_value_per_channel(self) -> None:
        """Avoid collapsing a rank-one channel tensor into one global range."""
        values = torch.tensor([0.01, 0.10, 0.25, -0.05], dtype=torch.float32)

        minimum, maximum = channelwise_minmax(values, channel_axis=0)

        torch.testing.assert_close(minimum, values, rtol=0.0, atol=0.0)
        torch.testing.assert_close(maximum, values, rtol=0.0, atol=0.0)

    def test_multi_dimensional_tensor_reduces_non_channel_axes(self) -> None:
        """Preserve the existing reduction behavior for regular weight tensors."""
        values = torch.tensor(
            [
                [[1.0, -2.0], [3.0, 4.0]],
                [[-5.0, 6.0], [7.0, 8.0]],
            ]
        )

        minimum, maximum = channelwise_minmax(values, channel_axis=0)

        torch.testing.assert_close(minimum, torch.tensor([-2.0, -5.0]))
        torch.testing.assert_close(maximum, torch.tensor([4.0, 8.0]))


class QuantPReLUChannelwiseTest(unittest.TestCase):
    """Verify that one affine qparam is generated for every PReLU slope."""

    @staticmethod
    def _config(
        dtype: DType,
        activation_qscheme: QScheme,
        weight_qscheme: QScheme,
    ) -> PTQConfig:
        """Create a compact affine configuration for one wrapper test."""
        return PTQConfig(
            activation=affine(dtype, qscheme=activation_qscheme),
            weight=affine(dtype, qscheme=weight_qscheme),
            strict_wrap=False,
        )

    def _run_case(
        self,
        *,
        dtype: DType,
        activation_qscheme: QScheme,
        configured_weight_qscheme: QScheme,
        expected_weight_qscheme: QScheme,
    ) -> None:
        """Calibrate one wrapper and validate its slope quantization metadata."""
        module = nn.PReLU(num_parameters=4)
        slopes = torch.tensor([0.01, 0.10, 0.25, -0.05], dtype=torch.float32)
        module.weight.data.copy_(slopes)
        wrapper = QuantPReLU(
            module,
            qcfg=self._config(
                dtype,
                activation_qscheme,
                configured_weight_qscheme,
            ),
        )

        self.assertEqual(wrapper.obs_weight.qscheme, expected_weight_qscheme)
        self.assertEqual(wrapper.obs_weight.channel_axis, 0)

        wrapper.enable_calibration()
        wrapper(torch.randn(1, 4, 3, 3))
        wrapper.freeze_qparams()

        scale, zero_point = wrapper.obs_weight.compute_qparams()
        self.assertEqual(tuple(scale.shape), (4,))
        self.assertEqual(tuple(zero_point.shape), (4,))

        quantized_slopes = wrapper.obs_weight.fake_quant(module.weight.detach())
        torch.testing.assert_close(quantized_slopes, slopes, rtol=0.0, atol=0.0)

    def test_uint8_slope_uses_per_channel_asymmetric_quantization(self) -> None:
        """Convert an unsigned per-tensor role policy to per-channel slopes."""
        self._run_case(
            dtype=DType.uint(8),
            activation_qscheme=QScheme.PER_TENSOR_ASYMM,
            configured_weight_qscheme=QScheme.PER_TENSOR_ASYMM,
            expected_weight_qscheme=QScheme.PER_CHANNEL_ASYMM,
        )

    def test_int16_slope_uses_per_channel_symmetric_quantization(self) -> None:
        """Convert a signed per-tensor role policy to symmetric per-channel slopes."""
        self._run_case(
            dtype=DType.int(16),
            activation_qscheme=QScheme.PER_TENSOR_SYMM,
            configured_weight_qscheme=QScheme.PER_TENSOR_SYMM,
            expected_weight_qscheme=QScheme.PER_CHANNEL_SYMM,
        )

    def test_shared_slope_still_uses_one_channel_qparam(self) -> None:
        """Represent a shared one-element slope as one per-channel qparam."""
        module = nn.PReLU(num_parameters=1)
        wrapper = QuantPReLU(
            module,
            qcfg=self._config(
                DType.uint(8),
                QScheme.PER_TENSOR_ASYMM,
                QScheme.PER_CHANNEL_ASYMM,
            ),
        )

        wrapper.enable_calibration()
        wrapper(torch.randn(1, 3, 2, 2))
        wrapper.freeze_qparams()
        scale, zero_point = wrapper.obs_weight.compute_qparams()

        self.assertEqual(tuple(scale.shape), (1,))
        self.assertEqual(tuple(zero_point.shape), (1,))
        self.assertEqual(wrapper.obs_weight.channel_axis, 0)


if __name__ == "__main__":
    unittest.main()

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

"""Tests for the autograd-safe SAME-padding Conv2d path."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from tico.ops.same_padding_conv2d import _same_padding_2d, SamePaddingConv2d


class SamePaddingConv2dAutogradTest(unittest.TestCase):
    """Verify native regular/depthwise convolution parity and gradients."""

    def test_depthwise_stride_two_supports_activation_backward(self) -> None:
        module = SamePaddingConv2d(
            4,
            4,
            kernel_size=5,
            stride=2,
            groups=4,
            bias=True,
        )
        input_ = torch.randn(2, 4, 17, 19, requires_grad=True)

        output = module(input_)
        reference = _reference(module, input_)
        torch.testing.assert_close(output, reference)

        output.square().mean().backward()
        self.assertIsNotNone(input_.grad)
        self.assertTrue(torch.isfinite(input_.grad).all())
        self.assertGreater(float(input_.grad.abs().sum()), 0.0)

    def test_regular_dilated_convolution_supports_activation_backward(self) -> None:
        module = SamePaddingConv2d(
            3,
            6,
            kernel_size=3,
            stride=2,
            dilation=2,
            groups=1,
            bias=False,
        )
        input_ = torch.randn(1, 3, 16, 15, requires_grad=True)

        output = module(input_)
        reference = _reference(module, input_)
        torch.testing.assert_close(output, reference)

        output.sum().backward()
        self.assertIsNotNone(input_.grad)
        self.assertTrue(torch.isfinite(input_.grad).all())

    def test_same_padding_uses_extra_cell_on_bottom_and_right(self) -> None:
        self.assertEqual(
            _same_padding_2d(
                input_height=6,
                input_width=8,
                kernel_height=3,
                kernel_width=3,
                stride=(2, 2),
                dilation=(1, 1),
            ),
            (0, 1, 0, 1),
        )


def _reference(module: SamePaddingConv2d, input_: torch.Tensor) -> torch.Tensor:
    padding = _same_padding_2d(
        input_height=input_.shape[-2],
        input_width=input_.shape[-1],
        kernel_height=module.weight.shape[-2],
        kernel_width=module.weight.shape[-1],
        stride=module.stride,
        dilation=module.dilation,
    )
    return F.conv2d(
        F.pad(input_, padding),
        module.weight,
        module.bias,
        stride=module.stride,
        padding=0,
        dilation=module.dilation,
        groups=module.groups,
    )


if __name__ == "__main__":
    unittest.main()

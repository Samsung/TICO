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

"""Unit tests for Circle-compatible SAME-padding Conv2d."""

import unittest
from collections import Counter

import torch
import torch.nn.functional as F

from tico.ops import SamePaddingConv2d


def _same_pad(
    input_size: int,
    kernel_size: int,
    stride: int,
    dilation: int,
) -> tuple[int, int]:
    """Return TensorFlow Lite SAME padding before and after one dimension."""
    output_size = (input_size + stride - 1) // stride
    effective_kernel = (kernel_size - 1) * dilation + 1
    total = max((output_size - 1) * stride + effective_kernel - input_size, 0)
    before = total // 2
    return before, total - before


def _reference(module: SamePaddingConv2d, input_: torch.Tensor) -> torch.Tensor:
    """Run an explicit-pad PyTorch reference for one SAME convolution."""
    top, bottom = _same_pad(
        input_.shape[2],
        module.kernel_size[0],
        module.stride[0],
        module.dilation[0],
    )
    left, right = _same_pad(
        input_.shape[3],
        module.kernel_size[1],
        module.stride[1],
        module.dilation[1],
    )
    padded = F.pad(input_, (left, right, top, bottom))
    return F.conv2d(
        padded,
        module.weight,
        module.bias,
        module.stride,
        0,
        module.dilation,
        module.groups,
    )


class SamePaddingConv2dTest(unittest.TestCase):
    """Verify regular and depthwise SAME-padding semantics and export."""

    def test_regular_stride_two_matches_reference(self) -> None:
        """Match asymmetric 5x5 stride-two SAME padding."""
        module = SamePaddingConv2d(3, 4, kernel_size=5, stride=2).eval()
        input_ = torch.randn(2, 3, 12, 14)
        torch.testing.assert_close(module(input_), _reference(module, input_))

    def test_depthwise_stride_two_matches_reference(self) -> None:
        """Match depthwise asymmetric SAME padding."""
        module = SamePaddingConv2d(
            4,
            8,
            kernel_size=5,
            stride=2,
            groups=4,
        ).eval()
        input_ = torch.randn(2, 4, 11, 13)
        torch.testing.assert_close(module(input_), _reference(module, input_))

    def test_torch_export_keeps_circle_same_conv(self) -> None:
        """Keep one opaque Circle Conv2D and avoid an explicit pad node."""
        module = SamePaddingConv2d(3, 4, kernel_size=5, stride=2).eval()
        exported = torch.export.export(
            module,
            (torch.randn(1, 3, 12, 14),),
            strict=True,
        )
        targets = Counter(
            str(node.target)
            for node in exported.graph.nodes
            if node.op == "call_function"
        )
        self.assertEqual(targets["circle_custom.conv2d.padding"], 1)
        self.assertEqual(targets["aten.constant_pad_nd.default"], 0)

    def test_rejects_general_grouped_convolution(self) -> None:
        """Reject grouped convolution that is neither regular nor depthwise."""
        with self.assertRaisesRegex(ValueError, "regular or depthwise"):
            SamePaddingConv2d(4, 8, kernel_size=3, groups=2)


if __name__ == "__main__":
    unittest.main()

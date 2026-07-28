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

import torch

from tico.passes.legalize_predefined_layout_operators import (
    LegalizePreDefinedLayoutOperators,
)

from test.support.helper import num_of_ops
from test.support.pass_value_test import SinglePassValueTest


class ChannelWisePReLUNet(torch.nn.Module):
    """Apply one PReLU slope per channel to an NCHW input."""

    def __init__(self) -> None:
        super().__init__()
        self.prelu = torch.nn.PReLU(num_parameters=4)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Return the channel-wise PReLU output."""
        return self.prelu(input_)

    def get_example_inputs(self):
        """Return representative inputs for export and value comparison."""
        return (torch.randn(1, 4, 5, 7),), {}


class SharedPReLUNet(torch.nn.Module):
    """Apply one PReLU slope shared by every input element."""

    def __init__(self) -> None:
        super().__init__()
        self.prelu = torch.nn.PReLU(num_parameters=1)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Return the shared-slope PReLU output."""
        return self.prelu(input_)

    def get_example_inputs(self):
        """Return representative inputs for export and value comparison."""
        return (torch.randn(1, 4, 5, 7),), {}


class ConvChannelWisePReLUNet(torch.nn.Module):
    """Apply channel-wise PReLU directly after an NCHW convolution."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.prelu = torch.nn.PReLU(num_parameters=4)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Return the convolution followed by channel-wise PReLU."""
        return self.prelu(self.conv(input_))

    def get_example_inputs(self):
        """Return representative inputs for export and value comparison."""
        return (torch.randn(1, 3, 5, 7),), {}


class LegalizeChannelWisePReLUTest(SinglePassValueTest):
    """Verify channel-wise PReLU layout legalization and numerical parity."""

    def test_pass(self):
        """Replace aten.prelu with the channel-last Circle custom operator."""
        self.setup(ChannelWisePReLUNet())
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.prelu.default]), 1
        )

        self.run_value_test(LegalizePreDefinedLayoutOperators())

        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.prelu.default]), 0
        )
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.circle_custom.prelu]), 1
        )
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.permute.default]), 2
        )

    def test_shared_slope_is_not_rewritten(self):
        """Keep a broadcast-safe shared PReLU slope in native aten form."""
        self.setup(SharedPReLUNet())
        self.run_value_test(LegalizePreDefinedLayoutOperators())

        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.prelu.default]), 1
        )
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.circle_custom.prelu]), 0
        )

    def test_reuses_channel_last_convolution_output(self):
        """Avoid an inverse permutation pair between Conv2D and PReLU."""
        self.setup(ConvChannelWisePReLUNet())
        self.run_value_test(LegalizePreDefinedLayoutOperators())

        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.circle_custom.conv2d]), 1
        )
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.circle_custom.prelu]), 1
        )

        graph_nodes = list(self.exported_program().graph.nodes)
        circle_conv = next(
            node
            for node in graph_nodes
            if node.target == torch.ops.circle_custom.conv2d
        )
        circle_prelu = next(
            node for node in graph_nodes if node.target == torch.ops.circle_custom.prelu
        )

        self.assertIs(circle_prelu.args[0], circle_conv)

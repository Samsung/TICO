# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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
from tico.passes import ops
from tico.passes.convert_conv3d_to_conv2d import ConvertConv3dToConv2d

from test.utils.helper import num_of_ops
from test.utils.pass_value_test import SinglePassValueTest


class Conv3dBasic(torch.nn.Module):
    """Basic Conv3D with padding"""

    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=3,
            out_channels=4,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(2, 3, 8, 16, 16),), {}


class ConvertConv3dBasicTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(Conv3dBasic())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv3d), 1)
        self.run_value_test(ConvertConv3dToConv2d())
        # Check whether Conv3D is transformed to Conv2D after removal
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv3d), 0)
        self.assertGreaterEqual(num_of_ops(self.exported_program(), ops.aten.conv2d), 1)


class Conv3dNoPadding(torch.nn.Module):
    """Conv3D without padding"""

    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=2,
            out_channels=3,
            kernel_size=(2, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 0, 0),
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 2, 10, 32, 32),), {}


class ConvertConv3dNoPaddingTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(Conv3dNoPadding())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv3d), 1)
        self.run_value_test(ConvertConv3dToConv2d())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv3d), 0)
        self.assertGreaterEqual(num_of_ops(self.exported_program(), ops.aten.conv2d), 1)


class Conv3dWithDilation(torch.nn.Module):
    """Conv3D with dilation"""

    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=4,
            out_channels=6,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            dilation=(2, 1, 1),
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 4, 12, 24, 24),), {}


class ConvertConv3dWithDilationTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(Conv3dWithDilation())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv3d), 1)
        self.run_value_test(ConvertConv3dToConv2d())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv3d), 0)
        self.assertGreaterEqual(num_of_ops(self.exported_program(), ops.aten.conv2d), 1)


class Conv3dWithStride(torch.nn.Module):
    """Conv3D with different strides"""

    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=3,
            out_channels=5,
            kernel_size=(2, 4, 4),
            stride=(2, 2, 2),
            padding=(0, 1, 1),
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(2, 3, 16, 32, 32),), {}


class ConvertConv3dWithStrideTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(Conv3dWithStride())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv3d), 1)
        self.run_value_test(ConvertConv3dToConv2d())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv3d), 0)
        self.assertGreaterEqual(num_of_ops(self.exported_program(), ops.aten.conv2d), 1)


class Conv3dWithoutBias(torch.nn.Module):
    """Conv3D without bias"""

    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=3,
            out_channels=4,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False,
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 3, 8, 16, 16),), {}


class ConvertConv3dWithoutBiasTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(Conv3dWithoutBias())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv3d), 1)
        self.run_value_test(ConvertConv3dToConv2d())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv3d), 0)
        self.assertGreaterEqual(num_of_ops(self.exported_program(), ops.aten.conv2d), 1)


class Conv3dGroups(torch.nn.Module):
    """Conv3D with groups (depthwise-like)"""

    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=8,
            out_channels=8,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            groups=8,
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 8, 6, 12, 12),), {}


class ConvertConv3dGroupsTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(Conv3dGroups())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv3d), 1)
        self.run_value_test(ConvertConv3dToConv2d())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv3d), 0)
        self.assertGreaterEqual(num_of_ops(self.exported_program(), ops.aten.conv2d), 1)


class Conv3dLargeKernel(torch.nn.Module):
    """Conv3D with large temporal kernel"""

    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=2,
            out_channels=3,
            kernel_size=(5, 3, 3),
            stride=(1, 1, 1),
            padding=(2, 1, 1),
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 2, 10, 16, 16),), {}


class ConvertConv3dLargeKernelTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(Conv3dLargeKernel())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv3d), 1)
        self.run_value_test(ConvertConv3dToConv2d())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv3d), 0)
        self.assertGreaterEqual(num_of_ops(self.exported_program(), ops.aten.conv2d), 1)


class Conv3dMultipleLayers(torch.nn.Module):
    """Model with multiple Conv3D layers"""

    def __init__(self):
        super().__init__()
        self.conv3d_1 = torch.nn.Conv3d(
            in_channels=3,
            out_channels=4,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        self.conv3d_2 = torch.nn.Conv3d(
            in_channels=4,
            out_channels=6,
            kernel_size=(2, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

    def forward(self, input):
        x = self.conv3d_1(input)
        x = torch.relu(x)
        x = self.conv3d_2(x)
        return x

    def get_example_inputs(self):
        return (torch.randn(1, 3, 8, 32, 32),), {}


class ConvertConv3dMultipleLayersTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(Conv3dMultipleLayers())
        initial_conv3d_count = num_of_ops(self.exported_program(), ops.aten.conv3d)
        self.assertEqual(initial_conv3d_count, 2)
        self.run_value_test(ConvertConv3dToConv2d())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv3d), 0)
        self.assertGreaterEqual(num_of_ops(self.exported_program(), ops.aten.conv2d), 2)

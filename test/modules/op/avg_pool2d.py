# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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
from packaging.version import Version
from torch.export import Dim

from test.modules.base import TestModuleBase

from test.utils.tag import skip, skip_if, use_onert


class SimpleAvgPool(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=2)

    def forward(self, tensor):
        result = self.avgpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),), {}


@use_onert
class SimpleAvgPoolDynamicShape(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=2)

    def forward(self, tensor):
        result = self.avgpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),), {}

    def get_dynamic_shapes(self):
        batch = Dim("batch", min=1, max=128)
        dynamic_shapes = {
            "tensor": {0: batch},
        }
        return dynamic_shapes


@use_onert
class SimpleAvgPoolWithAddDynamicShape(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=2)

    def forward(self, x, y):
        z = x + y
        result = self.avgpool(z)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),), {"y": torch.randn(2, 4, 8, 16)}

    def get_dynamic_shapes(self):
        batch = Dim("batch", min=1, max=128)
        dynamic_shapes = {
            "x": {0: batch},
            "y": {0: batch},
        }
        return dynamic_shapes


class AdaptiveAvgPool(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, tensor):
        result = self.avgpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(1, 128, 7, 7),), {}

    # This test should be done after applying NCHW_to_NHWC with onecc.
    # def get_calibration_data(self):
    #     for _ in range(100):
    #         yield self.get_example_inputs()


class AvgPoolWithPaddingKwargs(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, tensor0, tensor1):
        result = self.avgpool(tensor0) + tensor1
        return result

    def get_example_inputs(self):
        # Reverse-ordered kwargs
        return (), {
            "tensor1": torch.randn(2, 4, 4, 8),
            "tensor0": torch.randn(2, 4, 8, 16),
        }


class AvgPoolWithPadding(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, tensor):
        result = self.avgpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),), {}


class AvgPoolWithSamePadding(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, tensor):
        result = self.avgpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),), {}


class AvgPoolWithoutStride(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3)

    def forward(self, tensor):
        result = self.avgpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 17),), {}


class AvgPoolFunctionalWithoutStride(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        result = torch.nn.functional.avg_pool2d(tensor, kernel_size=3)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 17),), {}


class AvgPoolNonSquareWindow(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=(3, 2), stride=(2, 1))

    def forward(self, tensor):
        result = self.avgpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),), {}


class AvgPoolWithSamePaddingNoCountIncludePad(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), count_include_pad=False
        )

    def forward(self, tensor):
        result = self.avgpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(1, 3, 56, 56),), {}


class AvgPoolWithNoPaddingNoCountIncludePad(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(
            kernel_size=(3, 3), stride=(1, 1), count_include_pad=False
        )

    def forward(self, tensor):
        result = self.avgpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(1, 3, 56, 56),), {}


@skip(reason="Not supported yet")
class AvgPoolWithNoSamePaddingNoCountIncludePad(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), count_include_pad=False
        )

    def forward(self, tensor):
        result = self.avgpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(1, 3, 56, 56),), {}

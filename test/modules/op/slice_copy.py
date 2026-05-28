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

from test.modules.base import TestModuleBase

from test.support import tag


class SimpleSliceCopy(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        # equivalent to:              ... dim=1, start=1, end=5,    step=2)
        z = torch.slice_copy(input=input_, dim=1, start=1, end=None, step=2)
        return z

    def get_example_inputs(self):
        return (torch.randn(5, 5),), {}


class SimpleSliceCopyWithMinus(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        # equivalent to:              ... dim= 1, start= 4, end=5, step=1)
        z = torch.slice_copy(input=input_, dim=-3, start=-1, end=5, step=1)
        return z

    def get_example_inputs(self):
        return (torch.randn(5, 5, 5, 5),), {}


class SimpleSliceCopyWithOutOfBound(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        # equivalent to:              ... dim=1, start=2, end=5, step=2)
        z = torch.slice_copy(input=input_, dim=1, start=2, end=6, step=2)
        return z

    def get_example_inputs(self):
        return (torch.randn(5, 5, 5),), {}


class SimpleSliceOperator(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        # z = torch.slice_copy(input=input_, dim=1, start=1, end=None, step=2)
        z = input_[:, 1::2]
        return z

    def get_example_inputs(self):
        return (torch.randn(5, 5),), {}


class SimpleSliceOperatorWithMinus(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        # z = torch.slice_copy(input=input_, dim=3, start=-1, end=5, step=1)
        z = input_[:, :, :, -1:5:1]
        return z

    def get_example_inputs(self):
        return (torch.randn(5, 5, 5, 5),), {}


class SimpleSliceOperatorWithOutOfBound(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        # z = torch.slice_copy(input=input_, dim=1, start=2, end=6, step=2)
        z = input_[:, 2:6:2, :]
        return z

    def get_example_inputs(self):
        return (torch.randn(5, 5, 5),), {}


@tag.test_negative(expected_err="end(2) must be greater than start (3)")
class SimpleSliceCopyWithInvalidArgs(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        # output tensor must be non-zero for circle strided slice
        z = torch.slice_copy(input=input_, dim=1, start=3, end=2, step=2)
        return z

    def get_example_inputs(self):
        return (torch.randn(5, 5, 5),), {}

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
from torch.export import Dim

from test.utils import tag


class SimpleRepeat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = x.repeat((128, 1, 1, 1))
        return z

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(1, 4),)


@tag.use_onert
class SimpleRepeatDynamicShape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = x.repeat((128, 1, 1, 1))
        return z

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(1, 4),)

    def get_dynamic_shapes(self):
        dim = Dim("dim", min=1, max=128)
        dynamic_shapes = {
            "x": {1: dim},
        }
        return dynamic_shapes


class SimpleRepeat2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = x.repeat((4, 2, 1))
        return z

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(1, 3),)


class RepeatTwiceH(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = x.repeat(1, 1, 2, 1)
        return z

    def get_example_inputs(self):
        return (torch.randn(1, 2, 3, 3),)


class RepeatTwiceHW(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = x.repeat(1, 1, 2, 2)
        return z

    def get_example_inputs(self):
        return (torch.randn(1, 2, 3, 3),)


class RepeatThreetimesHW(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = x.repeat(1, 1, 3, 3)
        return z

    def get_example_inputs(self):
        return (torch.randn(1, 2, 3, 3),)


class RepeatTwiceHThreeTimesW(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = x.repeat(1, 1, 2, 3)
        return z

    def get_example_inputs(self):
        return (torch.randn(1, 2, 3, 3),)


@tag.skip(reason="Not Support Operator")
class RepeatLongerRepeats(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = x.repeat(1, 2, 3, 3)
        return z

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(1, 2, 3),)

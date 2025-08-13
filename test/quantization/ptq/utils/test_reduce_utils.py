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

import unittest

import torch
from tico.experimental.quantization.ptq.utils.reduce_utils import channelwise_minmax


class TestChannelwiseMinMax(unittest.TestCase):
    def test_keep_dim0(self):
        # (C, N) : keep channel dim = 0
        x = torch.tensor([[1.0, 2.0, -3.0], [4.0, -5.0, 0.5]])

        mins, maxs = channelwise_minmax(x, channel_axis=0)

        self.assertTrue(torch.equal(mins, torch.tensor([-3.0, -5.0])))
        self.assertTrue(torch.equal(maxs, torch.tensor([2.0, 4.0])))

    def test_keep_middle_dim(self):
        # Shape (B, C, L) – keep dim 1
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        mins, maxs = channelwise_minmax(x, 1)

        # Reference using PyTorch’s built-in reductions
        ref_min = x.amin(dim=(0, 2))
        ref_max = x.amax(dim=(0, 2))

        self.assertTrue(torch.equal(mins, ref_min))
        self.assertTrue(torch.equal(maxs, ref_max))

    def test_keep_negative_index(self):
        # Negative index should work the same as positive one
        x = torch.tensor([[7.0, -2.0, 3.0]])
        mins_pos, maxs_pos = channelwise_minmax(x, 1)  # keep last dim
        mins_neg, maxs_neg = channelwise_minmax(x, -1)  # same dim via -1

        self.assertTrue(
            torch.equal(mins_pos, mins_neg),
            f"mins_pos: {mins_pos}, mins_neg: {mins_neg}",
        )
        self.assertTrue(torch.equal(maxs_pos, maxs_neg))

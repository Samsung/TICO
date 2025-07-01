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
from tico.passes.const_prop_pass import ConstPropPass
from tico.passes.segment_index_select import SegmentIndexSelectConst

from test.utils.helper import num_of_ops
from test.utils.pass_value_test import SinglePassValueTest


class SimpleIndexSelectWithConstIndex(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("idx", torch.tensor([3, 1, 2]))

    def forward(self, x, y):
        result = torch.index_select(x, 2, self.idx) + y
        return result

    def get_example_inputs(self):
        return (torch.rand(3, 4, 5), torch.rand(3, 4, 3))


class TestSegmentIndexSelect(SinglePassValueTest):
    def test_pass(self):
        self.setup(SimpleIndexSelectWithConstIndex())
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.index_select.default]),
            1,
        )

        self.run_pass(ConstPropPass())
        self.run_value_test(SegmentIndexSelectConst())

        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.index_select.default]),
            3,
        )
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.cat.default]), 1
        )

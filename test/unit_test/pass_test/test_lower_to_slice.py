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
from tico.passes.fill_meta_val import FillMetaVal
from tico.passes.lower_to_slice import LowerIndexSelectToSlice, LowerSelectCopyToSlice
from tico.passes.remove_nop import RemoveNop
from tico.passes.segment_index_select import SegmentIndexSelectConst

from test.modules.op.index_select import (
    SimpleIndexSelectWithConstIndex,
    SimpleIndexSelectWithConstScalarIndex,
)
from test.modules.op.select import SimpleConstIndex
from test.utils.helper import num_of_ops
from test.utils.pass_value_test import SinglePassValueTest


class TestLowerSelectCopyToSlice(SinglePassValueTest):
    def test_pass(self):
        self.setup(SimpleConstIndex())
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.select.int]), 1
        )

        num_slice_before = num_of_ops(
            self.exported_program(), [torch.ops.aten.slice.Tensor]
        )
        self.run_value_test(LowerSelectCopyToSlice())
        num_slice_after = num_of_ops(
            self.exported_program(), [torch.ops.aten.slice.Tensor]
        )

        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.select.int]), 0
        )
        self.assertEqual(num_slice_after - num_slice_before, 1)


class TestLowerIndexSelectToSliceWithScalarIndex(SinglePassValueTest):
    def test_pass(self):
        self.setup(SimpleIndexSelectWithConstScalarIndex())
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.index_select.default]),
            1,
        )

        self.run_pass(ConstPropPass())
        self.run_pass(RemoveNop())
        self.run_value_test(LowerIndexSelectToSlice())
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.index_select.default]),
            0,
        )
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.slice.Tensor]), 1
        )
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.reshape.default]), 1
        )


class TestLowerIndexSelectToSliceWithLongIndice(SinglePassValueTest):
    def test_pass(self):
        self.setup(SimpleIndexSelectWithConstIndex())
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.index_select.default]),
            1,
        )

        self.run_pass(ConstPropPass())
        self.run_pass(RemoveNop())
        self.run_pass(SegmentIndexSelectConst())
        self.run_pass(FillMetaVal())
        self.run_value_test(LowerIndexSelectToSlice())

        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.index_select.default]),
            0,
        )
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.slice.Tensor]), 3
        )
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.reshape.default]), 3
        )
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.cat.default]), 1
        )

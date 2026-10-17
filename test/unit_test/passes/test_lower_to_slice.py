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

import tico
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
from test.support.helper import num_of_ops
from test.support.pass_value_test import SinglePassValueTest


class _Select(torch.nn.Module):
    def __init__(self, index, dim=1, copy=False, reuse=False):
        super().__init__()
        self.index = index
        self.dim = dim
        self.copy = copy
        self.reuse = reuse

    def forward(self, x):
        if self.copy:
            selected = torch.select_copy(x, self.dim, self.index)
        else:
            selected = torch.select(x, self.dim, self.index)
        if self.reuse:
            return selected, selected + 1, x + 1
        return selected


class TestLowerSelectCopyToSliceBounds(unittest.TestCase):
    def _assert_equal(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.equal(actual, expected))

    def _lower(self, model, inputs, dynamic_shapes=None):
        ep = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
        select = next(
            n
            for n in ep.graph.nodes
            if n.target in (torch.ops.aten.select.int, torch.ops.aten.select_copy.int)
        )
        metadata = select.meta["val"]
        signature = tuple(ep.graph_signature.input_specs)
        constraints = dict(ep.range_constraints)
        with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
            self.assertTrue(LowerSelectCopyToSlice().call(ep).modified)
        reshape = next(
            n for n in ep.graph.nodes if n.target == torch.ops.aten.reshape.default
        )
        self.assertIs(reshape.meta["val"], metadata)
        self.assertEqual(tuple(ep.graph_signature.input_specs), signature)
        self.assertEqual(ep.range_constraints, constraints)
        ep.graph.lint()
        graph = str(ep.graph)
        self.assertFalse(LowerSelectCopyToSlice().call(ep).modified)
        self.assertEqual(str(ep.graph), graph)
        return ep

    def test_negative_and_positive_indices_preserve_values_shape_and_dtype(self):
        for dtype in (torch.float32, torch.int32):
            inputs = (torch.arange(24, dtype=dtype).reshape(2, 3, 4),)
            for index in (-3, -2, -1, 0, 2):
                for dim in (1, -2):
                    for copy in (False, True):
                        with self.subTest(dtype=dtype, index=index, dim=dim, copy=copy):
                            model = _Select(index, dim, copy)
                            ep = self._lower(model, inputs)
                            self._assert_equal(ep.module()(*inputs), model(*inputs))
                            slice_node = next(
                                n
                                for n in ep.graph.nodes
                                if n.target == torch.ops.aten.slice.Tensor
                            )
                            expected_end = None if index == -1 else index + 1
                            self.assertEqual(
                                slice_node.args[1:], (1, index, expected_end, 1)
                            )
                            FillMetaVal().call(ep)
                            self.assertEqual(slice_node.meta["val"].shape, (2, 1, 4))
                            self.assertEqual(slice_node.meta["val"].dtype, dtype)
                            self.assertEqual(
                                num_of_ops(
                                    ep,
                                    [
                                        torch.ops.aten.select.int,
                                        torch.ops.aten.select_copy.int,
                                    ],
                                ),
                                0,
                            )

    def test_other_users_and_input_nonmutation(self):
        for copy in (False, True):
            with self.subTest(copy=copy):
                inputs = (torch.arange(6, dtype=torch.int32).reshape(2, 3),)
                original = inputs[0].clone()
                model = _Select(-1, copy=copy, reuse=True)
                ep = self._lower(model, inputs)
                for actual, expected in zip(ep.module()(*inputs), model(*inputs)):
                    self._assert_equal(actual, expected)
                self._assert_equal(inputs[0], original)

    def test_dynamic_selected_axis_is_not_specialized(self):
        width = torch.export.Dim("width", min=3, max=8)
        inputs = (torch.arange(8, dtype=torch.float32).reshape(2, 4),)
        for index in (-3, -2, -1, 0, 2):
            with self.subTest(index=index):
                model = _Select(index, dim=-1)
                ep = self._lower(model, inputs, ({1: width},))
                placeholder = next(n for n in ep.graph.nodes if n.op == "placeholder")
                self.assertIsInstance(placeholder.meta["val"].shape[1], torch.SymInt)
                for size in (3, 5, 8):
                    actual_inputs = (
                        torch.arange(2 * size, dtype=torch.float32).reshape(2, size),
                    )
                    self._assert_equal(
                        ep.module()(*actual_inputs), model(*actual_inputs)
                    )

    def test_default_conversion_reference_parity(self):
        for dtype in (torch.float32, torch.int32):
            inputs = (torch.arange(6, dtype=dtype).reshape(2, 3),)
            original = inputs[0].clone()
            for index in (-3, -1):
                for copy in (False, True):
                    with self.subTest(dtype=dtype, index=index, copy=copy):
                        model = _Select(index, dim=-1, copy=copy).eval()
                        circle = tico.convert(model, inputs)
                        self.assertEqual(circle.runtime, "reference")
                        self._assert_equal(
                            torch.from_numpy(circle(*inputs)), model(*inputs)
                        )
                        self._assert_equal(inputs[0], original)

    def test_unrelated_slice_is_unchanged(self):
        class Slice(torch.nn.Module):
            def forward(self, x):
                return x[:, -1:]

        ep = torch.export.export(Slice(), (torch.arange(6).reshape(2, 3),))
        graph = str(ep.graph)
        self.assertFalse(LowerSelectCopyToSlice().call(ep).modified)
        self.assertEqual(str(ep.graph), graph)


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

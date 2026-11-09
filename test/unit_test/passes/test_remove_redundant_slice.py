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
from tico.passes import ops
from tico.passes.remove_redundant_slice import RemoveRedundantSlice
from tico.utils.compat.torch import export_produces_slice

from test.support.helper import num_of_ops
from test.support.pass_value_test import SinglePassValueTest


class RedundantSliceNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = x[0, :]
        return z

    def get_example_inputs(self):
        return (torch.randn(1, 4),), {}


class ExplicitSliceNet(torch.nn.Module):
    def forward(self, x):
        return x.clone()

    def get_example_inputs(self):
        return (torch.arange(8).reshape(2, 4),), {}


class RemoveRedundantSliceTest(SinglePassValueTest):
    def _setup_explicit_slice(self, start, end, step):
        # Export only the surrounding graph; inject the pattern so coverage does
        # not depend on whether this Torch exporter retains a redundant slice.
        self.setup(ExplicitSliceNet())
        ep = self.exported_program()
        clone = next(
            node
            for node in ep.graph.nodes
            if node.target == torch.ops.aten.clone.default
        )
        source = clone.args[0]
        with ep.graph.inserting_before(clone):
            sliced = ep.graph.call_function(
                torch.ops.aten.slice.Tensor, (source, 1, start, end, step)
            )
        sliced.meta["val"] = torch.ops.aten.slice.Tensor(
            source.meta["val"], 1, start, end, step
        )
        clone.update_arg(0, sliced)
        clone.meta["val"] = sliced.meta["val"].clone()
        ep.graph.lint()
        ep.graph_module.recompile()
        self.assertEqual(num_of_ops(ep, ops.aten.slice), 1)

    def test_explicit_full_slice(self):
        self._setup_explicit_slice(0, 9223372036854775807, 1)
        self.run_value_test(RemoveRedundantSlice())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.slice), 0)

    def test_partial_slice_is_preserved(self):
        self._setup_explicit_slice(1, 3, 1)
        self.run_value_test(RemoveRedundantSlice())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.slice), 1)

    def test_non_unit_step_is_preserved(self):
        self._setup_explicit_slice(0, 9223372036854775807, 2)
        self.run_value_test(RemoveRedundantSlice())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.slice), 1)

    @unittest.skipUnless(
        export_produces_slice(),
        "Skip when torch doesn't produce redundant slices.",
    )
    def test_pass(self):
        self.setup(RedundantSliceNet())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.slice), 1)

        self.run_value_test(RemoveRedundantSlice())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.slice), 0)

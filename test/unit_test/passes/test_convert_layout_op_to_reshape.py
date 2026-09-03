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

import unittest

import torch

from tico.passes import ops
from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape

from test.support.helper import num_of_ops
from test.support.pass_value_test import SinglePassValueTest


def _reshape_sizes(exported_program):
    """Return the size argument of every ``aten.reshape`` node."""
    return [
        list(node.args[1])
        for node in exported_program.graph.nodes
        if node.op == "call_function" and node.target in ops.aten.reshape
    ]


class StaticView(torch.nn.Module):
    def forward(self, x):
        # (2, 12) -> (2, 3, 4): the user's -1 is resolved from the static shape.
        return x.view(2, -1, 4)

    def get_example_inputs(self):
        return (torch.randn(2, 12),), {}


class ConvertStaticViewToReshapeTest(SinglePassValueTest):
    def test_pass(self):
        """Static graphs keep the fully resolved output shape."""
        self.setup(StaticView())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 0)

        self.run_value_test(ConvertLayoutOpToReshape())

        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 1)
        self.assertEqual(_reshape_sizes(self.exported_program()), [[2, 3, 4]])


class DynamicSeqView(torch.nn.Module):
    def forward(self, x):
        # (1, S, 8) -> (1, S, 2, 4) written with an explicit -1 for S.
        return (x * 2.0).view(1, -1, 2, 4)


class DynamicSeqUnsqueeze(torch.nn.Module):
    def forward(self, x):
        # (1, S, 8) -> (1, 1, S, 8): no explicit size is available.
        return (x * 2.0).unsqueeze(1)


class ConvertDynamicLayoutOpToReshapeTest(unittest.TestCase):
    def _export(self, mod, x):
        seq_dim = torch.export.Dim("seq", min=1, max=16)
        with torch.no_grad():
            ep = torch.export.export(
                mod.eval(),
                (x,),
                dynamic_shapes={"x": {1: seq_dim}},
            )
        return ep.run_decompositions()

    def _assert_values(self, ep, mod):
        for seq_len in (1, 4, 16):
            x = torch.randn(1, seq_len, 8)
            self.assertTrue(torch.equal(ep.module()(x), mod(x)))

    def test_view_keeps_explicit_static_size_with_minus_one(self):
        """A dynamic view must keep the user's all-int size instead of SymInt."""
        mod = DynamicSeqView()
        ep = self._export(mod, torch.randn(1, 6, 8))

        result = ConvertLayoutOpToReshape().call(ep)

        self.assertTrue(result.modified)
        sizes = _reshape_sizes(ep)
        self.assertEqual(sizes, [[1, -1, 2, 4]])
        self.assertTrue(all(isinstance(dim, int) for dim in sizes[0]))
        self._assert_values(ep, mod)

    def test_single_symbolic_dim_is_inferred_as_minus_one(self):
        """Layout ops without an explicit size infer the one symbolic dim."""
        mod = DynamicSeqUnsqueeze()
        ep = self._export(mod, torch.randn(1, 6, 8))

        result = ConvertLayoutOpToReshape().call(ep)

        self.assertTrue(result.modified)
        sizes = _reshape_sizes(ep)
        self.assertEqual(len(sizes), 1)
        self.assertEqual(sizes[0], [1, 1, -1, 8])
        self._assert_values(ep, mod)


if __name__ == "__main__":
    unittest.main()

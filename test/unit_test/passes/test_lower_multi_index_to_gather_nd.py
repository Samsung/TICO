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

import torch
from tico.passes.lower_multi_index_to_gather_nd import LowerMultiIndexToGatherNd

from test.support.helper import num_of_ops
from test.support.pass_value_test import SinglePassValueTest

INDEX = [torch.ops.aten.index.Tensor]
GATHER_ND = [torch.ops.circle_custom.gather_nd]


def _single_node(ep, targets):
    nodes = [
        n for n in ep.graph.nodes if n.op == "call_function" and n.target in targets
    ]
    assert len(nodes) == 1, nodes
    return nodes[0]


class BroadcastIndexNet(torch.nn.Module):
    """
    Pattern produced by transformers' padding mask: `mask[batch_idx, kv_idx]` with
    `batch_idx` of shape [1, 1, 1, 1] and `kv_idx` of shape [1, 1, 1, L].
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("batch_idx", torch.zeros(1, 1, 1, 1, dtype=torch.int64))
        self.register_buffer("kv_idx", torch.arange(7).reshape(1, 1, 1, 7))

    def forward(self, mask):
        return mask[self.batch_idx, self.kv_idx]

    def get_example_inputs(self):
        torch.manual_seed(0)
        return (torch.randint(0, 2, (1, 7), dtype=torch.int32),), {}


class BroadcastIndexTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(BroadcastIndexNet())
        ep = self.exported_program()
        self.assertEqual(num_of_ops(ep, INDEX), 1)
        self.assertEqual(num_of_ops(ep, GATHER_ND), 0)

        self.run_value_test(LowerMultiIndexToGatherNd())

        ep = self.exported_program()
        self.assertEqual(num_of_ops(ep, INDEX), 0)
        gather_nd = _single_node(ep, GATHER_ND)
        params, indices = gather_nd.args
        self.assertEqual(params.op, "placeholder")
        self.assertEqual(tuple(indices.meta["val"].shape), (1, 1, 1, 7, 2))
        self.assertEqual(indices.meta["val"].dtype, torch.int32)
        self.assertEqual(tuple(gather_nd.meta["val"].shape), (1, 1, 1, 7))
        self.assertEqual(gather_nd.meta["val"].dtype, torch.int32)

    def test_idempotent(self):
        self.setup(BroadcastIndexNet())
        test_pass = LowerMultiIndexToGatherNd()
        self.assertTrue(test_pass.call(self.exported_program()).modified)
        self.assertFalse(test_pass.call(self.exported_program()).modified)
        self.assertEqual(num_of_ops(self.exported_program(), GATHER_ND), 1)


class SameShapeIndexNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("rows", torch.tensor([0, 2, 1, 2, 0]))
        self.register_buffer("cols", torch.tensor([3, 0, 1, 2, 3]))

    def forward(self, x):
        return x[self.rows, self.cols]

    def get_example_inputs(self):
        torch.manual_seed(0)
        return (torch.randn(3, 4),), {}


class SameShapeIndexTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(SameShapeIndexNet())

        self.run_value_test(LowerMultiIndexToGatherNd())

        ep = self.exported_program()
        self.assertEqual(num_of_ops(ep, INDEX), 0)
        gather_nd = _single_node(ep, GATHER_ND)
        self.assertEqual(tuple(gather_nd.args[1].meta["val"].shape), (5, 2))
        self.assertEqual(tuple(gather_nd.meta["val"].shape), (5,))
        # Indices of equal shape need no broadcast.
        self.assertEqual(num_of_ops(ep, [torch.ops.aten.expand.default]), 0)


class ThreeIndexTrailingDimsNet(torch.nn.Module):
    """Three broadcast indices on a rank-4 input keep the trailing dimension."""

    def __init__(self):
        super().__init__()
        self.register_buffer("i0", torch.tensor([[1], [0]]))
        self.register_buffer("i1", torch.tensor([[0, 2, 1]]))
        self.register_buffer("i2", torch.tensor([[3, 0, 1], [2, 2, 0]]))

    def forward(self, x):
        return x[self.i0, self.i1, self.i2]

    def get_example_inputs(self):
        torch.manual_seed(0)
        return (torch.randn(2, 3, 4, 5),), {}


class ThreeIndexTrailingDimsTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(ThreeIndexTrailingDimsNet())

        self.run_value_test(LowerMultiIndexToGatherNd())

        ep = self.exported_program()
        self.assertEqual(num_of_ops(ep, INDEX), 0)
        gather_nd = _single_node(ep, GATHER_ND)
        self.assertEqual(tuple(gather_nd.args[1].meta["val"].shape), (2, 3, 3))
        self.assertEqual(tuple(gather_nd.meta["val"].shape), (2, 3, 5))


class SingleIndexNet(torch.nn.Module):
    """A single index tensor is serialized as GATHER and is not lowered here."""

    def __init__(self):
        super().__init__()
        self.register_buffer("idx", torch.tensor([1, 2, 3]))

    def forward(self, x):
        return x[:, self.idx]

    def get_example_inputs(self):
        torch.manual_seed(0)
        return (torch.randn(3, 20, 4),), {}


class SingleIndexTest(SinglePassValueTest):
    def test_pass_neg(self):
        self.setup(SingleIndexNet())
        self.assertFalse(
            LowerMultiIndexToGatherNd().call(self.exported_program()).modified
        )
        self.assertEqual(num_of_ops(self.exported_program(), INDEX), 1)
        self.assertEqual(num_of_ops(self.exported_program(), GATHER_ND), 0)


class NonLeadingIndexNet(torch.nn.Module):
    """Advanced indices after a sliced dimension are out of scope."""

    def __init__(self):
        super().__init__()
        self.register_buffer("i0", torch.tensor([0, 1]))
        self.register_buffer("i1", torch.tensor([2, 3]))

    def forward(self, x):
        return x[:, self.i0, self.i1]

    def get_example_inputs(self):
        torch.manual_seed(0)
        return (torch.randn(2, 3, 4),), {}


class NonLeadingIndexTest(SinglePassValueTest):
    def test_pass_neg(self):
        self.setup(NonLeadingIndexNet())
        self.assertFalse(
            LowerMultiIndexToGatherNd().call(self.exported_program()).modified
        )
        self.assertEqual(num_of_ops(self.exported_program(), INDEX), 1)
        self.assertEqual(num_of_ops(self.exported_program(), GATHER_ND), 0)

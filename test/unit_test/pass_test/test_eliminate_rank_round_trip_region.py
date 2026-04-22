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

from tico.passes import ops
from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape
from tico.passes.eliminate_rank_round_trip_region import EliminateRankRoundTripRegion

from test.utils.helper import num_of_ops
from test.utils.pass_value_test import SinglePassValueTest


class DecomposedLayerNormPermuteNet(torch.nn.Module):
    """
    A module that mimics the decomposed LayerNorm-style pattern followed by
    a 3D permute and a weak sink reshape to 4D.

    Expected behavior of the pass:
      - remove the source reshape from 4D to 3D
      - rewrite the internal region in 4D
      - keep the final weak sink reshape, but update its input
    """

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(32))
        self.bias = torch.nn.Parameter(torch.rand(32))

    def forward(self, x):
        """
        Build a graph similar to the decomposed LayerNorm subgraph that
        originally motivated this pass.
        """
        x = torch.reshape(x, [1, 16, 32])

        mean = torch.mean(x, dim=[-1], keepdim=True)
        sub = torch.sub(x, mean)
        sq = torch.mul(sub, sub)
        var = torch.mean(sq, dim=[-1], keepdim=True)
        var_eps = torch.add(var, 1e-5)
        inv_std = torch.rsqrt(var_eps)
        normed = torch.mul(sub, inv_std)
        scaled = torch.mul(normed, self.weight)
        shifted = torch.add(scaled, self.bias)

        y = torch.permute(shifted, [0, 2, 1])
        y = torch.reshape(y, [1, 32, 16, 1])
        return y

    def get_example_inputs(self):
        """
        Return example inputs.
        """
        return (torch.rand([1, 1, 16, 32]),), {}


class SplitWithSizesGLUNet(torch.nn.Module):
    """
    A module that tests the split_with_sizes/getitem/sigmoid/mul DAG pattern.

    Expected behavior of the pass:
      - remove the source reshape from 4D to 3D
      - rewrite split/getitem/sigmoid/mul directly on 4D
      - keep the final weak sink reshape, but update its input
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Build a GLU-like pattern using split_with_sizes.
        """
        x = torch.reshape(x, [1, 64, 16])
        xs = torch.ops.aten.split_with_sizes.default(x, [32, 32], 1)
        x1 = xs[0]
        x2 = xs[1]
        y = torch.mul(x1, torch.sigmoid(x2))
        y = torch.reshape(y, [1, 32, 16, 1])
        return y

    def get_example_inputs(self):
        """
        Return example inputs.
        """
        return (torch.rand([1, 64, 16, 1]),), {}


class ChunkTransposeNet(torch.nn.Module):
    """
    A module that tests transpose and chunk in the same region.

    Expected behavior of the pass:
      - remove the source reshape from 4D to 3D
      - remap transpose and chunk dimensions into 4D
      - keep the final weak sink reshape, but update its input
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Build a region that includes transpose, chunk, getitem, and add.
        """
        x = torch.reshape(x, [1, 16, 32])
        x = torch.transpose(x, 1, 2)
        xs = torch.chunk(x, 2, dim=1)
        x1 = xs[0]
        x2 = xs[1]
        y = torch.add(x1, x2)
        y = torch.reshape(y, [1, 16, 16, 1])
        return y

    def get_example_inputs(self):
        """
        Return example inputs.
        """
        return (torch.rand([1, 1, 16, 32]),), {}


class UnsupportedMatmulNet(torch.nn.Module):
    """
    A module that contains an unsupported internal operator.

    Expected behavior of the pass:
      - do not rewrite the region
      - keep both reshape operators unchanged
    """

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand(32, 8))

    def forward(self, x):
        """
        Build a graph that contains matmul, which is intentionally unsupported
        by the current pass.
        """
        x = torch.reshape(x, [1, 16, 32])
        y = torch.matmul(x, self.w)
        y = torch.reshape(y, [1, 16, 8, 1])
        return y

    def get_example_inputs(self):
        """
        Return example inputs.
        """
        return (torch.rand([1, 1, 16, 32]),), {}


class EliminateRankRoundTripRegionLayerNormStyleTest(SinglePassValueTest):
    """
    Test the decomposed LayerNorm-style region.
    """

    def test_pass(self):
        """
        Verify that the pass removes only the source reshape and preserves
        value correctness.
        """
        self.setup(DecomposedLayerNormPermuteNet())
        self.ep = self.ep.run_decompositions()
        self.run_pass(ConvertLayoutOpToReshape())

        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 2)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.mean), 2)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.rsqrt), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.permute), 1)

        self.run_value_test(EliminateRankRoundTripRegion(enabled=True))

        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.mean), 2)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.rsqrt), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.permute), 1)


class EliminateRankRoundTripRegionSplitWithSizesTest(SinglePassValueTest):
    """
    Test the split_with_sizes/getitem/sigmoid/mul DAG pattern.
    """

    def test_pass(self):
        """
        Verify that the pass handles the GLU-like split pattern correctly.
        """
        self.setup(SplitWithSizesGLUNet())
        self.ep = self.ep.run_decompositions()
        self.run_pass(ConvertLayoutOpToReshape())

        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 2)
        self.assertEqual(
            num_of_ops(self.exported_program(), ops.aten.split_with_sizes), 1
        )
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.sigmoid), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.mul_tensor), 1)

        self.run_value_test(EliminateRankRoundTripRegion(enabled=True))

        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 1)
        self.assertEqual(
            num_of_ops(self.exported_program(), ops.aten.split_with_sizes), 1
        )
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.sigmoid), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.mul_tensor), 1)


class EliminateRankRoundTripRegionChunkTransposeTest(SinglePassValueTest):
    """
    Test a region containing transpose, chunk, getitem, and add.
    """

    def test_pass(self):
        """
        Verify that the pass remaps transpose and chunk correctly.
        """
        self.setup(ChunkTransposeNet())
        self.ep = self.ep.run_decompositions()
        self.run_pass(ConvertLayoutOpToReshape())

        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 2)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.permute), 1)
        self.assertEqual(
            num_of_ops(self.exported_program(), ops.aten.split_with_sizes), 1
        )
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.add), 1)

        self.run_value_test(EliminateRankRoundTripRegion(enabled=True))

        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.permute), 1)
        self.assertEqual(
            num_of_ops(self.exported_program(), ops.aten.split_with_sizes), 1
        )
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.add), 1)


class EliminateRankRoundTripRegionUnsupportedOpTest(SinglePassValueTest):
    """
    Test that the pass does not rewrite a region containing an unsupported op.
    """

    def test_pass(self):
        """
        Verify that the graph remains unchanged when an unsupported operator
        appears inside the candidate region.
        """
        self.setup(UnsupportedMatmulNet())
        self.ep = self.ep.run_decompositions()
        self.run_pass(ConvertLayoutOpToReshape())

        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 4)
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.mm.default]), 1
        )

        self.run_value_test(EliminateRankRoundTripRegion(enabled=True))

        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 4)
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.mm.default]), 1
        )

# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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
from tico.passes.convert_gather_to_gather_nd import ConvertGatherToGatherNd

from test.support.helper import num_of_ops
from test.support.pass_value_test import SinglePassValueTest


class GatherBasic(torch.nn.Module):
    """Basic gather """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, input, index):
        return torch.gather(input, dim=self.dim, index=index)

    def get_example_inputs(self):
        input = torch.randn(5, 1, 3, 6)
        index = torch.tensor([[[[0, 1, 2]]], [[[3, 4, 5]]]])
        return (input, index), {}


class GatherBasicTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(GatherBasic(dim=3))
        self.assertEqual(num_of_ops(
            self.exported_program(), ops.aten.gather), 1)
        # Run conversion without actually running the model because after the
        # ConvertGatherToGatherNd pass the graph is not a valid PyTorch model
        # any more. Therefore we just check that the conversion is correct
        # based on the graph structure.
        ConvertGatherToGatherNd().call(self.ep)
        # Check whether indices preprocessing nodes have been added. The number
        # of arange, reshape and expand ops should be equal to the rank of the
        # input tensor minus 1 (the original index is used for the gather dim).
        self.assertEqual(num_of_ops(
            self.exported_program(), ops.aten.arange), 3)
        self.assertEqual(num_of_ops(
            self.exported_program(), ops.aten.reshape), 3)
        self.assertEqual(num_of_ops(
            self.exported_program(), ops.aten.expand), 3)
        self.assertEqual(num_of_ops(
            self.exported_program(), ops.aten.unsqueeze), 4)
        self.assertEqual(num_of_ops(
            self.exported_program(), ops.aten.cat), 1)
        self.assertEqual(num_of_ops(
            self.exported_program(), ops.aten.gather), 1)

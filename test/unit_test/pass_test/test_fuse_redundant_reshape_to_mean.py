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
from packaging.version import Version

from tico.passes import ops
from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape
from tico.passes.fuse_redundant_reshape_to_mean import FuseRedundantReshapeToMean

from test.utils.helper import num_of_ops
from test.utils.pass_value_test import SinglePassValueTest


class MeanRedundantViewNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.mean(x, dim=[1])
        new_shape = [*x.size(), 1]
        z = x.reshape(new_shape)
        return z

    def get_example_inputs(self):
        return (torch.randn(3, 4),)


class FuseRedundantReshapeToMeanTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(MeanRedundantViewNet())

        if Version(torch.__version__) <= Version("2.6.0.dev20241015"):
            self.run_value_test(ConvertLayoutOpToReshape())

        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 1)

        self.run_value_test(FuseRedundantReshapeToMean())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 0)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.mean), 1)

        for node in self.exported_program().graph.nodes:
            if not node.op == "call_function":
                continue

            if node.target in ops.aten.mean:
                tensor, dim, *keepdim = node.args
                self.assertEqual(1, len(keepdim))
                self.assertTrue(keepdim[0])

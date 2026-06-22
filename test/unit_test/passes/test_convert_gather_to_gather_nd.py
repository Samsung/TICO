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

import unittest

import torch

from tico.passes.convert_gather_to_gather_nd import ConvertGatherToGatherNd

from test.support.helper import num_of_ops
from test.support.pass_value_test import SinglePassValueTest


class GatherBasic(torch.nn.Module):
    """A module that calls torch.gather for one dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, input, index):
        """Return torch.gather for the configured dimension."""
        return torch.gather(input, dim=self.dim, index=index)

    def get_example_inputs(self):
        """Return an input tensor and a valid gather index tensor."""
        input_shape = [3, 4, 5, 6]
        input_tensor = torch.randn(input_shape)
        dim = self.dim if self.dim >= 0 else len(input_shape) + self.dim
        index_shape = list(input_shape)
        index_shape[dim] = 2
        index = torch.randint(0, input_shape[dim], index_shape, dtype=torch.long)
        return (input_tensor, index), {}


class GatherBasicTest(SinglePassValueTest):
    """Tests for lowering torch.gather to explicit GatherNd IR."""

    def test_pass_preserves_values_for_all_static_dims(self):
        """Check that the pass is value-preserving for supported static shapes."""
        for dim in [0, 1, 2, 3, -1]:
            with self.subTest(dim=dim):
                self.setup(GatherBasic(dim=dim))
                self.assertEqual(
                    num_of_ops(
                        self.exported_program(), [torch.ops.aten.gather.default]
                    ),
                    1,
                )

                self.run_value_test(ConvertGatherToGatherNd())

                self.assertEqual(
                    num_of_ops(
                        self.exported_program(), [torch.ops.aten.gather.default]
                    ),
                    0,
                )
                self.assertTrue(self._has_circle_gather_nd())

    def _has_circle_gather_nd(self) -> bool:
        """Return True if the exported program contains the internal GatherNd op."""
        for node in self.exported_program().graph.nodes:
            if node.op != "call_function":
                continue
            if "circle_custom.gather_nd" in str(node.target):
                return True
        return False


if __name__ == "__main__":
    unittest.main()

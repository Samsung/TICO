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

from tico.serialize.circle_graph import CircleModel, CircleSubgraph

from tico.utils.serialize import validate_tensor_shapes


class CircleSerializeTest(unittest.TestCase):
    def test_validate_circle_shape(self):
        mod = CircleModel()
        g = CircleSubgraph(mod)
        g.add_tensor_from_scratch(
            prefix="name", shape=[1, 2, 3], shape_signature=None, dtype=0
        )
        g.add_tensor_from_scratch(
            prefix="name", shape=[1, 2, 3], shape_signature=None, dtype=0
        )
        validate_tensor_shapes(g)

    def test_validate_tensor_shape_neg(self):
        mod = CircleModel()
        g = CircleSubgraph(mod)
        g.add_tensor_from_scratch(
            prefix="tensor0",
            shape=[1, 2, 3],
            shape_signature=[-1, 0, 0],  # Invalid shape pair
            dtype=0,
        )
        g.add_tensor_from_scratch(
            prefix="tensor1", shape=[1, 2, 3], shape_signature=None, dtype=0
        )
        with self.assertRaises(ValueError):
            validate_tensor_shapes(g)

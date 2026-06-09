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
from circle_schema import circle

from test.unit_test.serialize.operator.fixture import SingleOpGraphFixture


class RMSNormNet(torch.nn.Module):
    def __init__(self, normalized_shape=16, eps=1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.rms_norm = torch.nn.RMSNorm(normalized_shape, eps=eps)

    def forward(self, x):
        return self.rms_norm(x)

    def get_example_inputs(self):
        return (torch.randn(1, 8, self.normalized_shape),), {}


class TestRMSNormVisitor(unittest.TestCase):
    """Test RMSNormVisitor that serializes aten.rms_norm.default directly."""

    def test_define_node(self):
        fixture = SingleOpGraphFixture(RMSNormNet(), torch.ops.aten.rms_norm.default)
        visitor = fixture.target_visitor()
        node = fixture.target_node()
        operator = visitor.define_node(node)

        # Check that the operator is RMS_NORM
        self.assertEqual(
            operator.builtinOptionsType,
            circle.BuiltinOptions.BuiltinOptions.RmsNormOptions,
        )
        # Check epsilon value
        self.assertAlmostEqual(operator.builtinOptions.epsilon, 1e-6)
        # Check inputs: [input, weight]
        self.assertEqual(len(operator.inputs), 2)

    def test_custom_eps(self):
        fixture = SingleOpGraphFixture(
            RMSNormNet(normalized_shape=32, eps=1e-5),
            torch.ops.aten.rms_norm.default,
        )
        visitor = fixture.target_visitor()
        node = fixture.target_node()
        operator = visitor.define_node(node)

        self.assertAlmostEqual(operator.builtinOptions.epsilon, 1e-5)


if __name__ == "__main__":
    unittest.main()

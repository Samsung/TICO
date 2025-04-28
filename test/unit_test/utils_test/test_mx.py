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
from tico.experimental.quantization.evaluation.metric import compute_peir
from tico.utils.mx.mx_ops import quantize_mx

from torch.export import export

from test.utils.helper import num_of_ops


# TODO Move this to test/modules/op
class SimpleMXINT8(torch.nn.Module):
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return quantize_mx(x, "int8", axis=self.axis)

    def get_example_inputs(self):
        return (torch.ones(1, 2, 3),)


class QuantizeMXTest(unittest.TestCase):
    # Test with a tensor filled with the same value
    # Fake-quant values must be the same with the original values
    def test_ones(self):
        input_ = torch.ones(1, 2, 3, dtype=torch.float32)

        output = quantize_mx(input_, "int8", 0)
        self.assertTrue(torch.allclose(input_, output))

        output = quantize_mx(input_, "int8", 1)
        self.assertTrue(torch.allclose(input_, output))

        output = quantize_mx(input_, "int8", 2)
        self.assertTrue(torch.allclose(input_, output))

    # Test with randomly generated values
    # Fake-quant values must be
    # (1) different from the original values
    # (2) but the difference should be small (PEIR < 0.01)
    def test_random_values(self):
        torch.manual_seed(0)
        input_ = torch.randn(1, 32, 32)

        output = quantize_mx(input_, "int8", 0)
        self.assertFalse(torch.allclose(input_, output))
        self.assertLess(compute_peir(input_, output), 0.01)

        output = quantize_mx(input_, "int8", 1)
        self.assertFalse(torch.allclose(input_, output))
        self.assertLess(compute_peir(input_, output), 0.01)

        output = quantize_mx(input_, "int8", 2)
        self.assertFalse(torch.allclose(input_, output))
        self.assertLess(compute_peir(input_, output), 0.01)

    # Check if exported program includes circle_custom::quantize_mx Op
    def test_export(self):
        m = SimpleMXINT8(axis=2)

        with torch.no_grad():
            ep = export(m.eval(), m.get_example_inputs())

        self.assertEqual(
            1, num_of_ops(ep, [torch.ops.circle_custom.quantize_mx.default])
        )

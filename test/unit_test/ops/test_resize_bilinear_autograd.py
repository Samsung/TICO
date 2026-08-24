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

"""Autograd regression tests for the ResizeBilinear public facade."""

from __future__ import annotations

import unittest

import torch

from tico.ops.resize_bilinear import ResizeBilinear2d


class ResizeBilinearAutogradTest(unittest.TestCase):
    def test_eager_gradient_uses_differentiable_reference(self) -> None:
        input_ = torch.randn(1, 2, 3, 4, requires_grad=True)
        module = ResizeBilinear2d(
            (5, 7),
            half_pixel_centers=True,
        )
        output = module(input_)
        self.assertEqual(tuple(output.shape), (1, 2, 5, 7))
        output.square().mean().backward()
        self.assertIsNotNone(input_.grad)
        self.assertTrue(torch.isfinite(input_.grad).all())
        self.assertGreater(float(input_.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

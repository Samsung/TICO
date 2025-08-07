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

import numpy as np

import tico
import torch

from test.modules.op.add import SimpleAdd
from test.modules.op.avg_pool2d import AvgPoolWithPaddingKwargs
from test.modules.op.cat import SimpleCatDefault, SimpleCatWithDim


class InferSimpleAddTest(unittest.TestCase):
    def setUp(self):
        # Input: torch.ones(1), torch.ones(1)
        m = SimpleAdd()
        self.torch_model = m
        args, kwargs = m.get_example_inputs()
        self.circle_model = tico.convert(m.eval(), args, kwargs)

    def test_add_float(self):
        x = torch.randn(1)
        y = torch.randn(1)
        with torch.no_grad():
            out_np = self.torch_model(x, y).numpy()
        np.testing.assert_allclose(
            actual=self.circle_model(x, y), desired=out_np, rtol=1e-4, atol=1e-4
        )

    def test_add_float_numpy(self):
        x = torch.randn(1).numpy()
        y = torch.randn(1).numpy()
        with torch.no_grad():
            out_np = self.torch_model(x, y)
        np.testing.assert_allclose(
            actual=self.circle_model(x, y), desired=out_np, rtol=1e-4, atol=1e-4
        )

    def test_add_float_builtin(self):
        x = torch.tensor([1.0])
        y = torch.tensor([2.0])
        np.testing.assert_allclose(
            actual=self.circle_model(x, y), desired=[10.0], rtol=1e-4, atol=1e-4
        )


class InferCatTest(unittest.TestCase):
    def test_concat(self):
        # convert
        m = SimpleCatDefault()
        torch_model = m
        args, kwargs = m.get_example_inputs()
        circle_model = tico.convert(m.eval(), args, kwargs)
        # test
        x = torch.randn(3)
        y = torch.randn(2)
        with torch.no_grad():
            out_np = torch_model((x, y)).numpy()
        np.testing.assert_allclose(
            actual=circle_model((x, y)), desired=out_np, rtol=1e-4, atol=1e-4
        )

    def test_concat_with_dim(self):
        # convert
        m = SimpleCatWithDim()
        torch_model = m
        args, kwargs = m.get_example_inputs()
        circle_model = tico.convert(m.eval(), args, kwargs)
        # test
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        with torch.no_grad():
            out_np = torch_model((x, y)).numpy()
        np.testing.assert_allclose(
            actual=circle_model((x, y)), desired=out_np, rtol=1e-4, atol=1e-4
        )


class InferAvgPoolReverseKwargsTest(unittest.TestCase):
    def test_avgpool_reverse_kwargs(self):
        # convert
        m = AvgPoolWithPaddingKwargs()
        torch_model = m
        args, kwargs = m.get_example_inputs()
        circle_model = tico.convert(m.eval(), args, kwargs)

        # test
        tensor0 = torch.randn(2, 4, 8, 16)
        tensor1 = torch.randn(2, 4, 4, 8)
        with torch.no_grad():
            out_np = torch_model(tensor1=tensor1, tensor0=tensor0).numpy()

        kwargs = {
            "tensor1": tensor1,
            "tensor0": tensor0,
        }
        np.testing.assert_allclose(
            actual=circle_model(**kwargs), desired=out_np, rtol=1e-4, atol=1e-4
        )

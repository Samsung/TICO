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
from tico.passes import ops
from tico.passes.cast_clamp_mixed_type_args import CastClampMixedTypeArgs

from test.utils.helper import num_of_ops
from test.utils.pass_value_test import SinglePassValueTest


class CastClampFloatInputIntMin(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, -10, None)
        return x

    def get_example_inputs(self):
        return (torch.randn(5, 3) * 20,)


class CastClampFloatInputIntMinTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(CastClampFloatInputIntMin())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.clamp), 1)

        self.run_value_test(CastClampMixedTypeArgs())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.clamp), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten._to_copy), 0)


class CastClampFloatInputIntMax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, None, 10)
        return x

    def get_example_inputs(self):
        return (torch.randn(5, 3) * 20,)


class CastClampFloatInputIntMaxTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(CastClampFloatInputIntMax())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.clamp), 1)

        self.run_value_test(CastClampMixedTypeArgs())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.clamp), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten._to_copy), 0)


class CastClampIntInputFloatMin(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, -10.0, None)
        return x

    def get_example_inputs(self):
        return (torch.randint(-20, 20, (5, 3)),)


class CastClampIntInputFloatMinTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(CastClampIntInputFloatMin())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.clamp), 1)

        self.run_value_test(CastClampMixedTypeArgs())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.clamp), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten._to_copy), 1)


class CastClampIntInputFloatMax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, None, 10.0)
        return x

    def get_example_inputs(self):
        return (torch.randint(-20, 20, (5, 3)),)


class CastClampIntInputFloatMaxTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(CastClampIntInputFloatMax())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.clamp), 1)

        self.run_value_test(CastClampMixedTypeArgs())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.clamp), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten._to_copy), 1)

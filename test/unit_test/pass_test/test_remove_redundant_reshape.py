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
from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape
from tico.passes.remove_redundant_reshape import (
    RemoveRedundantReshapePattern1,
    RemoveRedundantReshapePattern2,
    RemoveRedundantReshapePattern3,
    RemoveRedundantReshapePattern4,
    RemoveRedundantReshapePattern5,
)

from test.utils.helper import num_of_ops
from test.utils.pass_value_test import SinglePassValueTest


class RedundantReshapePattern1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        new_shape = [1, *x.size()]
        # AxBxC -> 1xAxBxC
        reshape = torch.reshape(x, new_shape)
        # 1xAxBxC -> 1xAxCxB
        permute = torch.permute(reshape, (0, 1, 3, 2))
        mul = torch.mul(permute, 3.0)
        # 1xAxCxB -> AxCxB
        reshape_2 = torch.reshape(mul, [*mul.size()[1:]])
        return reshape_2

    def get_example_inputs(self):
        return (torch.randn(2, 3, 4),), {}


class RemoveRedundantReshapePattern1Test(SinglePassValueTest):
    def test_pass(self):
        self.setup(RedundantReshapePattern1())
        self.run_value_test(ConvertLayoutOpToReshape())

        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 2)

        self.run_value_test(RemoveRedundantReshapePattern1())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 0)

        for node in self.exported_program().graph.nodes:
            if not node.op == "call_function":
                continue


class RedundantReshapePattern2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        new_shape = [1, *x.size()]
        # AxBxC -> 1xAxBxC
        reshape = torch.reshape(x, new_shape)
        # 1xAxBxC -> Bx1xAxC
        permute = torch.permute(reshape, (2, 0, 1, 3))
        # Bx1xAxC -> Bx(A*C)
        permute_size = permute.size()
        reshape_2 = permute.contiguous().reshape(
            [permute_size[0], (permute_size[2] * permute_size[3])]
        )
        return reshape_2

    def get_example_inputs(self):
        return (torch.randn(2, 3, 4),), {}


class RemoveRedundantReshapePattern2Test(SinglePassValueTest):
    def test_pass(self):
        self.setup(RedundantReshapePattern2())
        self.run_value_test(ConvertLayoutOpToReshape())

        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 2)

        self.run_value_test(RemoveRedundantReshapePattern2())

        # TODO comment out this after introducing circle IR.
        # self.assertEqual(num_of_ops(ep, ReshapeVisitor.target), 1)


class RedundantReshapePattern3(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        new_shape = [1, *x.size()]
        # AxBxC -> 1xAxBxC
        x_reshape = torch.reshape(x, new_shape)
        y_reshape = torch.reshape(y, new_shape)
        # add
        add = torch.add(x_reshape, y_reshape)
        # softmax
        softmax = torch.softmax(add, dim=3)
        # 1xAxBxC -> AxBxC
        reshape = torch.reshape(softmax, x.size())
        return reshape

    def get_example_inputs(self):
        return (torch.randn(2, 3, 4), torch.randn(2, 3, 4)), {}


class RemoveRedundantReshapePattern3Test(SinglePassValueTest):
    def test_pass(self):
        self.setup(RedundantReshapePattern3())

        from tico.utils.utils import SuppressWarning

        # torch.ops.aten.softmax.int -> torch.ops.aten._softmax.default
        with SuppressWarning(UserWarning, ".*quantize_per_tensor"):
            # Warning details:
            #   ...site-packages/torch/_subclasses/functional_tensor.py:364
            #   UserWarning: At pre-dispatch tracing, we assume that any custom op marked with
            #     CompositeImplicitAutograd and have functional schema are safe to not decompose.
            self.ep = self.exported_program().run_decompositions()
        self.run_value_test(ConvertLayoutOpToReshape())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 3)

        self.run_value_test(RemoveRedundantReshapePattern3())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 0)


class RedundantReshapePattern3Broadcasted(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # AxBxC -> 1xAxBxC
        x_reshape = torch.reshape(x, [1, *x.size()])
        y_reshape = torch.reshape(y, [1, *y.size()])
        # add
        add = torch.add(x_reshape, y_reshape)
        # softmax
        softmax = torch.softmax(add, dim=3)
        # 1xAxBxC -> AxBxC
        reshape = torch.reshape(softmax, y.size())
        return reshape

    def get_example_inputs(self):
        return (torch.randn(16, 1, 64), torch.randn(16, 64, 64)), {}


class RemoveRedundantReshapePattern3BroadcastedTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(RedundantReshapePattern3Broadcasted())

        from tico.utils.utils import SuppressWarning

        # torch.ops.aten.softmax.int -> torch.ops.aten._softmax.default
        with SuppressWarning(UserWarning, ".*quantize_per_tensor"):
            # Warning details:
            #   ...site-packages/torch/_subclasses/functional_tensor.py:364
            #   UserWarning: At pre-dispatch tracing, we assume that any custom op marked with
            #     CompositeImplicitAutograd and have functional schema are safe to not decompose.
            self.ep = self.exported_program().run_decompositions()
        self.run_value_test(ConvertLayoutOpToReshape())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 3)

        self.run_value_test(RemoveRedundantReshapePattern3())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 0)


class RedundantReshapePattern3DifferentSoftmaxLength(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x_reshape = torch.reshape(x, [1, 2, 3])
        y_reshape = torch.reshape(y, [1, 2, 3])
        # add
        add = torch.add(x_reshape, y_reshape)
        # softmax
        softmax = torch.softmax(add, dim=-1)
        reshape = torch.reshape(softmax, y.size())
        return reshape

    def get_example_inputs(self):
        return (
            torch.randn(1, 1, 6),
            torch.randn(1, 1, 6),
        ), {}


class RemoveRedundantReshapePattern3DifferentSoftmaxLengthTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(RedundantReshapePattern3DifferentSoftmaxLength())

        from tico.utils.utils import SuppressWarning

        # torch.ops.aten.softmax.int -> torch.ops.aten._softmax.default
        with SuppressWarning(UserWarning, ".*quantize_per_tensor"):
            # Warning details:
            #   ...site-packages/torch/_subclasses/functional_tensor.py:364
            #   UserWarning: At pre-dispatch tracing, we assume that any custom op marked with
            #     CompositeImplicitAutograd and have functional schema are safe to not decompose.
            self.ep = self.exported_program().run_decompositions()
        self.run_value_test(ConvertLayoutOpToReshape())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 3)

        self.run_value_test(RemoveRedundantReshapePattern3())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 3)


class RedundantReshapePattern4(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        reshape1_shape = [4, 12, 4]
        # AxBxC -> AxB'xC'
        reshape_1 = torch.reshape(x, reshape1_shape)

        reshape2_shape = [4, 4, 12]
        # AxB'xC' -> AxB''xC'
        reshape_2 = torch.reshape(reshape_1, reshape2_shape)

        return reshape_2

    def get_example_inputs(self):
        return (torch.randn(4, 6, 8),), {}


class RemoveRedundantReshapePattern4Test(SinglePassValueTest):
    def test_pass(self):
        self.setup(RedundantReshapePattern4())
        self.run_value_test(ConvertLayoutOpToReshape())

        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 2)

        self.run_value_test(RemoveRedundantReshapePattern4())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 1)


class RedundantReshapePattern5(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        reshape_shape = [4, 6, 8]
        # AxBxC -> AxBxC
        reshape = torch.reshape(x, reshape_shape)
        return reshape

    def get_example_inputs(self):
        return (torch.randn(4, 6, 8),), {}


class RemoveRedundantReshapePattern5Test(SinglePassValueTest):
    def test_pass(self):
        self.setup(RedundantReshapePattern5())
        self.run_value_test(ConvertLayoutOpToReshape())

        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 1)

        self.run_value_test(RemoveRedundantReshapePattern5())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 0)

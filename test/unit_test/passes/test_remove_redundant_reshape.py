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

import math
import unittest

import numpy as np
import tico
import torch
from circle_schema import circle

from tico.circle import CircleDocument
from tico.passes import ops
from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape
from tico.passes.remove_redundant_reshape import (
    RemoveRedundantReshapePattern1,
    RemoveRedundantReshapePattern2,
    RemoveRedundantReshapePattern3,
    RemoveRedundantReshapePattern4,
    RemoveRedundantReshapePattern5,
)

from test.support.helper import num_of_ops
from test.support.pass_value_test import SinglePassValueTest


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


class _ReshapePermuteMul(torch.nn.Module):
    def __init__(self, reverse=False, square=False, share=None):
        super().__init__()
        self.reverse = reverse
        self.square = square
        self.share = share

    def forward(self, x, other):
        expanded = x.reshape(1, *x.shape)
        permuted = expanded.permute(0, 1, 3, 2)
        multiplier = permuted if self.square else other
        product = multiplier * permuted if self.reverse else permuted * multiplier
        result = product.reshape(*product.shape[1:])
        if self.share == "reshape":
            return result, expanded
        if self.share == "permute":
            return result, permuted
        if self.share == "mul":
            return result, product
        return result


class RemoveRedundantReshapePattern1BroadcastTest(unittest.TestCase):
    def _apply(self, model, args, expected_modified, dynamic_shapes=None):
        ep = torch.export.export(model.eval(), args, dynamic_shapes=dynamic_shapes)
        ConvertLayoutOpToReshape().call(ep)
        self.assertEqual(num_of_ops(ep, ops.aten.reshape), 2)
        expected = model(*args)
        with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
            result = RemoveRedundantReshapePattern1().call(ep)
        self.assertEqual(result.modified, expected_modified)
        self.assertEqual(
            num_of_ops(ep, ops.aten.reshape), 0 if expected_modified else 2
        )
        ep.validate()
        torch.testing.assert_close(ep.module()(*args), expected, rtol=0, atol=0)
        graph = str(ep.graph)
        self.assertFalse(RemoveRedundantReshapePattern1().call(ep).modified)
        self.assertEqual(str(ep.graph), graph)
        return ep

    def test_rank_four_multiplier_preserves_final_reshape(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        for shape in ((1, 2, 4, 3), (1, 1, 1, 3), (1, 1, 4, 1)):
            for reverse in (False, True):
                with self.subTest(shape=shape, reverse=reverse):
                    other = torch.arange(math.prod(shape), dtype=x.dtype).reshape(shape)
                    self._apply(_ReshapePermuteMul(reverse), (x, other), False)

    def test_scalar_and_lower_rank_multipliers_still_fuse(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        for other in (
            3.0,
            torch.tensor(3.0),
            torch.arange(3, dtype=x.dtype),
            torch.arange(4, dtype=x.dtype).reshape(4, 1),
            torch.arange(24, dtype=x.dtype).reshape(2, 4, 3),
        ):
            for reverse in (False, True):
                with self.subTest(other=other, reverse=reverse):
                    self._apply(_ReshapePermuteMul(reverse), (x, other), True)

    def test_repeated_permute_operand_still_fuses(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        self._apply(_ReshapePermuteMul(square=True), (x, torch.tensor(1.0)), True)

    def test_shared_intermediates_are_unchanged(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        for share in ("reshape", "permute", "mul"):
            with self.subTest(share=share):
                self._apply(
                    _ReshapePermuteMul(share=share), (x, torch.tensor(3.0)), False
                )

    def test_dynamic_batch_preserves_rank_for_all_batch_sizes(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        batch = torch.export.Dim("batch", min=1, max=4)
        for shape, expected_modified in (((4, 3), True), ((1, 1, 4, 3), False)):
            with self.subTest(shape=shape):
                other = torch.arange(math.prod(shape), dtype=x.dtype).reshape(shape)
                model = _ReshapePermuteMul()
                ep = self._apply(
                    model,
                    (x, other),
                    expected_modified,
                    dynamic_shapes=({0: batch}, None),
                )
                for size in (1, 3, 4):
                    value = torch.arange(size * 12, dtype=x.dtype).reshape(size, 3, 4)
                    torch.testing.assert_close(
                        ep.module()(value, other), model(value, other), rtol=0, atol=0
                    )

    def test_default_o1_circle_output_shape_and_values(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        for shape in ((2, 4, 3), (1, 2, 4, 3), (1, 1, 1, 3)):
            for reverse in (False, True):
                with self.subTest(shape=shape, reverse=reverse):
                    other = torch.arange(math.prod(shape), dtype=x.dtype).reshape(shape)
                    model = _ReshapePermuteMul(reverse).eval()
                    converted = tico.convert(model, (x, other))
                    document = CircleDocument.from_bytes(converted.circle_binary)
                    document.verify()
                    graph = document.model.subgraphs[0]
                    self.assertEqual(len(graph.outputs), 1)
                    output = graph.tensors[graph.outputs[0]]
                    self.assertEqual(list(output.shape), [2, 4, 3])
                    self.assertEqual(output.type, circle.TensorType.TensorType.FLOAT32)
                    np.testing.assert_array_equal(
                        converted(x, other), model(x, other).numpy()
                    )


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


class RedundantReshapePattern4DynamicSeq(torch.nn.Module):
    """Consecutive reshapes whose sizes depend on a dynamic sequence dimension."""

    def forward(self, x):
        batch, seq = x.shape[0], x.shape[1]
        # (B, S, 8) -> (B*S, 8) -> (B, S, 2, 4): both sizes carry SymInt.
        reshape_1 = torch.reshape(x, [batch * seq, 8])
        reshape_2 = torch.reshape(reshape_1, [batch, seq, 2, 4])
        return reshape_2

    def get_example_inputs(self):
        return (torch.randn(1, 6, 8),), {}


class RemoveRedundantReshapePattern4DynamicSeqTest(unittest.TestCase):
    def test_pass_skips_symbolic_sizes(self):
        """The static fusion must skip, not crash on, symbolic reshape sizes."""
        mod = RedundantReshapePattern4DynamicSeq()
        args, kwargs = mod.get_example_inputs()
        seq_dim = torch.export.Dim("seq", min=1, max=16)
        with torch.no_grad():
            ep = torch.export.export(
                mod.eval(),
                args,
                kwargs,
                dynamic_shapes={"x": {1: seq_dim}},
            )
        ConvertLayoutOpToReshape().call(ep)
        self.assertEqual(num_of_ops(ep, ops.aten.reshape), 2)

        result = RemoveRedundantReshapePattern4().call(ep)

        self.assertFalse(result.modified)
        self.assertEqual(num_of_ops(ep, ops.aten.reshape), 2)
        for seq_len in (1, 3, 16):
            x = torch.randn(1, seq_len, 8)
            self.assertTrue(torch.equal(ep.module()(x), mod(x)))

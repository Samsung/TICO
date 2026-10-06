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

import torch
from tico.passes import ops
from tico.passes.remove_empty_cat_inputs import RemoveEmptyCatInputs
from tico.utils.convert import traced_run_decompositions
from tico.utils.validate_args_kwargs import CatArgs
from torch.export.graph_signature import InputKind

from test.support.helper import num_of_ops
from test.support.pass_value_test import SinglePassValueTest


class DecomposedSinglePassValueTest(SinglePassValueTest):
    """
    Run the pass on the decomposed program, as the conversion pipeline does.

    Without decompositions, `torch.export` keeps a constant created inside `forward`
    behind an in-place `detach_` that dead-code elimination never removes, which
    hides whether the pass releases the lifted constant.
    """

    def setup(self, mod: torch.nn.Module):
        super().setup(mod)
        self.ep = traced_run_decompositions(self.ep)


def _cat_nodes(ep):
    return [
        node
        for node in ep.graph.nodes
        if node.op == "call_function" and node.target in ops.aten.cat
    ]


def _cat_input_shapes(cat_node):
    args = CatArgs(*cat_node.args, **cat_node.kwargs)
    return [tuple(t.meta["val"].shape) for t in args.tensors]


def _constant_placeholder_names(ep):
    return [
        spec.arg.name
        for spec in ep.graph_signature.input_specs
        if spec.kind == InputKind.CONSTANT_TENSOR
    ]


def _user_input_names(ep):
    return [
        spec.arg.name
        for spec in ep.graph_signature.input_specs
        if spec.kind == InputKind.USER_INPUT
    ]


def _placeholder_names(ep):
    return [n.name for n in ep.graph.nodes if n.op == "placeholder"]


class EmptyConstantCatNet(torch.nn.Module):
    """DynamicCache-style pattern: a lifted 1-D empty constant next to a 4-D input."""

    def forward(self, x):
        empty = torch.tensor([], dtype=x.dtype)
        return torch.cat([empty, x], dim=-2)

    def get_example_inputs(self):
        torch.manual_seed(0)
        return (torch.randn(1, 2, 3, 4),), {}


class EmptyConstantCatTest(DecomposedSinglePassValueTest):
    def test_pass(self):
        self.setup(EmptyConstantCatNet())
        ep = self.exported_program()
        self.assertEqual(num_of_ops(ep, ops.aten.cat), 1)
        self.assertEqual(len(_constant_placeholder_names(ep)), 1)

        self.run_value_test(RemoveEmptyCatInputs())

        ep = self.exported_program()
        self.assertEqual(num_of_ops(ep, ops.aten.cat), 0)
        # The lifted empty constant is gone from the graph, the signature, and the
        # program constants; the user input is untouched.
        self.assertEqual(_constant_placeholder_names(ep), [])
        self.assertEqual(len(ep.constants), 0)
        self.assertEqual(_user_input_names(ep), ["x"])
        self.assertEqual(_placeholder_names(ep), ["x"])

        # The graph output is the user input itself.
        output_node = list(ep.graph.nodes)[-1]
        self.assertEqual(output_node.op, "output")
        (out,) = output_node.args[0]
        self.assertEqual(out.name, "x")

    def test_idempotent(self):
        self.setup(EmptyConstantCatNet())
        test_pass = RemoveEmptyCatInputs()
        self.assertTrue(test_pass.call(self.exported_program()).modified)
        self.assertFalse(test_pass.call(self.exported_program()).modified)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.cat), 0)


class EmptyConstantCatThreeInputsNet(torch.nn.Module):
    def forward(self, x, y):
        empty = torch.tensor([], dtype=x.dtype)
        return torch.cat([x, empty, y], dim=1)

    def get_example_inputs(self):
        torch.manual_seed(0)
        return (torch.randn(2, 3), torch.randn(2, 1)), {}


class EmptyConstantCatThreeInputsTest(DecomposedSinglePassValueTest):
    def test_pass(self):
        self.setup(EmptyConstantCatThreeInputsNet())
        (cat,) = _cat_nodes(self.exported_program())
        self.assertEqual(_cat_input_shapes(cat), [(2, 3), (0,), (2, 1)])

        self.run_value_test(RemoveEmptyCatInputs())

        ep = self.exported_program()
        (cat,) = _cat_nodes(ep)
        self.assertEqual(_cat_input_shapes(cat), [(2, 3), (2, 1)])
        self.assertEqual(CatArgs(*cat.args, **cat.kwargs).dim, 1)
        self.assertEqual(cat.kwargs, {})
        self.assertEqual(tuple(cat.meta["val"].shape), (2, 4))
        self.assertEqual(_constant_placeholder_names(ep), [])
        self.assertEqual(_user_input_names(ep), ["x", "y"])


class EmptySliceCatNet(torch.nn.Module):
    """A non-constant 1-D empty input produced from a user input, 1-D concatenation."""

    def forward(self, x, y):
        return torch.cat([y[:0], x], dim=0)

    def get_example_inputs(self):
        torch.manual_seed(0)
        return (torch.randn(3), torch.randn(2)), {}


class EmptySliceCatTest(DecomposedSinglePassValueTest):
    def test_pass(self):
        self.setup(EmptySliceCatNet())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.cat), 1)

        self.run_value_test(RemoveEmptyCatInputs())

        ep = self.exported_program()
        self.assertEqual(num_of_ops(ep, ops.aten.cat), 0)
        # User input placeholders are preserved even when they become unused.
        self.assertEqual(_user_input_names(ep), ["x", "y"])
        self.assertEqual(_placeholder_names(ep), ["x", "y"])


class NarrowerEmptyCatNet(torch.nn.Module):
    """An int64 empty input does not change the float32 result; it is removed."""

    def forward(self, x):
        empty = torch.tensor([], dtype=torch.int64)
        return torch.cat([empty, x], dim=0)

    def get_example_inputs(self):
        torch.manual_seed(0)
        return (torch.randn(2, 3),), {}


class NarrowerEmptyCatTest(DecomposedSinglePassValueTest):
    def test_pass(self):
        self.setup(NarrowerEmptyCatNet())
        (cat,) = _cat_nodes(self.exported_program())
        self.assertEqual(cat.meta["val"].dtype, torch.float32)

        self.run_value_test(RemoveEmptyCatInputs())

        ep = self.exported_program()
        self.assertEqual(num_of_ops(ep, ops.aten.cat), 0)
        self.assertEqual(_constant_placeholder_names(ep), [])


class SharedEmptyConstantCatNet(torch.nn.Module):
    """The empty constant also feeds a non-cat user and must survive the pass."""

    def forward(self, x):
        empty = torch.tensor([], dtype=x.dtype)
        y = torch.cat([empty, x], dim=-2)
        return y + empty.sum()

    def get_example_inputs(self):
        torch.manual_seed(0)
        return (torch.randn(1, 2, 3, 4),), {}


class SharedEmptyConstantCatTest(DecomposedSinglePassValueTest):
    def test_pass(self):
        self.setup(SharedEmptyConstantCatNet())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.cat), 1)
        self.assertEqual(len(_constant_placeholder_names(self.exported_program())), 1)

        self.run_value_test(RemoveEmptyCatInputs())

        ep = self.exported_program()
        self.assertEqual(num_of_ops(ep, ops.aten.cat), 0)
        self.assertEqual(len(_constant_placeholder_names(ep)), 1)
        self.assertEqual(len(ep.constants), 1)


class NoEmptyInputCatNet(torch.nn.Module):
    def forward(self, x, y):
        return torch.cat([x, y], dim=0)

    def get_example_inputs(self):
        torch.manual_seed(0)
        return (torch.randn(2, 3), torch.randn(1, 3)), {}


class NoEmptyInputCatTest(DecomposedSinglePassValueTest):
    def test_pass_neg(self):
        self.setup(NoEmptyInputCatNet())
        self.assertFalse(RemoveEmptyCatInputs().call(self.exported_program()).modified)
        (cat,) = _cat_nodes(self.exported_program())
        self.assertEqual(_cat_input_shapes(cat), [(2, 3), (1, 3)])


class PromotingEmptyCatNet(torch.nn.Module):
    """The empty float64 input promotes the result dtype; it must be kept."""

    def forward(self, x):
        empty = torch.tensor([], dtype=torch.float64)
        return torch.cat([empty, x], dim=0)

    def get_example_inputs(self):
        torch.manual_seed(0)
        return (torch.randn(2, 3),), {}


class PromotingEmptyCatTest(DecomposedSinglePassValueTest):
    def test_pass_neg(self):
        self.setup(PromotingEmptyCatNet())
        (cat,) = _cat_nodes(self.exported_program())
        self.assertEqual(cat.meta["val"].dtype, torch.float64)

        self.assertFalse(RemoveEmptyCatInputs().call(self.exported_program()).modified)

        ep = self.exported_program()
        (cat,) = _cat_nodes(ep)
        self.assertEqual(_cat_input_shapes(cat), [(0,), (2, 3)])
        self.assertEqual(cat.meta["val"].dtype, torch.float64)
        self.assertEqual(len(_constant_placeholder_names(ep)), 1)


class AllEmptyCatNet(torch.nn.Module):
    def forward(self, x):
        a = torch.tensor([], dtype=x.dtype)
        b = torch.tensor([], dtype=x.dtype)
        return x + torch.cat([a, b], dim=0).sum()

    def get_example_inputs(self):
        torch.manual_seed(0)
        return (torch.randn(2, 3),), {}


class AllEmptyCatTest(DecomposedSinglePassValueTest):
    def test_pass_neg(self):
        self.setup(AllEmptyCatNet())
        self.assertFalse(RemoveEmptyCatInputs().call(self.exported_program()).modified)
        (cat,) = _cat_nodes(self.exported_program())
        self.assertEqual(_cat_input_shapes(cat), [(0,), (0,)])


class SameRankEmptyCatNet(torch.nn.Module):
    """A zero-size input of the same rank is a regular Circle input; out of scope."""

    def forward(self, x):
        empty = torch.zeros(1, 2, 0, 4, dtype=x.dtype)
        return torch.cat([empty, x], dim=-2)

    def get_example_inputs(self):
        torch.manual_seed(0)
        return (torch.randn(1, 2, 3, 4),), {}


class SameRankEmptyCatTest(DecomposedSinglePassValueTest):
    def test_pass_neg(self):
        self.setup(SameRankEmptyCatNet())
        self.assertFalse(RemoveEmptyCatInputs().call(self.exported_program()).modified)
        (cat,) = _cat_nodes(self.exported_program())
        self.assertEqual(_cat_input_shapes(cat), [(1, 2, 0, 4), (1, 2, 3, 4)])

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

import tico
import torch
from tico.passes import ops
from tico.passes.cast_clamp_mixed_type_args import CastClampMixedTypeArgs

from test.support.helper import num_of_ops
from test.support.pass_value_test import SinglePassValueTest


class CastClampFloatInputIntMin(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, -10, None)
        return x

    def get_example_inputs(self):
        return (torch.randn(5, 3) * 20,), {}


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
        return (torch.randn(5, 3) * 20,), {}


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
        return (torch.randint(-20, 20, (5, 3)),), {}


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
        return (torch.randint(-20, 20, (5, 3)),), {}


class CastClampIntInputFloatMaxTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(CastClampIntInputFloatMax())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.clamp), 1)

        self.run_value_test(CastClampMixedTypeArgs())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.clamp), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten._to_copy), 1)


class CastClampTensorBounds(torch.nn.Module):
    """Use lifted scalar tensor buffers as Clamp min/max arguments."""

    def __init__(self):
        super().__init__()
        self.register_buffer("min_value", torch.tensor(-10.0))
        self.register_buffer("max_value", torch.tensor(10.0))

    def forward(self, x):
        return torch.clamp(x, self.min_value, self.max_value)

    def get_example_inputs(self):
        return (torch.randn(5, 3) * 20,), {}


class CastClampTensorBoundsTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(CastClampTensorBounds())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.clamp), 1)

        self.run_value_test(CastClampMixedTypeArgs())

        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.clamp), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten._to_copy), 0)


class _ScalarBounds(torch.nn.Module):
    def __init__(self, minimum, maximum):
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum

    def forward(self, value):
        return torch.clamp(value, self.minimum, self.maximum)


class _AliasedTensorBounds(torch.nn.Module):
    def __init__(self, input_is_minimum=False):
        super().__init__()
        self.input_is_minimum = input_is_minimum

    def forward(self, value, bound):
        minimum = value if self.input_is_minimum else bound
        return torch.clamp(value, minimum, bound)


class ClampArgumentPositionsTest(unittest.TestCase):
    def _export(self, model, inputs, positional_bounds):
        ep = torch.export.export(model.eval(), inputs)
        node = next(n for n in ep.graph.nodes if n.target in ops.aten.clamp)
        # Export normalizes kwargs to positional arguments. Re-express the same
        # ATen call to exercise both valid FX argument representations.
        arguments = (*node.args, *([None] * (3 - len(node.args))))
        node.args = arguments[: 1 + positional_bounds]
        node.kwargs = {
            name: arguments[index]
            for index, name in ((1, "min"), (2, "max"))
            if index > positional_bounds
        }
        ep.graph.lint()
        ep.graph_module.recompile()
        return ep, node

    def _assert_parity_and_stability(self, ep, inputs, before):
        after = ep.module()(*inputs)
        self.assertEqual(after.shape, before.shape)
        self.assertEqual(after.dtype, before.dtype)
        self.assertTrue(torch.equal(after, before))
        graph_before = str(ep.graph)
        self.assertFalse(CastClampMixedTypeArgs().call(ep).modified)
        self.assertEqual(str(ep.graph), graph_before)

    def test_equal_scalar_bounds_update_their_own_slots(self):
        inputs = (torch.tensor([-2.0, 0.0, 2.0]),)
        for minimum, maximum in ((0, 0), (0.0, 0), (0, 0.0), (-1, -1)):
            for positional_bounds in (0, 1, 2):
                with self.subTest(
                    bounds=(minimum, maximum), positional=positional_bounds
                ):
                    ep, node = self._export(
                        _ScalarBounds(minimum, maximum), inputs, positional_bounds
                    )
                    before = ep.module()(*inputs)
                    self.assertTrue(CastClampMixedTypeArgs().call(ep).modified)
                    for index, name, expected in (
                        (1, "min", minimum),
                        (2, "max", maximum),
                    ):
                        actual = (
                            node.args[index]
                            if index < len(node.args)
                            else node.kwargs[name]
                        )
                        self.assertIs(type(actual), float)
                        self.assertEqual(actual, expected)
                    self._assert_parity_and_stability(ep, inputs, before)

    def test_matching_scalar_dtypes_do_not_change(self):
        inputs = (torch.tensor([-2.0, 0.0, 2.0]),)
        for positional_bounds in (0, 1, 2):
            with self.subTest(positional=positional_bounds):
                ep, _ = self._export(_ScalarBounds(0.0, 0.0), inputs, positional_bounds)
                self._assert_parity_and_stability(ep, inputs, ep.module()(*inputs))

    def test_absent_scalar_bound_remains_absent(self):
        inputs = (torch.tensor([-2.0, 0.0, 2.0]),)
        for minimum, maximum in ((None, 0), (0, None)):
            for positional_bounds in (0, 1, 2):
                with self.subTest(
                    bounds=(minimum, maximum), positional=positional_bounds
                ):
                    ep, node = self._export(
                        _ScalarBounds(minimum, maximum), inputs, positional_bounds
                    )
                    before = ep.module()(*inputs)
                    self.assertTrue(CastClampMixedTypeArgs().call(ep).modified)
                    for index, name, expected in (
                        (1, "min", minimum),
                        (2, "max", maximum),
                    ):
                        actual = (
                            node.args[index]
                            if index < len(node.args)
                            else node.kwargs[name]
                        )
                        if expected is None:
                            self.assertIsNone(actual)
                        else:
                            self.assertIs(type(actual), float)
                            self.assertEqual(actual, expected)
                    self._assert_parity_and_stability(ep, inputs, before)

    def test_shared_tensor_bounds_keep_the_input_and_cast_both_bounds(self):
        inputs = (torch.tensor([-2.0, 0.0, 2.0]), torch.tensor(0))
        for positional_bounds in (0, 1, 2):
            with self.subTest(positional=positional_bounds):
                ep, node = self._export(
                    _AliasedTensorBounds(), inputs, positional_bounds
                )
                input_node = node.args[0]
                bound_node = next(n for n in ep.graph.nodes if n.target == "bound")
                before = ep.module()(*inputs)
                self.assertTrue(CastClampMixedTypeArgs().call(ep).modified)
                self.assertIs(node.args[0], input_node)
                for index, name in ((1, "min"), (2, "max")):
                    cast = (
                        node.args[index]
                        if index < len(node.args)
                        else node.kwargs[name]
                    )
                    self.assertEqual(cast.target, torch.ops.aten._to_copy.default)
                    self.assertEqual(cast.args, (bound_node,))
                    self.assertEqual(cast.kwargs["dtype"], torch.float32)
                self.assertEqual(num_of_ops(ep, ops.aten._to_copy), 2)
                self._assert_parity_and_stability(ep, inputs, before)

    def test_input_aliasing_minimum_casts_each_slot(self):
        inputs = (torch.tensor([-2, 0, 2]), torch.tensor(1.0))
        for positional_bounds in (0, 1, 2):
            with self.subTest(positional=positional_bounds):
                ep, node = self._export(
                    _AliasedTensorBounds(input_is_minimum=True),
                    inputs,
                    positional_bounds,
                )
                input_node = node.args[0]
                before = ep.module()(*inputs)
                self.assertTrue(CastClampMixedTypeArgs().call(ep).modified)
                minimum = node.args[1] if positional_bounds else node.kwargs["min"]
                for cast in (node.args[0], minimum):
                    self.assertEqual(cast.target, torch.ops.aten._to_copy.default)
                    self.assertEqual(cast.args, (input_node,))
                    self.assertEqual(cast.kwargs["dtype"], torch.float32)
                self.assertEqual(num_of_ops(ep, ops.aten._to_copy), 2)
                self._assert_parity_and_stability(ep, inputs, before)

    def test_equal_scalar_bounds_convert_with_default_optimization_and_runtime(self):
        model = _ScalarBounds(0, 0).eval()
        value = torch.tensor([-2.0, 0.0, 2.0])
        circle = tico.convert(model, (value,))
        self.assertEqual(circle.runtime, "reference")
        actual = torch.from_numpy(circle(value))
        expected = model(value)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.equal(actual, expected))

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
from tico.passes.convert_repeat_to_expand_copy import ConvertRepeatToExpandCopy
from torch.export import Dim

from test.support.helper import num_of_ops
from test.support.pass_value_test import SinglePassValueTest


class RepeatKeepsDynamicDimNet(torch.nn.Module):
    """repeat((128, 1, 1, 1)) on [1, dim]: the dynamic dim is kept."""

    def forward(self, x):
        return x.repeat((128, 1, 1, 1))

    def get_example_inputs(self):
        return (torch.randn(1, 4),), {}

    def get_dynamic_shapes(self):
        return {"x": {1: Dim("dim", min=1, max=128)}}


class RepeatMultipliesDynamicDimNet(torch.nn.Module):
    """repeat((1, 2)) on [1, dim]: the dynamic dim would be multiplied."""

    def forward(self, x):
        return x.repeat((1, 2))

    def get_example_inputs(self):
        return (torch.randn(1, 4),), {}

    def get_dynamic_shapes(self):
        return {"x": {1: Dim("dim", min=1, max=128)}}


class RepeatStaticNet(torch.nn.Module):
    def forward(self, x):
        return x.repeat((3, 1, 1))

    def get_example_inputs(self):
        return (torch.randn(1, 4),), {}


def _expand_size(exported_program) -> list:
    for node in exported_program.graph.nodes:
        if node.op == "call_function" and node.target in (
            torch.ops.aten.expand_copy.default,
        ):
            return list(node.args[1])
    raise AssertionError("expand_copy node not found")


class ConvertRepeatToExpandCopyTest(SinglePassValueTest):
    def _export_dynamic(self, module):
        self.forward_args, self.forward_kwargs = module.get_example_inputs()
        with torch.no_grad():
            self.ep = torch.export.export(
                module.eval(),
                self.forward_args,
                self.forward_kwargs,
                dynamic_shapes=module.get_dynamic_shapes(),
            )
        self.initialized = True

    def test_static_repeat_is_converted_to_concrete_sizes(self):
        self.setup(RepeatStaticNet())
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.repeat.default]), 1
        )

        self.run_value_test(ConvertRepeatToExpandCopy())

        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.repeat.default]), 0
        )
        self.assertEqual(_expand_size(self.exported_program()), [3, 1, 4])

    def test_dynamic_dim_with_repeat_one_is_kept_as_minus_one(self):
        """The symbolic size must not be frozen to the example value."""

        self._export_dynamic(RepeatKeepsDynamicDimNet())
        self.run_value_test(ConvertRepeatToExpandCopy())

        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.repeat.default]), 0
        )
        self.assertEqual(_expand_size(self.exported_program()), [128, 1, 1, -1])

        # The rewritten program must still run for a different dynamic size.
        other = torch.randn(1, 7)
        result = self.exported_program().module()(other)
        self.assertEqual(tuple(result.shape), (128, 1, 1, 7))
        self.assertTrue(torch.equal(result, other.repeat((128, 1, 1, 1))))

    def test_dynamic_dim_with_repeat_greater_than_one_is_not_converted(self):
        """A repeat that multiplies a dynamic dim cannot become an expand."""

        self._export_dynamic(RepeatMultipliesDynamicDimNet())
        before = num_of_ops(self.exported_program(), [torch.ops.aten.repeat.default])
        self.assertEqual(before, 1)

        ConvertRepeatToExpandCopy().call(self.exported_program())

        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.repeat.default]), 1
        )
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.expand_copy.default]), 0
        )


if __name__ == "__main__":
    unittest.main()

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
from typing import List

import torch
from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape
from tico.quantization.passes.propagate_qparam_backward import PropagateQParamBackward
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.validate_args_kwargs import CatArgs, PermuteArgs, ReshapeArgs


class SingleOpPropagateQParamBackwardTest(unittest.TestCase):
    """
    How to use?

    Example)

    class DummyTest(SingleOpPropagateQParamBackwardTest):
        def test_pass(self):
            self.setup(DummyNet(), target_op=torch.ops.aten.reshape.default, dtype="uint8")

            inputs = [self.target.args[0]]

            self.run_test(inputs)
    """

    initialized: bool = False

    def setup(self, mod: torch.nn.Module, target_op, scale=1.0, zp=0, dtype="uint8"):
        assert hasattr(mod, "get_example_inputs")
        self.args, self.kwargs = mod.get_example_inputs()  # type: ignore[operator]
        self.scale = scale
        self.zp = zp
        self.dtype = dtype

        with torch.no_grad():
            self.ep = torch.export.export(
                mod.eval(),
                self.args,
                self.kwargs,
            )

        # This is necessary for testing Reshape on torch 2.5
        ConvertLayoutOpToReshape().call(self.ep)

        # Find target node
        target_node = None
        for node in self.ep.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == target_op:
                target_node = node
                break

        assert target_node is not None
        self.target = target_node

        # Set qparam to all input nodes
        qparam = QuantParam()
        qparam.scale = [self.scale]
        qparam.zero_point = [self.zp]
        qparam.dtype = self.dtype

        self.target.meta[QPARAM_KEY] = qparam

        self.initialized = True

    def run_test(self, inputs: List[torch.fx.Node]):
        assert self.initialized, "setup() must be called first"

        # Before pass
        for input_ in inputs:
            self.assertFalse(QPARAM_KEY in input_.meta)

        target_pass = PropagateQParamBackward()
        target_pass.call(self.ep)

        # After pass
        for input_ in inputs:
            self.assertTrue(QPARAM_KEY in input_.meta)
            self.assertTrue(input_.meta[QPARAM_KEY].scale == [self.scale])
            self.assertTrue(input_.meta[QPARAM_KEY].zero_point == [self.zp])
            self.assertTrue(input_.meta[QPARAM_KEY].dtype == self.dtype)


class ReshapeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ops.aten.reshape.default(x, (3, 2))

    def get_example_inputs(self):
        return (torch.randn(2, 3),), {}


class ReshapeTest(SingleOpPropagateQParamBackwardTest):
    def test_u8(self):
        self.setup(ReshapeModule(), torch.ops.aten.reshape.default, dtype="uint8")
        args = ReshapeArgs(*self.target.args)
        inputs = [args.input]
        self.run_test(inputs)

    def test_s16(self):
        self.setup(ReshapeModule(), torch.ops.aten.reshape.default, dtype="int16")
        args = ReshapeArgs(*self.target.args)
        inputs = [args.input]
        self.run_test(inputs)


class CatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.cat([x, y])

    def get_example_inputs(self):
        return (torch.randn(2, 3), torch.randn(2, 3)), {}


class CatTest(SingleOpPropagateQParamBackwardTest):
    def test_u8(self):
        self.setup(CatModule(), torch.ops.aten.cat.default, dtype="uint8")
        args = CatArgs(*self.target.args)
        inputs = args.tensors
        self.run_test(inputs)

    def test_s16(self):
        self.setup(CatModule(), torch.ops.aten.cat.default, dtype="int16")
        args = CatArgs(*self.target.args)
        inputs = args.tensors
        self.run_test(inputs)


class PermuteModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ops.aten.permute.default(x, (0, 1))

    def get_example_inputs(self):
        return (torch.randn(2, 3),), {}


class PermuteTest(SingleOpPropagateQParamBackwardTest):
    def test_u8(self):
        self.setup(PermuteModule(), torch.ops.aten.permute.default, dtype="uint8")
        args = PermuteArgs(*self.target.args)
        inputs = [args.input]
        self.run_test(inputs)

    def test_s16(self):
        self.setup(PermuteModule(), torch.ops.aten.permute.default, dtype="int16")
        args = PermuteArgs(*self.target.args)
        inputs = [args.input]
        self.run_test(inputs)

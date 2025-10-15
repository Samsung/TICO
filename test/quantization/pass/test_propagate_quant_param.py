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
from tico.experimental.quantization import convert, prepare
from tico.experimental.quantization.config.pt2e import PT2EConfig
from tico.experimental.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.experimental.quantization.passes.propagate_qparam_forward import (
    PropagateQParamForward,
)
from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.torch_compat import export_produces_slice


class LinearPermuteModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, arg):
        z = self.linear(arg)
        return torch.permute(z, (1, 0))

    def get_example_inputs(self):
        return (torch.randn(2, 3),), {}


class PropagateQParamForwardTest(unittest.TestCase):
    def test_pass(self):
        m: LinearPermuteModule | torch.nn.Module = LinearPermuteModule().eval()
        assert isinstance(m, LinearPermuteModule)
        args, kwargs = m.get_example_inputs()

        q_m = prepare(m, PT2EConfig(), args=args, kwargs=kwargs, inplace=False)

        # Calibration
        for i in range(10):
            cal_args, cal_kwargs = m.get_example_inputs()
            q_m(*cal_args, **cal_kwargs)

        q_m = convert(q_m, inplace=False)

        ep = torch.export.export(q_m, args, kwargs)

        FoldQuantOps().call(ep)
        # Before pass
        for node in ep.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == torch.ops.aten.permute.default:
                self.assertFalse(QPARAM_KEY in node.meta)
        target_pass = PropagateQParamForward()
        target_pass.call(ep)
        # After pass
        for node in ep.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == torch.ops.aten.permute.default:
                self.assertTrue(QPARAM_KEY in node.meta)


class SingleOpPropagateQParamForwardTest(unittest.TestCase):
    """
    How to use?

    Example)

    class DummyTest(SingleOpPropagateQParamForwardTest):
        def test_pass(self):
            self.setup(DummyNet(), target_op=torch.ops.aten.reshape.default, dtype="uint8")
            self.run_test()
    """

    initialized: bool = False

    def setup(self, mod: torch.nn.Module, target_op, scale=1.0, zp=0, dtype="uint8"):
        assert hasattr(mod, "get_example_inputs")
        self.args, self.kwargs = mod.get_example_inputs()  # type: ignore[operator]
        self.scale = scale
        self.zp = zp
        self.dtype = dtype

        with torch.no_grad():
            self.ep = torch.export.export(mod.eval(), self.args, self.kwargs)

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
        user_inputs = []
        for node in self.ep.graph.nodes:
            if node.op != "placeholder":
                continue
            if node.name in self.ep.graph_signature.user_inputs:
                user_inputs.append(node)

        assert len(user_inputs) > 0

        for user_input in user_inputs:
            qparam = QuantParam()
            qparam.scale = [self.scale]
            qparam.zero_point = [self.zp]
            qparam.dtype = self.dtype

            user_input.meta[QPARAM_KEY] = qparam

        self.initialized = True

    def run_test(self):
        # Before pass
        self.assertFalse(QPARAM_KEY in self.target.meta)

        target_pass = PropagateQParamForward()
        target_pass.call(self.ep)

        # After pass
        self.assertTrue(QPARAM_KEY in self.target.meta)
        self.assertTrue(self.target.meta[QPARAM_KEY].scale == [self.scale])
        self.assertTrue(self.target.meta[QPARAM_KEY].zero_point == [self.zp])
        self.assertTrue(self.target.meta[QPARAM_KEY].dtype == self.dtype)


class PermuteModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.permute(x, (1, 0))

    def get_example_inputs(self):
        return (torch.randn(2, 3),), {}


class PermuteTest(SingleOpPropagateQParamForwardTest):
    def test_u8(self):
        self.setup(PermuteModule(), torch.ops.aten.permute.default, dtype="uint8")
        self.run_test()

    def test_s16(self):
        self.setup(PermuteModule(), torch.ops.aten.permute.default, dtype="int16")
        self.run_test()


class ReshapeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ops.aten.reshape.default(x, (3, 2))

    def get_example_inputs(self):
        return (torch.randn(2, 3),), {}


class ReshapeTest(SingleOpPropagateQParamForwardTest):
    def test_u8(self):
        self.setup(ReshapeModule(), torch.ops.aten.reshape.default, dtype="uint8")
        self.run_test()

    def test_s16(self):
        self.setup(ReshapeModule(), torch.ops.aten.reshape.default, dtype="int16")
        self.run_test()


class SliceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, 1]

    def get_example_inputs(self):
        return (torch.randn(2, 3),), {}


@unittest.skipUnless(
    export_produces_slice(),
    "Skip when torch doesn't produce redundant slices.",
)
class SliceTest(SingleOpPropagateQParamForwardTest):
    def test_u8(self):
        self.setup(SliceModule(), torch.ops.aten.slice.Tensor, dtype="uint8")
        self.run_test()

    def test_s16(self):
        self.setup(SliceModule(), torch.ops.aten.slice.Tensor, dtype="int16")
        self.run_test()


class NegModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return -x

    def get_example_inputs(self):
        return (torch.randn(2, 3),), {}


class NegTest(SingleOpPropagateQParamForwardTest):
    # TODO Support u8
    def test_s16(self):
        self.setup(NegModule(), torch.ops.aten.neg.default, dtype="int16")
        self.run_test()


class CatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.cat([x, y])

    def get_example_inputs(self):
        return (torch.randn(2, 3), torch.randn(2, 3)), {}


class CatTest(SingleOpPropagateQParamForwardTest):
    # TODO Support u8
    def test_s16(self):
        self.setup(CatModule(), torch.ops.aten.cat.default, dtype="int16")
        self.run_test()

    def test_s16_different_scale(self):
        self.setup(CatModule(), torch.ops.aten.cat.default, scale=1.0, dtype="int16")

        inputs = self.target.args[0]
        # First input scale = 1.0
        # Second input scale = 0.5
        inputs[1].meta[QPARAM_KEY].scale = [0.5]

        # The test will check cat's scale is 1.0, the larger one
        self.run_test()

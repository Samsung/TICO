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
from tico.experimental.quantization.passes.insert_quantize_on_dtype_mismatch import (
    InsertQuantizeOnDtypeMismatch,
)
from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape
from tico.serialize.quant_param import QPARAM_KEY, QuantParam

from test.modules.op.add import SimpleAdd
from test.modules.op.bmm import SimpleBatchMatMul
from test.modules.op.linear import SimpleLinear
from test.modules.op.mul import SimpleMulWithTensor
from test.modules.op.permute import SimplePermute
from test.modules.op.relu import SimpleRelu
from test.modules.op.reshape import ReshapeTorchAPI


class InsertQuantizeOnDtypeMismatchTest(unittest.TestCase):
    """
    This class runs a test for InsertQuantizeOpOnDtypeMismatch pass
    - Set up a network with target_op and dtype of input. Target node's
      dtype is set differently from input dtype
    - After the pass runs, node's dtype must be the same with desired_dtype

    How to use?

    Example)

    class DummyTest(InsertQuantizeOpOnDtypeMismatchTest):
        def test_pass(self):
            self.setup(DummyNet(), target_op=torch.ops.aten.reshape.default, input_dtype="uint8", desired_dtype="uint8")
            self.run_test()
    """

    initialized: bool = False

    def setup(
        self,
        mod: torch.nn.Module,
        target_op,
        scale=1.0,
        zp=0,
        input_dtype="uint8",
        desired_dtype="uint8",
    ):
        assert hasattr(mod, "get_example_inputs")
        self.inputs = mod.get_example_inputs()  # type: ignore[operator]
        self.scale = scale
        self.zp = zp
        self.input_dtype = input_dtype
        self.desired_dtype = desired_dtype

        with torch.no_grad():
            self.ep = torch.export.export(mod.eval(), self.inputs)

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
            qparam.dtype = self.input_dtype

            user_input.meta[QPARAM_KEY] = qparam

        node_qparam = QuantParam()
        if self.input_dtype == "int16":
            node_qparam.scale = [1.0]
            node_qparam.zero_point = [0]
            node_qparam.dtype = "uint8"
        elif self.input_dtype == "uint8":
            node_qparam.scale = [1.0]
            node_qparam.zero_point = [0]
            node_qparam.dtype = "int16"
        else:
            raise RuntimeError("Unsupported dtype")

        self.target.meta[QPARAM_KEY] = node_qparam

        self.initialized = True

    def run_test(self):
        # Before pass
        # InsertQuantizeOnDtypeMismatch inserts Quantize Op when input dtype does not match with output dtype
        self.assertNotEqual(self.target.meta[QPARAM_KEY].dtype, self.input_dtype)

        target_pass = InsertQuantizeOnDtypeMismatch()
        target_pass.call(self.ep)

        # After pass
        # Check whether dtype of target is the same with desired_dtype
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, self.desired_dtype)

        # Check whether input/output of target has the same dtype
        # TODO Update args[0] if an Op does not have its input as args[0]
        self.assertEqual(
            self.target.meta[QPARAM_KEY].dtype,
            self.target.args[0].meta[QPARAM_KEY].dtype,
        )


class PermuteTest(InsertQuantizeOnDtypeMismatchTest):
    def test_i8o16(self):
        self.setup(
            SimplePermute(),
            torch.ops.aten.permute.default,
            input_dtype="uint8",
            desired_dtype="uint8",
        )
        self.run_test()

    # Quantize Op is inserted before Permute, so desired_dtype of Reshape is uint8
    # After conversion)
    # Input (s16) -> Quantize (u8) -> Permute (u8)
    def test_i16o8(self):
        self.setup(
            SimplePermute(),
            torch.ops.aten.permute.default,
            input_dtype="int16",
            desired_dtype="uint8",
        )
        self.run_test()


class LinearTest(InsertQuantizeOnDtypeMismatchTest):
    def test_i8o16(self):
        self.setup(
            SimpleLinear(),
            torch.ops.aten.linear.default,
            input_dtype="uint8",
            desired_dtype="uint8",
        )
        self.run_test()


class MulTest(InsertQuantizeOnDtypeMismatchTest):
    def test_i16o8(self):
        self.setup(
            SimpleMulWithTensor(),
            torch.ops.aten.mul.Tensor,
            input_dtype="int16",
            desired_dtype="int16",
        )
        self.run_test()


class BMMTest(InsertQuantizeOnDtypeMismatchTest):
    def test_i16o8(self):
        self.setup(
            SimpleBatchMatMul(),
            torch.ops.aten.bmm.default,
            input_dtype="int16",
            desired_dtype="int16",
        )
        self.run_test()

    def test_i8o16(self):
        self.setup(
            SimpleBatchMatMul(),
            torch.ops.aten.bmm.default,
            input_dtype="uint8",
            desired_dtype="uint8",
        )
        self.run_test()


class ReshapeTest(InsertQuantizeOnDtypeMismatchTest):
    def test_i8o16(self):
        self.setup(
            ReshapeTorchAPI(),
            torch.ops.aten.reshape.default,
            input_dtype="uint8",
            desired_dtype="uint8",
        )
        self.run_test()

    # Quantize Op is inserted before Reshape, so desired_dtype of Reshape is uint8
    # After conversion)
    # Input (s16) -> Quantize (u8) -> Reshape (u8)
    def test_i16o8(self):
        self.setup(
            ReshapeTorchAPI(),
            torch.ops.aten.reshape.default,
            input_dtype="int16",
            desired_dtype="uint8",
        )
        self.run_test()


class ReluTest(InsertQuantizeOnDtypeMismatchTest):
    def test_i16o8(self):
        self.setup(
            SimpleRelu(),
            torch.ops.aten.relu.default,
            input_dtype="int16",
            desired_dtype="int16",
        )
        self.run_test()


class AddTest(InsertQuantizeOnDtypeMismatchTest):
    def test_i16o8(self):
        self.setup(
            SimpleAdd(),
            torch.ops.aten.add.Tensor,
            input_dtype="int16",
            desired_dtype="int16",
        )
        self.run_test()

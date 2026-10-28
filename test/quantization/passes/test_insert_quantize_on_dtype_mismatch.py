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

import copy
import unittest

import torch
from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape
from tico.quantization.passes.insert_quantize_on_dtype_mismatch import (
    InsertQuantizeOnDtypeMismatch,
)
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.errors import NotYetSupportedError

from test.modules.op.add import SimpleAdd
from test.modules.op.bmm import SimpleBatchMatMul
from test.modules.op.cat import SimpleCatDefault
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
        self.args, self.kwargs = mod.get_example_inputs()  # type: ignore[operator]
        self.scale = scale
        self.zp = zp
        self.input_dtype = input_dtype
        self.desired_dtype = desired_dtype

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
    def test_i8o16(self):
        self.setup(
            SimpleRelu(),
            torch.ops.aten.relu.default,
            input_dtype="uint8",
            desired_dtype="uint8",
        )
        self.run_test()

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

    def test_no_mismatch_add(self):
        # Test case where input and output dtypes are the same (uint8)
        self.setup(
            SimpleAdd(),
            torch.ops.aten.add.Tensor,
            input_dtype="uint8",
            desired_dtype="uint8",
        )
        # Manually set output dtype to be same as input for this test
        self.target.meta[QPARAM_KEY].dtype = "uint8"
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, self.input_dtype)
        target_pass = InsertQuantizeOnDtypeMismatch()
        target_pass.call(self.ep)
        # Dtypes should remain unchanged
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "uint8")
        self.assertEqual(self.target.args[0].meta[QPARAM_KEY].dtype, "uint8")

    def test_mismatch_input_dtypes_add(self):
        # Test case where inputs x and y have different dtypes
        self.setup(
            SimpleAdd(),
            torch.ops.aten.add.Tensor,
            input_dtype="uint8",  # This will be set for the first placeholder
            desired_dtype="int16",
        )
        # Manually set a different dtype for the second input
        # Find the second placeholder node
        user_inputs = []
        for node in self.ep.graph.nodes:
            if (
                node.op == "placeholder"
                and node.name in self.ep.graph_signature.user_inputs
            ):
                user_inputs.append(node)
        self.assertGreaterEqual(len(user_inputs), 2)
        second_input_qparam = QuantParam()
        second_input_qparam.scale = [1.0]
        second_input_qparam.zero_point = [0]
        second_input_qparam.dtype = "int16"
        user_inputs[1].meta[QPARAM_KEY] = second_input_qparam

        self.assertEqual(self.target.args[0].meta[QPARAM_KEY].dtype, "uint8")
        self.assertEqual(
            self.target.args[1].meta[QPARAM_KEY].dtype, "int16"
        )  # Assuming args[1] is the second input

        target_pass = InsertQuantizeOnDtypeMismatch()
        target_pass.call(self.ep)
        # Dtypes should remain unchanged as handler should return early
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "int16")

    def test_unsupported_add_dtype(self):
        self.setup(
            SimpleAdd(),
            torch.ops.aten.add.Tensor,
            input_dtype="uint8",
            desired_dtype="float32",  # An unsupported target dtype
        )
        # Manually set output dtype to an unsupported type for conversion
        self.target.meta[QPARAM_KEY].dtype = "float32"
        target_pass = InsertQuantizeOnDtypeMismatch()
        with self.assertRaises(NotYetSupportedError):
            target_pass.call(self.ep)

    def test_i16_to_u8_add(self):
        self.setup(
            SimpleAdd(),
            torch.ops.aten.add.Tensor,
            input_dtype="int16",
            desired_dtype="uint8",  # Pass will change node's qparam to i16
        )
        # The pass expects output dtype to be uint8 if input is int16 for this path
        # The setup method sets output dtype based on input_dtype, which is int16 -> uint8.
        # So desired_dtype should be uint8, and the pass will modify node's qparam to int16.
        # Let's adjust the setup for this specific case.
        self.target.meta[QPARAM_KEY].dtype = "uint8"  # Target output is uint8

        target_pass = InsertQuantizeOnDtypeMismatch()
        target_pass.call(self.ep)

        # After pass, node's dtype should be int16 because it was converted from u8
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "int16")
        # The inserted quantize op should have uint8
        self.assertEqual(list(self.target.users)[0].meta[QPARAM_KEY].dtype, "uint8")


class CatTest(InsertQuantizeOnDtypeMismatchTest):
    def test_no_mismatch_cat(self):
        # Test case where input and output dtypes are the same (uint8)
        self.setup(
            SimpleCatDefault(),
            torch.ops.aten.cat.default,
            input_dtype="uint8",
            desired_dtype="uint8",
        )
        # Manually set output dtype to be same as input
        self.target.meta[QPARAM_KEY].dtype = "uint8"
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, self.input_dtype)
        target_pass = InsertQuantizeOnDtypeMismatch()
        target_pass.call(self.ep)
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "uint8")
        self.assertEqual(
            self.target.args[0][0].meta[QPARAM_KEY].dtype, "uint8"
        )  # args[0] is a list of tensors

    def test_i16_to_u8_cat(self):
        self.setup(
            SimpleCatDefault(),
            torch.ops.aten.cat.default,
            input_dtype="int16",
            desired_dtype="uint8",  # Pass will change node's qparam to i16
        )
        self.target.meta[QPARAM_KEY].dtype = "uint8"  # Target output is uint8

        target_pass = InsertQuantizeOnDtypeMismatch()
        target_pass.call(self.ep)

        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "int16")
        self.assertEqual(list(self.target.users)[0].meta[QPARAM_KEY].dtype, "uint8")

    def test_unsupported_cat_dtype(self):
        self.setup(
            SimpleCatDefault(),
            torch.ops.aten.cat.default,
            input_dtype="uint8",
            desired_dtype="float32",
        )
        self.target.meta[QPARAM_KEY].dtype = "float32"
        target_pass = InsertQuantizeOnDtypeMismatch()
        with self.assertRaises(NotYetSupportedError):
            target_pass.call(self.ep)


class _ConcatContract(torch.nn.Module):
    def __init__(self, order, shared_producer=False):
        super().__init__()
        self.order = order
        self.shared_producer = shared_producer

    def forward(self, *inputs):
        tensors = list(inputs)
        if self.shared_producer:
            tensors[0] = torch.relu(tensors[0])
        concat = torch.cat([tensors[i] for i in self.order], dim=-1)
        return concat, tensors[0], torch.neg(tensors[0])


class MixedCatDtypeMismatchTest(unittest.TestCase):
    def _export(self, input_qparams, output_qparam, order=None, shared_producer=False):
        if order is None:
            order = tuple(range(len(input_qparams)))
        module = _ConcatContract(order, shared_producer).eval()
        inputs = tuple(torch.full((1, 2), float(i)) for i in range(len(input_qparams)))
        ep = torch.export.export(module, inputs)
        placeholders = [node for node in ep.graph.nodes if node.op == "placeholder"]
        for node, qparam in zip(placeholders, input_qparams):
            node.meta[QPARAM_KEY] = copy.deepcopy(qparam)
        concat = next(
            node for node in ep.graph.nodes if node.target == torch.ops.aten.cat.default
        )
        concat.meta[QPARAM_KEY] = copy.deepcopy(output_qparam)
        for node in ep.graph.nodes:
            if node.target in (torch.ops.aten.relu.default, torch.ops.aten.neg.default):
                node.meta[QPARAM_KEY] = copy.deepcopy(input_qparams[0])
        return ep, concat

    def _assert_conversion(self, converted, source, destination):
        self.assertEqual(
            converted.target,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        )
        limits = torch.iinfo(getattr(torch, destination.dtype))
        self.assertEqual(
            converted.args,
            (
                source,
                destination.scale[0],
                destination.zero_point[0],
                limits.min,
                limits.max,
                getattr(torch, destination.dtype),
            ),
        )
        self.assertEqual(converted.meta[QPARAM_KEY], destination)
        self.assertIsNot(converted.meta[QPARAM_KEY], destination)

    def _assert_stable(self, ep):
        nodes = list(ep.graph.nodes)
        graph = str(ep.graph)
        qparams = {
            node: copy.deepcopy(node.meta[QPARAM_KEY])
            for node in nodes
            if QPARAM_KEY in node.meta
        }
        InsertQuantizeOnDtypeMismatch().call(ep)
        self.assertEqual(list(ep.graph.nodes), nodes)
        self.assertEqual(str(ep.graph), graph)
        for node, qparam in qparams.items():
            self.assertEqual(node.meta[QPARAM_KEY], qparam)

    def test_all_mixed_dtype_orders_use_actual_destination_grid(self):
        for input_dtypes in (("uint8", "int16"), ("int16", "uint8")):
            for output_dtype in ("uint8", "int16"):
                with self.subTest(inputs=input_dtypes, output=output_dtype):
                    sources = [
                        QuantParam(scale=[0.125], zero_point=[91], dtype="uint8"),
                        QuantParam(scale=[0.0625], zero_point=[0], dtype="int16"),
                    ]
                    if input_dtypes[0] == "int16":
                        sources.reverse()
                    destination = QuantParam(
                        scale=[0.375 if output_dtype == "uint8" else 0.03125],
                        zero_point=[137 if output_dtype == "uint8" else 0],
                        dtype=output_dtype,
                    )
                    ep, concat = self._export(sources, destination)
                    operands = tuple(concat.args[0])
                    # Circle dtypes come from QPARAM_KEY, not the float FX values.
                    self.assertTrue(
                        all(
                            node.meta["val"].dtype == torch.float32 for node in operands
                        )
                    )
                    InsertQuantizeOnDtypeMismatch().call(ep)
                    self.assertEqual(concat.meta[QPARAM_KEY], destination)
                    for converted, source, qparam in zip(
                        concat.args[0], operands, sources
                    ):
                        self._assert_conversion(converted, source, destination)
                        self.assertEqual(source.meta[QPARAM_KEY], qparam)
                    self._assert_stable(ep)

    def test_integer_fx_values_support_both_destination_dtypes(self):
        sources = [
            QuantParam(scale=[0.125], zero_point=[91], dtype="uint8"),
            QuantParam(scale=[0.0625], zero_point=[0], dtype="int16"),
        ]
        for output_dtype in ("uint8", "int16"):
            with self.subTest(output=output_dtype):
                destination = QuantParam(
                    scale=[0.25],
                    zero_point=[137 if output_dtype == "uint8" else 0],
                    dtype=output_dtype,
                )
                ep, concat = self._export(sources, destination)
                operands = tuple(concat.args[0])
                for operand, qparam in zip(operands, sources):
                    operand.meta["val"] = operand.meta["val"].to(
                        dtype=getattr(torch, qparam.dtype)
                    )
                InsertQuantizeOnDtypeMismatch().call(ep)
                for converted, source in zip(concat.args[0], operands):
                    self._assert_conversion(converted, source, destination)
                    self.assertEqual(
                        converted.meta["val"].dtype, getattr(torch, output_dtype)
                    )
                    self.assertEqual(
                        converted.meta["val"].shape, source.meta["val"].shape
                    )
                self._assert_stable(ep)

    def test_same_dtype_distinct_domains_repeated_edges_are_stable(self):
        destination = QuantParam(scale=[0.25], zero_point=[117], dtype="uint8")
        sources = [
            destination,
            QuantParam(scale=[0.5], zero_point=[29], dtype="uint8"),
        ]
        ep, concat = self._export(
            sources, destination, order=(0, 1, 1, 0), shared_producer=True
        )
        operands = tuple(concat.args[0])
        original_domains = [copy.deepcopy(node.meta[QPARAM_KEY]) for node in operands]
        InsertQuantizeOnDtypeMismatch().call(ep)
        converted = concat.args[0]
        self.assertIs(converted[0], operands[0])
        self.assertIs(converted[3], operands[3])
        self.assertIs(converted[1], converted[2])
        self._assert_conversion(converted[1], operands[1], destination)
        for node, domain in zip(operands, original_domains):
            self.assertEqual(node.meta[QPARAM_KEY], domain)
        self.assertEqual(
            sum(
                node.target
                == torch.ops.quantized_decomposed.quantize_per_tensor.default
                for node in ep.graph.nodes
            ),
            1,
        )
        self._assert_stable(ep)

    def test_later_mismatch_repeated_operands_and_matching_grid(self):
        destination = QuantParam(scale=[0.25], zero_point=[117], dtype="uint8")
        sources = [
            destination,
            QuantParam(scale=[0.25], zero_point=[29], dtype="uint8"),
            QuantParam(scale=[0.03125], zero_point=[0], dtype="int16"),
        ]
        ep, concat = self._export(sources, destination, order=(0, 1, 2, 2, 0))
        operands = tuple(concat.args[0])
        axis = concat.args[1]
        InsertQuantizeOnDtypeMismatch().call(ep)
        converted = concat.args[0]
        self.assertEqual(len(converted), len(operands))
        self.assertIs(converted[0], operands[0])
        self.assertIs(converted[4], operands[4])
        self._assert_conversion(converted[1], operands[1], destination)
        self._assert_conversion(converted[2], operands[2], destination)
        self._assert_conversion(converted[3], operands[3], destination)
        self.assertEqual(concat.args[1], axis)
        self.assertEqual(concat.meta[QPARAM_KEY], destination)
        self._assert_stable(ep)

    def test_shared_producer_inputs_and_other_outputs_are_preserved(self):
        sources = [
            QuantParam(scale=[0.0625], zero_point=[0], dtype="int16"),
            QuantParam(scale=[0.5], zero_point=[83], dtype="uint8"),
        ]
        destination = QuantParam(scale=[0.25], zero_point=[149], dtype="uint8")
        ep, concat = self._export(sources, destination, shared_producer=True)
        operands = tuple(concat.args[0])
        shared = operands[0]
        self.assertEqual(shared.target, torch.ops.aten.relu.default)
        output = next(node for node in ep.graph.nodes if node.op == "output")
        branch = next(
            node for node in ep.graph.nodes if node.target == torch.ops.aten.neg.default
        )
        output_args = output.args
        branch_args = branch.args
        signature = copy.deepcopy(ep.graph_signature)
        preserved = {
            node: copy.deepcopy(node.meta[QPARAM_KEY])
            for node in ep.graph.nodes
            if QPARAM_KEY in node.meta and node is not concat
        }
        InsertQuantizeOnDtypeMismatch().call(ep)
        self.assertEqual(ep.graph_signature, signature)
        self.assertEqual(output.args, output_args)
        self.assertEqual(branch.args, branch_args)
        self.assertIs(branch.args[0], shared)
        self.assertIs(output.args[0][1], shared)
        for node, qparam in preserved.items():
            self.assertEqual(node.meta[QPARAM_KEY], qparam)
        for converted, source in zip(concat.args[0], operands):
            self._assert_conversion(converted, source, destination)
        self._assert_stable(ep)

    def test_uniform_matching_inputs_remain_noops(self):
        for dtype in ("uint8", "int16"):
            with self.subTest(dtype=dtype):
                qparam = QuantParam(
                    scale=[0.125],
                    zero_point=[113 if dtype == "uint8" else 0],
                    dtype=dtype,
                )
                ep, concat = self._export([qparam, qparam, qparam], qparam)
                graph = str(ep.graph)
                InsertQuantizeOnDtypeMismatch().call(ep)
                self.assertEqual(str(ep.graph), graph)
                self.assertEqual(concat.meta[QPARAM_KEY], qparam)
                self._assert_stable(ep)

    def test_uniform_int16_to_uint8_keeps_conversion_after_concat(self):
        source = QuantParam(scale=[0.0625], zero_point=[0], dtype="int16")
        destination = QuantParam(scale=[0.25], zero_point=[149], dtype="uint8")
        ep, concat = self._export([source, source], destination)
        operands = tuple(concat.args[0])
        InsertQuantizeOnDtypeMismatch().call(ep)
        self.assertEqual(tuple(concat.args[0]), operands)
        self.assertEqual(concat.meta[QPARAM_KEY].dtype, "int16")
        output = next(node for node in ep.graph.nodes if node.op == "output")
        self._assert_conversion(output.args[0][0], concat, destination)
        for operand in operands:
            self.assertEqual(operand.meta[QPARAM_KEY], source)
        self._assert_stable(ep)

    def test_unsupported_fully_quantized_mixed_inputs_fail_explicitly(self):
        destination = QuantParam(scale=[0.25], zero_point=[117], dtype="uint8")
        unsupported = [
            QuantParam(scale=[0.125], zero_point=[0], dtype="int8"),
            QuantParam(
                scale=[0.125, 0.25],
                zero_point=[0, 0],
                quantized_dimension=1,
                dtype="int16",
            ),
            QuantParam(
                scale=[0.125], zero_point=[0], quantized_dimension=1, dtype="int16"
            ),
        ]
        for source in unsupported:
            with self.subTest(source=source):
                ep, _ = self._export([destination, source], destination)
                with self.assertRaisesRegex(
                    NotYetSupportedError,
                    "mixed-dtype concat.*per-tensor affine uint8/int16",
                ):
                    InsertQuantizeOnDtypeMismatch().call(ep)

    def test_missing_or_incomplete_qparams_remain_unchanged(self):
        sources = [
            QuantParam(scale=[0.25], zero_point=[117], dtype="uint8"),
            QuantParam(scale=[0.125], zero_point=[0], dtype="int16"),
        ]
        for missing_from in ("input", "output"):
            for field in (None, "dtype", "scale", "zero_point"):
                with self.subTest(missing_from=missing_from, field=field):
                    ep, concat = self._export(sources, sources[0])
                    node = concat.args[0][1] if missing_from == "input" else concat
                    if field is None:
                        del node.meta[QPARAM_KEY]
                    else:
                        setattr(
                            node.meta[QPARAM_KEY],
                            field,
                            "" if field == "dtype" else None,
                        )
                    graph = str(ep.graph)
                    InsertQuantizeOnDtypeMismatch().call(ep)
                    self.assertEqual(str(ep.graph), graph)


class LinearDtypeMismatchTest(InsertQuantizeOnDtypeMismatchTest):
    def test_u8_to_i16_linear(self):
        self.setup(
            SimpleLinear(),
            torch.ops.aten.linear.default,
            input_dtype="uint8",
            # desired_dtype will be int16 after pass modifies node's qparam
        )
        # Setup already makes node's qparam int16 if input is uint8
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "int16")
        # Input dtype is uint8
        self.assertEqual(self.target.args[0].meta[QPARAM_KEY].dtype, "uint8")

        target_pass = InsertQuantizeOnDtypeMismatch()
        target_pass.call(self.ep)

        # After pass, node's dtype should be u8 (converted from i16)
        # A quantize op (i16) should be inserted after
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "uint8")
        self.assertEqual(list(self.target.users)[0].meta[QPARAM_KEY].dtype, "int16")

    def test_no_mismatch_linear(self):
        self.setup(
            SimpleLinear(),
            torch.ops.aten.linear.default,
            input_dtype="uint8",
            desired_dtype="uint8",
        )
        self.target.meta[QPARAM_KEY].dtype = "uint8"
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, self.input_dtype)
        target_pass = InsertQuantizeOnDtypeMismatch()
        target_pass.call(self.ep)
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "uint8")

    def test_unsupported_linear_dtype(self):
        self.setup(
            SimpleLinear(),
            torch.ops.aten.linear.default,
            input_dtype="uint8",
            desired_dtype="float32",
        )
        self.target.meta[QPARAM_KEY].dtype = "float32"
        target_pass = InsertQuantizeOnDtypeMismatch()
        with self.assertRaises(NotYetSupportedError):
            target_pass.call(self.ep)


class MulDtypeMismatchTest(InsertQuantizeOnDtypeMismatchTest):
    def test_no_mismatch_mul(self):
        self.setup(
            SimpleMulWithTensor(),
            torch.ops.aten.mul.Tensor,
            input_dtype="uint8",
            desired_dtype="uint8",
        )
        self.target.meta[QPARAM_KEY].dtype = "uint8"
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, self.input_dtype)
        target_pass = InsertQuantizeOnDtypeMismatch()
        target_pass.call(self.ep)
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "uint8")

    def test_unsupported_mul_dtype(self):
        self.setup(
            SimpleMulWithTensor(),
            torch.ops.aten.mul.Tensor,
            input_dtype="uint8",
            desired_dtype="float32",
        )
        self.target.meta[QPARAM_KEY].dtype = "float32"
        target_pass = InsertQuantizeOnDtypeMismatch()
        with self.assertRaises(NotYetSupportedError):
            target_pass.call(self.ep)


class BMMDtypeMismatchTest(InsertQuantizeOnDtypeMismatchTest):
    def test_u8_to_i16_bmm(self):
        self.setup(
            SimpleBatchMatMul(),
            torch.ops.aten.bmm.default,
            input_dtype="uint8",
        )
        # Setup makes node's qparam int16 if input is uint8
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "int16")
        self.assertEqual(self.target.args[0].meta[QPARAM_KEY].dtype, "uint8")

        target_pass = InsertQuantizeOnDtypeMismatch()
        target_pass.call(self.ep)

        # After pass, node's dtype should be u8 (converted from i16)
        # A quantize op (i16) should be inserted after
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "uint8")
        self.assertEqual(list(self.target.users)[0].meta[QPARAM_KEY].dtype, "int16")

    def test_no_mismatch_bmm(self):
        self.setup(
            SimpleBatchMatMul(),
            torch.ops.aten.bmm.default,
            input_dtype="uint8",
            desired_dtype="uint8",
        )
        self.target.meta[QPARAM_KEY].dtype = "uint8"
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, self.input_dtype)
        target_pass = InsertQuantizeOnDtypeMismatch()
        target_pass.call(self.ep)
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "uint8")

    def test_unsupported_bmm_dtype(self):
        self.setup(
            SimpleBatchMatMul(),
            torch.ops.aten.bmm.default,
            input_dtype="uint8",
            desired_dtype="float32",
        )
        self.target.meta[QPARAM_KEY].dtype = "float32"
        target_pass = InsertQuantizeOnDtypeMismatch()
        with self.assertRaises(NotYetSupportedError):
            target_pass.call(self.ep)


class PermuteDtypeMismatchTest(InsertQuantizeOnDtypeMismatchTest):
    def test_no_mismatch_permute(self):
        self.setup(
            SimplePermute(),
            torch.ops.aten.permute.default,
            input_dtype="uint8",
            desired_dtype="uint8",
        )
        self.target.meta[QPARAM_KEY].dtype = "uint8"
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, self.input_dtype)
        target_pass = InsertQuantizeOnDtypeMismatch()
        target_pass.call(self.ep)
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "uint8")

    def test_u8_to_i16_permute(self):
        self.setup(
            SimplePermute(),
            torch.ops.aten.permute.default,
            input_dtype="uint8",
        )
        # Setup makes node's qparam int16 if input is uint8
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "int16")
        self.assertEqual(self.target.args[0].meta[QPARAM_KEY].dtype, "uint8")

        target_pass = InsertQuantizeOnDtypeMismatch()
        target_pass.call(self.ep)

        # After pass, node's dtype should be u8 (converted from i16)
        # A quantize op (i16) should be inserted after
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "uint8")
        self.assertEqual(list(self.target.users)[0].meta[QPARAM_KEY].dtype, "int16")

    def test_unsupported_permute_dtype(self):
        self.setup(
            SimplePermute(),
            torch.ops.aten.permute.default,
            input_dtype="uint8",
            desired_dtype="float32",
        )
        self.target.meta[QPARAM_KEY].dtype = "float32"
        target_pass = InsertQuantizeOnDtypeMismatch()
        with self.assertRaises(NotYetSupportedError):
            target_pass.call(self.ep)


class ReshapeDtypeMismatchTest(InsertQuantizeOnDtypeMismatchTest):
    def test_no_mismatch_reshape(self):
        self.setup(
            ReshapeTorchAPI(),
            torch.ops.aten.reshape.default,
            input_dtype="uint8",
            desired_dtype="uint8",
        )
        self.target.meta[QPARAM_KEY].dtype = "uint8"
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, self.input_dtype)
        target_pass = InsertQuantizeOnDtypeMismatch()
        target_pass.call(self.ep)
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "uint8")

    def test_u8_to_i16_reshape(self):
        self.setup(
            ReshapeTorchAPI(),
            torch.ops.aten.reshape.default,
            input_dtype="uint8",
        )
        # Setup makes node's qparam int16 if input is uint8
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "int16")
        self.assertEqual(self.target.args[0].meta[QPARAM_KEY].dtype, "uint8")

        target_pass = InsertQuantizeOnDtypeMismatch()
        target_pass.call(self.ep)

        # After pass, node's dtype should be u8 (converted from i16)
        # A quantize op (i16) should be inserted after
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "uint8")
        self.assertEqual(list(self.target.users)[0].meta[QPARAM_KEY].dtype, "int16")

    def test_unsupported_reshape_dtype(self):
        self.setup(
            ReshapeTorchAPI(),
            torch.ops.aten.reshape.default,
            input_dtype="uint8",
            desired_dtype="float32",
        )
        self.target.meta[QPARAM_KEY].dtype = "float32"
        target_pass = InsertQuantizeOnDtypeMismatch()
        with self.assertRaises(NotYetSupportedError):
            target_pass.call(self.ep)


class ReluDtypeMismatchTest(InsertQuantizeOnDtypeMismatchTest):
    def test_no_mismatch_relu(self):
        self.setup(
            SimpleRelu(),
            torch.ops.aten.relu.default,
            input_dtype="uint8",
            desired_dtype="uint8",
        )
        self.target.meta[QPARAM_KEY].dtype = "uint8"
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, self.input_dtype)
        target_pass = InsertQuantizeOnDtypeMismatch()
        target_pass.call(self.ep)
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "uint8")

    def test_u8_to_i16_relu(self):
        self.setup(
            SimpleRelu(),
            torch.ops.aten.relu.default,
            input_dtype="uint8",
        )
        # Setup makes node's qparam int16 if input is uint8
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "int16")
        self.assertEqual(self.target.args[0].meta[QPARAM_KEY].dtype, "uint8")

        target_pass = InsertQuantizeOnDtypeMismatch()
        target_pass.call(self.ep)

        # After pass, node's dtype should be u8 (converted from i16)
        # A quantize op (i16) should be inserted after
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "uint8")
        self.assertEqual(list(self.target.users)[0].meta[QPARAM_KEY].dtype, "int16")

    def test_unsupported_relu_dtype(self):
        self.setup(
            SimpleRelu(),
            torch.ops.aten.relu.default,
            input_dtype="uint8",
            desired_dtype="float32",
        )
        self.target.meta[QPARAM_KEY].dtype = "float32"
        target_pass = InsertQuantizeOnDtypeMismatch()
        with self.assertRaises(NotYetSupportedError):
            target_pass.call(self.ep)

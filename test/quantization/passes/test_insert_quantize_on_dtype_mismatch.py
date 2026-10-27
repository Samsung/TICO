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
from tico.ops import Concat
from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape
from tico.quantization.passes.insert_quantize_on_dtype_mismatch import (
    InsertQuantizeOnDtypeMismatch,
)
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.errors import NotYetSupportedError
from tico.utils.mx.dtypes import is_mx_dtype

from test.modules.op.add import SimpleAdd
from test.modules.op.bmm import SimpleBatchMatMul
from test.modules.op.cat import SimpleCatDefault
from test.modules.op.linear import SimpleLinear
from test.modules.op.mul import SimpleMulWithTensor
from test.modules.op.permute import SimplePermute
from test.modules.op.relu import SimpleRelu
from test.modules.op.reshape import ReshapeTorchAPI
from test.support.helper import num_of_ops


def _make_qparam(dtype: str, scale: float = 1.0, zp: int = 0) -> QuantParam:
    """Return a per-tensor affine qparam, or an MX qparam grouped on the last axis."""
    qparam = QuantParam()
    qparam.dtype = dtype
    if is_mx_dtype(dtype):
        qparam.quantized_dimension = -1
    else:
        qparam.scale = [scale]
        qparam.zero_point = [zp]
    return qparam


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
        node_dtype=None,
    ):
        """Build the graph and set mismatching qparams on inputs and the target node.

        ``node_dtype`` defaults to the historical uint8/int16 counterpart of
        ``input_dtype``. Pass it explicitly (or use ``input_dtype="mxint8"``)
        for MX cases; MX qparams carry a dtype and a quantized dimension only.
        """
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
            user_input.meta[QPARAM_KEY] = _make_qparam(
                self.input_dtype, scale=self.scale, zp=self.zp
            )

        if node_dtype is None:
            if self.input_dtype == "int16":
                node_dtype = "uint8"
            elif self.input_dtype == "uint8":
                node_dtype = "int16"
            else:
                raise RuntimeError("Unsupported dtype")
        self.target.meta[QPARAM_KEY] = _make_qparam(node_dtype)

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

    def assert_mx_quantize_after_target(self, *, elem_format: str, axis: int):
        """The target's only user must be a logical MX quantize carrying the MX qparam."""
        axis %= self.target.meta["val"].ndim
        users = list(self.target.users)
        self.assertEqual(len(users), 1)
        quantize = users[0]
        self.assertEqual(quantize.target, torch.ops.circle_custom.quantize_mx.default)
        self.assertIs(quantize.args[0], self.target)
        self.assertEqual(quantize.args[1], elem_format)
        self.assertEqual(quantize.args[2], axis)
        qparam = quantize.meta[QPARAM_KEY]
        self.assertEqual(qparam.dtype, "mxint8")
        self.assertEqual(qparam.quantized_dimension, axis)
        self.assertIsNone(qparam.scale)
        # The affine qparam of the input must have moved onto the view op unchanged.
        inp = self.target.args[0]
        self.assertEqual(self.target.meta[QPARAM_KEY].scale, inp.meta[QPARAM_KEY].scale)
        self.assertEqual(
            self.target.meta[QPARAM_KEY].zero_point, inp.meta[QPARAM_KEY].zero_point
        )

    def assert_affine_quantize_before_target(self, *, dtype: str):
        """The target's input must be an affine quantize fed by the MX placeholder."""
        quantize = self.target.args[0]
        self.assertEqual(
            quantize.target,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        )
        self.assertEqual(quantize.meta[QPARAM_KEY].dtype, dtype)
        self.assertEqual(quantize.args[5], getattr(torch, dtype))
        source = quantize.args[0]
        self.assertEqual(source.op, "placeholder")
        self.assertEqual(source.meta[QPARAM_KEY].dtype, "mxint8")


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

    # Input (s16) -> Permute (s16) -> QuantizeMX (mxint8)
    def test_i16omx(self):
        self.setup(
            SimplePermute(),
            torch.ops.aten.permute.default,
            input_dtype="int16",
            desired_dtype="int16",
            node_dtype="mxint8",
        )
        self.run_test()
        self.assert_mx_quantize_after_target(elem_format="int8", axis=-1)

    # Input (mxint8) -> Quantize (s16) -> Permute (s16)
    def test_imxo16(self):
        self.setup(
            SimplePermute(),
            torch.ops.aten.permute.default,
            input_dtype="mxint8",
            desired_dtype="int16",
            node_dtype="int16",
        )
        self.run_test()
        self.assert_affine_quantize_before_target(dtype="int16")


class LinearTest(InsertQuantizeOnDtypeMismatchTest):
    def test_i8o16(self):
        self.setup(
            SimpleLinear(),
            torch.ops.aten.linear.default,
            input_dtype="uint8",
            desired_dtype="uint8",
        )
        self.run_test()

    # MX handling exists only for value-preserving view ops; a Linear whose
    # input and output disagree on MX-ness must still be rejected explicitly.
    def test_imx_o16_is_rejected(self):
        self.setup(
            SimpleLinear(),
            torch.ops.aten.linear.default,
            input_dtype="mxint8",
            desired_dtype="int16",
            node_dtype="int16",
        )
        with self.assertRaisesRegex(NotYetSupportedError, "From mxint8 to int16"):
            InsertQuantizeOnDtypeMismatch().call(self.ep)


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


class BMMMXTest(InsertQuantizeOnDtypeMismatchTest):
    def test_mx_inputs_preserve_output_domain(self):
        for output_dtype in ("int16", "mxint8"):
            with self.subTest(output_dtype=output_dtype):
                self.setup(
                    SimpleBatchMatMul(),
                    torch.ops.aten.bmm.default,
                    input_dtype="mxint8",
                    desired_dtype=output_dtype,
                    node_dtype=output_dtype,
                )
                self.target.args[1].meta[QPARAM_KEY].quantized_dimension = -2
                InsertQuantizeOnDtypeMismatch().call(self.ep)
                self.assertEqual(self.target.meta[QPARAM_KEY].dtype, output_dtype)
                x, y = self.target.args[:2]
                self.assertEqual(x.meta[QPARAM_KEY].dtype, "mxint8")
                self.assertEqual(y.meta[QPARAM_KEY].dtype, "mxint8")
                self.assertEqual(
                    num_of_ops(
                        self.ep,
                        [
                            torch.ops.quantized_decomposed.quantize_per_tensor.default,
                            torch.ops.circle_custom.quantize_mx.default,
                        ],
                    ),
                    0,
                )

    def test_mixed_mx_affine_operands_are_rejected(self):
        # Validate both operands even when the left input matches the output.
        for left_dtype, right_dtype in (("mxint8", "int16"), ("int16", "mxint8")):
            for output_dtype in ("int16", "mxint8"):
                with self.subTest(
                    left_dtype=left_dtype,
                    right_dtype=right_dtype,
                    output_dtype=output_dtype,
                ):
                    self.setup(
                        SimpleBatchMatMul(),
                        torch.ops.aten.bmm.default,
                        input_dtype=left_dtype,
                        desired_dtype=output_dtype,
                        node_dtype=output_dtype,
                    )
                    _, y = self.target.args[:2]
                    y.meta[QPARAM_KEY] = _make_qparam(right_dtype)
                    with self.assertRaisesRegex(
                        NotYetSupportedError,
                        f"{left_dtype} x {right_dtype} -> {output_dtype}",
                    ):
                        InsertQuantizeOnDtypeMismatch().call(self.ep)

    def test_same_format_mxfp4_bmm_is_preserved(self):
        self.setup(
            SimpleBatchMatMul(),
            torch.ops.aten.bmm.default,
            input_dtype="mxfp4",
            node_dtype="mxfp4",
        )
        self.target.args[1].meta[QPARAM_KEY].quantized_dimension = -2
        InsertQuantizeOnDtypeMismatch().call(self.ep)
        self.assertEqual(self.target.meta[QPARAM_KEY].dtype, "mxfp4")

    def test_unsupported_mx_formats_and_outputs_are_rejected(self):
        for left, right, output in (
            ("mxint8", "mxfp4", "mxint8"),
            ("mxfp4", "mxint8", "mxfp4"),
            ("mxfp4", "mxfp4", "int16"),
            ("mxint8", "mxint8", "uint8"),
            ("mxfp6_e3m2", "mxfp6_e3m2", "mxfp6_e3m2"),
        ):
            with self.subTest(left=left, right=right, output=output):
                self.setup(
                    SimpleBatchMatMul(),
                    torch.ops.aten.bmm.default,
                    input_dtype=left,
                    node_dtype=output,
                )
                self.target.args[1].meta[QPARAM_KEY] = _make_qparam(right)
                with self.assertRaisesRegex(
                    NotYetSupportedError, "MX matmul combination"
                ):
                    InsertQuantizeOnDtypeMismatch().call(self.ep)

    def test_mx_bmm_requires_reduction_axis_blocks(self):
        for left_axis, right_axis in ((-2, -2), (-1, -1)):
            with self.subTest(left_axis=left_axis, right_axis=right_axis):
                self.setup(
                    SimpleBatchMatMul(),
                    torch.ops.aten.bmm.default,
                    input_dtype="mxint8",
                    node_dtype="int16",
                )
                self.target.args[0].meta[QPARAM_KEY].quantized_dimension = left_axis
                self.target.args[1].meta[QPARAM_KEY].quantized_dimension = right_axis
                with self.assertRaisesRegex(
                    NotYetSupportedError, "reduction dimension"
                ):
                    InsertQuantizeOnDtypeMismatch().call(self.ep)


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

    # Input (s16) -> Reshape (s16) -> QuantizeMX (mxint8)
    def test_i16omx(self):
        self.setup(
            ReshapeTorchAPI(),
            torch.ops.aten.reshape.default,
            input_dtype="int16",
            desired_dtype="int16",
            node_dtype="mxint8",
        )
        self.run_test()
        self.assert_mx_quantize_after_target(elem_format="int8", axis=-1)

    # Input (mxint8) -> Quantize (s16) -> Reshape (s16)
    def test_imxo16(self):
        self.setup(
            ReshapeTorchAPI(),
            torch.ops.aten.reshape.default,
            input_dtype="mxint8",
            desired_dtype="int16",
            node_dtype="int16",
        )
        self.run_test()
        self.assert_affine_quantize_before_target(dtype="int16")


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


class _CatWithSharedSources(torch.nn.Module):
    def __init__(self, *, width=False, repeat=False):
        super().__init__()
        self.width = width
        self.repeat = repeat
        self.concat = Concat(dim=1, allow_distinct_input_qparams=True)

    def forward(self, left, right):
        inputs = (left, left, right) if self.repeat else (left, right)
        combined = self.concat(inputs) if self.width else torch.cat(inputs, dim=1)
        return combined, left, right


class ConcatAffineDomainTest(unittest.TestCase):
    @staticmethod
    def _export(dtype="int16", *, width=False, repeat=False):
        program = torch.export.export(
            _CatWithSharedSources(width=width, repeat=repeat),
            (torch.ones(1, 2, 4), torch.ones(1, 3, 4)),
        )
        concat = next(
            node
            for node in program.graph.nodes
            if node.target == torch.ops.aten.cat.default
        )
        left, right = tuple(
            node for node in program.graph.nodes if node.op == "placeholder"
        )
        target = _make_qparam(dtype, scale=0.5, zp=29)
        left.meta[QPARAM_KEY] = _make_qparam(dtype, scale=0.125, zp=3)
        right.meta[QPARAM_KEY] = copy.deepcopy(target)
        concat.meta[QPARAM_KEY] = target
        return program, concat, left, right

    def test_scale_and_zero_point_mismatches_use_real_consumer_edge_quantize(self):
        for dtype in ("uint8", "int16"):
            for scale_only in (False, True):
                with self.subTest(dtype=dtype, scale_only=scale_only):
                    program, concat, left, right = self._export(dtype)
                    target = concat.meta[QPARAM_KEY]
                    original = left.meta[QPARAM_KEY]
                    if scale_only:
                        original.zero_point = copy.deepcopy(target.zero_point)
                    else:
                        original.scale = copy.deepcopy(target.scale)
                    saved = copy.deepcopy(original)

                    InsertQuantizeOnDtypeMismatch().call(program)

                    quantize, unchanged = concat.args[0]
                    self.assertIs(unchanged, right)
                    self.assertEqual(
                        quantize.target,
                        torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    )
                    self.assertIs(quantize.args[0], left)
                    self.assertEqual(quantize.meta[QPARAM_KEY], target)
                    self.assertIsNot(quantize.meta[QPARAM_KEY], target)
                    self.assertEqual(quantize.args[1:3], (0.5, 29))
                    self.assertEqual(quantize.args[5], getattr(torch, dtype))
                    self.assertIs(left.meta[QPARAM_KEY], original)
                    self.assertEqual(left.meta[QPARAM_KEY], saved)
                    output = next(
                        node for node in program.graph.nodes if node.op == "output"
                    )
                    self.assertEqual(output.args[0][1:], (left, right))
                    values = torch.tensor([-0.25, 0, 0.75])
                    codes = quantize.target(values, *quantize.args[1:])
                    lower, upper = (0, 255) if dtype == "uint8" else (-32768, 32767)
                    expected = (torch.round(values / 0.5) + 29).clamp(lower, upper)
                    torch.testing.assert_close(
                        codes, expected.to(getattr(torch, dtype))
                    )

    def test_same_domain_ignores_calibration_range_metadata_and_is_idempotent(self):
        program, concat, left, right = self._export()
        left.meta[QPARAM_KEY] = copy.deepcopy(concat.meta[QPARAM_KEY])
        left.meta[QPARAM_KEY].min = [-1]
        right.meta[QPARAM_KEY].max = [2]
        original_inputs = concat.args[0]
        for _ in range(2):
            InsertQuantizeOnDtypeMismatch().call(program)
            self.assertEqual(concat.args[0], original_inputs)
            self.assertFalse(
                any(
                    node.target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                    for node in program.graph.nodes
                )
            )

    def test_repeated_shared_input_uses_one_quantize_without_changing_other_users(self):
        program, concat, left, right = self._export(repeat=True)
        for _ in range(2):
            InsertQuantizeOnDtypeMismatch().call(program)
            first, repeated, unchanged = concat.args[0]
            self.assertIs(first, repeated)
            self.assertIs(first.args[0], left)
            self.assertIs(unchanged, right)
            self.assertEqual(left.meta[QPARAM_KEY].scale, [0.125])
            self.assertEqual(
                sum(
                    node.target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                    for node in program.graph.nodes
                ),
                1,
            )

    def test_output_dtype_bridge_also_aligns_the_new_concat_input_domain(self):
        program, concat, left, right = self._export()
        concat.meta[QPARAM_KEY] = _make_qparam("uint8", scale=0.25, zp=128)
        original = copy.deepcopy(left.meta[QPARAM_KEY])

        InsertQuantizeOnDtypeMismatch().call(program)

        self.assertEqual(concat.meta[QPARAM_KEY].dtype, "int16")
        for edge in concat.args[0]:
            self.assertEqual(
                edge.target, torch.ops.quantized_decomposed.quantize_per_tensor.default
            )
            self.assertEqual(edge.meta[QPARAM_KEY], concat.meta[QPARAM_KEY])
            self.assertEqual(edge.args[5], torch.int16)
        output = next(node for node in program.graph.nodes if node.op == "output")
        final_quantize = output.args[0][0]
        self.assertIs(final_quantize.args[0], concat)
        self.assertEqual(final_quantize.meta[QPARAM_KEY].dtype, "uint8")
        self.assertEqual(final_quantize.args[1:3], (0.25, 128))
        self.assertEqual(left.meta[QPARAM_KEY], original)

    def test_width_direction_exception_keeps_distinct_input_grids(self):
        program, concat, left, right = self._export(width=True)
        original = concat.args[0]

        InsertQuantizeOnDtypeMismatch().call(program)

        self.assertEqual(concat.args[0], original)
        self.assertEqual(left.meta[QPARAM_KEY].scale, [0.125])
        self.assertEqual(right.meta[QPARAM_KEY].scale, [0.5])

    def test_non_affine_or_missing_domains_do_not_gain_affine_bridges(self):
        for kind in ("per_channel", "mixed_mx", "missing"):
            with self.subTest(kind=kind):
                program, concat, left, right = self._export()
                if kind == "per_channel":
                    left.meta[QPARAM_KEY].quantized_dimension = 1
                elif kind == "mixed_mx":
                    right.meta[QPARAM_KEY] = _make_qparam("mxint8")
                else:
                    del right.meta[QPARAM_KEY]
                original = concat.args[0]

                InsertQuantizeOnDtypeMismatch().call(program)

                self.assertEqual(concat.args[0], original)
                self.assertFalse(
                    any(
                        node.target
                        == torch.ops.quantized_decomposed.quantize_per_tensor.default
                        for node in program.graph.nodes
                    )
                )


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


class _Int16ViewToMX(torch.nn.Module):
    """int16 fake-quant -> permute -> reshape -> MX fake-quant (linear act_in pattern)."""

    def forward(self, x):
        x = torch.fake_quantize_per_tensor_affine(x, 0.01, 0, -32768, 32767)
        x = torch.permute(x, (0, 2, 1, 3))
        x = torch.reshape(x, (2, 4, 64))
        return torch.ops.circle_custom.mx_fake_quantize(x, "int8", -1)

    def get_example_inputs(self):
        torch.manual_seed(0)
        return (torch.randn(2, 2, 4, 32),), {}


class _MXViewToInt16(torch.nn.Module):
    """MX fake-quant -> reshape -> permute -> int16 fake-quant (linear act_out pattern)."""

    def forward(self, x):
        x = torch.ops.circle_custom.mx_fake_quantize(x, "int8", -1)
        x = torch.reshape(x, (2, 4, 2, 32))
        x = torch.permute(x, (0, 2, 1, 3))
        return torch.fake_quantize_per_tensor_affine(x, 0.01, 0, -32768, 32767)

    def get_example_inputs(self):
        torch.manual_seed(0)
        return (torch.randn(2, 4, 64),), {}


class _MXCircleExportTestCase(unittest.TestCase):
    """End-to-end: an int16/MX boundary across view ops becomes a Circle QUANTIZE."""

    @staticmethod
    def _ops(model):
        from circle_schema import circle

        graph = model.subgraphs[0]
        result = []
        for operator in graph.operators:
            code = model.operatorCodes[operator.opcodeIndex]
            builtin = (
                code.builtinCode
                if code.builtinCode != 0 or code.deprecatedBuiltinCode == 0
                else code.deprecatedBuiltinCode
            )
            ins = [graph.tensors[i].type for i in operator.inputs if i >= 0]
            outs = [graph.tensors[i].type for i in operator.outputs]
            result.append((builtin, ins, outs))
        return result, graph

    @staticmethod
    def _builtin(model, operator):
        code = model.operatorCodes[operator.opcodeIndex]
        return (
            code.builtinCode
            if code.builtinCode != 0 or code.deprecatedBuiltinCode == 0
            else code.deprecatedBuiltinCode
        )

    def _export(self, module):
        import tico
        from tico.circle.io import model_from_bytes

        args, kwargs = module.get_example_inputs()
        with torch.no_grad():
            module.eval()
            ep = torch.export.export(module, args, kwargs)
            torch.testing.assert_close(
                ep.module()(*args, **kwargs), module(*args, **kwargs), rtol=0, atol=0
            )
            circle_model = tico.convert(module, args, kwargs)
        return model_from_bytes(circle_model.circle_binary)


class MXViewCircleExportTest(_MXCircleExportTestCase):
    def test_int16_views_then_mx(self):
        from circle_schema import circle

        T = circle.TensorType.TensorType
        B = circle.BuiltinOperator.BuiltinOperator
        model = self._export(_Int16ViewToMX())
        ops, graph = self._ops(model)
        kinds = [op[0] for op in ops]
        # Views stay INT16 and a single INT16 -> MXINT8 QUANTIZE closes the chain.
        self.assertEqual(kinds.count(B.QUANTIZE), 1)
        for builtin, ins, outs in ops:
            if builtin == B.QUANTIZE:
                self.assertEqual(ins, [T.INT16])
                self.assertEqual(outs, [T.MXINT8])
            else:
                self.assertEqual(outs, [T.INT16], builtin)
        self.assertEqual(ops[-1][0], B.QUANTIZE)
        storage_input = graph.tensors[graph.inputs[0]]
        self.assertEqual(storage_input.type, T.INT16)
        self.assertAlmostEqual(float(storage_input.quantization.scale[0]), 0.01)
        self.assertEqual(list(storage_input.quantization.zeroPoint), [0])
        output = graph.tensors[graph.outputs[0]]
        self.assertEqual(output.type, T.MXINT8)
        # The MX block axis (-1 in the fake-quant op) is stored as the last
        # axis of the rank-3 output tensor.
        self.assertEqual(output.quantization.quantizedDimension, 2)
        self.assertIsNone(output.quantization.scale)
        self.assertIsNone(output.quantization.zeroPoint)

    def test_mx_then_int16_views(self):
        from circle_schema import circle

        T = circle.TensorType.TensorType
        B = circle.BuiltinOperator.BuiltinOperator
        model = self._export(_MXViewToInt16())
        ops, graph = self._ops(model)
        kinds = [op[0] for op in ops]
        # FoldQuantOps remaps the MX block axis through the views (reshape:
        # last axis 64 -> 32 keeps whole blocks; transpose moves axis 3 to 3),
        # so the views run in the MX domain and a single MXINT8 -> INT16
        # QUANTIZE closes the chain at the int16 fake-quant.
        self.assertEqual(kinds.count(B.QUANTIZE), 1)
        self.assertEqual(kinds[-1], B.QUANTIZE)
        self.assertEqual(ops[-1][1], [T.MXINT8])
        self.assertEqual(ops[-1][2], [T.INT16])
        for builtin, ins, outs in ops[:-1]:
            self.assertEqual(ins[:1], [T.MXINT8], builtin)
            self.assertEqual(outs, [T.MXINT8], builtin)
        quantize_input = graph.tensors[graph.operators[-1].inputs[0]]
        self.assertEqual(list(quantize_input.shape), [2, 2, 4, 32])
        self.assertEqual(quantize_input.quantization.quantizedDimension, 3)
        self.assertEqual(graph.tensors[graph.inputs[0]].type, T.MXINT8)
        self.assertEqual(
            graph.tensors[graph.inputs[0]].quantization.quantizedDimension, 2
        )
        affine_output = graph.tensors[graph.outputs[0]]
        self.assertEqual(affine_output.type, T.INT16)
        self.assertAlmostEqual(float(affine_output.quantization.scale[0]), 0.01)
        self.assertEqual(list(affine_output.quantization.zeroPoint), [0])


class _MXAttentionMatmul(torch.nn.Module):
    """softmax(INT16) -> MX(axis -1); V(INT16) -> MX(axis -2); bmm -> INT16.

    The attention-probability blocks run along the key axis and the V blocks
    along its key (reduction) axis, so both matmul operands share the
    reduction dimension of their MX blocks. The KV-cache side stays INT16.
    """

    def forward(self, probs, value):
        probs = torch.fake_quantize_per_tensor_affine(
            probs, 1.0 / 32767, 0, -32768, 32767
        )
        probs = torch.ops.circle_custom.mx_fake_quantize(probs, "int8", -1)
        value = torch.fake_quantize_per_tensor_affine(value, 0.01, 0, -32768, 32767)
        value = torch.ops.circle_custom.mx_fake_quantize(value, "int8", -2)
        out = torch.bmm(probs, value)
        return torch.fake_quantize_per_tensor_affine(out, 0.02, 0, -32768, 32767)

    def get_example_inputs(self):
        torch.manual_seed(0)
        probs = torch.softmax(torch.randn(2, 8, 64), dim=-1)
        return (probs, torch.randn(2, 64, 32)), {}


class MXAttentionMatmulCircleExportTest(_MXCircleExportTestCase):
    def test_mx_probabilities_times_mx_value_to_int16(self):
        from circle_schema import circle

        T = circle.TensorType.TensorType
        B = circle.BuiltinOperator.BuiltinOperator
        model = self._export(_MXAttentionMatmul())
        ops, graph = self._ops(model)
        kinds = [op[0] for op in ops]
        self.assertEqual(kinds.count(B.QUANTIZE), 2)
        self.assertEqual(kinds.count(B.BATCH_MATMUL), 1)
        for builtin, ins, outs in ops:
            if builtin == B.QUANTIZE:
                self.assertEqual(ins, [T.INT16])
                self.assertEqual(outs, [T.MXINT8])
            elif builtin == B.BATCH_MATMUL:
                self.assertEqual(ins[:2], [T.MXINT8, T.MXINT8])
                self.assertEqual(outs, [T.INT16])
        bmm = next(
            op for op in graph.operators if self._builtin(model, op) == B.BATCH_MATMUL
        )
        probs_t, value_t = (graph.tensors[i] for i in bmm.inputs[:2])
        self.assertEqual(
            probs_t.quantization.quantizedDimension, 2
        )  # key axis of [2, 8, 64]
        self.assertEqual(
            value_t.quantization.quantizedDimension, 1
        )  # key axis of [2, 64, 32]
        for index in graph.inputs:
            self.assertEqual(graph.tensors[index].type, T.INT16)
        self.assertEqual(graph.tensors[graph.outputs[0]].type, T.INT16)


class MXViewDomainTest(unittest.TestCase):
    def _program(self, shape, output_shape=None):
        class View(torch.nn.Module):
            def forward(self, value):
                if output_shape is not None:
                    return value.reshape(output_shape)
                return value.permute(1, 0)

        ep = torch.export.export(View(), (torch.zeros(shape),))
        ConvertLayoutOpToReshape().call(ep)
        source = next(node for node in ep.graph.nodes if node.op == "placeholder")
        view = next(node for node in ep.graph.nodes if node.op == "call_function")
        source.meta[QPARAM_KEY] = QuantParam(dtype="mxint8", quantized_dimension=1)
        return ep, source, view

    def test_same_format_view_remaps_axis_without_requantize(self):
        ep, source, view = self._program((2, 64))
        view.meta[QPARAM_KEY] = QuantParam(dtype="mxint8", quantized_dimension=0)
        InsertQuantizeOnDtypeMismatch().call(ep)
        self.assertEqual(view.meta[QPARAM_KEY].quantized_dimension, 0)
        self.assertEqual(source.meta[QPARAM_KEY].quantized_dimension, 1)
        self.assertEqual(list(view.users)[0].op, "output")

    def test_format_or_axis_change_keeps_requantize_after_view(self):
        for dtype, axis, elem_format in (("mxfp4", 0, "fp4"), ("mxint8", 1, "int8")):
            with self.subTest(dtype=dtype, axis=axis):
                ep, source, view = self._program((2, 64))
                view.meta[QPARAM_KEY] = QuantParam(
                    dtype=dtype, quantized_dimension=axis
                )
                InsertQuantizeOnDtypeMismatch().call(ep)
                self.assertEqual(view.meta[QPARAM_KEY].dtype, "mxint8")
                self.assertEqual(view.meta[QPARAM_KEY].quantized_dimension, 0)
                quantize = next(iter(view.users))
                self.assertEqual(
                    quantize.target, torch.ops.circle_custom.quantize_mx.default
                )
                self.assertEqual(quantize.args, (view, elem_format, axis))
                self.assertEqual(quantize.meta[QPARAM_KEY].dtype, dtype)
                self.assertEqual(quantize.meta[QPARAM_KEY].quantized_dimension, axis)
                self.assertEqual(source.meta[QPARAM_KEY].quantized_dimension, 1)

    def test_same_dtype_reshape_cannot_regroup_blocks(self):
        ep, _, view = self._program((4, 8), (2, 16))
        view.meta[QPARAM_KEY] = QuantParam(dtype="mxint8", quantized_dimension=1)
        with self.assertRaisesRegex(NotYetSupportedError, "changes the groups"):
            InsertQuantizeOnDtypeMismatch().call(ep)

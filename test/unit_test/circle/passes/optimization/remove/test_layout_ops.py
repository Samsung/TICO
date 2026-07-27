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

import struct
import unittest
from unittest.mock import Mock

from tico.circle.document import CircleDocument
from tico.circle.passes.base import CirclePassContext
from tico.circle.passes.cleanup import DeadCodeEliminationPass
from tico.circle.passes.optimization.remove.layout_ops import (
    _check_perm,
    _get_const_data,
    _is_reshape_op,
    _is_transpose_op,
    RemoveRedundantLayoutOpsPass,
)

from test.unit_test.circle.fixture import (
    FakeBuffer,
    FakeModel,
    FakeOperator,
    FakeOperatorCode,
    FakeSignatureDef,
    FakeSubGraph,
    FakeTensor,
    FakeTensorMap,
)


def _make_inverse_transpose_document() -> CircleDocument:
    """Create a non-square graph containing two inverse transposes."""
    permutation = struct.pack("<ii", 1, 0)
    subgraph = FakeSubGraph(
        name="main",
        tensors=[
            FakeTensor("input", shape=[2, 3]),
            FakeTensor("first_perm", buffer=1, shape=[2]),
            FakeTensor("first_transpose", shape=[3, 2]),
            FakeTensor("second_perm", buffer=2, shape=[2]),
            FakeTensor("second_transpose", shape=[2, 3]),
        ],
        inputs=[0],
        outputs=[4],
        operators=[
            FakeOperator(opcodeIndex=0, inputs=[0, 1], outputs=[2]),
            FakeOperator(opcodeIndex=0, inputs=[2, 3], outputs=[4]),
        ],
    )
    model = FakeModel(
        subgraphs=[subgraph],
        buffers=[
            FakeBuffer(),
            FakeBuffer(data=permutation),
            FakeBuffer(data=permutation),
        ],
        operatorCodes=[FakeOperatorCode(builtinCode=54)],
        signatureDefs=[
            FakeSignatureDef(
                signatureKey="main",
                subgraphIndex=0,
                inputs=[FakeTensorMap("input", 0)],
                outputs=[FakeTensorMap("output", 4)],
            )
        ],
    )
    return CircleDocument(model)


class TestPermutationHelpers(unittest.TestCase):
    """Test permutation checking helper functions."""

    def test_check_perm_identity(self):
        """Test identity permutation."""
        # perm = [0, 1, 2]
        result = _check_perm([0, 1, 2], [0, 1, 2])
        self.assertTrue(result)

    def test_check_perm_inverse_2d(self):
        """Test 2D inverse permutation (swap)."""
        # perm1 = [1, 0], perm2 = [1, 0]
        result = _check_perm([1, 0], [1, 0])
        self.assertTrue(result)

    def test_check_perm_inverse_3d(self):
        """Test 3D inverse permutation."""
        # perm1 = [0, 2, 1], perm2 = [0, 2, 1]
        result = _check_perm([0, 2, 1], [0, 2, 1])
        self.assertTrue(result)

    def test_check_perm_non_inverse(self):
        """Test non-inverse permutations."""
        result = _check_perm([1, 0, 2], [0, 2, 1])
        self.assertFalse(result)

    def test_check_perm_length_mismatch(self):
        """Test permutations of different lengths."""
        result = _check_perm([1, 0], [0, 2, 1])
        self.assertFalse(result)


class TestOperatorDetection(unittest.TestCase):
    """Test operator type detection."""

    def test_is_reshape_valid(self):
        """Test valid Reshape operator detection."""
        operator = Mock()
        operator.opcodeIndex = 0

        opcode = Mock()
        opcode.builtinCode = 26  # RESHAPE

        result = _is_reshape_op(operator, [opcode])
        self.assertTrue(result)

    def test_is_reshape_invalid_type(self):
        """Test non-Reshape operator."""
        operator = Mock()
        operator.opcodeIndex = 0

        opcode = Mock()
        opcode.builtinCode = 1  # ADD

        result = _is_reshape_op(operator, [opcode])
        self.assertFalse(result)

    def test_is_reshape_invalid_index(self):
        """Test Reshape with invalid opcode index."""
        operator = Mock()
        operator.opcodeIndex = 5

        result = _is_reshape_op(operator, [])
        self.assertFalse(result)

    def test_is_transpose_valid(self):
        """Test valid Transpose operator detection."""
        operator = Mock()
        operator.opcodeIndex = 0

        opcode = Mock()
        opcode.builtinCode = 54  # TRANSPOSE

        result = _is_transpose_op(operator, [opcode])
        self.assertTrue(result)

    def test_is_transpose_invalid_type(self):
        """Test non-Transpose operator."""
        operator = Mock()
        operator.opcodeIndex = 0

        opcode = Mock()
        opcode.builtinCode = 26  # RESHAPE

        result = _is_transpose_op(operator, [opcode])
        self.assertFalse(result)


class TestConstDataExtraction(unittest.TestCase):
    """Test constant data extraction from graph."""

    def test_get_const_data_valid(self):
        """Test extracting valid int32 constant data."""
        graph = Mock()

        tensor = Mock()
        tensor.buffer = 1
        subgraph = Mock()
        subgraph.tensors = [None, tensor]
        graph.subgraph = subgraph

        data = struct.pack("<ii", 1, 0)
        buffer = Mock()
        buffer.data = data
        model = Mock()
        model.buffers = [None, buffer]
        graph.model = model

        result = _get_const_data(graph, 1)
        self.assertEqual(result, [1, 0])

    def test_get_const_data_invalid_tensor_index(self):
        """Test with invalid tensor index."""
        graph = Mock()
        graph.subgraph = Mock()
        graph.subgraph.tensors = []

        result = _get_const_data(graph, 5)
        self.assertIsNone(result)

    def test_get_const_data_no_buffer(self):
        """Test with missing buffer data."""
        graph = Mock()

        tensor = Mock()
        tensor.buffer = 0
        subgraph = Mock()
        subgraph.tensors = [tensor]
        graph.subgraph = subgraph

        model = Mock()
        model.buffers = []
        graph.model = model

        result = _get_const_data(graph, 0)
        self.assertIsNone(result)


class TestRemoveRedundantLayoutOpsPass(unittest.TestCase):
    """Test RemoveRedundantLayoutOpsPass functionality."""

    def test_pass_name(self):
        """Test pass name property."""
        pass_obj = RemoveRedundantLayoutOpsPass()
        self.assertEqual(pass_obj.name, "RemoveRedundantLayoutOpsPass")

    def test_run_with_mock_document(self):
        """Test pass run method with minimal mock."""
        document = Mock()
        document.subgraph_count = 1

        graph = Mock()
        graph.subgraph = Mock()
        # Make operators iterable for as_list()
        operators: list[object] = []
        graph.subgraph.operators = operators

        # Make operatorCodes iterable
        operator_codes: list[object] = []
        model = Mock()
        model.operatorCodes = operator_codes
        document.model = model

        document.graph = Mock(return_value=graph)

        context = Mock()
        context.logger = Mock()

        pass_obj = RemoveRedundantLayoutOpsPass()
        result = pass_obj.run(document, context)

        self.assertFalse(result.modified)
        self.assertEqual(result.changes, 0)

    def test_remove_redundant_reshape_simple(self):
        """Test removing redundant Reshape operation."""
        pass_obj = RemoveRedundantLayoutOpsPass()

        graph = Mock()

        # reshape1 is a Reshape operation
        reshape1 = Mock()
        reshape1.inputs = [0, 1]  # input=0, shape=1
        reshape1.opcodeIndex = 0

        # reshape2 is also a Reshape operation that takes reshape1's output as input
        reshape2 = Mock()
        reshape2.inputs = [2, 3]  # input=2 (reshape1's output), shape=3
        reshape2.opcodeIndex = 0

        graph.subgraph = Mock()
        graph.subgraph.operators = [reshape1, reshape2]
        graph.producer = Mock(return_value=0)  # reshape1 produces tensor 2

        opcode = Mock()
        opcode.builtinCode = 26  # RESHAPE

        graph.model = Mock()
        graph.model.operatorCodes = [opcode]

        result = pass_obj._remove_redundant_reshape(graph, reshape2)

        self.assertTrue(result)
        self.assertEqual(reshape2.inputs[0], 0)

    def test_remove_redundant_reshape_no_producer(self):
        """Test Reshape with no producer (graph input)."""
        pass_obj = RemoveRedundantLayoutOpsPass()

        graph = Mock()
        reshape = Mock()
        reshape.inputs = [0, 1]
        graph.producer = Mock(return_value=None)

        result = pass_obj._remove_redundant_reshape(graph, reshape)

        self.assertFalse(result)

    def test_remove_redundant_transpose_inverse(self):
        """Test inverse Transpose cancellation and dead-code removal."""
        document = _make_inverse_transpose_document()
        subgraph = document.subgraph()
        second_transpose = subgraph.operators[1]
        context = CirclePassContext()

        result = RemoveRedundantLayoutOpsPass().run(document, context)

        self.assertTrue(result.modified)
        self.assertEqual(result.changes, 1)
        self.assertEqual(second_transpose.inputs, [2, 3])
        self.assertEqual(subgraph.outputs, [0])
        self.assertEqual(
            document.model.signatureDefs[0].outputs[0].tensorIndex,
            0,
        )
        self.assertEqual(subgraph.tensors[subgraph.outputs[0]].shape, [2, 3])
        self.assertTrue(document.verify(raise_on_error=False).ok)

        dce_result = DeadCodeEliminationPass().run(document, context)

        self.assertTrue(dce_result.modified)
        self.assertEqual(subgraph.operators, [])
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_inverse_transpose_preserves_shared_first_transpose(self):
        """Test that a shared first Transpose remains live after cancellation."""
        document = _make_inverse_transpose_document()
        subgraph = document.subgraph()
        first_transpose = subgraph.operators[0]

        fanout_operator = FakeOperator(opcodeIndex=1, inputs=[2], outputs=[5])
        subgraph.tensors.append(FakeTensor("fanout_output", shape=[3, 2]))
        subgraph.operators.append(fanout_operator)
        subgraph.outputs.append(5)
        document.model.operatorCodes.append(FakeOperatorCode(builtinCode=0))
        document.model.signatureDefs[0].outputs.append(
            FakeTensorMap("fanout_output", 5)
        )

        context = CirclePassContext()
        RemoveRedundantLayoutOpsPass().run(document, context)
        DeadCodeEliminationPass().run(document, context)

        self.assertEqual(len(subgraph.operators), 2)
        self.assertIs(subgraph.operators[0], first_transpose)
        self.assertIs(subgraph.operators[1], fanout_operator)
        self.assertEqual(subgraph.outputs, [0, 5])
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_remove_redundant_transpose_missing_perm(self):
        """Test Transpose with missing permutation data."""
        pass_obj = RemoveRedundantLayoutOpsPass()

        graph = Mock()
        graph.subgraph = Mock()
        graph.subgraph.tensors = [None]  # Minimal tensors list
        graph.subgraph.operators = [Mock()]

        graph.model = Mock()
        graph.model.buffers = [None]  # Minimal buffers list
        graph.model.operatorCodes = []

        transpose = Mock()
        transpose.inputs = [0, 1]
        transpose.outputs = [2]
        graph.producer = Mock(return_value=None)

        result = pass_obj._remove_redundant_transpose(graph, transpose, [])

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()

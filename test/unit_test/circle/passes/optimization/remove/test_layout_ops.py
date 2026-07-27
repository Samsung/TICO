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

from tico.circle.passes.optimization.remove.layout_ops import (
    _check_perm,
    _get_const_data,
    _is_reshape_op,
    _is_transpose_op,
    RemoveRedundantLayoutOpsPass,
)


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
        """Test inverse Transpose cancellation."""
        pass_obj = RemoveRedundantLayoutOpsPass()

        graph = Mock()

        # Create permutation data [1, 0]
        perm_data = struct.pack("<ii", 1, 0)

        # tensor 0: input to transpose1
        # tensor 1: permutation for transpose1
        # tensor 2: output of transpose1 (input to transpose2)
        # tensor 3: permutation for transpose2
        tensor0 = Mock()
        tensor1 = Mock()
        tensor1.buffer = 1
        tensor2 = Mock()
        tensor3 = Mock()
        tensor3.buffer = 2

        graph.subgraph = Mock()
        graph.subgraph.tensors = [tensor0, tensor1, tensor2, tensor3]

        buffer1 = Mock()
        buffer1.data = perm_data
        buffer2 = Mock()
        buffer2.data = perm_data

        graph.model = Mock()
        graph.model.buffers = [None, buffer1, buffer2]

        # transpose1: input=0, perm=1 -> output=2
        transpose1 = Mock()
        transpose1.inputs = [0, 1]
        transpose1.opcodeIndex = 0

        # transpose2: input=2, perm=3 -> should cancel with transpose1
        transpose2 = Mock()
        transpose2.inputs = [2, 3]
        transpose2.opcodeIndex = 0

        graph.subgraph.operators = [transpose1, transpose2]
        graph.producer = Mock(return_value=0)

        opcode = Mock()
        opcode.builtinCode = 54  # TRANSPOSE
        graph.model.operatorCodes = [opcode]

        result = pass_obj._remove_redundant_transpose(graph, transpose2, [opcode])

        self.assertTrue(result)
        self.assertEqual(transpose2.inputs[0], 0)

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
        graph.producer = Mock(return_value=None)

        result = pass_obj._remove_redundant_transpose(graph, transpose, [])

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()

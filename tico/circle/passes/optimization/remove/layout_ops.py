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

from __future__ import annotations

from typing import TYPE_CHECKING

from tico.circle.graph import as_indices, as_list
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult

if TYPE_CHECKING:
    from tico.circle.document import CircleDocument


def _get_builtin_code(operator: any) -> int:
    """Extract the builtin operator code from an operator."""
    builtin_options = getattr(operator, "builtinOptions", None)
    if builtin_options is None:
        return -1
    # The builtin operator type is stored in the BuiltinOperator enum
    # We need to extract it from the operator's BuiltinOptions
    builtin_ops = getattr(operator, "opcodeIndex", -1)
    return int(builtin_ops) if builtin_ops is not None else -1


def _is_reshape_op(operator: any, operator_codes: list[any]) -> bool:
    """Check if operator is a Reshape operation."""
    try:
        opcode_index = int(getattr(operator, "opcodeIndex", -1))
        if opcode_index < 0 or opcode_index >= len(operator_codes):
            return False
        opcode = operator_codes[opcode_index]
        builtin_code = int(getattr(opcode, "builtinCode", -1))
        # BuiltinOperator::RESHAPE = 26
        return builtin_code == 26
    except (TypeError, ValueError, AttributeError):
        return False


def _is_transpose_op(operator: any, operator_codes: list[any]) -> bool:
    """Check if operator is a Transpose operation."""
    try:
        opcode_index = int(getattr(operator, "opcodeIndex", -1))
        if opcode_index < 0 or opcode_index >= len(operator_codes):
            return False
        opcode = operator_codes[opcode_index]
        builtin_code = int(getattr(opcode, "builtinCode", -1))
        # BuiltinOperator::TRANSPOSE = 54
        return builtin_code == 54
    except (TypeError, ValueError, AttributeError):
        return False


def _check_perm(first_perm: list[int], second_perm: list[int]) -> bool:
    """Check if first_perm[second_perm[i]] == i for all i.

    This verifies if composing two permutations results in identity.
    """
    if len(first_perm) != len(second_perm):
        return False
    for i in range(len(second_perm)):
        if first_perm[second_perm[i]] != i:
            return False
    return True


def _get_const_data(graph: any, tensor_index: int) -> list[int] | None:
    """Extract constant data from a tensor, returns list of ints or None."""
    tensors = as_list(graph.subgraph.tensors)
    if tensor_index < 0 or tensor_index >= len(tensors):
        return None

    tensor = tensors[tensor_index]
    buffer_index = int(getattr(tensor, "buffer", 0) or 0)
    if buffer_index <= 0:
        return None

    buffers = as_list(graph.model.buffers)
    if buffer_index >= len(buffers):
        return None

    buffer = buffers[buffer_index]
    if buffer is None:
        return None

    data = getattr(buffer, "data", None)
    if data is None:
        return None

    try:
        # Convert buffer data to list of ints
        import struct

        result = []
        for i in range(0, len(data), 4):
            if i + 4 <= len(data):
                value = struct.unpack("<i", data[i : i + 4])[0]
                result.append(value)
        return result
    except (struct.error, TypeError):
        return None


class RemoveRedundantLayoutOpsPass(CirclePass):
    """Remove redundant Reshape and Transpose operations.

    This pass optimizes consecutive Reshape or Transpose operations that
    either cancel each other out or can be fused into a single operation.

    Handles two patterns:
    1. Reshape-Reshape: Multiple reshapes can be fused or simplified
    2. Transpose-Transpose: Consecutive transposes can be fused or eliminated
    """

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Remove redundant layout operations from all subgraphs."""

        changes = 0

        for subgraph_index in range(document.subgraph_count):
            graph = document.graph(subgraph_index)
            subgraph = graph.subgraph
            operators = as_list(subgraph.operators)
            operator_codes = as_list(document.model.operatorCodes)

            if not operators:
                continue

            # Process operators: reshape operations
            for operator in operators:
                if _is_reshape_op(operator, operator_codes):
                    if self._remove_redundant_reshape(graph, operator):
                        changes += 1

            # Process operators: transpose operations
            for operator in operators:
                if _is_transpose_op(operator, operator_codes):
                    if self._remove_redundant_transpose(
                        graph, operator, operator_codes
                    ):
                        changes += 1

        context.logger.debug(
            "RemoveRedundantLayoutOpsPass removed %d redundant layout operations.",
            changes,
        )
        return CirclePassResult(
            modified=changes > 0,
            changes=changes,
        )

    def _remove_redundant_reshape(self, graph: any, reshape_op: any) -> bool:
        """Remove redundant consecutive Reshape operations.

        Pattern:
            input → Reshape → Reshape → output

        Simplification:
            input → Reshape → output

        Args:
            graph: CircleGraph object with model structure
            reshape_op: Current Reshape operator

        Returns:
            True if optimization was applied, False otherwise
        """
        inputs = as_indices(getattr(reshape_op, "inputs", None))
        if len(inputs) < 1:
            return False

        input_tensor = inputs[0]
        producer = graph.producer(input_tensor)

        if producer is None:
            return False

        operators = as_list(graph.subgraph.operators)
        operator_codes = as_list(graph.model.operatorCodes)

        if producer >= len(operators):
            return False

        producer_op = operators[producer]

        # Check if producer is also a Reshape
        if not _is_reshape_op(producer_op, operator_codes):
            return False

        # Get producer's input
        producer_inputs = as_indices(getattr(producer_op, "inputs", None))
        if len(producer_inputs) < 1:
            return False

        # Connect current reshape to producer's input, skipping intermediate reshape
        reshape_op.inputs = [producer_inputs[0]] + list(inputs[1:])
        return True

    def _remove_redundant_transpose(
        self, graph: any, transpose_op: any, operator_codes: list[any]
    ) -> bool:
        """Remove or fuse consecutive Transpose operations.

        Patterns:
            1. Inverse transposes cancel out:
               Transpose(perm1) → Transpose(perm2) where perm1[perm2[i]] == i

            2. General composition:
               Transpose(perm1) → Transpose(perm2) → Transpose(composite)

        Args:
            graph: CircleGraph object with model structure
            transpose_op: Current Transpose operator
            operator_codes: List of operator codes from model

        Returns:
            True if optimization was applied, False otherwise
        """
        inputs = as_indices(getattr(transpose_op, "inputs", None))
        if len(inputs) < 2:
            return False

        input_tensor = inputs[0]
        perm_tensor = inputs[1]

        producer = graph.producer(input_tensor)

        if producer is None:
            return False

        operators = as_list(graph.subgraph.operators)

        if producer >= len(operators):
            return False

        producer_op = operators[producer]

        # Check if producer is also a Transpose
        if not _is_transpose_op(producer_op, operator_codes):
            return False

        producer_inputs = as_indices(getattr(producer_op, "inputs", None))
        if len(producer_inputs) < 2:
            return False

        producer_perm_tensor = producer_inputs[1]

        # Extract permutation data
        perm_data = _get_const_data(graph, perm_tensor)
        producer_perm_data = _get_const_data(graph, producer_perm_tensor)

        if perm_data is None or producer_perm_data is None:
            return False

        # Check if the composition is identity (inverse transpose)
        if _check_perm(perm_data, producer_perm_data):
            # The two transposes cancel out, connect to producer's input
            main_input = producer_inputs[0]
            transpose_op.inputs = [main_input] + list(inputs[1:])
            return True

        # TODO: Implement general composition optimization
        # This would create a new permutation constant for the composite transpose
        return False

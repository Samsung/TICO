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

from typing import Any, TYPE_CHECKING

from tico.circle._schema import circle_schema
from tico.circle.graph import as_indices, as_list
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult
from tico.circle.rewrite import replace_tensor_uses

if TYPE_CHECKING:
    from tico.circle.document import CircleDocument


def _builtin_operator_value(name: str) -> int:
    """Return a builtin operator value from the generated Circle schema."""

    schema = circle_schema()
    enum_module = getattr(schema, "BuiltinOperator", None)
    enum_type = (
        getattr(enum_module, "BuiltinOperator", None)
        if enum_module is not None
        else None
    )
    if enum_type is None or not hasattr(enum_type, name):
        raise RuntimeError(f"Circle schema does not provide BuiltinOperator.{name}.")
    return int(getattr(enum_type, name))


_RESHAPE_BUILTIN_CODE = _builtin_operator_value("RESHAPE")
_TRANSPOSE_BUILTIN_CODE = _builtin_operator_value("TRANSPOSE")


def _is_reshape_op(operator: Any, operator_codes: list[Any]) -> bool:
    """Return whether an operator references the Circle RESHAPE builtin."""

    try:
        opcode_index = int(getattr(operator, "opcodeIndex", -1))
        if opcode_index < 0 or opcode_index >= len(operator_codes):
            return False
        opcode = operator_codes[opcode_index]
        builtin_code = int(getattr(opcode, "builtinCode", -1))
        return builtin_code == _RESHAPE_BUILTIN_CODE
    except (TypeError, ValueError, AttributeError):
        return False


def _is_transpose_op(operator: Any, operator_codes: list[Any]) -> bool:
    """Return whether an operator references the Circle TRANSPOSE builtin."""

    try:
        opcode_index = int(getattr(operator, "opcodeIndex", -1))
        if opcode_index < 0 or opcode_index >= len(operator_codes):
            return False
        opcode = operator_codes[opcode_index]
        builtin_code = int(getattr(opcode, "builtinCode", -1))
        return builtin_code == _TRANSPOSE_BUILTIN_CODE
    except (TypeError, ValueError, AttributeError):
        return False


def _check_perm(first_perm: list[int], second_perm: list[int]) -> bool:
    """Return whether composing two permutations produces the identity."""

    if len(first_perm) != len(second_perm):
        return False
    for index in range(len(second_perm)):
        if first_perm[second_perm[index]] != index:
            return False
    return True


def _get_const_data(graph: Any, tensor_index: int) -> list[int] | None:
    """Decode an inline INT32 constant tensor as a list of Python integers."""

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
        import struct

        result = []
        for offset in range(0, len(data), 4):
            if offset + 4 <= len(data):
                value = struct.unpack("<i", data[offset : offset + 4])[0]
                result.append(value)
        return result
    except (struct.error, TypeError):
        return None


class RemoveRedundantLayoutOpsPass(CirclePass):
    """Remove redundant consecutive Reshape and Transpose operations.

    The pass handles two patterns:

    1. Consecutive Reshape operators, where the second Reshape can consume the
       first Reshape input directly.
    2. Consecutive inverse Transpose operators, where the pair can be bypassed.
    """

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Remove redundant layout operations from every subgraph."""

        changes = 0

        for subgraph_index in range(document.subgraph_count):
            graph = document.graph(subgraph_index)
            subgraph = graph.subgraph
            operators = as_list(subgraph.operators)
            operator_codes = as_list(document.model.operatorCodes)

            if not operators:
                continue

            for operator in operators:
                if _is_reshape_op(operator, operator_codes):
                    if self._remove_redundant_reshape(graph, operator):
                        changes += 1

            for operator in operators:
                if _is_transpose_op(operator, operator_codes):
                    if self._remove_redundant_transpose(
                        graph,
                        operator,
                        operator_codes,
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

    def _remove_redundant_reshape(self, graph: Any, reshape_op: Any) -> bool:
        """Bypass the first operator in a consecutive Reshape pair.

        Pattern::

            input -> Reshape -> Reshape -> output

        Simplification::

            input -----------> Reshape -> output
        """

        inputs = as_indices(getattr(reshape_op, "inputs", None))
        if not inputs:
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
        if not _is_reshape_op(producer_op, operator_codes):
            return False

        producer_inputs = as_indices(getattr(producer_op, "inputs", None))
        if not producer_inputs:
            return False

        reshape_op.inputs = [producer_inputs[0]] + list(inputs[1:])
        return True

    def _remove_redundant_transpose(
        self,
        graph: Any,
        transpose_op: Any,
        operator_codes: list[Any],
    ) -> bool:
        """Bypass two consecutive inverse Transpose operators."""

        inputs = as_indices(getattr(transpose_op, "inputs", None))
        if len(inputs) < 2:
            return False

        outputs = as_indices(getattr(transpose_op, "outputs", None))
        if len(outputs) != 1:
            return False

        input_tensor = inputs[0]
        permutation_tensor = inputs[1]
        producer = graph.producer(input_tensor)
        if producer is None:
            return False

        operators = as_list(graph.subgraph.operators)
        if producer >= len(operators):
            return False

        producer_op = operators[producer]
        if not _is_transpose_op(producer_op, operator_codes):
            return False

        producer_inputs = as_indices(getattr(producer_op, "inputs", None))
        if len(producer_inputs) < 2:
            return False

        producer_permutation_tensor = producer_inputs[1]
        permutation = _get_const_data(graph, permutation_tensor)
        producer_permutation = _get_const_data(
            graph,
            producer_permutation_tensor,
        )
        if permutation is None or producer_permutation is None:
            return False

        if _check_perm(permutation, producer_permutation):
            replacement = replace_tensor_uses(
                graph.model,
                subgraph_index=graph.subgraph_index,
                old_tensor_index=outputs[0],
                new_tensor_index=producer_inputs[0],
            )
            return replacement.modified

        return False

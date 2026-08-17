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

import struct
from typing import Any

from tico.circle._schema import circle_schema
from tico.circle.graph import as_list


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


_TRANSPOSE_BUILTIN_CODE = _builtin_operator_value("TRANSPOSE")


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
    """Return whether composing two valid permutations produces the identity."""

    if len(first_perm) != len(second_perm):
        return False
    expected = list(range(len(first_perm)))
    if sorted(first_perm) != expected or sorted(second_perm) != expected:
        return False
    return all(
        first_perm[second_perm[index]] == index for index in range(len(second_perm))
    )


def _get_const_data(graph: Any, tensor_index: int) -> list[int] | None:
    """Decode an inline INT32 constant tensor as Python integers."""

    tensors = as_list(graph.subgraph.tensors)
    if tensor_index < 0 or tensor_index >= len(tensors):
        return None

    tensor = tensors[tensor_index]
    buffer_index = int(getattr(tensor, "buffer", 0) or 0)
    buffers = as_list(graph.model.buffers)
    if buffer_index <= 0 or buffer_index >= len(buffers):
        return None

    buffer = buffers[buffer_index]
    data = getattr(buffer, "data", None)
    if data is None:
        return None
    if int(getattr(buffer, "offset", 0) or 0) or int(getattr(buffer, "size", 0) or 0):
        return None

    try:
        payload = memoryview(data).tobytes()
    except TypeError:
        try:
            payload = bytes(data)
        except (TypeError, ValueError):
            return None
    if not payload or len(payload) % 4 != 0:
        return None

    try:
        return [
            struct.unpack_from("<i", payload, offset)[0]
            for offset in range(0, len(payload), 4)
        ]
    except struct.error:
        return None

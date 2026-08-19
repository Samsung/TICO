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

from dataclasses import dataclass
from typing import Any

from tico.circle._object import freeze_object, FrozenValue
from tico.circle.analysis.tensor_contract import TensorContract
from tico.circle.errors import CircleRewriteError
from tico.circle.graph import as_indices, as_list, CircleGraph, OPTIONAL_TENSOR_INDEX


@dataclass(frozen=True)
class ExpressionKey:
    """Describe the semantic identity of one pure Circle operator expression."""

    builtin_code: int
    operator_version: int
    custom_code_fingerprint: FrozenValue
    inputs: tuple[int, ...]
    output_contracts: tuple[TensorContract, ...]
    intermediate_contracts: tuple[TensorContract | None, ...]
    mutating_variable_inputs: tuple[bool, ...]
    builtin_options_type: int
    builtin_options_fingerprint: FrozenValue
    builtin_options2_type: int
    builtin_options2_fingerprint: FrozenValue
    custom_options_format: int
    custom_options_fingerprint: FrozenValue
    large_custom_options_offset: int
    large_custom_options_size: int


def build_expression_key(
    model: Any,
    graph: CircleGraph,
    operator_index: int,
    *,
    builtin_code: int,
    operator_version: int,
    input_tensor_indices: tuple[int, ...] | None = None,
) -> ExpressionKey:
    """Build a stable expression key without using output tensor identities.

    Input tensor order remains significant, including for mathematically commutative
    operators. This preserves strict floating-point operand ordering and avoids adding
    algebraic assumptions to a structural optimization. Callers may supply canonical
    input tensor identities when earlier duplicate outputs are already known.
    """

    operators = as_list(getattr(graph.subgraph, "operators", None))
    if operator_index < 0 or operator_index >= len(operators):
        raise IndexError(
            f"Operator index {operator_index} is outside 0..{len(operators) - 1}."
        )
    operator = operators[operator_index]
    opcode_index = int(getattr(operator, "opcodeIndex", -1))
    operator_codes = as_list(getattr(model, "operatorCodes", None))
    if opcode_index < 0 or opcode_index >= len(operator_codes):
        raise CircleRewriteError(
            f"Operator {operator_index} references invalid opcode {opcode_index}."
        )
    operator_code = operator_codes[opcode_index]
    tensors = as_list(getattr(graph.subgraph, "tensors", None))

    def contract(tensor_index: int) -> TensorContract:
        """Capture one required tensor contract with a descriptive bounds check."""

        if tensor_index < 0 or tensor_index >= len(tensors):
            raise CircleRewriteError(
                f"Operator {operator_index} references invalid tensor {tensor_index}."
            )
        return TensorContract.from_tensor(tensors[tensor_index])

    raw_inputs = tuple(as_indices(getattr(operator, "inputs", None)))
    inputs = raw_inputs if input_tensor_indices is None else input_tensor_indices
    if len(inputs) != len(raw_inputs):
        raise CircleRewriteError(
            f"Operator {operator_index} has {len(raw_inputs)} inputs, but "
            f"{len(inputs)} canonical input identities were supplied."
        )

    outputs = tuple(as_indices(getattr(operator, "outputs", None)))
    intermediate_contracts: list[TensorContract | None] = []
    for tensor_index in as_indices(getattr(operator, "intermediates", None)):
        if tensor_index == OPTIONAL_TENSOR_INDEX:
            intermediate_contracts.append(None)
        else:
            intermediate_contracts.append(contract(tensor_index))

    return ExpressionKey(
        builtin_code=int(builtin_code),
        operator_version=max(1, int(operator_version)),
        custom_code_fingerprint=freeze_object(
            getattr(operator_code, "customCode", None)
        ),
        inputs=tuple(int(tensor_index) for tensor_index in inputs),
        output_contracts=tuple(contract(tensor_index) for tensor_index in outputs),
        intermediate_contracts=tuple(intermediate_contracts),
        mutating_variable_inputs=tuple(
            bool(value)
            for value in as_list(getattr(operator, "mutatingVariableInputs", None))
        ),
        builtin_options_type=int(getattr(operator, "builtinOptionsType", 0) or 0),
        builtin_options_fingerprint=freeze_object(
            getattr(operator, "builtinOptions", None)
        ),
        builtin_options2_type=int(getattr(operator, "builtinOptions2Type", 0) or 0),
        builtin_options2_fingerprint=freeze_object(
            getattr(operator, "builtinOptions2", None)
        ),
        custom_options_format=int(getattr(operator, "customOptionsFormat", 0) or 0),
        custom_options_fingerprint=freeze_object(
            getattr(operator, "customOptions", None)
        ),
        large_custom_options_offset=int(
            getattr(operator, "largeCustomOptionsOffset", 0) or 0
        ),
        large_custom_options_size=int(
            getattr(operator, "largeCustomOptionsSize", 0) or 0
        ),
    )

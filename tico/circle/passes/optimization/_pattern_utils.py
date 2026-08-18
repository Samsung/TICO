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
from typing import Any, Iterable

import numpy as np

from tico.circle._schema import decode_text
from tico.circle.analysis import TensorContract
from tico.circle.document import CircleDocument
from tico.circle.errors import CircleRewriteError
from tico.circle.graph import as_indices, as_list, CircleGraph
from tico.circle.passes.optimization._utils import (
    contract_is_dense_nonvariable,
    contract_is_fully_static,
    decode_constant_value,
    operator_builtin_code,
    operator_is_plain,
    tensor_contract,
)
from tico.circle.passes.rules import OperatorSnapshot, RewritePlan
from tico.circle.value import TensorValue, TensorValueCodec


@dataclass(frozen=True, kw_only=True)
class SupportingOperatorsPlan(RewritePlan):
    """Extend a rewrite plan with every non-anchor operator used by a match."""

    supporting_operators: tuple[OperatorSnapshot, ...] = ()

    def validate(self, document: CircleDocument) -> None:
        """Reject application when any supporting operator changed or disappeared."""

        super().validate(document)
        for expected in self.supporting_operators:
            try:
                current = OperatorSnapshot.capture(
                    document,
                    subgraph_index=self.subgraph_index,
                    operator_index=expected.operator_index,
                )
            except (CircleRewriteError, IndexError) as error:
                raise CircleRewriteError(
                    f"Rewrite plan is stale because supporting operator "
                    f"{expected.operator_index} no longer exists."
                ) from error
            if current != expected:
                raise CircleRewriteError(
                    f"Rewrite plan is stale because supporting operator "
                    f"{expected.operator_index} changed."
                )


def capture_supporting_operators(
    document: CircleDocument,
    *,
    subgraph_index: int,
    operator_indices: Iterable[int],
) -> tuple[OperatorSnapshot, ...]:
    """Capture unique supporting operators in deterministic index order."""

    return tuple(
        OperatorSnapshot.capture(
            document,
            subgraph_index=subgraph_index,
            operator_index=operator_index,
        )
        for operator_index in sorted({int(index) for index in operator_indices})
    )


def operator_is_live(graph: CircleGraph, operator_index: int) -> bool:
    """Return whether an operator contributes to a serialized graph output."""

    return int(operator_index) in graph.backward_operators(graph.outputs)


def operator_matches(
    document: CircleDocument,
    graph: CircleGraph,
    operator_index: int,
    *,
    builtin_code: int,
    input_count: int | None = None,
    output_count: int = 1,
    require_plain: bool = True,
) -> Any | None:
    """Return an operator when its serialized shape and builtin identity match."""

    operators = as_list(getattr(graph.subgraph, "operators", None))
    if operator_index < 0 or operator_index >= len(operators):
        return None
    operator = operators[operator_index]
    if operator_builtin_code(document.model, operator) != int(builtin_code):
        return None
    if require_plain and not operator_is_plain(operator):
        return None
    inputs = as_indices(getattr(operator, "inputs", None))
    outputs = as_indices(getattr(operator, "outputs", None))
    if input_count is not None and len(inputs) != input_count:
        return None
    if len(outputs) != output_count:
        return None
    return operator


def producer_matching(
    document: CircleDocument,
    graph: CircleGraph,
    tensor_index: int,
    *,
    builtin_code: int,
    input_count: int | None = None,
    output_count: int = 1,
    require_plain: bool = True,
) -> tuple[int, Any] | None:
    """Return a matching producer index and operator for one tensor."""

    producer_index = graph.producer(int(tensor_index))
    if producer_index is None:
        return None
    operator = operator_matches(
        document,
        graph,
        producer_index,
        builtin_code=builtin_code,
        input_count=input_count,
        output_count=output_count,
        require_plain=require_plain,
    )
    if operator is None:
        return None
    outputs = as_indices(getattr(operator, "outputs", None))
    if outputs != [int(tensor_index)]:
        return None
    return producer_index, operator


def has_no_fused_activation(operator: Any, activation_none: int) -> bool:
    """Return whether an operator options table declares no fused activation."""

    options = getattr(operator, "builtinOptions", None)
    return options is not None and int(
        getattr(options, "fusedActivationFunction", activation_none)
    ) == int(activation_none)


def supported_float_contract(
    contract: TensorContract,
    *,
    float32_type: int,
) -> bool:
    """Return whether a tensor is static dense unquantized FLOAT32."""

    return (
        contract.tensor_type == int(float32_type)
        and contract.quantization is None
        and contract_is_fully_static(contract)
        and contract_is_dense_nonvariable(contract)
    )


def decode_float32_constant(
    codec: TensorValueCodec,
    document: CircleDocument,
    graph: CircleGraph,
    tensor_index: int,
    *,
    float32_type: int,
    require_finite: bool = True,
) -> tuple[TensorValue, TensorContract] | None:
    """Decode one static dense unquantized FLOAT32 constant."""

    contract = tensor_contract(graph, tensor_index)
    if not supported_float_contract(contract, float32_type=float32_type):
        return None
    value = decode_constant_value(
        codec,
        document.model,
        subgraph_index=graph.subgraph_index,
        tensor_index=tensor_index,
    )
    if value is None or value.tensor_type != int(float32_type):
        return None
    if value.quantization is not None or value.data.dtype != np.dtype(np.float32):
        return None
    if require_finite and not np.all(np.isfinite(value.data)):
        return None
    return value, contract


def scalar_float32(
    codec: TensorValueCodec,
    document: CircleDocument,
    graph: CircleGraph,
    tensor_index: int,
    *,
    float32_type: int,
    require_finite: bool = True,
) -> float | None:
    """Decode one scalar-like FLOAT32 constant and return its Python value."""

    pair = decode_float32_constant(
        codec,
        document,
        graph,
        tensor_index,
        float32_type=float32_type,
        require_finite=require_finite,
    )
    if pair is None:
        return None
    value, _contract = pair
    if value.data.size != 1:
        return None
    return float(value.data.reshape(-1)[0])


def decode_integer_constant(
    codec: TensorValueCodec,
    document: CircleDocument,
    graph: CircleGraph,
    tensor_index: int,
) -> tuple[tuple[int, ...], TensorContract] | None:
    """Decode one static dense integer constant as a flattened tuple."""

    contract = tensor_contract(graph, tensor_index)
    if not (
        contract.quantization is None
        and contract_is_fully_static(contract)
        and contract_is_dense_nonvariable(contract)
    ):
        return None
    value = decode_constant_value(
        codec,
        document.model,
        subgraph_index=graph.subgraph_index,
        tensor_index=tensor_index,
    )
    if value is None or value.data.dtype.kind not in {"i", "u"}:
        return None
    return tuple(int(item) for item in value.data.reshape(-1)), contract


def normalize_axes(axes: Iterable[int], rank: int) -> tuple[int, ...] | None:
    """Normalize unique reduction axes while preserving their first occurrence."""

    normalized: list[int] = []
    seen: set[int] = set()
    for raw_axis in axes:
        axis = int(raw_axis)
        if axis < 0:
            axis += int(rank)
        if axis < 0 or axis >= int(rank):
            return None
        if axis not in seen:
            normalized.append(axis)
            seen.add(axis)
    return tuple(normalized)


def tensor_name(graph: CircleGraph, tensor_index: int, fallback: str) -> str:
    """Return a tensor name or a stable fallback for newly allocated values."""

    tensors = as_list(getattr(graph.subgraph, "tensors", None))
    if 0 <= tensor_index < len(tensors):
        name = decode_text(getattr(tensors[tensor_index], "name", ""))
        if name:
            return name
    return fallback


def tensor_has_single_consumer(
    graph: CircleGraph,
    tensor_index: int,
    consumer_index: int,
) -> bool:
    """Return whether a tensor is consumed only by the requested operator."""

    return graph.consumers(int(tensor_index)) == (int(consumer_index),)

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

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np

from tico.circle._object import ObjectFactory
from tico.circle._schema import circle_schema
from tico.circle.analysis import TensorContract
from tico.circle.builder import CircleBuilder
from tico.circle.document import CircleDocument
from tico.circle.graph import (
    as_indices,
    as_list,
    CircleGraph,
    is_constant_tensor,
    OPTIONAL_TENSOR_INDEX,
)
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult
from tico.circle.passes.optimization._utils import (
    AppendedObjectCheckpoint,
    operator_builtin_code,
    operator_is_plain,
    OptimizationSchemaResolver,
    tensor_contract,
)
from tico.circle.value import TensorValue, TensorValueCodec


@dataclass(frozen=True)
class DynamicFullyConnectedLegalizationPolicy:
    """Control the conservative FLOAT32 dynamic-weight legalization boundary."""

    require_rank_at_least_two: bool = True

    def __post_init__(self) -> None:
        """Normalize policy fields to plain bool values."""

        object.__setattr__(
            self,
            "require_rank_at_least_two",
            bool(self.require_rank_at_least_two),
        )


class LegalizeDynamicFullyConnectedPass(CirclePass):
    """Lower FLOAT32 FULLY_CONNECTED with dynamic weights to BATCH_MATMUL."""

    def __init__(
        self,
        *,
        policy: DynamicFullyConnectedLegalizationPolicy | None = None,
        builtin_codes: Mapping[str, int] | None = None,
        builtin_options_types: Mapping[str, int] | None = None,
        tensor_types: Mapping[str, int] | None = None,
        activation_values: Mapping[str, int] | None = None,
        activation_none: int | None = None,
        codec: TensorValueCodec | None = None,
        object_factory: ObjectFactory | None = None,
    ) -> None:
        """Create the legalization with injectable schema-independent enum values."""

        self.policy = policy or DynamicFullyConnectedLegalizationPolicy()
        self.resolver = OptimizationSchemaResolver(
            builtin_codes=builtin_codes,
            builtin_options_types=builtin_options_types,
            tensor_types=tensor_types,
            activation_none=activation_none,
            object_factory=object_factory,
        )
        self.codec = codec or TensorValueCodec()
        self.object_factory = object_factory
        self.codes = {
            name: self.resolver.builtin_code(name)
            for name in (
                "FULLY_CONNECTED",
                "BATCH_MATMUL",
                "RESHAPE",
                "ADD",
                "RELU",
                "RELU_N1_TO_1",
                "RELU6",
                "TANH",
            )
        }
        self.options_types = {
            name: self.resolver.builtin_options_type(name)
            for name in (
                "FullyConnectedOptions",
                "BatchMatMulOptions",
                "ReshapeOptions",
                "AddOptions",
            )
        }
        self.float32_type = self.resolver.tensor_type("FLOAT32")
        self.int32_type = self.resolver.tensor_type("INT32")
        configured_activations = {
            str(name).upper(): int(value)
            for name, value in (activation_values or {}).items()
        }
        self.activation_values = {
            "NONE": configured_activations.get(
                "NONE",
                self.resolver.activation_none,
            )
        }
        self.activation_values.update(
            {
                name: configured_activations.get(
                    name,
                    _maybe_schema_enum_value(
                        "ActivationFunctionType",
                        name,
                        fallback,
                    ),
                )
                for name, fallback in (
                    ("RELU", 1),
                    ("RELU_N1_TO_1", 2),
                    ("RELU6", 3),
                    ("TANH", 4),
                )
            }
        )

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Legalize all supported dynamic-weight FC operators in graph order."""

        del context
        changes = 0
        diagnostics: list[str] = []
        for subgraph_index, _subgraph in enumerate(as_list(document.model.subgraphs)):
            operator_index = 0
            while True:
                operators = as_list(document.subgraph(subgraph_index).operators)
                if operator_index >= len(operators):
                    break
                graph = CircleGraph(document.model, subgraph_index)
                operator = operators[operator_index]
                if (
                    operator_builtin_code(document.model, operator)
                    != self.codes["FULLY_CONNECTED"]
                ):
                    operator_index += 1
                    continue
                inserted = self._legalize(document, graph, operator_index)
                if inserted == 0:
                    operator_index += 1
                    continue
                changes += 1
                diagnostics.append(
                    "Legalized dynamic FULLY_CONNECTED at "
                    f"subgraphs[{subgraph_index}].operators[{operator_index}] "
                    f"to {inserted} operator(s)."
                )
                operator_index += inserted
        return CirclePassResult(
            modified=changes > 0,
            changes=changes,
            diagnostics=tuple(diagnostics),
        )

    def _legalize(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
    ) -> int:
        """Return the replacement operator count, or zero for an unsupported FC."""

        operator = as_list(graph.subgraph.operators)[operator_index]
        if not operator_is_plain(operator):
            return 0
        if (
            int(getattr(operator, "builtinOptionsType", 0) or 0)
            != self.options_types["FullyConnectedOptions"]
        ):
            return 0
        options = getattr(operator, "builtinOptions", None)
        if options is None:
            return 0
        if int(getattr(options, "weightsFormat", 0) or 0) != 0:
            return 0

        inputs = tuple(as_indices(operator.inputs))
        outputs = tuple(as_indices(operator.outputs))
        if len(inputs) != 3 or len(outputs) != 1:
            return 0
        input_index, weight_index, bias_index = inputs
        output_index = outputs[0]
        if weight_index == OPTIONAL_TENSOR_INDEX:
            return 0
        if is_constant_tensor(document.model, graph.subgraph, weight_index):
            return 0

        input_contract = tensor_contract(graph, input_index)
        weight_contract = tensor_contract(graph, weight_index)
        output_contract = tensor_contract(graph, output_index)
        if not all(
            _supported_float32(contract, self.float32_type)
            for contract in (input_contract, weight_contract, output_contract)
        ):
            return 0
        if weight_contract.rank != 2 or input_contract.rank == 0:
            return 0
        if self.policy.require_rank_at_least_two and input_contract.rank < 2:
            return 0

        units, input_size = weight_contract.shape
        if units <= 0 or input_size <= 0 or input_contract.shape[-1] != input_size:
            return 0
        keep_num_dims = bool(getattr(options, "keepNumDims", False))
        bmm_shape = input_contract.shape[:-1] + (units,)
        if keep_num_dims:
            expected_output_shape = bmm_shape
        else:
            expected_output_shape = (prod(input_contract.shape[:-1]), units)
        if output_contract.shape != expected_output_shape:
            return 0

        has_bias = bias_index != OPTIONAL_TENSOR_INDEX
        if has_bias:
            bias_contract = tensor_contract(graph, bias_index)
            if not _supported_float32(bias_contract, self.float32_type):
                return 0
            if bias_contract.shape != (units,):
                return 0
        activation = int(
            getattr(options, "fusedActivationFunction", self.activation_values["NONE"])
            or 0
        )
        activation_code = self._activation_builtin(activation)
        if activation != self.activation_values["NONE"] and activation_code is None:
            return 0

        need_reshape = bmm_shape != output_contract.shape
        checkpoint = AppendedObjectCheckpoint.capture(
            document,
            subgraph_index=graph.subgraph_index,
        )
        builder = CircleBuilder(
            document,
            subgraph_index=graph.subgraph_index,
            codec=self.codec,
            object_factory=self.object_factory,
        )
        try:
            replacement_operators = self._build_replacement(
                builder,
                graph,
                input_index=input_index,
                weight_index=weight_index,
                bias_index=bias_index,
                output_index=output_index,
                output_contract=output_contract,
                bmm_shape=bmm_shape,
                need_reshape=need_reshape,
                activation=activation,
                activation_code=activation_code,
                has_bias=has_bias,
            )
        except Exception:
            checkpoint.rollback(document)
            raise

        operators = as_list(graph.subgraph.operators)
        operators[operator_index : operator_index + 1] = replacement_operators
        graph.subgraph.operators = operators
        return len(replacement_operators)

    def _build_replacement(
        self,
        builder: CircleBuilder,
        graph: CircleGraph,
        *,
        input_index: int,
        weight_index: int,
        bias_index: int,
        output_index: int,
        output_contract: TensorContract,
        bmm_shape: tuple[int, ...],
        need_reshape: bool,
        activation: int,
        activation_code: int | None,
        has_bias: bool,
    ) -> list[Any]:
        """Allocate intermediate tensors and return a complete replacement sequence."""

        has_activation = activation != self.activation_values["NONE"]
        operations_after_bmm = need_reshape or has_bias or has_activation
        if operations_after_bmm:
            bmm_output = builder.add_tensor(
                _derived_name(graph, output_index, "batch_matmul"),
                TensorContract(tensor_type=self.float32_type, shape=bmm_shape),
            )
        else:
            bmm_output = output_index
        bmm_options = self.resolver.create("BatchMatMulOptions")
        bmm_options.adjointLhs = False
        bmm_options.adjointRhs = True
        if hasattr(bmm_options, "asymmetricQuantizeInputs"):
            bmm_options.asymmetricQuantizeInputs = False
        replacement = [
            builder.make_operator(
                self.codes["BATCH_MATMUL"],
                inputs=(input_index, weight_index),
                outputs=(bmm_output,),
                builtin_options_type=self.options_types["BatchMatMulOptions"],
                builtin_options=bmm_options,
            )
        ]
        current = bmm_output

        if need_reshape:
            operations_after_reshape = has_bias or has_activation
            if operations_after_reshape:
                reshape_output = builder.add_tensor(
                    _derived_name(graph, output_index, "reshape"),
                    TensorContract(
                        tensor_type=output_contract.tensor_type,
                        shape=output_contract.shape,
                    ),
                )
            else:
                reshape_output = output_index
            shape_value = TensorValue.from_values(
                self.int32_type,
                np.asarray(output_contract.shape, dtype=np.int32),
                dtype=np.int32,
            )
            shape_index = builder.add_constant(
                _derived_name(graph, output_index, "shape"),
                shape_value,
            )
            reshape_options = self.resolver.create("ReshapeOptions")
            reshape_options.newShape = list(output_contract.shape)
            replacement.append(
                builder.make_operator(
                    self.codes["RESHAPE"],
                    inputs=(current, shape_index),
                    outputs=(reshape_output,),
                    builtin_options_type=self.options_types["ReshapeOptions"],
                    builtin_options=reshape_options,
                )
            )
            current = reshape_output

        if has_bias:
            add_options = self.resolver.create("AddOptions")
            add_options.fusedActivationFunction = activation
            if hasattr(add_options, "potScaleInt16"):
                add_options.potScaleInt16 = False
            replacement.append(
                builder.make_operator(
                    self.codes["ADD"],
                    inputs=(current, bias_index),
                    outputs=(output_index,),
                    builtin_options_type=self.options_types["AddOptions"],
                    builtin_options=add_options,
                )
            )
        elif has_activation:
            assert activation_code is not None
            replacement.append(
                builder.make_operator(
                    activation_code,
                    inputs=(current,),
                    outputs=(output_index,),
                )
            )
        return replacement

    def _activation_builtin(self, activation: int) -> int | None:
        """Map a supported fused activation value to its standalone builtin code."""

        if activation == self.activation_values["NONE"]:
            return None
        for name in ("RELU", "RELU_N1_TO_1", "RELU6", "TANH"):
            if activation == self.activation_values[name]:
                return self.codes[name]
        return None


def _supported_float32(contract: TensorContract, float32_type: int) -> bool:
    """Return whether a tensor is static, dense, immutable, and non-quantized F32."""

    signature = contract.shape_signature
    return (
        contract.tensor_type == float32_type
        and (signature is None or all(dimension >= 0 for dimension in signature))
        and not contract.is_variable
        and contract.sparsity is None
        and contract.variant_tensors is None
        and contract.quantization is None
    )


def _derived_name(graph: CircleGraph, tensor_index: int, suffix: str) -> str:
    """Create a stable intermediate name from an existing output tensor."""

    tensors = as_list(graph.subgraph.tensors)
    raw_name = getattr(tensors[tensor_index], "name", None)
    if isinstance(raw_name, bytes):
        name = raw_name.decode("utf-8", errors="replace")
    else:
        name = str(raw_name or f"tensor_{tensor_index}")
    return f"{name}/{suffix}"


def _schema_enum_value(enum_name: str, member_name: str) -> int:
    """Return one generated Circle enum member by symbolic name."""

    schema = circle_schema()
    module = getattr(schema, enum_name, None)
    enum_type = getattr(module, enum_name, None) if module is not None else None
    if enum_type is None:
        enum_type = module
    if enum_type is None or not hasattr(enum_type, member_name):
        raise RuntimeError(f"Circle schema does not provide {enum_name}.{member_name}.")
    return int(getattr(enum_type, member_name))


def _maybe_schema_enum_value(
    enum_name: str,
    member_name: str,
    fallback: int,
) -> int:
    """Return a generated enum value or a stable legacy fallback."""

    try:
        return _schema_enum_value(enum_name, member_name)
    except (AttributeError, ImportError, RuntimeError):
        return int(fallback)


__all__ = [
    "DynamicFullyConnectedLegalizationPolicy",
    "LegalizeDynamicFullyConnectedPass",
]

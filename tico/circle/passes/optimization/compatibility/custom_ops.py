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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tico.circle._object import ObjectFactory
from tico.circle._schema import circle_schema
from tico.circle.analysis import TensorContract
from tico.circle.builder import CircleBuilder
from tico.circle.document import CircleDocument
from tico.circle.errors import CircleValueError
from tico.circle.graph import as_indices, as_list, CircleGraph, is_constant_tensor
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult
from tico.circle.passes.optimization._utils import (
    AppendedObjectCheckpoint,
    operator_builtin_code,
    OptimizationSchemaResolver,
    tensor_contract,
)
from tico.circle.value import TensorValue, TensorValueCodec

CustomOptionDecoder = Callable[[bytes], Mapping[str, Any]]


@dataclass(frozen=True)
class LegacyCustomOpPolicy:
    """Select legacy TensorFlow custom operators eligible for builtin recovery."""

    resolve_add_v2: bool = True
    resolve_batch_matmul_v2: bool = True
    resolve_matmul: bool = True
    resolve_split_v: bool = True
    resolve_unit_max_pool_with_argmax: bool = True

    def __post_init__(self) -> None:
        """Normalize every policy switch to a plain bool."""

        for field_name in (
            "resolve_add_v2",
            "resolve_batch_matmul_v2",
            "resolve_matmul",
            "resolve_split_v",
            "resolve_unit_max_pool_with_argmax",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))


class ResolveLegacyCustomOpsPass(CirclePass):
    """Recover selected former TensorFlow custom operators as Circle builtins."""

    def __init__(
        self,
        *,
        policy: LegacyCustomOpPolicy | None = None,
        builtin_codes: Mapping[str, int] | None = None,
        builtin_options_types: Mapping[str, int] | None = None,
        tensor_types: Mapping[str, int] | None = None,
        padding_values: Mapping[str, int] | None = None,
        activation_none: int | None = None,
        codec: TensorValueCodec | None = None,
        object_factory: ObjectFactory | None = None,
        custom_option_decoder: CustomOptionDecoder | None = None,
    ) -> None:
        """Create compatibility rewrites with injectable schema and option services."""

        self.policy = policy or LegacyCustomOpPolicy()
        self.resolver = OptimizationSchemaResolver(
            builtin_codes=builtin_codes,
            builtin_options_types=builtin_options_types,
            tensor_types=tensor_types,
            activation_none=activation_none,
            object_factory=object_factory,
        )
        self.codec = codec or TensorValueCodec()
        self.object_factory = object_factory
        self.custom_option_decoder = custom_option_decoder or _decode_flexbuffer_map
        self.padding_values = {
            str(name).upper(): int(value)
            for name, value in (padding_values or {}).items()
        }
        self.codes = {
            name: self.resolver.builtin_code(name)
            for name in (
                "CUSTOM",
                "ADD",
                "BATCH_MATMUL",
                "SPLIT_V",
                "MAX_POOL_2D",
            )
        }
        self.options_types = {
            name: self.resolver.builtin_options_type(name)
            for name in (
                "AddOptions",
                "BatchMatMulOptions",
                "SplitVOptions",
                "Pool2DOptions",
            )
        }

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Resolve every supported custom operator while preserving output tensors."""

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
                    != self.codes["CUSTOM"]
                ):
                    operator_index += 1
                    continue
                custom_name = _custom_code(document.model, operator)
                handler = self._handler(custom_name)
                if handler is None:
                    operator_index += 1
                    continue
                if handler(document, graph, operator_index):
                    changes += 1
                    diagnostics.append(
                        f"Resolved custom operator {custom_name} at "
                        f"subgraphs[{subgraph_index}].operators[{operator_index}]."
                    )
                operator_index += 1
        return CirclePassResult(
            modified=changes > 0,
            changes=changes,
            diagnostics=tuple(diagnostics),
        )

    def _handler(
        self,
        custom_name: str,
    ) -> Callable[[CircleDocument, CircleGraph, int], bool] | None:
        """Return the enabled resolver for one decoded custom-code string."""

        if custom_name == "AddV2" and self.policy.resolve_add_v2:
            return self._resolve_add_v2
        if custom_name == "BatchMatMulV2" and self.policy.resolve_batch_matmul_v2:
            return self._resolve_batch_matmul_v2
        if custom_name == "MatMul" and self.policy.resolve_matmul:
            return self._resolve_matmul
        if custom_name == "SplitV" and self.policy.resolve_split_v:
            return self._resolve_split_v
        if (
            custom_name == "MaxPoolWithArgmax"
            and self.policy.resolve_unit_max_pool_with_argmax
        ):
            return self._resolve_unit_max_pool_with_argmax
        return None

    def _resolve_add_v2(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
    ) -> bool:
        """Replace a two-input AddV2 custom operator with builtin ADD."""

        operator = as_list(graph.subgraph.operators)[operator_index]
        inputs = tuple(as_indices(operator.inputs))
        outputs = tuple(as_indices(operator.outputs))
        if len(inputs) != 2 or len(outputs) != 1:
            return False
        options = self.resolver.create("AddOptions")
        options.fusedActivationFunction = self.resolver.activation_none
        if hasattr(options, "potScaleInt16"):
            options.potScaleInt16 = False
        return self._replace_builtin(
            document,
            graph,
            operator_index,
            builtin_code=self.codes["ADD"],
            inputs=inputs,
            outputs=outputs,
            options_type=self.options_types["AddOptions"],
            options=options,
        )

    def _resolve_batch_matmul_v2(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
    ) -> bool:
        """Replace BatchMatMulV2 with builtin BATCH_MATMUL and decoded adjoints."""

        operator = as_list(graph.subgraph.operators)[operator_index]
        raw_options = self._custom_options(operator)
        adjoint_lhs = _first_bool(raw_options, ("adj_x", "adjoint_a"), False)
        adjoint_rhs = _first_bool(raw_options, ("adj_y", "adjoint_b"), False)
        return self._replace_batch_matmul(
            document,
            graph,
            operator_index,
            adjoint_lhs=adjoint_lhs,
            adjoint_rhs=adjoint_rhs,
        )

    def _resolve_matmul(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
    ) -> bool:
        """Replace MatMul with builtin BATCH_MATMUL and transpose attributes."""

        operator = as_list(graph.subgraph.operators)[operator_index]
        raw_options = self._custom_options(operator)
        transpose_lhs = _first_bool(raw_options, ("transpose_a",), False)
        transpose_rhs = _first_bool(raw_options, ("transpose_b",), False)
        return self._replace_batch_matmul(
            document,
            graph,
            operator_index,
            adjoint_lhs=transpose_lhs,
            adjoint_rhs=transpose_rhs,
        )

    def _replace_batch_matmul(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        *,
        adjoint_lhs: bool,
        adjoint_rhs: bool,
    ) -> bool:
        """Create one BATCH_MATMUL replacement while retaining serialized outputs."""

        operator = as_list(graph.subgraph.operators)[operator_index]
        inputs = tuple(as_indices(operator.inputs))
        outputs = tuple(as_indices(operator.outputs))
        if len(inputs) != 2 or len(outputs) != 1:
            return False
        if any(tensor_contract(graph, index).rank < 2 for index in (*inputs, *outputs)):
            return False
        options = self.resolver.create("BatchMatMulOptions")
        options.adjointLhs = bool(adjoint_lhs)
        options.adjointRhs = bool(adjoint_rhs)
        if hasattr(options, "asymmetricQuantizeInputs"):
            options.asymmetricQuantizeInputs = False
        return self._replace_builtin(
            document,
            graph,
            operator_index,
            builtin_code=self.codes["BATCH_MATMUL"],
            inputs=inputs,
            outputs=outputs,
            options_type=self.options_types["BatchMatMulOptions"],
            options=options,
        )

    def _resolve_split_v(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
    ) -> bool:
        """Replace custom SplitV and narrow S64 shape constants when representable."""

        operator = as_list(graph.subgraph.operators)[operator_index]
        inputs = tuple(as_indices(operator.inputs))
        outputs = tuple(as_indices(operator.outputs))
        if len(inputs) != 3 or not outputs:
            return False
        size_contract = tensor_contract(graph, inputs[1])
        axis_contract = tensor_contract(graph, inputs[2])
        if size_contract.rank != 1 or size_contract.shape != (len(outputs),):
            return False
        if axis_contract.element_count != 1:
            return False

        checkpoint = AppendedObjectCheckpoint.capture(
            document,
            subgraph_index=graph.subgraph_index,
        )
        try:
            size_splits = self._narrow_int64_constant(document, graph, inputs[1])
            split_dim = self._narrow_int64_constant(document, graph, inputs[2])
            if size_splits is None or split_dim is None:
                checkpoint.rollback(document)
                return False
            options = self.resolver.create("SplitVOptions")
            options.numSplits = len(outputs)
            builder = CircleBuilder(
                document,
                subgraph_index=graph.subgraph_index,
                codec=self.codec,
                object_factory=self.object_factory,
            )
            replacement = builder.make_operator(
                self.codes["SPLIT_V"],
                inputs=(inputs[0], size_splits, split_dim),
                outputs=outputs,
                builtin_options_type=self.options_types["SplitVOptions"],
                builtin_options=options,
            )
            builder.replace_operator(operator_index, replacement)
        except Exception:
            checkpoint.rollback(document)
            raise
        return True

    def _resolve_unit_max_pool_with_argmax(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
    ) -> bool:
        """Recover the exactly representable 1x1 MaxPoolWithArgmax special case."""

        operator = as_list(graph.subgraph.operators)[operator_index]
        inputs = tuple(as_indices(operator.inputs))
        outputs = tuple(as_indices(operator.outputs))
        if len(inputs) != 1 or len(outputs) != 2:
            return False
        raw_options = self._custom_options(operator)
        kernel = _int_vector(raw_options.get("ksize"))
        strides = _int_vector(raw_options.get("strides"))
        if kernel != (1, 1, 1, 1) or strides != (1, 1, 1, 1):
            return False
        padding_name = str(raw_options.get("padding", "VALID")).upper()
        if padding_name not in {"SAME", "VALID"}:
            return False

        input_contract = tensor_contract(graph, inputs[0])
        value_contract = tensor_contract(graph, outputs[0])
        index_contract = tensor_contract(graph, outputs[1])
        if input_contract.rank != 4 or value_contract != input_contract:
            return False
        if (
            not _dense_static(index_contract)
            or index_contract.shape != input_contract.shape
        ):
            return False
        if index_contract.quantization is not None:
            return False
        index_spec = self.codec.registry.get(index_contract.tensor_type)
        if index_spec is None or index_spec.name not in {"INT32", "INT64"}:
            return False

        include_batch = _first_bool(
            raw_options,
            ("include_batch_in_index", "includeBatchInIndex"),
            False,
        )
        batch, height, width, channels = input_contract.shape
        per_batch = height * width * channels
        if include_batch:
            indices = np.arange(batch * per_batch, dtype=index_spec.logical_dtype)
            indices = indices.reshape(index_contract.shape)
        else:
            one_batch = np.arange(per_batch, dtype=index_spec.logical_dtype)
            indices = np.broadcast_to(
                one_batch.reshape(1, height, width, channels),
                index_contract.shape,
            ).copy()
        index_value = TensorValue(
            tensor_type=index_contract.tensor_type,
            shape=index_contract.shape,
            data=indices,
            quantization=None,
        )

        options = self.resolver.create("Pool2DOptions")
        options.padding = self._padding_value(padding_name)
        options.strideH = 1
        options.strideW = 1
        options.filterHeight = 1
        options.filterWidth = 1
        options.fusedActivationFunction = self.resolver.activation_none

        checkpoint = AppendedObjectCheckpoint.capture(
            document,
            subgraph_index=graph.subgraph_index,
        )
        tensors = as_list(graph.subgraph.tensors)
        previous_buffer = int(getattr(tensors[outputs[1]], "buffer", 0) or 0)
        builder = CircleBuilder(
            document,
            subgraph_index=graph.subgraph_index,
            codec=self.codec,
            object_factory=self.object_factory,
        )
        try:
            buffer_index = builder.constant_pool.intern_buffer(
                self.codec.encode(index_value)
            )
            replacement = builder.make_operator(
                self.codes["MAX_POOL_2D"],
                inputs=inputs,
                outputs=(outputs[0],),
                builtin_options_type=self.options_types["Pool2DOptions"],
                builtin_options=options,
            )
            tensors[outputs[1]].buffer = buffer_index
            builder.replace_operator(
                operator_index,
                replacement,
                require_same_outputs=False,
            )
        except Exception:
            tensors[outputs[1]].buffer = previous_buffer
            checkpoint.rollback(document)
            raise
        return True

    def _replace_builtin(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        *,
        builtin_code: int,
        inputs: Sequence[int],
        outputs: Sequence[int],
        options_type: int,
        options: Any,
    ) -> bool:
        """Replace one custom operator transactionally with a builtin operator."""

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
            replacement = builder.make_operator(
                builtin_code,
                inputs=inputs,
                outputs=outputs,
                builtin_options_type=options_type,
                builtin_options=options,
            )
            builder.replace_operator(operator_index, replacement)
        except Exception:
            checkpoint.rollback(document)
            raise
        return True

    def _narrow_int64_constant(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        tensor_index: int,
    ) -> int | None:
        """Return an INT32 constant index, narrowing INT64 values when safe."""

        contract = tensor_contract(graph, tensor_index)
        spec = self.codec.registry.get(contract.tensor_type)
        if spec is None or spec.name not in {"INT32", "INT64"}:
            return None
        if not is_constant_tensor(document.model, graph.subgraph, tensor_index):
            return None
        if spec.name == "INT32":
            return tensor_index
        try:
            value = self.codec.decode_tensor(
                document.model,
                subgraph_index=graph.subgraph_index,
                tensor_index=tensor_index,
            )
        except (CircleValueError, IndexError, ValueError):
            return None
        limits = np.iinfo(np.int32)
        if value.data.size and (
            np.any(value.data < limits.min) or np.any(value.data > limits.max)
        ):
            return None
        int32_type = self.resolver.tensor_type("INT32")
        narrowed = TensorValue.from_values(
            int32_type,
            value.data.astype(np.int32),
            shape=value.shape,
            dtype=np.int32,
            quantization=None,
        )
        builder = CircleBuilder(
            document,
            subgraph_index=graph.subgraph_index,
            codec=self.codec,
            object_factory=self.object_factory,
        )
        tensor = as_list(graph.subgraph.tensors)[tensor_index]
        tensor_name = str(getattr(tensor, "name", ""))
        return builder.add_constant(
            f"{tensor_name or 'legacy_constant'}_int32",
            narrowed,
        )

    def _custom_options(self, operator: Any) -> Mapping[str, Any]:
        """Decode one custom-options byte vector into a mapping."""

        payload = _byte_vector(getattr(operator, "customOptions", None))
        if not payload:
            return {}
        try:
            return dict(self.custom_option_decoder(payload))
        except (KeyError, TypeError, ValueError):
            return {}

    def _padding_value(self, name: str) -> int:
        """Return a configured or generated Circle Padding enum value."""

        normalized = name.upper()
        if normalized in self.padding_values:
            return self.padding_values[normalized]
        return _schema_enum_value("Padding", normalized)


def _custom_code(model: Any, operator: Any) -> str:
    """Decode the custom-code field referenced by one operator."""

    operator_codes = as_list(getattr(model, "operatorCodes", None))
    opcode_index = int(getattr(operator, "opcodeIndex", -1))
    if opcode_index < 0 or opcode_index >= len(operator_codes):
        return ""
    raw = getattr(operator_codes[opcode_index], "customCode", None)
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, np.ndarray):
        return bytes(np.asarray(raw, dtype=np.uint8)).decode(
            "utf-8",
            errors="replace",
        )
    return str(raw)


def _decode_flexbuffer_map(payload: bytes) -> Mapping[str, Any]:
    """Decode FlexBuffers custom options using the installed flatbuffers package."""

    try:
        from flatbuffers import flexbuffers
    except ImportError as error:
        raise ValueError(
            "Resolving legacy custom options requires flatbuffers.flexbuffers."
        ) from error
    loads = getattr(flexbuffers, "Loads", None)
    if callable(loads):
        decoded = loads(payload)
        if isinstance(decoded, Mapping):
            return decoded
        raise ValueError("FlexBuffers root is not a mapping.")
    raise ValueError("The installed flatbuffers package does not provide Loads().")


def _byte_vector(value: Any) -> bytes:
    """Normalize a generated uint8 vector to immutable bytes."""

    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    return bytes(np.ascontiguousarray(np.asarray(value, dtype=np.uint8)).reshape(-1))


def _first_bool(
    values: Mapping[str, Any],
    names: Sequence[str],
    default: bool,
) -> bool:
    """Return the first present mapping value converted to bool."""

    for name in names:
        if name in values:
            return bool(values[name])
    return bool(default)


def _int_vector(value: Any) -> tuple[int, ...]:
    """Convert a decoded sequence to a tuple of plain integers."""

    if value is None or isinstance(value, (str, bytes, bytearray)):
        return ()
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return ()


def _dense_static(contract: TensorContract) -> bool:
    """Return whether one tensor contract is static, dense, and immutable."""

    signature = contract.shape_signature
    return (
        (signature is None or all(dimension >= 0 for dimension in signature))
        and not contract.is_variable
        and contract.sparsity is None
        and contract.variant_tensors is None
    )


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


__all__ = [
    "CustomOptionDecoder",
    "LegacyCustomOpPolicy",
    "ResolveLegacyCustomOpsPass",
]

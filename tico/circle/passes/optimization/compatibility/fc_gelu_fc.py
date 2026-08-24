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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np

from tico.circle._object import clone_object, ObjectFactory
from tico.circle.analysis import TensorContract
from tico.circle.builder import CircleBuilder
from tico.circle.document import CircleDocument
from tico.circle.errors import CircleValueError
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
class LegacyFCGeluFCFusionPolicy:
    """Control tolerance and resource limits for the legacy Erf GELU pattern."""

    absolute_tolerance: float = 1.0e-5
    maximum_weight_elements: int = 16_000_000
    require_finite_constants: bool = True

    def __post_init__(self) -> None:
        """Normalize policy values and reject invalid numerical limits."""

        object.__setattr__(self, "absolute_tolerance", float(self.absolute_tolerance))
        object.__setattr__(
            self,
            "maximum_weight_elements",
            int(self.maximum_weight_elements),
        )
        object.__setattr__(
            self,
            "require_finite_constants",
            bool(self.require_finite_constants),
        )
        if self.absolute_tolerance < 0.0:
            raise ValueError("absolute_tolerance must not be negative.")
        if self.maximum_weight_elements <= 0:
            raise ValueError("maximum_weight_elements must be positive.")


@dataclass(frozen=True)
class _LegacyFCGeluFCMatch:
    """Capture all tensor and operator indices needed by one replacement."""

    source_tensor: int
    front_fc_index: int
    front_output: int
    back_fc_index: int
    back_weight: int
    back_bias: int
    back_output: int


class FuseLegacyFCGeluFCPass(CirclePass):
    """Replace a legacy FC-Erf branch with exact GELU and a rescaled back FC."""

    def __init__(
        self,
        *,
        policy: LegacyFCGeluFCFusionPolicy | None = None,
        builtin_codes: Mapping[str, int] | None = None,
        builtin_options_types: Mapping[str, int] | None = None,
        tensor_types: Mapping[str, int] | None = None,
        activation_none: int | None = None,
        codec: TensorValueCodec | None = None,
        object_factory: ObjectFactory | None = None,
    ) -> None:
        """Create the optional fusion with injectable schema-independent values."""

        self.policy = policy or LegacyFCGeluFCFusionPolicy()
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
                "CUSTOM",
                "FULLY_CONNECTED",
                "ADD",
                "MUL",
                "GELU",
            )
        }
        self.options_types = {
            name: self.resolver.builtin_options_type(name)
            for name in (
                "FullyConnectedOptions",
                "AddOptions",
                "MulOptions",
                "GeluOptions",
            )
        }
        self.float32_type = self.resolver.tensor_type("FLOAT32")

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Fuse every supported legacy pattern and leave obsolete branches for DCE."""

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
                matched = self._match(document, graph, operator_index)
                if matched is None:
                    operator_index += 1
                    continue
                inserted = self._apply(document, graph, matched)
                changes += 1
                diagnostics.append(
                    "Fused a legacy FC-Erf GELU pattern at "
                    f"subgraphs[{subgraph_index}].operators[{operator_index}] "
                    f"into {inserted} operator(s)."
                )
                operator_index += inserted
        return CirclePassResult(
            modified=changes > 0,
            changes=changes,
            diagnostics=tuple(diagnostics),
        )

    def _match(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        back_fc_index: int,
    ) -> _LegacyFCGeluFCMatch | None:
        """Match FC3(Mul(FC1(x), Add(Erf(FC2(x)), 1))) conservatively."""

        operators = as_list(graph.subgraph.operators)
        back_fc = self._plain_fc(document, graph, back_fc_index, allow_activation=True)
        if back_fc is None:
            return None
        back_inputs = tuple(as_indices(back_fc.inputs))
        back_outputs = tuple(as_indices(back_fc.outputs))
        if len(back_inputs) != 3 or len(back_outputs) != 1:
            return None
        mul_tensor, back_weight, back_bias = back_inputs
        if not self._constant_float32(document, graph, back_weight):
            return None
        back_weight_value = self._decode_float32(document, graph, back_weight)
        if back_weight_value is None:
            return None
        if back_weight_value.data.size > self.policy.maximum_weight_elements:
            return None
        if self.policy.require_finite_constants and not np.all(
            np.isfinite(back_weight_value.data)
        ):
            return None
        if back_bias != OPTIONAL_TENSOR_INDEX:
            if not self._constant_float32(document, graph, back_bias):
                return None

        mul_index = graph.producer(mul_tensor)
        mul = self._plain_binary(
            document,
            graph,
            mul_index,
            code_name="MUL",
            options_name="MulOptions",
        )
        if mul is None:
            return None
        mul_inputs = tuple(as_indices(mul.inputs))
        front_match = self._match_front_and_add(document, graph, mul_inputs)
        if front_match is None:
            return None
        front_fc_index, front_output, add_index = front_match

        add = operators[add_index]
        add_inputs = tuple(as_indices(add.inputs))
        erf_match = self._match_erf_and_one(document, graph, add_inputs)
        if erf_match is None:
            return None
        erf_index, erf_output = erf_match
        erf = operators[erf_index]
        erf_inputs = tuple(as_indices(erf.inputs))
        if len(erf_inputs) != 1:
            return None
        scaled_fc_output = erf_inputs[0]
        scaled_fc_index = graph.producer(scaled_fc_output)
        scaled_fc = self._plain_fc(
            document,
            graph,
            scaled_fc_index,
            allow_activation=False,
        )
        front_fc = operators[front_fc_index]
        if scaled_fc is None:
            return None

        front_inputs = tuple(as_indices(front_fc.inputs))
        scaled_inputs = tuple(as_indices(scaled_fc.inputs))
        if len(front_inputs) != 3 or len(scaled_inputs) != 3:
            return None
        if front_inputs[0] != scaled_inputs[0]:
            return None
        front_options = getattr(front_fc, "builtinOptions", None)
        scaled_options = getattr(scaled_fc, "builtinOptions", None)
        if bool(getattr(front_options, "keepNumDims", False)) != bool(
            getattr(scaled_options, "keepNumDims", False)
        ):
            return None
        source_tensor = front_inputs[0]
        front_weight, front_bias = front_inputs[1], front_inputs[2]
        scaled_weight, scaled_bias = scaled_inputs[1], scaled_inputs[2]
        if not all(
            self._constant_float32(document, graph, tensor_index)
            for tensor_index in (front_weight, scaled_weight)
        ):
            return None
        if not self._bias_pair_matches(
            document,
            graph,
            front_bias,
            scaled_bias,
        ):
            return None
        if not self._scaled_constant_matches(
            document,
            graph,
            front_weight,
            scaled_weight,
            np.float32(sqrt(0.5)),
        ):
            return None
        if front_bias != OPTIONAL_TENSOR_INDEX and not self._scaled_constant_matches(
            document,
            graph,
            front_bias,
            scaled_bias,
            np.float32(sqrt(0.5)),
        ):
            return None

        front_contract = tensor_contract(graph, front_output)
        scaled_contract = tensor_contract(graph, scaled_fc_output)
        erf_contract = tensor_contract(graph, erf_output)
        mul_contract = tensor_contract(graph, mul_tensor)
        back_input_contract = tensor_contract(graph, mul_tensor)
        if not all(
            self._supported_float32(contract)
            for contract in (
                tensor_contract(graph, source_tensor),
                front_contract,
                scaled_contract,
                erf_contract,
                mul_contract,
                back_input_contract,
                tensor_contract(graph, back_outputs[0]),
            )
        ):
            return None
        if not (front_contract == scaled_contract == erf_contract == mul_contract):
            return None

        return _LegacyFCGeluFCMatch(
            source_tensor=source_tensor,
            front_fc_index=front_fc_index,
            front_output=front_output,
            back_fc_index=back_fc_index,
            back_weight=back_weight,
            back_bias=back_bias,
            back_output=back_outputs[0],
        )

    def _apply(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        match: _LegacyFCGeluFCMatch,
    ) -> int:
        """Insert GELU before a cloned back FC and double its constant weights."""

        operators = as_list(graph.subgraph.operators)
        back_fc = operators[match.back_fc_index]
        back_weight_value = self._decode_float32(document, graph, match.back_weight)
        if back_weight_value is None:
            raise ValueError(
                "Matched back-FC weight is no longer a decodable constant."
            )
        doubled_weight = TensorValue(
            tensor_type=back_weight_value.tensor_type,
            shape=back_weight_value.shape,
            data=np.asarray(back_weight_value.data * np.float32(2.0), dtype=np.float32),
            quantization=None,
        )

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
            output_contract = tensor_contract(graph, match.front_output)
            gelu_output = builder.add_tensor(
                _derived_name(graph, match.back_output, "gelu"),
                output_contract,
            )
            weight_index = builder.add_constant(
                _derived_name(graph, match.back_weight, "gelu_rescaled"),
                doubled_weight,
                contract=tensor_contract(graph, match.back_weight),
            )
            gelu_options = self.resolver.create("GeluOptions")
            gelu_options.approximate = False
            gelu_operator = builder.make_operator(
                self.codes["GELU"],
                inputs=(match.front_output,),
                outputs=(gelu_output,),
                builtin_options_type=self.options_types["GeluOptions"],
                builtin_options=gelu_options,
            )
            fused_back = clone_object(back_fc)
            fused_back.inputs = [gelu_output, weight_index, match.back_bias]
            fused_back.outputs = [match.back_output]
            replacement = (gelu_operator, fused_back)
            current = as_list(graph.subgraph.operators)
            current[match.back_fc_index : match.back_fc_index + 1] = replacement
            graph.subgraph.operators = current
        except Exception:
            checkpoint.rollback(document)
            raise
        return 2

    def _match_front_and_add(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        inputs: Sequence[int],
    ) -> tuple[int, int, int] | None:
        """Find FC and ADD producers among the commutative MUL inputs."""

        if len(inputs) != 2:
            return None
        for front_output, add_output in (inputs, tuple(reversed(inputs))):
            front_index = graph.producer(front_output)
            add_index = graph.producer(add_output)
            if (
                self._plain_fc(
                    document,
                    graph,
                    front_index,
                    allow_activation=False,
                )
                is None
            ):
                continue
            if (
                self._plain_binary(
                    document,
                    graph,
                    add_index,
                    code_name="ADD",
                    options_name="AddOptions",
                )
                is None
            ):
                continue
            return int(front_index or 0), int(front_output or 0), int(add_index or 0)
        return None

    def _match_erf_and_one(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        inputs: Sequence[int],
    ) -> tuple[int, int] | None:
        """Find a custom Erf producer and scalar one among ADD inputs."""

        if len(inputs) != 2:
            return None
        for erf_output, one_tensor in (inputs, tuple(reversed(inputs))):
            if not self._is_scalar_one(document, graph, one_tensor):
                continue
            erf_index = graph.producer(erf_output)
            if erf_index is None:
                continue
            operators = as_list(graph.subgraph.operators)
            erf = operators[erf_index]
            if operator_builtin_code(document.model, erf) != self.codes["CUSTOM"]:
                continue
            if _custom_code(document.model, erf) != "Erf":
                continue
            if not operator_is_plain(erf):
                continue
            if len(as_indices(erf.inputs)) != 1 or tuple(as_indices(erf.outputs)) != (
                erf_output,
            ):
                continue
            return erf_index, erf_output
        return None

    def _plain_fc(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int | None,
        *,
        allow_activation: bool,
    ) -> Any | None:
        """Return one default-format plain FC with the requested activation policy."""

        if operator_index is None:
            return None
        operators = as_list(graph.subgraph.operators)
        if operator_index < 0 or operator_index >= len(operators):
            return None
        operator = operators[operator_index]
        if (
            operator_builtin_code(document.model, operator)
            != self.codes["FULLY_CONNECTED"]
        ):
            return None
        if not operator_is_plain(operator):
            return None
        if (
            int(getattr(operator, "builtinOptionsType", 0) or 0)
            != self.options_types["FullyConnectedOptions"]
        ):
            return None
        options = getattr(operator, "builtinOptions", None)
        if options is None or int(getattr(options, "weightsFormat", 0) or 0) != 0:
            return None
        activation = int(getattr(options, "fusedActivationFunction", -1))
        if not allow_activation and activation != self.resolver.activation_none:
            return None
        return operator

    def _plain_binary(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int | None,
        *,
        code_name: str,
        options_name: str,
    ) -> Any | None:
        """Return one plain two-input binary operator without fused activation."""

        if operator_index is None:
            return None
        operators = as_list(graph.subgraph.operators)
        if operator_index < 0 or operator_index >= len(operators):
            return None
        operator = operators[operator_index]
        if operator_builtin_code(document.model, operator) != self.codes[code_name]:
            return None
        if not operator_is_plain(operator):
            return None
        if (
            int(getattr(operator, "builtinOptionsType", 0) or 0)
            != self.options_types[options_name]
        ):
            return None
        if (
            len(as_indices(operator.inputs)) != 2
            or len(as_indices(operator.outputs)) != 1
        ):
            return None
        options = getattr(operator, "builtinOptions", None)
        if options is None:
            return None
        if int(getattr(options, "fusedActivationFunction", -1)) != (
            self.resolver.activation_none
        ):
            return None
        return operator

    def _bias_pair_matches(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        front_bias: int,
        scaled_bias: int,
    ) -> bool:
        """Return whether both FC biases are absent or both are F32 constants."""

        if front_bias == OPTIONAL_TENSOR_INDEX or scaled_bias == OPTIONAL_TENSOR_INDEX:
            return front_bias == scaled_bias == OPTIONAL_TENSOR_INDEX
        return self._constant_float32(
            document,
            graph,
            front_bias,
        ) and self._constant_float32(document, graph, scaled_bias)

    def _scaled_constant_matches(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        source_index: int,
        scaled_index: int,
        multiplier: np.float32,
    ) -> bool:
        """Compare two F32 constants under one absolute tolerance."""

        source = self._decode_float32(document, graph, source_index)
        scaled = self._decode_float32(document, graph, scaled_index)
        if source is None or scaled is None or source.shape != scaled.shape:
            return False
        if self.policy.require_finite_constants and not (
            np.all(np.isfinite(source.data)) and np.all(np.isfinite(scaled.data))
        ):
            return False
        expected = np.asarray(source.data * multiplier, dtype=np.float32)
        return bool(
            np.allclose(
                expected,
                scaled.data,
                rtol=0.0,
                atol=self.policy.absolute_tolerance,
                equal_nan=False,
            )
        )

    def _constant_float32(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        tensor_index: int,
    ) -> bool:
        """Return whether one tensor is an inline plain FLOAT32 constant."""

        if tensor_index == OPTIONAL_TENSOR_INDEX:
            return False
        if not is_constant_tensor(document.model, graph.subgraph, tensor_index):
            return False
        return self._supported_float32(tensor_contract(graph, tensor_index))

    def _decode_float32(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        tensor_index: int,
    ) -> TensorValue | None:
        """Decode one supported FLOAT32 constant without mutating the graph."""

        if not self._constant_float32(document, graph, tensor_index):
            return None
        try:
            value = self.codec.decode_tensor(
                document.model,
                subgraph_index=graph.subgraph_index,
                tensor_index=tensor_index,
            )
        except (CircleValueError, IndexError, ValueError):
            return None
        if value.data.dtype != np.dtype(np.float32):
            return None
        return value

    def _is_scalar_one(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        tensor_index: int,
    ) -> bool:
        """Return whether one inline F32 constant contains exactly one value of one."""

        value = self._decode_float32(document, graph, tensor_index)
        if value is None or value.data.size != 1:
            return False
        return bool(value.data.reshape(-1)[0] == np.float32(1.0))

    def _supported_float32(self, contract: TensorContract) -> bool:
        """Return whether one tensor is static, dense, immutable, and plain F32."""

        signature = contract.shape_signature
        return (
            contract.tensor_type == self.float32_type
            and contract.quantization is None
            and (signature is None or all(dimension >= 0 for dimension in signature))
            and not contract.is_variable
            and contract.sparsity is None
            and contract.variant_tensors is None
        )


def _custom_code(model: Any, operator: Any) -> str:
    """Decode the custom code referenced by one Circle operator."""

    operator_codes = as_list(getattr(model, "operatorCodes", None))
    opcode_index = int(getattr(operator, "opcodeIndex", -1))
    if opcode_index < 0 or opcode_index >= len(operator_codes):
        return ""
    raw = getattr(operator_codes[opcode_index], "customCode", None)
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, np.ndarray):
        return bytes(np.asarray(raw, dtype=np.uint8)).decode(
            "utf-8",
            errors="replace",
        )
    return str(raw or "")


def _derived_name(graph: CircleGraph, tensor_index: int, suffix: str) -> str:
    """Create a stable name from one existing tensor and a descriptive suffix."""

    tensors = as_list(graph.subgraph.tensors)
    raw_name = getattr(tensors[tensor_index], "name", None)
    if isinstance(raw_name, bytes):
        name = raw_name.decode("utf-8", errors="replace")
    else:
        name = str(raw_name or f"tensor_{tensor_index}")
    return f"{name}/{suffix}"


__all__ = [
    "FuseLegacyFCGeluFCPass",
    "LegacyFCGeluFCFusionPolicy",
]

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
from typing import Any, cast

import numpy as np

from tico.circle._object import clone_object, freeze_object, ObjectFactory
from tico.circle.analysis import TensorContract
from tico.circle.builder import CircleBuilder
from tico.circle.document import CircleDocument
from tico.circle.errors import CircleRewriteError
from tico.circle.graph import as_indices, as_list, CircleGraph, OPTIONAL_TENSOR_INDEX
from tico.circle.passes.base import CirclePassContext
from tico.circle.passes.optimization._utils import (
    AppendedObjectCheckpoint,
    contract_is_dense_nonvariable,
    contract_is_fully_static,
    decode_constant_value,
    operator_builtin_code,
    operator_is_plain,
    operator_version,
    OptimizationSchemaResolver,
    tensor_contract,
)
from tico.circle.passes.rules import (
    CircleRewriteRule,
    CircleRulePass,
    OperatorSnapshot,
    RewriteApplication,
    RewritePlan,
)
from tico.circle.value import TensorValue, TensorValueCodec

_NEW_WEIGHT = -2
_NEW_BIAS = -3


@dataclass(frozen=True)
class LinearFusionPolicy:
    """Control bounded allocation and numerical preconditions for linear fusion."""

    maximum_replacement_bytes: int = 64 * 1024 * 1024
    require_finite_constants: bool = True
    allow_float_reassociation: bool = True

    def __post_init__(self) -> None:
        """Reject resource limits that cannot admit any replacement constant."""

        if self.maximum_replacement_bytes <= 0:
            raise ValueError("maximum_replacement_bytes must be positive.")


@dataclass(frozen=True)
class LinearOperatorDescriptor:
    """Describe Circle input positions and weight layout for one linear operator."""

    name: str
    builtin_code: int
    options_type: int
    data_position: int
    weight_position: int
    bias_position: int
    weight_output_axis: int


@dataclass(frozen=True)
class PlannedConstant:
    """Describe one immutable constant allocated while applying a fusion plan."""

    input_position: int
    name: str
    value: TensorValue
    contract: TensorContract


@dataclass(frozen=True, kw_only=True)
class LinearFusionPlan(RewritePlan):
    """Capture all operators, tensors, and replacement constants for one fusion."""

    template_operator_index: int
    supporting_operators: tuple[OperatorSnapshot, ...]
    replacement_inputs: tuple[int, ...]
    constants: tuple[PlannedConstant, ...]

    def validate(self, document: CircleDocument) -> None:
        """Validate the anchor and every supporting linear or affine operator."""

        super().validate(document)
        for expected in self.supporting_operators:
            try:
                current = OperatorSnapshot.capture(
                    document,
                    subgraph_index=self.subgraph_index,
                    operator_index=expected.operator_index,
                )
            except (IndexError, CircleRewriteError) as error:
                raise CircleRewriteError(
                    f"Linear fusion plan is stale because operator "
                    f"{expected.operator_index} no longer exists."
                ) from error
            if current != expected:
                raise CircleRewriteError(
                    f"Linear fusion plan is stale because operator "
                    f"{expected.operator_index} changed."
                )


@dataclass(frozen=True)
class _LinearState:
    """Collect validated float linear parameters used by fusion matchers."""

    descriptor: LinearOperatorDescriptor
    operator_index: int
    operator: Any
    inputs: tuple[int, ...]
    output_tensor: int
    data_tensor: int
    weight_tensor: int
    bias_tensor: int | None
    weight: TensorValue
    weight_contract: TensorContract
    bias: TensorValue | None
    bias_contract: TensorContract | None
    output_contract: TensorContract
    output_channels: int


class _LinearRuleBase(CircleRewriteRule[LinearFusionPlan]):
    """Provide shared schema, decoding, planning, and transactional apply helpers."""

    def __init__(
        self,
        *,
        descriptors: Mapping[int, LinearOperatorDescriptor],
        affine_codes: Mapping[str, int],
        affine_options_types: Mapping[str, int],
        float32_type: int,
        activation_none: int,
        codec: TensorValueCodec,
        object_factory: ObjectFactory | None,
        policy: LinearFusionPolicy,
    ) -> None:
        """Store immutable services shared by all linear-fusion rules."""

        self.descriptors = dict(descriptors)
        self.affine_codes = dict(affine_codes)
        self.affine_options_types = dict(affine_options_types)
        self.float32_type = int(float32_type)
        self.activation_none = int(activation_none)
        self.codec = codec
        self.object_factory = object_factory
        self.policy = policy

    def apply(
        self,
        document: CircleDocument,
        plan: LinearFusionPlan,
        context: CirclePassContext,
    ) -> RewriteApplication:
        """Allocate replacement constants and replace only the anchor operator."""

        del context
        checkpoint = AppendedObjectCheckpoint.capture(
            document,
            subgraph_index=plan.subgraph_index,
        )
        builder = CircleBuilder(
            document,
            subgraph_index=plan.subgraph_index,
            codec=self.codec,
            object_factory=self.object_factory,
        )
        try:
            inputs = list(plan.replacement_inputs)
            for constant in plan.constants:
                tensor_index = builder.add_constant(
                    constant.name,
                    constant.value,
                    contract=constant.contract,
                )
                inputs[constant.input_position] = tensor_index

            subgraph = document.subgraph(plan.subgraph_index)
            operators = as_list(getattr(subgraph, "operators", None))
            template = clone_object(operators[plan.template_operator_index])
            template.inputs = inputs
            template.outputs = list(plan.anchor.outputs)
            builder.replace_operator(plan.anchor_operator_index, template)
        except Exception:
            checkpoint.rollback(document)
            raise
        return RewriteApplication(changes=1 + len(plan.constants))

    def _is_live(self, graph: CircleGraph, operator_index: int) -> bool:
        """Return whether an operator contributes to a serialized graph output."""

        return operator_index in graph.backward_operators(graph.outputs)

    def _linear_state(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        *,
        require_no_activation: bool,
    ) -> _LinearState | None:
        """Validate and decode one supported dense FLOAT32 linear operator."""

        operators = as_list(getattr(graph.subgraph, "operators", None))
        if operator_index < 0 or operator_index >= len(operators):
            return None
        operator = operators[operator_index]
        descriptor = self.descriptors.get(
            operator_builtin_code(document.model, operator)
        )
        if descriptor is None or not operator_is_plain(operator):
            return None
        if int(getattr(operator, "builtinOptionsType", 0) or 0) != (
            descriptor.options_type
        ):
            return None
        if require_no_activation and not _has_no_fused_activation(
            operator,
            self.activation_none,
        ):
            return None

        inputs = tuple(as_indices(getattr(operator, "inputs", None)))
        outputs = tuple(as_indices(getattr(operator, "outputs", None)))
        required_position = max(
            descriptor.data_position,
            descriptor.weight_position,
        )
        if len(inputs) <= required_position or len(outputs) != 1:
            return None
        data_tensor = inputs[descriptor.data_position]
        weight_tensor = inputs[descriptor.weight_position]
        if data_tensor < 0 or weight_tensor < 0:
            return None
        bias_tensor = _optional_input(inputs, descriptor.bias_position)

        data_contract = tensor_contract(graph, data_tensor)
        output_contract = tensor_contract(graph, outputs[0])
        if not self._supported_float_contract(data_contract):
            return None
        if not self._supported_float_contract(output_contract):
            return None

        weight_pair = self._float_constant(
            document,
            graph,
            weight_tensor,
        )
        if weight_pair is None:
            return None
        weight, weight_contract = weight_pair
        bias: TensorValue | None = None
        bias_contract: TensorContract | None = None
        if bias_tensor is not None:
            bias_pair = self._float_constant(
                document,
                graph,
                bias_tensor,
            )
            if bias_pair is None:
                return None
            bias, bias_contract = bias_pair

        layout = _validate_linear_layout(
            descriptor,
            data_contract=data_contract,
            output_contract=output_contract,
            weight=weight,
            bias=bias,
            options=getattr(operator, "builtinOptions", None),
        )
        if layout is None:
            return None
        return _LinearState(
            descriptor=descriptor,
            operator_index=operator_index,
            operator=operator,
            inputs=inputs,
            output_tensor=outputs[0],
            data_tensor=data_tensor,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            weight=weight,
            weight_contract=weight_contract,
            bias=bias,
            bias_contract=bias_contract,
            output_contract=output_contract,
            output_channels=layout,
        )

    def _float_constant(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        tensor_index: int,
    ) -> tuple[TensorValue, TensorContract] | None:
        """Decode a static dense unquantized FLOAT32 constant."""

        contract = tensor_contract(graph, tensor_index)
        if not self._supported_float_contract(contract):
            return None
        value = decode_constant_value(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=tensor_index,
        )
        if value is None or value.tensor_type != self.float32_type:
            return None
        if value.quantization is not None or value.data.dtype != np.dtype(np.float32):
            return None
        if self.policy.require_finite_constants and not np.all(np.isfinite(value.data)):
            return None
        return value, contract

    def _supported_float_contract(self, contract: TensorContract) -> bool:
        """Return whether a tensor has a static dense unquantized FLOAT32 contract."""

        return (
            contract.tensor_type == self.float32_type
            and contract.quantization is None
            and contract_is_fully_static(contract)
            and contract_is_dense_nonvariable(contract)
        )

    def _replacement_fits(self, constants: Sequence[PlannedConstant]) -> bool:
        """Return whether planned constant payloads fit the configured byte bound."""

        return (
            sum(constant.value.nbytes for constant in constants)
            <= self.policy.maximum_replacement_bytes
        )

    def _constant(
        self,
        *,
        position: int,
        name: str,
        data: np.ndarray[Any, Any],
        contract: TensorContract,
    ) -> PlannedConstant | None:
        """Create one finite FLOAT32 planned constant with a preserved contract."""

        array = np.asarray(data, dtype=np.float32)
        if tuple(array.shape) != contract.shape:
            return None
        if self.policy.require_finite_constants and not np.all(np.isfinite(array)):
            return None
        value = TensorValue.from_values(
            self.float32_type,
            array,
            dtype=np.float32,
        )
        return PlannedConstant(
            input_position=int(position),
            name=name,
            value=value,
            contract=contract,
        )


class _PostAffineIntoLinearRule(_LinearRuleBase):
    """Fold channel-wise ADD, SUB, or MUL after a supported linear operator."""

    def match(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        context: CirclePassContext,
    ) -> LinearFusionPlan | None:
        """Match a live post-linear affine operator with one constant operand."""

        del context
        if not self.policy.allow_float_reassociation:
            return None
        if not self._is_live(graph, operator_index):
            return None
        operator = as_list(graph.subgraph.operators)[operator_index]
        affine_name = _lookup_name(
            self.affine_codes,
            operator_builtin_code(document.model, operator),
        )
        if affine_name not in {"ADD", "SUB", "MUL"}:
            return None
        if int(getattr(operator, "builtinOptionsType", 0) or 0) != (
            self.affine_options_types[affine_name]
        ):
            return None
        if not operator_is_plain(operator) or not _has_no_fused_activation(
            operator,
            self.activation_none,
        ):
            return None
        inputs = tuple(as_indices(getattr(operator, "inputs", None)))
        outputs = tuple(as_indices(getattr(operator, "outputs", None)))
        if len(inputs) != 2 or len(outputs) != 1:
            return None

        affine_match = self._affine_input(
            document,
            graph,
            affine_name,
            inputs,
        )
        if affine_match is None:
            return None
        linear_tensor, constant_tensor, constant, _constant_contract = affine_match
        producer = graph.producer(linear_tensor)
        if producer is None:
            return None
        linear = self._linear_state(
            document,
            graph,
            producer,
            require_no_activation=True,
        )
        if linear is None or linear.output_tensor != linear_tensor:
            return None

        result_contract = tensor_contract(graph, outputs[0])
        if result_contract != linear.output_contract:
            return None
        channel_values = _channel_vector(
            constant.data,
            result_contract.shape,
            linear.output_channels,
        )
        if channel_values is None:
            return None

        replacement_inputs = _with_optional_bias_slot(
            linear.inputs,
            linear.descriptor.bias_position,
        )
        constants: list[PlannedConstant] = []
        if affine_name in {"ADD", "SUB"}:
            if (
                linear.descriptor.name == "TRANSPOSE_CONV"
                and linear.bias is None
                and (
                    len(linear.inputs) <= linear.descriptor.bias_position
                    or operator_version(document.model, linear.operator) < 3
                )
            ):
                return None
            signed = channel_values if affine_name == "ADD" else -channel_values
            old_bias = (
                np.zeros(linear.output_channels, dtype=np.float32)
                if linear.bias is None
                else linear.bias.data
            )
            with np.errstate(over="ignore", invalid="ignore"):
                new_bias = np.asarray(old_bias + signed, dtype=np.float32)
            bias_contract = linear.bias_contract or _bias_contract(
                self.float32_type,
                linear.output_channels,
            )
            planned = self._constant(
                position=linear.descriptor.bias_position,
                name=_fused_name(graph, linear.bias_tensor, "bias"),
                data=new_bias,
                contract=bias_contract,
            )
            if planned is None:
                return None
            replacement_inputs[linear.descriptor.bias_position] = _NEW_BIAS
            constants.append(planned)
        else:
            weight_scale = _reshape_for_axis(
                channel_values,
                linear.weight.shape,
                linear.descriptor.weight_output_axis,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                new_weight = np.asarray(
                    linear.weight.data * weight_scale,
                    dtype=np.float32,
                )
            planned_weight = self._constant(
                position=linear.descriptor.weight_position,
                name=_fused_name(graph, linear.weight_tensor, "weight"),
                data=new_weight,
                contract=linear.weight_contract,
            )
            if planned_weight is None:
                return None
            replacement_inputs[linear.descriptor.weight_position] = _NEW_WEIGHT
            constants.append(planned_weight)
            if linear.bias is not None and linear.bias_contract is not None:
                with np.errstate(over="ignore", invalid="ignore"):
                    new_bias = np.asarray(
                        linear.bias.data * channel_values,
                        dtype=np.float32,
                    )
                planned_bias = self._constant(
                    position=linear.descriptor.bias_position,
                    name=_fused_name(graph, linear.bias_tensor, "bias"),
                    data=new_bias,
                    contract=linear.bias_contract,
                )
                if planned_bias is None:
                    return None
                replacement_inputs[linear.descriptor.bias_position] = _NEW_BIAS
                constants.append(planned_bias)

        if not self._replacement_fits(constants):
            return None
        tensor_indices = _unique_indices(
            (*inputs, *outputs, *linear.inputs, linear.output_tensor)
        )
        plan = LinearFusionPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=tensor_indices,
            template_operator_index=producer,
            supporting_operators=(
                OperatorSnapshot.capture(
                    document,
                    subgraph_index=graph.subgraph_index,
                    operator_index=producer,
                ),
            ),
            replacement_inputs=tuple(replacement_inputs),
            constants=tuple(constants),
        )
        return cast(LinearFusionPlan, plan)

    def _affine_input(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        affine_name: str,
        inputs: tuple[int, int],
    ) -> tuple[int, int, TensorValue, TensorContract] | None:
        """Return the linear input and constant input of one affine operator."""

        candidates: tuple[tuple[int, int], ...]
        if affine_name == "SUB":
            candidates = ((inputs[0], inputs[1]),)
        else:
            candidates = ((inputs[0], inputs[1]), (inputs[1], inputs[0]))
        for linear_tensor, constant_tensor in candidates:
            constant_pair = self._float_constant(
                document,
                graph,
                constant_tensor,
            )
            if constant_pair is None:
                continue
            constant, contract = constant_pair
            if graph.producer(linear_tensor) is None:
                continue
            return linear_tensor, constant_tensor, constant, contract
        return None


class _PreAffineIntoFullyConnectedRule(_LinearRuleBase):
    """Fold channel-wise ADD, SUB, or MUL before FULLY_CONNECTED."""

    def __init__(self, *, fully_connected_code: int, **kwargs: Any) -> None:
        """Create the rule for one schema-resolved FULLY_CONNECTED code."""

        super().__init__(**kwargs)
        self.fully_connected_code = int(fully_connected_code)

    def match(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        context: CirclePassContext,
    ) -> LinearFusionPlan | None:
        """Match a live FULLY_CONNECTED whose data input is a constant affine op."""

        del context
        if not self.policy.allow_float_reassociation:
            return None
        if not self._is_live(graph, operator_index):
            return None
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.fully_connected_code:
            return None
        linear = self._linear_state(
            document,
            graph,
            operator_index,
            require_no_activation=False,
        )
        if linear is None or linear.descriptor.name != "FULLY_CONNECTED":
            return None
        affine_index = graph.producer(linear.data_tensor)
        if affine_index is None:
            return None
        affine = as_list(graph.subgraph.operators)[affine_index]
        affine_name = _lookup_name(
            self.affine_codes,
            operator_builtin_code(document.model, affine),
        )
        if affine_name not in {"ADD", "SUB", "MUL"}:
            return None
        if int(getattr(affine, "builtinOptionsType", 0) or 0) != (
            self.affine_options_types[affine_name]
        ):
            return None
        if not operator_is_plain(affine) or not _has_no_fused_activation(
            affine,
            self.activation_none,
        ):
            return None
        affine_inputs = tuple(as_indices(getattr(affine, "inputs", None)))
        affine_outputs = tuple(as_indices(getattr(affine, "outputs", None)))
        if len(affine_inputs) != 2 or affine_outputs != (linear.data_tensor,):
            return None

        affine_match = self._affine_input(
            document,
            graph,
            affine_name,
            affine_inputs,
        )
        if affine_match is None:
            return None
        source_tensor, constant_tensor, constant, _constant_contract = affine_match
        source_contract = tensor_contract(graph, source_tensor)
        data_contract = tensor_contract(graph, linear.data_tensor)
        if source_contract != data_contract or source_contract.rank == 0:
            return None
        input_channels = int(linear.weight.shape[1])
        if source_contract.shape[-1] != input_channels:
            return None
        channel_values = _channel_vector(
            constant.data,
            source_contract.shape,
            input_channels,
        )
        if channel_values is None:
            return None

        replacement_inputs = _with_optional_bias_slot(
            linear.inputs,
            linear.descriptor.bias_position,
        )
        replacement_inputs[linear.descriptor.data_position] = source_tensor
        constants: list[PlannedConstant] = []
        if affine_name == "MUL":
            with np.errstate(over="ignore", invalid="ignore"):
                new_weight = np.asarray(
                    linear.weight.data * channel_values.reshape(1, -1),
                    dtype=np.float32,
                )
            planned_weight = self._constant(
                position=linear.descriptor.weight_position,
                name=_fused_name(graph, linear.weight_tensor, "weight"),
                data=new_weight,
                contract=linear.weight_contract,
            )
            if planned_weight is None:
                return None
            replacement_inputs[linear.descriptor.weight_position] = _NEW_WEIGHT
            constants.append(planned_weight)
        else:
            signed = channel_values if affine_name == "ADD" else -channel_values
            old_bias = (
                np.zeros(linear.output_channels, dtype=np.float32)
                if linear.bias is None
                else linear.bias.data
            )
            with np.errstate(over="ignore", invalid="ignore"):
                delta = np.matmul(linear.weight.data, signed)
                new_bias = np.asarray(old_bias + delta, dtype=np.float32)
            bias_contract = linear.bias_contract or _bias_contract(
                self.float32_type,
                linear.output_channels,
            )
            planned_bias = self._constant(
                position=linear.descriptor.bias_position,
                name=_fused_name(graph, linear.bias_tensor, "bias"),
                data=new_bias,
                contract=bias_contract,
            )
            if planned_bias is None:
                return None
            replacement_inputs[linear.descriptor.bias_position] = _NEW_BIAS
            constants.append(planned_bias)

        if not self._replacement_fits(constants):
            return None
        tensor_indices = _unique_indices(
            (*affine_inputs, *affine_outputs, *linear.inputs, *linear.operator.outputs)
        )
        plan = LinearFusionPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=tensor_indices,
            template_operator_index=operator_index,
            supporting_operators=(
                OperatorSnapshot.capture(
                    document,
                    subgraph_index=graph.subgraph_index,
                    operator_index=affine_index,
                ),
            ),
            replacement_inputs=tuple(replacement_inputs),
            constants=tuple(constants),
        )
        return cast(LinearFusionPlan, plan)

    def _affine_input(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        affine_name: str,
        inputs: tuple[int, int],
    ) -> tuple[int, int, TensorValue, TensorContract] | None:
        """Return the runtime source and constant of a pre-FC affine operator."""

        candidates: tuple[tuple[int, int], ...]
        if affine_name == "SUB":
            candidates = ((inputs[0], inputs[1]),)
        else:
            candidates = ((inputs[0], inputs[1]), (inputs[1], inputs[0]))
        for source_tensor, constant_tensor in candidates:
            if graph.is_constant(source_tensor):
                continue
            constant_pair = self._float_constant(
                document,
                graph,
                constant_tensor,
            )
            if constant_pair is None:
                continue
            constant, contract = constant_pair
            return source_tensor, constant_tensor, constant, contract
        return None


class _HorizontalFullyConnectedRule(_LinearRuleBase):
    """Replace ADD of two compatible FULLY_CONNECTED branches by one branch."""

    def __init__(
        self,
        *,
        fully_connected_code: int,
        add_code: int,
        **kwargs: Any,
    ) -> None:
        """Create the horizontal fusion rule from resolved builtin codes."""

        super().__init__(**kwargs)
        self.fully_connected_code = int(fully_connected_code)
        self.add_code = int(add_code)

    def match(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        context: CirclePassContext,
    ) -> LinearFusionPlan | None:
        """Match a live ADD whose two inputs are compatible FC outputs."""

        del context
        if not self.policy.allow_float_reassociation:
            return None
        if not self._is_live(graph, operator_index):
            return None
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.add_code:
            return None
        if int(getattr(operator, "builtinOptionsType", 0) or 0) != (
            self.affine_options_types["ADD"]
        ):
            return None
        if not operator_is_plain(operator) or not _has_no_fused_activation(
            operator,
            self.activation_none,
        ):
            return None
        inputs = tuple(as_indices(getattr(operator, "inputs", None)))
        outputs = tuple(as_indices(getattr(operator, "outputs", None)))
        if len(inputs) != 2 or len(outputs) != 1:
            return None
        producer_indices = tuple(graph.producer(index) for index in inputs)
        if any(index is None for index in producer_indices):
            return None
        first_index = int(cast(int, producer_indices[0]))
        second_index = int(cast(int, producer_indices[1]))
        first = self._linear_state(
            document,
            graph,
            first_index,
            require_no_activation=True,
        )
        second = self._linear_state(
            document,
            graph,
            second_index,
            require_no_activation=True,
        )
        if first is None or second is None:
            return None
        if (
            first.descriptor.name != "FULLY_CONNECTED"
            or second.descriptor.name != "FULLY_CONNECTED"
            or first.data_tensor != second.data_tensor
            or first.output_tensor != inputs[0]
            or second.output_tensor != inputs[1]
            or first.output_contract != second.output_contract
            or tensor_contract(graph, outputs[0]) != first.output_contract
            or first.weight.shape != second.weight.shape
            or first.weight_contract != second.weight_contract
            or not _bias_contracts_compatible(first, second)
            or operator_version(document.model, first.operator)
            != operator_version(document.model, second.operator)
            or not _same_linear_options(first.operator, second.operator)
        ):
            return None

        with np.errstate(over="ignore", invalid="ignore"):
            new_weight = np.asarray(
                first.weight.data + second.weight.data,
                dtype=np.float32,
            )
        replacement_inputs = _with_optional_bias_slot(
            first.inputs,
            first.descriptor.bias_position,
        )
        planned_weight = self._constant(
            position=first.descriptor.weight_position,
            name=_fused_name(graph, first.weight_tensor, "horizontal_weight"),
            data=new_weight,
            contract=first.weight_contract,
        )
        if planned_weight is None:
            return None
        replacement_inputs[first.descriptor.weight_position] = _NEW_WEIGHT
        constants: list[PlannedConstant] = [planned_weight]

        if first.bias is not None or second.bias is not None:
            first_bias = (
                np.zeros(first.output_channels, dtype=np.float32)
                if first.bias is None
                else first.bias.data
            )
            second_bias = (
                np.zeros(second.output_channels, dtype=np.float32)
                if second.bias is None
                else second.bias.data
            )
            with np.errstate(over="ignore", invalid="ignore"):
                new_bias = np.asarray(first_bias + second_bias, dtype=np.float32)
            bias_contract = (
                first.bias_contract
                or second.bias_contract
                or _bias_contract(self.float32_type, first.output_channels)
            )
            planned_bias = self._constant(
                position=first.descriptor.bias_position,
                name=_fused_name(graph, first.bias_tensor, "horizontal_bias"),
                data=new_bias,
                contract=bias_contract,
            )
            if planned_bias is None:
                return None
            replacement_inputs[first.descriptor.bias_position] = _NEW_BIAS
            constants.append(planned_bias)
        else:
            replacement_inputs[first.descriptor.bias_position] = OPTIONAL_TENSOR_INDEX

        if not self._replacement_fits(constants):
            return None
        supporting = tuple(
            OperatorSnapshot.capture(
                document,
                subgraph_index=graph.subgraph_index,
                operator_index=index,
            )
            for index in dict.fromkeys((first_index, second_index))
        )
        tensor_indices = _unique_indices(
            (*inputs, *outputs, *first.inputs, *second.inputs)
        )
        plan = LinearFusionPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=tensor_indices,
            template_operator_index=first_index,
            supporting_operators=supporting,
            replacement_inputs=tuple(replacement_inputs),
            constants=tuple(constants),
        )
        return cast(LinearFusionPlan, plan)


class FuseLinearOpsPass(CircleRulePass):
    """Fuse safe FLOAT32 affine patterns into Conv, TransposeConv, and FC operators."""

    def __init__(
        self,
        *,
        builtin_codes: Mapping[str, int] | None = None,
        builtin_options_types: Mapping[str, int] | None = None,
        tensor_types: Mapping[str, int] | None = None,
        activation_none: int | None = None,
        codec: TensorValueCodec | None = None,
        object_factory: ObjectFactory | None = None,
        policy: LinearFusionPolicy | None = None,
        maximum_rewrites: int = 10_000,
    ) -> None:
        """Create linear-fusion rules with schema-derived or injected identities."""

        resolver = OptimizationSchemaResolver(
            builtin_codes=builtin_codes,
            builtin_options_types=builtin_options_types,
            tensor_types=tensor_types,
            activation_none=activation_none,
            object_factory=object_factory,
        )
        descriptors = _linear_descriptors(resolver)
        affine_codes = {
            name: code
            for name in ("ADD", "MUL", "SUB")
            if (code := resolver.maybe_builtin_code(name)) is not None
        }
        affine_options_types = {
            name: resolver.builtin_options_type(f"{name.title()}Options")
            for name in affine_codes
        }
        fully_connected_code = resolver.builtin_code("FULLY_CONNECTED")
        add_code = resolver.builtin_code("ADD")
        shared: dict[str, Any] = {
            "descriptors": descriptors,
            "affine_codes": affine_codes,
            "affine_options_types": affine_options_types,
            "float32_type": resolver.tensor_type("FLOAT32"),
            "activation_none": resolver.activation_none,
            "codec": codec or TensorValueCodec(),
            "object_factory": object_factory,
            "policy": policy or LinearFusionPolicy(),
        }
        super().__init__(
            (
                _HorizontalFullyConnectedRule(
                    fully_connected_code=fully_connected_code,
                    add_code=add_code,
                    **shared,
                ),
                _PostAffineIntoLinearRule(**shared),
                _PreAffineIntoFullyConnectedRule(
                    fully_connected_code=fully_connected_code,
                    **shared,
                ),
            ),
            maximum_rewrites=maximum_rewrites,
        )


def _linear_descriptors(
    resolver: OptimizationSchemaResolver,
) -> dict[int, LinearOperatorDescriptor]:
    """Return supported linear descriptors keyed by builtin operator code."""

    layouts = (
        ("FULLY_CONNECTED", "FullyConnectedOptions", 0, 1, 2, 0),
        ("CONV_2D", "Conv2DOptions", 0, 1, 2, 0),
        (
            "DEPTHWISE_CONV_2D",
            "DepthwiseConv2DOptions",
            0,
            1,
            2,
            3,
        ),
        ("TRANSPOSE_CONV", "TransposeConvOptions", 2, 1, 3, 0),
    )
    descriptors: dict[int, LinearOperatorDescriptor] = {}
    for (
        name,
        options_name,
        data_position,
        weight_position,
        bias_position,
        output_axis,
    ) in layouts:
        code = resolver.maybe_builtin_code(name)
        if code is None:
            continue
        descriptors[code] = LinearOperatorDescriptor(
            name=name,
            builtin_code=code,
            options_type=resolver.builtin_options_type(options_name),
            data_position=data_position,
            weight_position=weight_position,
            bias_position=bias_position,
            weight_output_axis=output_axis,
        )
    return descriptors


def _validate_linear_layout(
    descriptor: LinearOperatorDescriptor,
    *,
    data_contract: TensorContract,
    output_contract: TensorContract,
    weight: TensorValue,
    bias: TensorValue | None,
    options: Any,
) -> int | None:
    """Validate known Circle linear weight layouts and return output channels."""

    if data_contract.rank == 0 or output_contract.rank == 0:
        return None
    shape = weight.shape
    if descriptor.name == "FULLY_CONNECTED":
        weights_format = int(getattr(options, "weightsFormat", 0) or 0)
        if (
            len(shape) != 2
            or weights_format != 0
            or data_contract.shape[-1] != shape[1]
        ):
            return None
    elif descriptor.name in {"CONV_2D", "TRANSPOSE_CONV"}:
        if (
            len(shape) != 4
            or data_contract.rank != 4
            or output_contract.rank != 4
            or data_contract.shape[-1] != shape[3]
        ):
            return None
    elif descriptor.name == "DEPTHWISE_CONV_2D":
        if (
            len(shape) != 4
            or data_contract.rank != 4
            or output_contract.rank != 4
            or shape[0] != 1
        ):
            return None
        multiplier = int(getattr(options, "depthMultiplier", 0) or 0)
        if multiplier <= 0 or data_contract.shape[-1] * multiplier != shape[3]:
            return None
    else:
        return None

    output_axis = descriptor.weight_output_axis
    if output_axis < 0:
        output_axis += len(shape)
    if output_axis < 0 or output_axis >= len(shape):
        return None
    output_channels = int(shape[output_axis])
    if output_channels <= 0 or output_contract.shape[-1] != output_channels:
        return None
    if bias is not None and bias.shape != (output_channels,):
        return None
    return output_channels


def _bias_contracts_compatible(
    first: _LinearState,
    second: _LinearState,
) -> bool:
    """Return whether horizontal FC biases can share one result contract."""

    if first.bias_contract is None or second.bias_contract is None:
        return True
    return first.bias_contract == second.bias_contract


def _has_no_fused_activation(operator: Any, activation_none: int) -> bool:
    """Return whether an operator has no observable fused activation."""

    options = getattr(operator, "builtinOptions", None)
    fused_activation = int(
        getattr(options, "fusedActivationFunction", activation_none) or 0
    )
    return fused_activation == int(activation_none)


def _same_linear_options(first: Any, second: Any) -> bool:
    """Compare serialized option and auxiliary state for horizontal FC fusion."""

    return (
        int(getattr(first, "builtinOptionsType", 0) or 0)
        == int(getattr(second, "builtinOptionsType", 0) or 0)
        and freeze_object(getattr(first, "builtinOptions", None))
        == freeze_object(getattr(second, "builtinOptions", None))
        and int(getattr(first, "builtinOptions2Type", 0) or 0)
        == int(getattr(second, "builtinOptions2Type", 0) or 0)
        and freeze_object(getattr(first, "builtinOptions2", None))
        == freeze_object(getattr(second, "builtinOptions2", None))
        and int(getattr(first, "customOptionsFormat", 0) or 0)
        == int(getattr(second, "customOptionsFormat", 0) or 0)
        and freeze_object(getattr(first, "customOptions", None))
        == freeze_object(getattr(second, "customOptions", None))
    )


def _optional_input(inputs: Sequence[int], position: int) -> int | None:
    """Return one optional input tensor index or None when omitted."""

    if position >= len(inputs):
        return None
    value = int(inputs[position])
    return None if value == OPTIONAL_TENSOR_INDEX else value


def _with_optional_bias_slot(inputs: Sequence[int], bias_position: int) -> list[int]:
    """Return mutable inputs extended through the optional bias position."""

    result = [int(index) for index in inputs]
    while len(result) <= bias_position:
        result.append(OPTIONAL_TENSOR_INDEX)
    return result


def _channel_vector(
    data: np.ndarray[Any, Any],
    output_shape: Sequence[int],
    channels: int,
) -> np.ndarray[Any, Any] | None:
    """Extract a scalar or last-axis broadcast value as a channel vector."""

    shape = tuple(int(dimension) for dimension in np.asarray(data).shape)
    target = tuple(int(dimension) for dimension in output_shape)
    if not target or channels <= 0 or target[-1] != channels:
        return None
    if len(shape) > len(target):
        return None
    aligned = (1,) * (len(target) - len(shape)) + shape
    if any(dimension != 1 for dimension in aligned[:-1]):
        return None
    if aligned[-1] not in {1, channels}:
        return None
    flattened = np.asarray(data, dtype=np.float32).reshape(-1)
    if flattened.size == 1:
        return np.full(channels, flattened[0], dtype=np.float32)
    if flattened.size != channels:
        return None
    return flattened.copy()


def _reshape_for_axis(
    values: np.ndarray[Any, Any],
    target_shape: Sequence[int],
    axis: int,
) -> np.ndarray[Any, Any]:
    """Reshape one channel vector for broadcasting over a weight tensor."""

    rank = len(tuple(target_shape))
    normalized_axis = int(axis)
    if normalized_axis < 0:
        normalized_axis += rank
    shape = [1] * rank
    shape[normalized_axis] = int(values.size)
    return np.asarray(values, dtype=np.float32).reshape(shape)


def _bias_contract(tensor_type: int, channels: int) -> TensorContract:
    """Create the static dense contract for a synthesized FLOAT32 bias."""

    return TensorContract(
        tensor_type=int(tensor_type),
        shape=(int(channels),),
        shape_signature=(int(channels),),
    )


def _fused_name(graph: CircleGraph, tensor_index: int | None, suffix: str) -> str:
    """Create a readable base name for one fused parameter constant."""

    base = "linear"
    if tensor_index is not None:
        candidate = graph.tensor_name(tensor_index)
        if candidate:
            base = candidate
    return f"{base}_{suffix}_fused"


def _lookup_name(mapping: Mapping[str, int], value: int) -> str | None:
    """Return the symbolic key associated with one builtin code."""

    for name, code in mapping.items():
        if int(code) == int(value):
            return name
    return None


def _unique_indices(values: Sequence[int]) -> tuple[int, ...]:
    """Return unique non-optional tensor indices in stable order."""

    return tuple(
        dict.fromkeys(
            int(value) for value in values if int(value) != OPTIONAL_TENSOR_INDEX
        )
    )

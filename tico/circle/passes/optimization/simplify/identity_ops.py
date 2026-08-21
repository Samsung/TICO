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
from typing import Mapping

import numpy as np

from tico.circle._object import ObjectFactory
from tico.circle.document import CircleDocument
from tico.circle.graph import as_indices, as_list
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult
from tico.circle.passes.optimization._utils import (
    constant_represents_real_zero,
    contract_is_fully_static,
    decode_constant_value,
    decode_integer_vector,
    normalize_axis,
    operator_builtin_code,
    operator_is_plain,
    OptimizationSchemaResolver,
    strided_slice_view_shape,
    tensor_contract,
)
from tico.circle.passes.rules import (
    CircleRewriteRule,
    CircleRulePass,
    RewriteApplication,
    RewriteDiagnostic,
    RewritePlan,
)
from tico.circle.rewrite import replace_tensor_uses
from tico.circle.value import TensorValueCodec


@dataclass(frozen=True, kw_only=True)
class _BypassPlan(RewritePlan):
    """Carry a replacement input for one removable operator output."""

    replacement_input: int
    diagnostic_code: str
    diagnostic_message: str


class _BypassRule(CircleRewriteRule[_BypassPlan]):
    """Remove one operator after redirecting all output uses to an input."""

    def _apply_bypass(
        self,
        document: CircleDocument,
        plan: _BypassPlan,
    ) -> RewriteApplication:
        """Redirect one output and delete the now-unnecessary operator."""

        old_output = plan.anchor.outputs[0]
        stats = replace_tensor_uses(
            document.model,
            subgraph_index=plan.subgraph_index,
            old_tensor_index=old_output,
            new_tensor_index=plan.replacement_input,
        )
        operators = as_list(
            getattr(document.subgraph(plan.subgraph_index), "operators", None)
        )
        del operators[plan.anchor_operator_index]
        document.subgraph(plan.subgraph_index).operators = operators
        return RewriteApplication(
            changes=1 + stats.remapped_references,
            diagnostics=(
                RewriteDiagnostic(
                    code=plan.diagnostic_code,
                    message=plan.diagnostic_message,
                    object_path=(
                        f"subgraphs[{plan.subgraph_index}].operators"
                        f"[{plan.anchor_operator_index}]"
                    ),
                ),
            ),
        )

    def apply(self, document, plan, context):
        """Apply a validated bypass plan."""

        del context
        return self._apply_bypass(document, plan)


class _RemoveAddZeroRule(_BypassRule):
    """Remove ADD when exactly one operand is exact real zero."""

    def __init__(self, schema: OptimizationSchemaResolver, codec: TensorValueCodec):
        """Bind ADD schema and constant decoding services."""

        self.schema = schema
        self.codec = codec
        self.source_code = schema.builtin_code("ADD")

    def match(self, document, graph, operator_index, context):
        """Match an activation-free ADD whose output contract equals its data input."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.source_code:
            return None
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if len(inputs) != 2 or len(outputs) != 1 or not operator_is_plain(operator):
            return None
        options = getattr(operator, "builtinOptions", None)
        if options is None:
            return None
        if int(getattr(options, "fusedActivationFunction", 0) or 0) != (
            self.schema.activation_none
        ):
            return None
        if bool(getattr(options, "potScaleInt16", False)):
            return None
        constant_positions = [
            position
            for position, tensor_index in enumerate(inputs)
            if graph.is_constant(tensor_index)
        ]
        if len(constant_positions) != 1:
            return None
        constant_position = constant_positions[0]
        data_position = 1 - constant_position
        data_input = inputs[data_position]
        data_contract = tensor_contract(graph, data_input)
        output_contract = tensor_contract(graph, outputs[0])
        if data_contract != output_contract:
            return None
        if data_contract.quantization is not None:
            return None
        constant = decode_constant_value(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[constant_position],
        )
        if constant is None or constant.tensor_type != data_contract.tensor_type:
            return None
        try:
            broadcast_shape = tuple(
                np.broadcast_shapes(data_contract.shape, constant.shape)
            )
        except ValueError:
            return None
        if broadcast_shape != output_contract.shape:
            return None
        if not constant_represents_real_zero(
            constant,
            reference_contract=data_contract,
        ):
            return None
        return _BypassPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*inputs, *outputs),
            replacement_input=data_input,
            diagnostic_code="REMOVE_ADD_ZERO",
            diagnostic_message="Removed ADD with an exact real-zero operand.",
        )


class _RemoveSameTypeCastRule(_BypassRule):
    """Remove CAST when input and output contracts are identical."""

    def __init__(self, schema: OptimizationSchemaResolver):
        """Bind the CAST source opcode."""

        self.source_code = schema.builtin_code("CAST")

    def match(self, document, graph, operator_index, context):
        """Match a CAST that cannot change type, shape, qparams, or metadata."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.source_code:
            return None
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if len(inputs) != 1 or len(outputs) != 1 or not operator_is_plain(operator):
            return None
        input_contract = tensor_contract(graph, inputs[0])
        output_contract = tensor_contract(graph, outputs[0])
        if input_contract != output_contract:
            return None
        options = getattr(operator, "builtinOptions", None)
        if options is None:
            return None
        if hasattr(options, "inDataType") and int(options.inDataType) != (
            input_contract.tensor_type
        ):
            return None
        if hasattr(options, "outDataType") and int(options.outDataType) != (
            output_contract.tensor_type
        ):
            return None
        return _BypassPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*inputs, *outputs),
            replacement_input=inputs[0],
            diagnostic_code="REMOVE_SAME_TYPE_CAST",
            diagnostic_message=(
                "Removed CAST with identical input and output contracts."
            ),
        )


class _RemoveFullRangeSliceRule(_BypassRule):
    """Remove SLICE when it selects the complete input tensor."""

    def __init__(self, schema: OptimizationSchemaResolver, codec: TensorValueCodec):
        """Bind SLICE schema and constant decoding services."""

        self.codec = codec
        self.source_code = schema.builtin_code("SLICE")

    def match(self, document, graph, operator_index, context):
        """Match a static full-range SLICE with an identical output contract."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.source_code:
            return None
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if len(inputs) != 3 or len(outputs) != 1 or not operator_is_plain(operator):
            return None
        input_contract = tensor_contract(graph, inputs[0])
        if input_contract != tensor_contract(graph, outputs[0]):
            return None
        begin = decode_integer_vector(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[1],
            expected_count=input_contract.rank,
        )
        size = decode_integer_vector(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[2],
            expected_count=input_contract.rank,
        )
        if begin is None or size is None or any(value != 0 for value in begin):
            return None
        signature = input_contract.shape_signature
        for axis, value in enumerate(size):
            dynamic = signature is not None and signature[axis] == -1
            allowed_sizes = {-1} if dynamic else {-1, input_contract.shape[axis]}
            if value not in allowed_sizes:
                return None
        return _BypassPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*inputs, *outputs),
            replacement_input=inputs[0],
            diagnostic_code="REMOVE_FULL_RANGE_SLICE",
            diagnostic_message="Removed SLICE that selected the full input tensor.",
        )


class _RemoveIdentityStridedSliceRule(_BypassRule):
    """Remove STRIDED_SLICE when it is a rank-preserving full slice."""

    def __init__(self, schema: OptimizationSchemaResolver, codec: TensorValueCodec):
        """Bind STRIDED_SLICE schema and constant decoding services."""

        self.codec = codec
        self.source_code = schema.builtin_code("STRIDED_SLICE")

    def match(self, document, graph, operator_index, context):
        """Match a full STRIDED_SLICE whose output contract is identical."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.source_code:
            return None
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if len(inputs) != 4 or len(outputs) != 1 or not operator_is_plain(operator):
            return None
        input_contract = tensor_contract(graph, inputs[0])
        if input_contract != tensor_contract(graph, outputs[0]):
            return None
        if not contract_is_fully_static(input_contract):
            return None
        begin = decode_integer_vector(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[1],
        )
        end = decode_integer_vector(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[2],
        )
        strides = decode_integer_vector(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[3],
        )
        if begin is None or end is None or strides is None:
            return None
        options = getattr(operator, "builtinOptions", None)
        if options is None:
            return None
        expected = strided_slice_view_shape(
            input_contract.shape,
            begin,
            end,
            strides,
            begin_mask=int(getattr(options, "beginMask", 0) or 0),
            end_mask=int(getattr(options, "endMask", 0) or 0),
            ellipsis_mask=int(getattr(options, "ellipsisMask", 0) or 0),
            new_axis_mask=int(getattr(options, "newAxisMask", 0) or 0),
            shrink_axis_mask=int(getattr(options, "shrinkAxisMask", 0) or 0),
        )
        if expected != input_contract.shape:
            return None
        return _BypassPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*inputs, *outputs),
            replacement_input=inputs[0],
            diagnostic_code="REMOVE_IDENTITY_STRIDED_SLICE",
            diagnostic_message=(
                "Removed STRIDED_SLICE that selected the complete input tensor."
            ),
        )


class _RemoveSingleOutputSplitRule(_BypassRule):
    """Remove SPLIT or SPLIT_V when exactly one complete output is requested."""

    def __init__(self, schema: OptimizationSchemaResolver, codec: TensorValueCodec):
        """Bind split opcodes and constant decoding services."""

        self.codec = codec
        self.split_code = schema.builtin_code("SPLIT")
        self.split_v_code = schema.builtin_code("SPLIT_V")

    def match(self, document, graph, operator_index, context):
        """Match one-output split forms whose output contract equals the input."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        builtin_code = operator_builtin_code(document.model, operator)
        if builtin_code not in {self.split_code, self.split_v_code}:
            return None
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if len(outputs) != 1 or not operator_is_plain(operator):
            return None
        options = getattr(operator, "builtinOptions", None)
        if options is None:
            return None
        if int(getattr(options, "numSplits", 0) or 0) != 1:
            return None
        if builtin_code == self.split_code:
            if len(inputs) != 2:
                return None
            axis_input, data_input = inputs
            size_input = None
        else:
            if len(inputs) != 3:
                return None
            data_input, size_input, axis_input = inputs
        data_contract = tensor_contract(graph, data_input)
        if data_contract != tensor_contract(graph, outputs[0]):
            return None
        axis_values = decode_integer_vector(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=axis_input,
            expected_count=1,
        )
        if axis_values is None:
            return None
        axis = normalize_axis(axis_values[0], data_contract.rank)
        if axis is None:
            return None
        if size_input is not None:
            size_values = decode_integer_vector(
                self.codec,
                document.model,
                subgraph_index=graph.subgraph_index,
                tensor_index=size_input,
                expected_count=1,
            )
            if size_values is None:
                return None
            signature = data_contract.shape_signature
            dynamic = signature is not None and signature[axis] == -1
            allowed_sizes = {-1} if dynamic else {-1, data_contract.shape[axis]}
            if size_values[0] not in allowed_sizes:
                return None
        return _BypassPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*inputs, *outputs),
            replacement_input=data_input,
            diagnostic_code="REMOVE_SINGLE_OUTPUT_SPLIT",
            diagnostic_message="Removed one-output split that returned the full input.",
        )


class EliminateIdentityOpsPass(CirclePass):
    """Remove operators that provably preserve the complete tensor contract."""

    def __init__(
        self,
        *,
        maximum_rewrites: int = 10_000,
        builtin_codes: Mapping[str, int] | None = None,
        activation_none: int | None = None,
        codec: TensorValueCodec | None = None,
        object_factory: ObjectFactory | None = None,
    ) -> None:
        """Create no-op rules with schema or test enum mappings."""

        self.maximum_rewrites = int(maximum_rewrites)
        if self.maximum_rewrites <= 0:
            raise ValueError("maximum_rewrites must be positive.")
        self.codec = codec or TensorValueCodec()
        self.schema = OptimizationSchemaResolver(
            builtin_codes=builtin_codes,
            activation_none=activation_none,
            object_factory=object_factory,
        )
        self.rules = (
            _RemoveAddZeroRule(self.schema, self.codec),
            _RemoveSameTypeCastRule(self.schema),
            _RemoveFullRangeSliceRule(self.schema, self.codec),
            _RemoveIdentityStridedSliceRule(self.schema, self.codec),
            _RemoveSingleOutputSplitRule(self.schema, self.codec),
        )

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Remove supported no-op operators to a fixed point."""

        return CircleRulePass(
            self.rules,
            maximum_rewrites=self.maximum_rewrites,
        ).run(document, context)


# Backward-compatible class spelling. The implementation lives under the
# semantic identity-elimination name.
RemoveNoOpOperatorsPass = EliminateIdentityOpsPass

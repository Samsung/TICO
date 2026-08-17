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

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from tico.circle._object import clone_object, ObjectFactory
from tico.circle.analysis import TensorContract
from tico.circle.builder import CircleBuilder
from tico.circle.document import CircleDocument
from tico.circle.errors import CircleRewriteError
from tico.circle.graph import as_indices, as_list
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult
from tico.circle.passes.optimization._utils import (
    AppendedObjectCheckpoint,
    contract_is_fully_static,
    decode_constant_value,
    decode_integer_vector,
    infer_reshape_shape,
    normalize_axis,
    operator_builtin_code,
    operator_is_plain,
    OptimizationSchemaResolver,
    output_shape_matches_transpose,
    tensor_contract,
    tensor_is_signature_bound,
    view_contracts_compatible,
)
from tico.circle.passes.rules import (
    CircleRewriteRule,
    CircleRulePass,
    OperatorSnapshot,
    RewriteApplication,
    RewriteDiagnostic,
    RewritePlan,
)
from tico.circle.rewrite import replace_tensor_uses
from tico.circle.value import TensorValue, TensorValueCodec


@dataclass(frozen=True, kw_only=True)
class _BypassViewPlan(RewritePlan):
    """Carry a replacement tensor for one removable view operator."""

    replacement_input: int
    diagnostic_code: str
    diagnostic_message: str


@dataclass(frozen=True, kw_only=True)
class _ProducerPlan(RewritePlan):
    """Capture a second producer operator participating in a two-op rewrite."""

    producer: OperatorSnapshot

    def validate(self, document: CircleDocument) -> None:
        """Validate the anchor and the separately captured producer operator."""

        super().validate(document)
        try:
            current = OperatorSnapshot.capture(
                document,
                subgraph_index=self.subgraph_index,
                operator_index=self.producer.operator_index,
            )
        except (IndexError, CircleRewriteError) as error:
            raise CircleRewriteError(
                "View rewrite plan is stale because its producer no longer exists."
            ) from error
        if current != self.producer:
            raise CircleRewriteError(
                "View rewrite plan is stale because its producer changed."
            )


@dataclass(frozen=True, kw_only=True)
class _ProducerBypassViewPlan(_ProducerPlan):
    """Carry a replacement tensor for a bypass involving a producer op."""

    replacement_input: int
    diagnostic_code: str
    diagnostic_message: str


@dataclass(frozen=True, kw_only=True)
class _ComposeReshapePlan(_ProducerPlan):
    """Carry the original data input into a consecutive RESHAPE composition."""

    data_input: int


@dataclass(frozen=True, kw_only=True)
class _ComposeTransposePlan(_ProducerPlan):
    """Carry a composed permutation into a consecutive TRANSPOSE rewrite."""

    data_input: int
    permutation: tuple[int, ...]
    permutation_name: str


@dataclass(frozen=True, kw_only=True)
class _CommuteElementwisePlan(_ProducerPlan):
    """Carry contracts and tensor positions for moving RESHAPE after elementwise."""

    data_input: int
    intermediate_tensor: int
    anchor_output: int
    anchor_input_position: int
    intermediate_contract: TensorContract


@dataclass(frozen=True, kw_only=True)
class _CommuteMeanPlan(_ProducerPlan):
    """Carry contracts and target shape for moving RESHAPE after MEAN."""

    data_input: int
    axis_input: int
    intermediate_tensor: int
    anchor_output: int
    intermediate_contract: TensorContract
    output_shape: tuple[int, ...]
    output_name: str


class _ViewRule(CircleRewriteRule[RewritePlan]):
    """Share schema and tensor value services across view simplification rules."""

    def __init__(
        self,
        schema: OptimizationSchemaResolver,
        codec: TensorValueCodec,
    ) -> None:
        """Bind schema and constant decoding services."""

        self.schema = schema
        self.codec = codec
        self.reshape_code = schema.builtin_code("RESHAPE")
        self.transpose_code = schema.builtin_code("TRANSPOSE")
        self.reshape_options_type = schema.builtin_options_type("ReshapeOptions")
        self.int32_type = schema.tensor_type("INT32")

    @staticmethod
    def _apply_bypass(
        document: CircleDocument,
        plan: _BypassViewPlan | _ProducerBypassViewPlan,
    ) -> RewriteApplication:
        """Redirect output uses and leave dead operators for an external DCE pass."""

        old_output = plan.anchor.outputs[0]
        stats = replace_tensor_uses(
            document.model,
            subgraph_index=plan.subgraph_index,
            old_tensor_index=old_output,
            new_tensor_index=plan.replacement_input,
        )
        return RewriteApplication(
            changes=stats.remapped_references,
            diagnostics=(
                RewriteDiagnostic(
                    code=plan.diagnostic_code,
                    message=plan.diagnostic_message,
                ),
            ),
        )


class _RemoveIdentityReshapeRule(_ViewRule):
    """Bypass RESHAPE when the complete tensor contract is unchanged."""

    def match(self, document, graph, operator_index, context):
        """Match a static identity RESHAPE."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.reshape_code:
            return None
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if not inputs or len(outputs) != 1 or not operator_is_plain(operator):
            return None
        if not _tensor_has_uses(document, graph, outputs[0]):
            return None
        input_contract = tensor_contract(graph, inputs[0])
        if input_contract != tensor_contract(graph, outputs[0]):
            return None
        if not contract_is_fully_static(input_contract):
            return None
        target = _reshape_target(self.codec, document, graph, operator)
        if target != input_contract.shape:
            return None
        return _BypassViewPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*inputs, *outputs),
            replacement_input=inputs[0],
            diagnostic_code="BYPASS_IDENTITY_RESHAPE",
            diagnostic_message="Bypassed RESHAPE with an identical tensor contract.",
        )

    def apply(self, document, plan, context):
        """Bypass one matched identity RESHAPE."""

        del context
        return self._apply_bypass(document, plan)


class _ComposeReshapeChainRule(_ViewRule):
    """Bypass the first RESHAPE in a consecutive RESHAPE chain."""

    def match(self, document, graph, operator_index, context):
        """Match two storage-compatible RESHAPE operators."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.reshape_code:
            return None
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if not inputs or len(outputs) != 1 or not operator_is_plain(operator):
            return None
        producer_index = graph.producer(inputs[0])
        if producer_index is None or producer_index >= operator_index:
            return None
        producer = as_list(graph.subgraph.operators)[producer_index]
        if operator_builtin_code(document.model, producer) != self.reshape_code:
            return None
        producer_inputs = as_indices(producer.inputs)
        producer_outputs = as_indices(producer.outputs)
        if (
            not producer_inputs
            or len(producer_outputs) != 1
            or producer_outputs[0] != inputs[0]
            or not operator_is_plain(producer)
        ):
            return None
        source = tensor_contract(graph, producer_inputs[0])
        middle = tensor_contract(graph, producer_outputs[0])
        output = tensor_contract(graph, outputs[0])
        if not (
            view_contracts_compatible(source, middle)
            and view_contracts_compatible(middle, output)
        ):
            return None
        if _reshape_target(self.codec, document, graph, producer) != middle.shape:
            return None
        if _reshape_target(self.codec, document, graph, operator) != output.shape:
            return None
        return _ComposeReshapePlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*producer_inputs, *producer_outputs, *inputs, *outputs),
            producer=OperatorSnapshot.capture(
                document,
                subgraph_index=graph.subgraph_index,
                operator_index=producer_index,
            ),
            data_input=producer_inputs[0],
        )

    def apply(self, document, plan, context):
        """Reconnect the second RESHAPE directly to the original data input."""

        del context
        operator = as_list(document.subgraph(plan.subgraph_index).operators)[
            plan.anchor_operator_index
        ]
        inputs = as_indices(operator.inputs)
        inputs[0] = plan.data_input
        operator.inputs = inputs
        return RewriteApplication(
            changes=1,
            diagnostics=(
                RewriteDiagnostic(
                    code="COMPOSE_RESHAPE_CHAIN",
                    message="Composed consecutive RESHAPE operators.",
                ),
            ),
        )


class _RemoveIdentityTransposeRule(_ViewRule):
    """Bypass TRANSPOSE when its permutation and tensor contract are identities."""

    def match(self, document, graph, operator_index, context):
        """Match an identity permutation with an identical output contract."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.transpose_code:
            return None
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if len(inputs) != 2 or len(outputs) != 1 or not operator_is_plain(operator):
            return None
        if not _tensor_has_uses(document, graph, outputs[0]):
            return None
        input_contract = tensor_contract(graph, inputs[0])
        if input_contract != tensor_contract(graph, outputs[0]):
            return None
        permutation = decode_integer_vector(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[1],
            expected_count=input_contract.rank,
        )
        if permutation != tuple(range(input_contract.rank)):
            return None
        return _BypassViewPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*inputs, *outputs),
            replacement_input=inputs[0],
            diagnostic_code="BYPASS_IDENTITY_TRANSPOSE",
            diagnostic_message="Bypassed TRANSPOSE with an identity permutation.",
        )

    def apply(self, document, plan, context):
        """Bypass one matched identity TRANSPOSE."""

        del context
        return self._apply_bypass(document, plan)


class _RemoveInverseTransposePairRule(_ViewRule):
    """Bypass an inverse TRANSPOSE pair without deleting its operators."""

    def match(self, document, graph, operator_index, context):
        """Match consecutive transposes whose composed permutation is identity."""

        del context
        match = _match_transpose_chain(
            self.codec,
            document,
            graph,
            operator_index,
            self.transpose_code,
        )
        if match is None:
            return None
        producer_index, producer_inputs, inputs, outputs, composed = match
        if composed != tuple(range(len(composed))):
            return None
        if not _tensor_has_uses(document, graph, outputs[0]):
            return None
        if tensor_contract(graph, producer_inputs[0]) != tensor_contract(
            graph, outputs[0]
        ):
            return None
        return _ProducerBypassViewPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*producer_inputs, *inputs, *outputs),
            producer=OperatorSnapshot.capture(
                document,
                subgraph_index=graph.subgraph_index,
                operator_index=producer_index,
            ),
            replacement_input=producer_inputs[0],
            diagnostic_code="BYPASS_INVERSE_TRANSPOSE_PAIR",
            diagnostic_message="Bypassed an inverse pair of TRANSPOSE operations.",
        )

    def apply(self, document, plan, context):
        """Bypass the second transpose in an inverse pair."""

        del context
        return self._apply_bypass(document, plan)


class _ComposeTransposeChainRule(_ViewRule):
    """Compose two consecutive TRANSPOSE permutations into the second operator."""

    def match(self, document, graph, operator_index, context):
        """Match a non-identity consecutive TRANSPOSE chain."""

        del context
        match = _match_transpose_chain(
            self.codec,
            document,
            graph,
            operator_index,
            self.transpose_code,
        )
        if match is None:
            return None
        producer_index, producer_inputs, inputs, outputs, composed = match
        if composed == tuple(range(len(composed))):
            return None
        source = tensor_contract(graph, producer_inputs[0])
        output = tensor_contract(graph, outputs[0])
        if not view_contracts_compatible(source, output):
            return None
        if not output_shape_matches_transpose(source.shape, output.shape, composed):
            return None
        return _ComposeTransposePlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*producer_inputs, *inputs, *outputs),
            producer=OperatorSnapshot.capture(
                document,
                subgraph_index=graph.subgraph_index,
                operator_index=producer_index,
            ),
            data_input=producer_inputs[0],
            permutation=composed,
            permutation_name=f"transpose_{operator_index}_permutation",
        )

    def apply(self, document, plan, context):
        """Replace the second TRANSPOSE with the composed permutation."""

        del context
        checkpoint = AppendedObjectCheckpoint.capture(
            document,
            subgraph_index=plan.subgraph_index,
        )
        subgraph = document.subgraph(plan.subgraph_index)
        operators = as_list(getattr(subgraph, "operators", None))
        original_operator = clone_object(operators[plan.anchor_operator_index])
        try:
            builder = CircleBuilder(
                document,
                subgraph_index=plan.subgraph_index,
                codec=self.codec,
                object_factory=self.schema.object_factory,
            )
            permutation_input = builder.add_constant(
                plan.permutation_name,
                TensorValue.from_values(
                    self.int32_type,
                    np.asarray(plan.permutation, dtype=np.int32),
                    dtype=np.int32,
                ),
            )
            operators[plan.anchor_operator_index].inputs = [
                plan.data_input,
                permutation_input,
            ]
        except Exception:
            checkpoint.rollback(document)
            restored_operators = as_list(
                getattr(document.subgraph(plan.subgraph_index), "operators", None)
            )
            restored_operators[plan.anchor_operator_index] = original_operator
            document.subgraph(plan.subgraph_index).operators = restored_operators
            raise
        return RewriteApplication(
            changes=1,
            diagnostics=(
                RewriteDiagnostic(
                    code="COMPOSE_TRANSPOSE_CHAIN",
                    message="Composed consecutive TRANSPOSE permutations.",
                ),
            ),
        )


class _CommuteReshapeThroughElementwiseRule(_ViewRule):
    """Move RESHAPE after a shape-preserving unary or scalar binary operator."""

    def __init__(self, schema: OptimizationSchemaResolver, codec: TensorValueCodec):
        """Build optional elementwise builtin-code sets from the active schema."""

        super().__init__(schema, codec)
        unary_names = (
            "ABS",
            "CAST",
            "CEIL",
            "COS",
            "ELU",
            "EXP",
            "FLOOR",
            "LEAKY_RELU",
            "LOG",
            "LOGICAL_NOT",
            "LOGISTIC",
            "NEG",
            "RELU",
            "RELU6",
            "RELU_N1_TO_1",
            "RSQRT",
            "SIN",
            "SQRT",
            "SQUARE",
            "TANH",
            "ZEROS_LIKE",
        )
        binary_names = (
            "ADD",
            "DIV",
            "MAXIMUM",
            "MINIMUM",
            "MUL",
            "POW",
            "SQUARED_DIFFERENCE",
            "SUB",
        )
        self.unary_codes = _available_codes(schema, unary_names)
        self.binary_codes = _available_codes(schema, binary_names)

    def match(self, document, graph, operator_index, context):
        """Match elementwise operation fed by a single-consumer RESHAPE output."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        builtin_code = operator_builtin_code(document.model, operator)
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if len(outputs) != 1 or not operator_is_plain(operator):
            return None
        broadcast_value = None
        if builtin_code in self.unary_codes:
            if len(inputs) != 1:
                return None
            reshape_position = 0
        elif builtin_code in self.binary_codes:
            if len(inputs) != 2:
                return None
            reshape_positions = [
                position
                for position, tensor_index in enumerate(inputs)
                if graph.producer(tensor_index) is not None
                and operator_builtin_code(
                    document.model,
                    as_list(graph.subgraph.operators)[graph.producer(tensor_index)],
                )
                == self.reshape_code
            ]
            if len(reshape_positions) != 1:
                return None
            reshape_position = reshape_positions[0]
            scalar_position = 1 - reshape_position
            broadcast_value = decode_constant_value(
                self.codec,
                document.model,
                subgraph_index=graph.subgraph_index,
                tensor_index=inputs[scalar_position],
            )
            if broadcast_value is None or broadcast_value.element_count != 1:
                return None
        else:
            return None

        intermediate = inputs[reshape_position]
        producer_index = graph.producer(intermediate)
        if (
            producer_index is None
            or producer_index >= operator_index
            or graph.consumers(intermediate) != (operator_index,)
        ):
            return None
        if _tensor_is_boundary(document, graph, intermediate):
            return None
        producer = as_list(graph.subgraph.operators)[producer_index]
        if operator_builtin_code(document.model, producer) != self.reshape_code:
            return None
        if not operator_is_plain(producer):
            return None
        producer_inputs = as_indices(producer.inputs)
        producer_outputs = as_indices(producer.outputs)
        if not producer_inputs or producer_outputs != [intermediate]:
            return None
        source = tensor_contract(graph, producer_inputs[0])
        reshaped = tensor_contract(graph, intermediate)
        output = tensor_contract(graph, outputs[0])
        if _reshape_target(self.codec, document, graph, producer) != reshaped.shape:
            return None
        if reshaped.shape != output.shape:
            return None
        if broadcast_value is not None:
            try:
                source_broadcast = tuple(
                    np.broadcast_shapes(source.shape, broadcast_value.shape)
                )
                reshaped_broadcast = tuple(
                    np.broadcast_shapes(reshaped.shape, broadcast_value.shape)
                )
            except ValueError:
                return None
            if source_broadcast != source.shape or reshaped_broadcast != reshaped.shape:
                return None
        if not view_contracts_compatible(source, reshaped):
            return None
        new_intermediate = replace(
            output,
            shape=source.shape,
            shape_signature=source.shape_signature,
        )
        if not view_contracts_compatible(new_intermediate, output):
            return None
        return _CommuteElementwisePlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*producer_inputs, intermediate, *inputs, *outputs),
            producer=OperatorSnapshot.capture(
                document,
                subgraph_index=graph.subgraph_index,
                operator_index=producer_index,
            ),
            data_input=producer_inputs[0],
            intermediate_tensor=intermediate,
            anchor_output=outputs[0],
            anchor_input_position=reshape_position,
            intermediate_contract=new_intermediate,
        )

    def apply(self, document, plan, context):
        """Swap RESHAPE and elementwise dataflow while preserving operator options."""

        del context
        subgraph = document.subgraph(plan.subgraph_index)
        operators = as_list(subgraph.operators)
        producer_index = plan.producer.operator_index
        reshape = operators[producer_index]
        elementwise = operators[plan.anchor_operator_index]

        original_producer = clone_object(reshape)
        original_anchor = clone_object(elementwise)
        tensors = as_list(getattr(subgraph, "tensors", None))
        original_intermediate = clone_object(tensors[plan.intermediate_tensor])
        try:
            plan.intermediate_contract.apply_to_tensor(
                tensors[plan.intermediate_tensor],
                factory=self.schema.object_factory,
            )
            elementwise_inputs = as_indices(elementwise.inputs)
            elementwise_inputs[plan.anchor_input_position] = plan.data_input
            elementwise.inputs = elementwise_inputs
            elementwise.outputs = [plan.intermediate_tensor]

            reshape_inputs = as_indices(reshape.inputs)
            reshape_inputs[0] = plan.intermediate_tensor
            reshape.inputs = reshape_inputs
            reshape.outputs = [plan.anchor_output]

            operators[producer_index] = elementwise
            operators[plan.anchor_operator_index] = reshape
            subgraph.operators = operators
        except Exception:
            tensors[plan.intermediate_tensor] = original_intermediate
            operators[producer_index] = original_producer
            operators[plan.anchor_operator_index] = original_anchor
            subgraph.tensors = tensors
            subgraph.operators = operators
            raise
        return RewriteApplication(
            changes=3,
            diagnostics=(
                RewriteDiagnostic(
                    code="COMMUTE_RESHAPE_ELEMENTWISE",
                    message="Moved RESHAPE after a shape-preserving elementwise op.",
                ),
            ),
        )


class _CommuteReshapeThroughMeanRule(_ViewRule):
    """Move rank-preserving RESHAPE after one-axis MEAN with keep_dims enabled."""

    def __init__(self, schema: OptimizationSchemaResolver, codec: TensorValueCodec):
        """Bind the optional MEAN builtin code."""

        super().__init__(schema, codec)
        self.mean_code = schema.builtin_code("MEAN")

    def match(self, document, graph, operator_index, context):
        """Match MEAN when the RESHAPE leaves all dimensions through its axis intact."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.mean_code:
            return None
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if len(inputs) != 2 or len(outputs) != 1 or not operator_is_plain(operator):
            return None
        options = getattr(operator, "builtinOptions", None)
        if options is None or not bool(getattr(options, "keepDims", False)):
            return None
        intermediate = inputs[0]
        producer_index = graph.producer(intermediate)
        if (
            producer_index is None
            or producer_index >= operator_index
            or graph.consumers(intermediate) != (operator_index,)
        ):
            return None
        if _tensor_is_boundary(document, graph, intermediate):
            return None
        producer = as_list(graph.subgraph.operators)[producer_index]
        if operator_builtin_code(document.model, producer) != self.reshape_code:
            return None
        if not operator_is_plain(producer):
            return None
        producer_inputs = as_indices(producer.inputs)
        producer_outputs = as_indices(producer.outputs)
        if not producer_inputs or producer_outputs != [intermediate]:
            return None
        source = tensor_contract(graph, producer_inputs[0])
        reshaped = tensor_contract(graph, intermediate)
        output = tensor_contract(graph, outputs[0])
        if _reshape_target(self.codec, document, graph, producer) != reshaped.shape:
            return None
        if source.rank != reshaped.rank or source.quantization is not None:
            return None
        if reshaped.quantization is not None or output.quantization is not None:
            return None
        if not view_contracts_compatible(source, reshaped):
            return None
        axes = decode_integer_vector(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[1],
            expected_count=1,
        )
        if axes is None:
            return None
        axis = normalize_axis(axes[0], reshaped.rank)
        if axis is None:
            return None
        if any(
            source.shape[index] != reshaped.shape[index] for index in range(axis + 1)
        ):
            return None
        expected_output = list(reshaped.shape)
        expected_output[axis] = 1
        if output.shape != tuple(expected_output) or not output.shape:
            return None
        new_shape = list(source.shape)
        new_shape[axis] = 1
        new_intermediate = replace(
            output,
            shape=tuple(new_shape),
            shape_signature=(
                None if source.shape_signature is None else tuple(new_shape)
            ),
        )
        if not view_contracts_compatible(new_intermediate, output):
            return None
        return _CommuteMeanPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*producer_inputs, intermediate, *inputs, *outputs),
            producer=OperatorSnapshot.capture(
                document,
                subgraph_index=graph.subgraph_index,
                operator_index=producer_index,
            ),
            data_input=producer_inputs[0],
            axis_input=inputs[1],
            intermediate_tensor=intermediate,
            anchor_output=outputs[0],
            intermediate_contract=new_intermediate,
            output_shape=output.shape,
            output_name=_tensor_name(graph, outputs[0]),
        )

    def apply(self, document, plan, context):
        """Swap RESHAPE and MEAN and update the final static reshape target."""

        del context
        subgraph = document.subgraph(plan.subgraph_index)
        operators = as_list(subgraph.operators)
        producer_index = plan.producer.operator_index
        reshape = operators[producer_index]
        mean = operators[plan.anchor_operator_index]

        checkpoint = AppendedObjectCheckpoint.capture(
            document,
            subgraph_index=plan.subgraph_index,
        )
        original_producer = clone_object(reshape)
        original_anchor = clone_object(mean)
        tensors = as_list(getattr(subgraph, "tensors", None))
        original_intermediate = clone_object(tensors[plan.intermediate_tensor])
        try:
            plan.intermediate_contract.apply_to_tensor(
                tensors[plan.intermediate_tensor],
                factory=self.schema.object_factory,
            )
            mean.inputs = [plan.data_input, plan.axis_input]
            mean.outputs = [plan.intermediate_tensor]

            builder = CircleBuilder(
                document,
                subgraph_index=plan.subgraph_index,
                codec=self.codec,
                object_factory=self.schema.object_factory,
            )
            shape_input = builder.add_constant(
                f"{plan.output_name}_shape",
                TensorValue.from_values(
                    self.int32_type,
                    np.asarray(plan.output_shape, dtype=np.int32),
                    dtype=np.int32,
                ),
            )
            reshape.inputs = [plan.intermediate_tensor, shape_input]
            reshape.outputs = [plan.anchor_output]
            options = self.schema.create("ReshapeOptions")
            options.newShape = list(plan.output_shape)
            reshape.builtinOptionsType = self.reshape_options_type
            reshape.builtinOptions = options

            operators[producer_index] = mean
            operators[plan.anchor_operator_index] = reshape
            subgraph.operators = operators
        except Exception:
            checkpoint.rollback(document)
            restored_subgraph = document.subgraph(plan.subgraph_index)
            restored_tensors = as_list(getattr(restored_subgraph, "tensors", None))
            restored_operators = as_list(getattr(restored_subgraph, "operators", None))
            restored_tensors[plan.intermediate_tensor] = original_intermediate
            restored_operators[producer_index] = original_producer
            restored_operators[plan.anchor_operator_index] = original_anchor
            restored_subgraph.tensors = restored_tensors
            restored_subgraph.operators = restored_operators
            raise
        return RewriteApplication(
            changes=4,
            diagnostics=(
                RewriteDiagnostic(
                    code="COMMUTE_RESHAPE_MEAN",
                    message="Moved RESHAPE after a compatible keep-dims MEAN.",
                ),
            ),
        )


class SimplifyViewOpsPass(CirclePass):
    """Simplify identity, chained, and movable RESHAPE and TRANSPOSE operations."""

    def __init__(
        self,
        *,
        maximum_rewrites: int = 10_000,
        builtin_codes: Mapping[str, int] | None = None,
        builtin_options_types: Mapping[str, int] | None = None,
        tensor_types: Mapping[str, int] | None = None,
        codec: TensorValueCodec | None = None,
        object_factory: ObjectFactory | None = None,
    ) -> None:
        """Create view rules with schema or test enum mappings."""

        self.maximum_rewrites = int(maximum_rewrites)
        if self.maximum_rewrites <= 0:
            raise ValueError("maximum_rewrites must be positive.")
        self.codec = codec or TensorValueCodec()
        self.schema = OptimizationSchemaResolver(
            builtin_codes=builtin_codes,
            builtin_options_types=builtin_options_types,
            tensor_types=tensor_types,
            object_factory=object_factory,
        )
        self.rules = (
            _RemoveIdentityReshapeRule(self.schema, self.codec),
            _ComposeReshapeChainRule(self.schema, self.codec),
            _RemoveIdentityTransposeRule(self.schema, self.codec),
            _RemoveInverseTransposePairRule(self.schema, self.codec),
            _ComposeTransposeChainRule(self.schema, self.codec),
            _CommuteReshapeThroughElementwiseRule(self.schema, self.codec),
            _CommuteReshapeThroughMeanRule(self.schema, self.codec),
        )

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Rewire supported view patterns without performing dead-code cleanup."""

        return CircleRulePass(
            self.rules,
            maximum_rewrites=self.maximum_rewrites,
        ).run(document, context)


def _reshape_target(
    codec: TensorValueCodec,
    document: CircleDocument,
    graph,
    operator,
) -> tuple[int, ...] | None:
    """Resolve the effective static RESHAPE target from its input or options."""

    inputs = as_indices(operator.inputs)
    requested: tuple[int, ...] | None = None
    if len(inputs) >= 2:
        requested = decode_integer_vector(
            codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[1],
        )
    options = getattr(operator, "builtinOptions", None)
    option_shape = tuple(
        int(value) for value in as_list(getattr(options, "newShape", None))
    )
    if requested is None:
        requested = option_shape or None
    elif option_shape and option_shape != requested:
        return None
    if requested is None:
        return None
    input_contract = tensor_contract(graph, inputs[0])
    return infer_reshape_shape(input_contract.shape, requested)


def _match_transpose_chain(
    codec: TensorValueCodec,
    document: CircleDocument,
    graph,
    operator_index: int,
    transpose_code: int,
):
    """Return consecutive transpose metadata and their composed permutation."""

    operator = as_list(graph.subgraph.operators)[operator_index]
    if operator_builtin_code(document.model, operator) != transpose_code:
        return None
    inputs = as_indices(operator.inputs)
    outputs = as_indices(operator.outputs)
    if len(inputs) != 2 or len(outputs) != 1 or not operator_is_plain(operator):
        return None
    producer_index = graph.producer(inputs[0])
    if producer_index is None or producer_index >= operator_index:
        return None
    producer = as_list(graph.subgraph.operators)[producer_index]
    if operator_builtin_code(document.model, producer) != transpose_code:
        return None
    producer_inputs = as_indices(producer.inputs)
    producer_outputs = as_indices(producer.outputs)
    if (
        len(producer_inputs) != 2
        or producer_outputs != [inputs[0]]
        or not operator_is_plain(producer)
    ):
        return None
    source = tensor_contract(graph, producer_inputs[0])
    first = decode_integer_vector(
        codec,
        document.model,
        subgraph_index=graph.subgraph_index,
        tensor_index=producer_inputs[1],
        expected_count=source.rank,
    )
    second = decode_integer_vector(
        codec,
        document.model,
        subgraph_index=graph.subgraph_index,
        tensor_index=inputs[1],
        expected_count=source.rank,
    )
    if first is None or second is None:
        return None
    if sorted(first) != list(range(source.rank)) or sorted(second) != list(
        range(source.rank)
    ):
        return None
    middle = tensor_contract(graph, inputs[0])
    output = tensor_contract(graph, outputs[0])
    if not output_shape_matches_transpose(source.shape, middle.shape, first):
        return None
    if not output_shape_matches_transpose(middle.shape, output.shape, second):
        return None
    composed = tuple(first[axis] for axis in second)
    return producer_index, producer_inputs, inputs, outputs, composed


def _available_codes(
    schema: OptimizationSchemaResolver,
    names: tuple[str, ...],
) -> frozenset[int]:
    """Return schema-supported builtin codes from a symbolic name list."""

    return frozenset(
        code for name in names if (code := schema.maybe_builtin_code(name)) is not None
    )


def _tensor_has_uses(document, graph, tensor_index: int) -> bool:
    """Return whether a tensor has consumer or public-boundary references."""

    return bool(graph.consumers(tensor_index)) or _tensor_is_boundary(
        document,
        graph,
        tensor_index,
    )


def _tensor_is_boundary(document, graph, tensor_index: int) -> bool:
    """Return whether a temporary tensor participates in a public graph interface."""

    return (
        tensor_index in set(graph.inputs)
        or tensor_index in set(graph.outputs)
        or tensor_is_signature_bound(
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=tensor_index,
        )
    )


def _tensor_name(graph, tensor_index: int) -> str:
    """Return a stable output-derived name for generated constants."""

    name = getattr(as_list(graph.subgraph.tensors)[tensor_index], "name", "")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return str(name) or f"tensor_{tensor_index}"

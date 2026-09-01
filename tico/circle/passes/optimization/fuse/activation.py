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

import copy
from dataclasses import dataclass
from typing import Mapping

from tico.circle._object import ObjectFactory
from tico.circle.document import CircleDocument
from tico.circle.errors import CircleRewriteError
from tico.circle.graph import as_indices, as_list
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult
from tico.circle.passes.optimization._utils import (
    operator_builtin_code,
    operator_is_plain,
    OptimizationSchemaResolver,
    tensor_contract,
)
from tico.circle.passes.rules import (
    CircleRewriteRule,
    CircleRulePass,
    OperatorSnapshot,
    RewriteApplication,
    RewriteDiagnostic,
    RewritePlan,
)

# Operators whose builtin options carry a fused activation slot that the
# runtime applies while rescaling the integer accumulator into the output
# quantization grid.
_FUSABLE_PRODUCERS = (
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
)

# Circle ActivationFunctionType values for standalone activation operators
# that one producer can absorb.
_FUSED_ACTIVATION_NONE = 0
_ACTIVATION_FUNCTION_CODES: Mapping[str, int] = {
    "RELU": 1,
    "RELU6": 3,
}


@dataclass(frozen=True, kw_only=True)
class _ActivationFusePlan(RewritePlan):
    """Carry the producer operator that absorbs the standalone activation."""

    producer: OperatorSnapshot
    producer_output_position: int
    activation_output: int
    activation_function: int

    def validate(self, document: CircleDocument) -> None:
        """Validate the anchor activation and the captured producer operator."""

        super().validate(document)
        try:
            current = OperatorSnapshot.capture(
                document,
                subgraph_index=self.subgraph_index,
                operator_index=self.producer.operator_index,
            )
        except (IndexError, CircleRewriteError) as error:
            raise CircleRewriteError(
                f"Activation fuse plan is stale because producer operator "
                f"{self.producer.operator_index} no longer exists."
            ) from error
        if current != self.producer:
            raise CircleRewriteError(
                f"Activation fuse plan is stale because producer operator "
                f"{self.producer.operator_index} changed."
            )


class _FuseActivationRule(CircleRewriteRule[_ActivationFusePlan]):
    """Fold a sole-consumer RELU or RELU6 into its producer's fused slot."""

    def __init__(self, schema: OptimizationSchemaResolver):
        """Bind the activation opcodes and the fusable producer opcodes."""

        self.activation_codes = {
            schema.builtin_code(name): function_code
            for name, function_code in _ACTIVATION_FUNCTION_CODES.items()
        }
        self.producer_codes = frozenset(
            schema.builtin_code(name) for name in _FUSABLE_PRODUCERS
        )

    def match(self, document, graph, operator_index, context):
        """Match RELU/RELU6 whose only-consumed input has a fusable producer."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        function_code = self.activation_codes.get(
            operator_builtin_code(document.model, operator)
        )
        if function_code is None:
            return None
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if len(inputs) != 1 or len(outputs) != 1 or not operator_is_plain(operator):
            return None
        source = inputs[0]
        target = outputs[0]
        if graph.consumers(source) != (operator_index,):
            return None
        if source in graph.outputs:
            return None
        producer_index = graph.producer(source)
        if producer_index is None:
            return None
        producer = as_list(graph.subgraph.operators)[producer_index]
        if operator_builtin_code(document.model, producer) not in self.producer_codes:
            return None
        if not operator_is_plain(producer):
            return None
        producer_options = getattr(producer, "builtinOptions", None)
        if producer_options is None:
            return None
        fused = int(getattr(producer_options, "fusedActivationFunction", 0) or 0)
        if fused != _FUSED_ACTIVATION_NONE:
            return None
        source_contract = tensor_contract(graph, source)
        target_contract = tensor_contract(graph, target)
        if source_contract.shape != target_contract.shape:
            return None
        if source_contract.shape_signature != target_contract.shape_signature:
            return None
        if source_contract.tensor_type != target_contract.tensor_type:
            return None
        # The activation output becomes the producer's output contract; the
        # runtime rescales the accumulator into it, so only per-tensor or
        # float encodings are accepted.
        target_quantization = target_contract.quantization
        if target_quantization is not None and len(target_quantization.scale) != 1:
            return None
        producer_outputs = as_indices(producer.outputs)
        return _ActivationFusePlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(source, target),
            producer=OperatorSnapshot.capture(
                document,
                subgraph_index=graph.subgraph_index,
                operator_index=producer_index,
            ),
            producer_output_position=producer_outputs.index(source),
            activation_output=target,
            activation_function=function_code,
        )

    def apply(self, document, plan, context):
        """Set the fused slot, retarget the output, and delete the activation."""

        del context
        subgraph = document.subgraph(plan.subgraph_index)
        operators = as_list(getattr(subgraph, "operators", None))
        producer = operators[plan.producer.operator_index]
        options = copy.deepcopy(getattr(producer, "builtinOptions", None))
        if options is None or not hasattr(options, "fusedActivationFunction"):
            raise CircleRewriteError(
                "Activation fusion expected producer builtin options with a "
                "fused activation slot."
            )
        options.fusedActivationFunction = plan.activation_function
        producer.builtinOptions = options
        producer_outputs = as_indices(producer.outputs)
        producer_outputs[plan.producer_output_position] = plan.activation_output
        producer.outputs = producer_outputs
        del operators[plan.anchor_operator_index]
        subgraph.operators = operators
        return RewriteApplication(
            changes=2,
            diagnostics=(
                RewriteDiagnostic(
                    code="FUSE_ACTIVATION_FUNCTION",
                    message=(
                        "Folded a standalone activation into the producer's "
                        "fused activation slot."
                    ),
                    object_path=(
                        f"subgraphs[{plan.subgraph_index}].operators"
                        f"[{plan.anchor_operator_index}]"
                    ),
                ),
            ),
        )


class FuseActivationFunctionPass(CirclePass):
    """Fold standalone RELU/RELU6 operators into fusable producer options.

    Fusing removes the producer's intermediate pre-activation tensor, so a
    quantized producer rescales its accumulator directly into the activation
    output grid instead of quantizing the unbounded pre-activation range.
    """

    def __init__(
        self,
        *,
        maximum_rewrites: int = 10_000,
        builtin_codes: Mapping[str, int] | None = None,
        object_factory: ObjectFactory | None = None,
    ) -> None:
        """Create the fuse rule with schema or test enum mappings."""

        self.maximum_rewrites = int(maximum_rewrites)
        if self.maximum_rewrites <= 0:
            raise ValueError("maximum_rewrites must be positive.")
        self.schema = OptimizationSchemaResolver(
            builtin_codes=builtin_codes,
            object_factory=object_factory,
        )
        self.rules = (_FuseActivationRule(self.schema),)

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Fuse supported activation operators to a fixed point."""

        return CircleRulePass(
            self.rules,
            maximum_rewrites=self.maximum_rewrites,
        ).run(document, context)

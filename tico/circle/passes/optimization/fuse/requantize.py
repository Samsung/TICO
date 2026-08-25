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

# Operators whose output tensor carries a free requantization target: the
# runtime rescales an integer accumulator into the output quantization, so a
# trailing per-tensor QUANTIZE can become the operator's own output contract.
_REQUANTIZING_PRODUCERS = (
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "FULLY_CONNECTED",
    "TRANSPOSE_CONV",
)


@dataclass(frozen=True, kw_only=True)
class _RequantizeFoldPlan(RewritePlan):
    """Carry the producer operator whose output adopts the QUANTIZE target."""

    producer: OperatorSnapshot
    producer_output_position: int
    quantize_output: int

    def validate(self, document: CircleDocument) -> None:
        """Validate the anchor QUANTIZE and the captured producer operator."""

        super().validate(document)
        try:
            current = OperatorSnapshot.capture(
                document,
                subgraph_index=self.subgraph_index,
                operator_index=self.producer.operator_index,
            )
        except (IndexError, CircleRewriteError) as error:
            raise CircleRewriteError(
                f"Requantize fold plan is stale because producer operator "
                f"{self.producer.operator_index} no longer exists."
            ) from error
        if current != self.producer:
            raise CircleRewriteError(
                f"Requantize fold plan is stale because producer operator "
                f"{self.producer.operator_index} changed."
            )


class _FuseOutputRequantizeRule(CircleRewriteRule[_RequantizeFoldPlan]):
    """Fold a sole-consumer QUANTIZE into its requantizing producer output."""

    def __init__(self, schema: OptimizationSchemaResolver):
        """Bind the QUANTIZE opcode and the requantizing producer opcodes."""

        self.quantize_code = schema.builtin_code("QUANTIZE")
        self.producer_codes = frozenset(
            schema.builtin_code(name) for name in _REQUANTIZING_PRODUCERS
        )

    def match(self, document, graph, operator_index, context):
        """Match QUANTIZE whose only-quantized input feeds no other consumer."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.quantize_code:
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
        if operator_builtin_code(document.model, producer) not in (self.producer_codes):
            return None
        if not operator_is_plain(producer):
            return None
        source_contract = tensor_contract(graph, source)
        target_contract = tensor_contract(graph, target)
        if source_contract.shape != target_contract.shape:
            return None
        if source_contract.shape_signature != target_contract.shape_signature:
            return None
        for contract in (source_contract, target_contract):
            quantization = contract.quantization
            if quantization is None or len(quantization.scale) != 1:
                return None
        producer_outputs = as_indices(producer.outputs)
        return _RequantizeFoldPlan.capture(
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
            quantize_output=target,
        )

    def apply(self, document, plan, context):
        """Retarget the producer output and delete the folded QUANTIZE."""

        del context
        subgraph = document.subgraph(plan.subgraph_index)
        operators = as_list(getattr(subgraph, "operators", None))
        producer = operators[plan.producer.operator_index]
        producer_outputs = as_indices(producer.outputs)
        producer_outputs[plan.producer_output_position] = plan.quantize_output
        producer.outputs = producer_outputs
        del operators[plan.anchor_operator_index]
        subgraph.operators = operators
        return RewriteApplication(
            changes=2,
            diagnostics=(
                RewriteDiagnostic(
                    code="FUSE_OUTPUT_REQUANTIZE",
                    message=("Folded QUANTIZE into the requantizing producer output."),
                    object_path=(
                        f"subgraphs[{plan.subgraph_index}].operators"
                        f"[{plan.anchor_operator_index}]"
                    ),
                ),
            ),
        )


class FuseOutputRequantizePass(CirclePass):
    """Fold trailing per-tensor QUANTIZE operators into requantizing producers."""

    def __init__(
        self,
        *,
        maximum_rewrites: int = 10_000,
        builtin_codes: Mapping[str, int] | None = None,
        object_factory: ObjectFactory | None = None,
    ) -> None:
        """Create the fold rule with schema or test enum mappings."""

        self.maximum_rewrites = int(maximum_rewrites)
        if self.maximum_rewrites <= 0:
            raise ValueError("maximum_rewrites must be positive.")
        self.schema = OptimizationSchemaResolver(
            builtin_codes=builtin_codes,
            object_factory=object_factory,
        )
        self.rules = (_FuseOutputRequantizeRule(self.schema),)

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Fold supported QUANTIZE operators to a fixed point."""

        return CircleRulePass(
            self.rules,
            maximum_rewrites=self.maximum_rewrites,
        ).run(document, context)

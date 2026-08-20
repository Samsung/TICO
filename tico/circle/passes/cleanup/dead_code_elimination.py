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

from collections import deque
from collections.abc import Iterable

from tico.circle.analysis import OperatorEffectAnalysis
from tico.circle.document import CircleDocument
from tico.circle.graph import as_list, OPTIONAL_TENSOR_INDEX
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult
from tico.circle.passes.cleanup.graph_inputs import prune_unused_graph_inputs


class DeadCodeEliminationPass(CirclePass):
    """Remove operators that cannot affect graph outputs or observable state."""

    def __init__(
        self,
        *,
        subgraph_indices: Iterable[int] | None = None,
        prune_unused_inputs: bool = True,
        preserve_zero_output_operators: bool = True,
        preserve_effectful_operators: bool = True,
        effect_analysis: OperatorEffectAnalysis | None = None,
    ):
        self.subgraph_indices = (
            tuple(dict.fromkeys(int(index) for index in subgraph_indices))
            if subgraph_indices is not None
            else None
        )
        self.prune_unused_inputs = bool(prune_unused_inputs)
        self.preserve_zero_output_operators = bool(preserve_zero_output_operators)
        self.preserve_effectful_operators = bool(preserve_effectful_operators)
        self.effect_analysis = effect_analysis or OperatorEffectAnalysis.from_schema()

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Remove unreachable pure operators from selected subgraphs."""

        indices = (
            tuple(range(document.subgraph_count))
            if self.subgraph_indices is None
            else self.subgraph_indices
        )
        removed_operators = 0
        diagnostics: list[str] = []

        for subgraph_index in indices:
            graph = document.graph(subgraph_index)
            subgraph = graph.subgraph
            operators = as_list(subgraph.operators)
            if not operators:
                continue

            live: set[int] = set()
            queue: deque[int] = deque()
            for tensor_index in graph.outputs:
                producer = graph.producer(tensor_index)
                if producer is not None:
                    queue.append(producer)

            for operator_index in range(graph.operator_count):
                outputs = [
                    index
                    for index in graph.operator_outputs(operator_index)
                    if index != OPTIONAL_TENSOR_INDEX
                ]
                if self.preserve_zero_output_operators and not outputs:
                    queue.append(operator_index)
                if (
                    self.preserve_effectful_operators
                    and self.effect_analysis.has_observable_effect(
                        graph,
                        operator_index,
                    )
                ):
                    queue.append(operator_index)

            while queue:
                operator_index = queue.popleft()
                if operator_index in live:
                    continue
                live.add(operator_index)
                queue.extend(graph.predecessors(operator_index))

            dead = sorted(set(range(graph.operator_count)) - live)
            if dead:
                subgraph.operators = [
                    operator
                    for operator_index, operator in enumerate(operators)
                    if operator_index in live
                ]
                removed_operators += len(dead)
                diagnostics.append(
                    f"Subgraph {subgraph_index}: removed operators {dead}."
                )

        input_cleanup = CirclePassResult(modified=False)
        if self.prune_unused_inputs:
            input_cleanup = prune_unused_graph_inputs(document, indices)

        changes = removed_operators + input_cleanup.changes
        context.logger.debug(
            "Dead-code elimination removed %d operators and %d graph inputs.",
            removed_operators,
            input_cleanup.changes,
        )
        return CirclePassResult(
            modified=changes > 0,
            changes=changes,
            diagnostics=(*diagnostics, *input_cleanup.diagnostics),
        )

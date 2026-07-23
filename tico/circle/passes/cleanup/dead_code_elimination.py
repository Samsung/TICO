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

from tico.circle.document import CircleDocument
from tico.circle.graph import as_indices, as_list, CircleGraph, OPTIONAL_TENSOR_INDEX
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult


class DeadCodeEliminationPass(CirclePass):
    """Remove operators that cannot contribute to any graph output."""

    def __init__(
        self,
        *,
        subgraph_indices: Iterable[int] | None = None,
        prune_unused_inputs: bool = True,
        preserve_zero_output_operators: bool = True,
    ):
        self.subgraph_indices = (
            tuple(dict.fromkeys(int(index) for index in subgraph_indices))
            if subgraph_indices is not None
            else None
        )
        self.prune_unused_inputs = prune_unused_inputs
        self.preserve_zero_output_operators = preserve_zero_output_operators

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Remove unreachable operators from selected subgraphs."""

        indices = self.subgraph_indices or tuple(range(document.subgraph_count))
        removed_operators = 0
        removed_inputs = 0
        diagnostics: list[str] = []

        for subgraph_index in indices:
            graph = document.graph(subgraph_index)
            subgraph = graph.subgraph
            operators = as_list(subgraph.operators)
            if not operators or not graph.outputs:
                continue

            live: set[int] = set()
            queue: deque[int] = deque()
            for tensor_index in graph.outputs:
                producer = graph.producer(tensor_index)
                if producer is not None:
                    queue.append(producer)

            if self.preserve_zero_output_operators:
                for operator_index in range(graph.operator_count):
                    outputs = [
                        index
                        for index in graph.operator_outputs(operator_index)
                        if index != OPTIONAL_TENSOR_INDEX
                    ]
                    if not outputs:
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

            if self.prune_unused_inputs:
                consumed = {
                    tensor_index
                    for operator in as_list(subgraph.operators)
                    for tensor_index in as_indices(getattr(operator, "inputs", None))
                    if tensor_index != OPTIONAL_TENSOR_INDEX
                }
                output_set = set(as_indices(subgraph.outputs))
                old_inputs = as_indices(subgraph.inputs)
                subgraph.inputs = [
                    tensor_index
                    for tensor_index in old_inputs
                    if tensor_index in consumed or tensor_index in output_set
                ]
                removed_inputs += len(old_inputs) - len(subgraph.inputs)

        changes = removed_operators + removed_inputs
        context.logger.debug(
            "Dead-code elimination removed %d operators and %d graph inputs.",
            removed_operators,
            removed_inputs,
        )
        return CirclePassResult(
            modified=changes > 0,
            changes=changes,
            diagnostics=tuple(diagnostics),
        )

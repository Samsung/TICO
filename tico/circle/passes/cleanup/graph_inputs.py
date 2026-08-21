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

from collections.abc import Iterable

from tico.circle.document import CircleDocument
from tico.circle.graph import as_indices, as_list, OPTIONAL_TENSOR_INDEX
from tico.circle.passes.base import CirclePassResult
from tico.circle.rewrite import iter_subgraph_references


def prune_unused_graph_inputs(
    document: CircleDocument,
    subgraph_indices: Iterable[int] | None = None,
    *,
    preserve_signature_inputs: bool = True,
    preserve_referenced_subgraph_interfaces: bool = True,
) -> CirclePassResult:
    """Remove unused graph inputs without changing observable graph interfaces.

    Signature-bound inputs remain public even after their data dependency disappears.
    Inputs of subgraphs referenced by control-flow or call operators are all retained
    because pruning them would require updating the caller-side argument contract.
    """

    indices = _normalize_subgraph_indices(document, subgraph_indices)
    signature_inputs = _signature_inputs(document) if preserve_signature_inputs else {}
    referenced_subgraphs = (
        _referenced_subgraph_indices(document)
        if preserve_referenced_subgraph_interfaces
        else set()
    )

    removed_inputs = 0
    modified_subgraphs: set[int] = set()
    diagnostics: list[str] = []
    for subgraph_index in indices:
        subgraph = document.subgraph(subgraph_index)
        old_inputs = as_indices(getattr(subgraph, "inputs", None))
        if not old_inputs:
            continue

        consumed = {
            tensor_index
            for operator in as_list(getattr(subgraph, "operators", None))
            for tensor_index in as_indices(getattr(operator, "inputs", None))
            if tensor_index != OPTIONAL_TENSOR_INDEX
        }
        protected = consumed | set(as_indices(getattr(subgraph, "outputs", None)))
        protected.update(signature_inputs.get(subgraph_index, ()))
        if subgraph_index in referenced_subgraphs:
            protected.update(old_inputs)

        removed = [
            tensor_index for tensor_index in old_inputs if tensor_index not in protected
        ]
        if not removed:
            continue
        removed_set = set(removed)
        subgraph.inputs = [
            tensor_index
            for tensor_index in old_inputs
            if tensor_index not in removed_set
        ]
        removed_inputs += len(removed)
        modified_subgraphs.add(subgraph_index)
        diagnostics.append(
            f"Subgraph {subgraph_index}: removed graph inputs {removed}."
        )

    if modified_subgraphs:
        from tico.circle.session import existing_optimization_session

        session = existing_optimization_session(document.model)
        if session is not None:
            session.mark_modified(tuple(sorted(modified_subgraphs)))
    return CirclePassResult(
        modified=removed_inputs > 0,
        changes=removed_inputs,
        diagnostics=tuple(diagnostics),
    )


def _normalize_subgraph_indices(
    document: CircleDocument,
    subgraph_indices: Iterable[int] | None,
) -> tuple[int, ...]:
    """Return unique validated subgraph indices in stable order."""

    indices = (
        tuple(range(document.subgraph_count))
        if subgraph_indices is None
        else tuple(dict.fromkeys(int(index) for index in subgraph_indices))
    )
    for subgraph_index in indices:
        document.subgraph(subgraph_index)
    return indices


def _signature_inputs(document: CircleDocument) -> dict[int, set[int]]:
    """Collect every tensor exposed as a signature input by subgraph."""

    result: dict[int, set[int]] = {}
    for signature in as_list(getattr(document.model, "signatureDefs", None)):
        subgraph_index = int(getattr(signature, "subgraphIndex", -1))
        mapped = result.setdefault(subgraph_index, set())
        mapped.update(
            int(getattr(tensor_map, "tensorIndex", -1))
            for tensor_map in as_list(getattr(signature, "inputs", None))
        )
    return result


def _referenced_subgraph_indices(document: CircleDocument) -> set[int]:
    """Return subgraphs whose complete input arity is owned by a caller."""

    return {
        int(subgraph_index)
        for _path, _container, _field_name, subgraph_index in (
            iter_subgraph_references(document.model)
        )
        if 0 <= int(subgraph_index) < document.subgraph_count
    }


__all__ = ["prune_unused_graph_inputs"]

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
from enum import Enum
from typing import Any, Iterable, Sequence

from tico.circle._buffer import clone_model_borrowing_payloads, detach_borrowed_payloads
from tico.circle.document import CircleDocument
from tico.circle.errors import CircleSelectionError
from tico.circle.graph import as_indices, as_list, GraphBoundary
from tico.circle.passes.base import CirclePassContext
from tico.circle.passes.cleanup.dead_code_elimination import DeadCodeEliminationPass
from tico.circle.rewrite import compact_model, keep_subgraphs, RewriteStats
from tico.circle.selector import (
    resolve_tensor_patterns,
    select_operators_by_tensor_boundaries,
)


class SignaturePolicy(str, Enum):
    """Control how extraction handles source model signatures."""

    DROP = "drop"
    PRESERVE_COMPATIBLE = "preserve-compatible"


class PayloadOwnership(str, Enum):
    """Control who owns the constant payload storage of an extracted document.

    ``DETACHED`` (the default) copies only the payloads that survive extraction, so
    the result is independent of the source document and of any bytes or file
    mapping backing it. ``BORROWED`` keeps the surviving payload storage shared
    with the source; the result is valid only while the source document and its
    backing storage stay alive and unmodified. It is intended for callers that own
    the source exclusively and save the result immediately.
    """

    DETACHED = "detached"
    BORROWED = "borrowed"


@dataclass(frozen=True)
class ExtractionResult:
    """Return an extracted document together with selection and cleanup metadata."""

    document: CircleDocument
    selected_operator_indices: tuple[int, ...]
    source_boundary: GraphBoundary
    boundary: GraphBoundary
    removed_operators: int
    rewrite_stats: RewriteStats
    payload_ownership: PayloadOwnership = PayloadOwnership.DETACHED


def _signature_tensor_indices(signature: object, field_name: str) -> set[int]:
    return {
        int(getattr(tensor_map, "tensorIndex", -1))
        for tensor_map in as_list(getattr(signature, field_name, None))
    }


def _apply_signature_policy(
    model: Any,
    *,
    target_subgraph_index: int,
    policy: SignaturePolicy,
) -> RewriteStats:
    signatures = getattr(model, "signatureDefs", None)
    if signatures is None:
        return RewriteStats()

    retained = []
    removed = 0
    target = model.subgraphs[target_subgraph_index]
    target_inputs = set(as_indices(target.inputs))
    target_outputs = set(as_indices(target.outputs))

    for signature in signatures:
        if int(getattr(signature, "subgraphIndex", -1)) != target_subgraph_index:
            retained.append(signature)
            continue
        if policy is SignaturePolicy.DROP:
            removed += 1
            continue

        signature_inputs = _signature_tensor_indices(signature, "inputs")
        signature_outputs = _signature_tensor_indices(signature, "outputs")
        if signature_inputs == target_inputs and signature_outputs == target_outputs:
            retained.append(signature)
        else:
            removed += 1

    model.signatureDefs = retained
    return RewriteStats(removed_signatures=removed)


def extract_by_operator_indices(
    document: CircleDocument,
    operator_indices: Iterable[int],
    *,
    subgraph_index: int = 0,
    keep_other_subgraphs: bool = False,
    signature_policy: SignaturePolicy = SignaturePolicy.DROP,
    verify: bool = True,
    payload_ownership: PayloadOwnership = PayloadOwnership.DETACHED,
) -> ExtractionResult:
    """Extract an operator-induced region into a new Circle document.

    Graph and buffer-table metadata are cloned first; constant payload storage is
    borrowed from the source while the selection, signature policy, dead-code
    elimination, and index compaction run. None of those steps modifies payload
    bytes, so discarded constants are never copied. With the default
    ``PayloadOwnership.DETACHED`` the payloads that remain are then copied once
    per buffer storage object, making the result independent of the source.
    """

    ownership = PayloadOwnership(payload_ownership)
    selected = tuple(sorted(set(int(index) for index in operator_indices)))
    source_graph = document.graph(subgraph_index)
    boundary = source_graph.region_boundary(selected)
    if not boundary.outputs:
        raise CircleSelectionError(
            "The selected operator region does not expose any output tensors."
        )

    cloned_model, borrowed = clone_model_borrowing_payloads(document.model)
    result = CircleDocument(cloned_model, source=document.source)
    target_subgraph = result.subgraph(subgraph_index)
    source_operator_count = len(as_list(target_subgraph.operators))
    selected_set = set(selected)
    target_subgraph.operators = [
        operator
        for operator_index, operator in enumerate(as_list(target_subgraph.operators))
        if operator_index in selected_set
    ]
    target_subgraph.inputs = list(boundary.inputs)
    target_subgraph.outputs = list(boundary.outputs)
    removed_operators = source_operator_count - len(target_subgraph.operators)

    stats = RewriteStats()
    target_index_after_rewrite = subgraph_index
    if not keep_other_subgraphs:
        stats += keep_subgraphs(result.model, (subgraph_index,))
        target_index_after_rewrite = 0

    stats += _apply_signature_policy(
        result.model,
        target_subgraph_index=target_index_after_rewrite,
        policy=signature_policy,
    )

    DeadCodeEliminationPass(
        subgraph_indices=(target_index_after_rewrite,),
        prune_unused_inputs=True,
    ).run(
        result,
        CirclePassContext(verify_after_each_pass=False),
    )
    final_operator_count = len(
        as_list(result.subgraph(target_index_after_rewrite).operators)
    )
    removed_operators = source_operator_count - final_operator_count
    stats += compact_model(result.model, subgraph_indices=(target_index_after_rewrite,))

    if verify:
        result.verify(raise_on_error=True)

    if ownership is PayloadOwnership.DETACHED:
        # Only buffers that survived compaction are still present, so this copies
        # exactly the retained payloads and releases the source's backing storage.
        detach_borrowed_payloads(result.model, borrowed)

    final_subgraph = result.subgraph(target_index_after_rewrite)
    final_boundary = GraphBoundary(
        tuple(as_indices(final_subgraph.inputs)),
        tuple(as_indices(final_subgraph.outputs)),
    )
    return ExtractionResult(
        document=result,
        selected_operator_indices=selected,
        source_boundary=boundary,
        boundary=final_boundary,
        removed_operators=removed_operators,
        rewrite_stats=stats,
        payload_ownership=ownership,
    )


def extract_by_tensor_indices(
    document: CircleDocument,
    *,
    from_tensors: Iterable[int] = (),
    to_tensors: Iterable[int] = (),
    subgraph_index: int = 0,
    keep_other_subgraphs: bool = False,
    signature_policy: SignaturePolicy = SignaturePolicy.DROP,
    verify: bool = True,
    payload_ownership: PayloadOwnership = PayloadOwnership.DETACHED,
) -> ExtractionResult:
    """Extract all operators on paths between tensor index boundaries."""

    graph = document.graph(subgraph_index)
    selected = select_operators_by_tensor_boundaries(
        graph,
        from_tensors=from_tensors,
        to_tensors=to_tensors,
    )
    return extract_by_operator_indices(
        document,
        selected,
        subgraph_index=subgraph_index,
        keep_other_subgraphs=keep_other_subgraphs,
        signature_policy=signature_policy,
        verify=verify,
        payload_ownership=payload_ownership,
    )


def extract_by_tensor_patterns(
    document: CircleDocument,
    *,
    from_patterns: Sequence[str] = (),
    to_patterns: Sequence[str] = (),
    subgraph_index: int = 0,
    full_match: bool = False,
    keep_other_subgraphs: bool = False,
    signature_policy: SignaturePolicy = SignaturePolicy.DROP,
    verify: bool = True,
    payload_ownership: PayloadOwnership = PayloadOwnership.DETACHED,
) -> ExtractionResult:
    """Extract all operators on paths between tensor name patterns."""

    graph = document.graph(subgraph_index)
    from_tensors = resolve_tensor_patterns(
        graph,
        from_patterns,
        full_match=full_match,
    )
    to_tensors = resolve_tensor_patterns(
        graph,
        to_patterns,
        full_match=full_match,
    )
    return extract_by_tensor_indices(
        document,
        from_tensors=from_tensors,
        to_tensors=to_tensors,
        subgraph_index=subgraph_index,
        keep_other_subgraphs=keep_other_subgraphs,
        signature_policy=signature_policy,
        verify=verify,
        payload_ownership=payload_ownership,
    )

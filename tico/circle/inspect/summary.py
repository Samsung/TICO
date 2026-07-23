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

from dataclasses import asdict, dataclass
from typing import Any

from tico.circle._schema import decode_text
from tico.circle.document import CircleDocument
from tico.circle.graph import (
    as_indices,
    as_list,
    has_buffer_payload,
    is_constant_tensor,
)


def _buffer_size(buffer: Any) -> int:
    data = getattr(buffer, "data", None)
    if data is not None:
        if hasattr(data, "nbytes") and int(data.nbytes) > 0:
            return int(data.nbytes)
        try:
            if len(data) > 0:
                return len(data)
        except TypeError:
            pass
    return int(getattr(buffer, "size", 0) or 0)


@dataclass(frozen=True)
class CircleSubgraphSummary:
    """Summarize one Circle subgraph without exposing schema objects."""

    index: int
    name: str
    inputs: int
    outputs: int
    tensors: int
    constant_tensors: int
    operators: int


@dataclass(frozen=True)
class CircleModelSummary:
    """Summarize model-level and subgraph-level Circle structure."""

    source: str | None
    version: int
    description: str
    subgraphs: tuple[CircleSubgraphSummary, ...]
    operator_codes: int
    buffers: int
    buffers_with_payload: int
    buffer_bytes: int
    signatures: int
    metadata_entries: int

    def to_dict(self) -> dict[str, Any]:
        """Convert the summary to a JSON-serializable dictionary."""

        return asdict(self)


def summarize_document(document: CircleDocument) -> CircleModelSummary:
    """Build a stable summary for a Circle document."""

    model = document.model
    subgraph_summaries: list[CircleSubgraphSummary] = []
    for subgraph_index, subgraph in enumerate(as_list(model.subgraphs)):
        constant_tensors = sum(
            is_constant_tensor(model, subgraph, tensor_index)
            for tensor_index in range(len(as_list(subgraph.tensors)))
        )
        subgraph_summaries.append(
            CircleSubgraphSummary(
                index=subgraph_index,
                name=decode_text(getattr(subgraph, "name", "")),
                inputs=len(as_indices(subgraph.inputs)),
                outputs=len(as_indices(subgraph.outputs)),
                tensors=len(as_list(subgraph.tensors)),
                constant_tensors=constant_tensors,
                operators=len(as_list(subgraph.operators)),
            )
        )

    buffers = as_list(model.buffers)
    return CircleModelSummary(
        source=str(document.source) if document.source is not None else None,
        version=int(getattr(model, "version", 0) or 0),
        description=decode_text(getattr(model, "description", "")),
        subgraphs=tuple(subgraph_summaries),
        operator_codes=len(as_list(model.operatorCodes)),
        buffers=len(buffers),
        buffers_with_payload=sum(has_buffer_payload(buffer) for buffer in buffers),
        buffer_bytes=sum(_buffer_size(buffer) for buffer in buffers),
        signatures=len(as_list(getattr(model, "signatureDefs", None))),
        metadata_entries=len(as_list(getattr(model, "metadata", None))),
    )

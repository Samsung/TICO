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
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

from tico.circle.errors import CircleIOError
from tico.circle.graph import as_list, CircleGraph
from tico.circle.io import (
    BinaryDestination,
    BinarySource,
    CirclePayloadMapping,
    load_model,
    load_model_mapped,
    model_from_bytes,
    model_to_bytes,
    PathLike,
    save_model,
)

if TYPE_CHECKING:
    from tico.circle.inspect.summary import CircleModelSummary
    from tico.circle.verify import VerificationReport


class CircleDocument:
    """Own a mutable Circle Object API model and its optional source path.

    A document created by ``load_mapped()`` additionally owns the read-only file
    mapping that its external constant payloads borrow. ``clone()``, deep copies,
    and detached extraction results copy payloads and therefore never depend on
    that mapping; call ``release_payloads()`` (or use the document as a context
    manager) once the mapped document and any borrowed results are done.
    """

    def __init__(
        self,
        model: Any,
        *,
        source: str | os.PathLike[str] | None = None,
        payload_mapping: CirclePayloadMapping | None = None,
    ):
        if model is None:
            raise TypeError("CircleDocument requires a Circle Object API model.")
        if not hasattr(model, "subgraphs"):
            raise TypeError("Circle model must expose a subgraphs field.")
        self._model = model
        self._source = Path(source) if source is not None else None
        self._payload_mapping = payload_mapping

    @property
    def model(self) -> Any:
        """Return the mutable generated Circle Object API model."""

        return self._model

    @property
    def source(self) -> Path | None:
        """Return the source path when the document was loaded from a file."""

        return self._source

    @property
    def payload_mapping(self) -> CirclePayloadMapping | None:
        """Return the file mapping this document's payloads borrow, if any."""

        return self._payload_mapping

    @property
    def subgraph_count(self) -> int:
        """Return the number of subgraphs in the model."""

        return len(as_list(self._model.subgraphs))

    @classmethod
    def load(cls, source: BinarySource) -> CircleDocument:
        """Load a Circle document from a path, standard input, or binary stream.

        The complete input is read into memory. Use ``load_mapped()`` to borrow
        external payloads from a read-only mapping of a regular file instead.
        """

        source_path: str | os.PathLike[str] | None = None
        if isinstance(source, (str, os.PathLike)) and os.fspath(source) != "-":
            source_path = source
        return cls(load_model(source), source=source_path)

    @classmethod
    def load_mapped(cls, path: PathLike) -> CircleDocument:
        """Load a regular file through a read-only mapping instead of ``bytes``.

        Graph metadata is unpacked into ordinary writable Object API objects, but
        appended constant payloads remain views of the mapping, so no Python
        allocation the size of the file is made. Files whose constants are inline
        still copy the complete FlatBuffer; see ``docs/large_circle_export.md``.
        The returned document owns the mapping; see ``release_payloads()``.
        """

        model, mapping = load_model_mapped(path)
        return cls(model, source=path, payload_mapping=mapping)

    @classmethod
    def from_bytes(cls, data: bytes) -> CircleDocument:
        """Deserialize a Circle document from binary data."""

        return cls(model_from_bytes(data))

    def to_bytes(self) -> bytes:
        """Serialize the document into Circle binary data."""

        return model_to_bytes(self._model)

    def save(
        self,
        destination: BinaryDestination,
        *,
        atomic: bool = True,
    ) -> None:
        """Save the document to a path, standard output, or binary stream.

        File destinations are written by streaming the header, padding, and
        payload views; the complete binary is not materialized first. An atomic
        save writes a temporary file beside the destination and replaces it, which
        keeps a mapped source readable until the replacement is complete.
        """

        mapping = self._payload_mapping
        if (
            mapping is not None
            and not atomic
            and isinstance(destination, (str, os.PathLike))
            and os.fspath(destination) != "-"
            and mapping.same_file(destination)
        ):
            raise CircleIOError(
                f"Refusing to overwrite the mapped source {mapping.path} in place; "
                "use an atomic save so its payloads stay readable."
            )
        save_model(self._model, destination, atomic=atomic)

    def release_payloads(self) -> None:
        """Release the owned file mapping; a no-op for documents without one.

        Payload views that are still referenced keep the mapping alive until they
        are dropped, so releasing never invalidates live data.
        """

        mapping, self._payload_mapping = self._payload_mapping, None
        if mapping is not None:
            mapping.release()

    def clone(self) -> CircleDocument:
        """Return a deep, independently mutable copy of the document.

        Payload views are copied into owned arrays, so the clone does not borrow
        from the source's bytes or file mapping.
        """

        return CircleDocument(copy.deepcopy(self._model), source=self._source)

    def subgraph(self, index: int = 0) -> Any:
        """Return a subgraph by index with a descriptive bounds check."""

        subgraphs = as_list(self._model.subgraphs)
        if index < 0 or index >= len(subgraphs):
            raise IndexError(
                f"Subgraph index {index} is outside the valid range "
                f"0..{len(subgraphs) - 1}."
            )
        return subgraphs[index]

    def graph(self, index: int = 0) -> CircleGraph:
        """Return an active session cache or build a standalone graph index."""

        from tico.circle.session import active_optimization_session

        session = active_optimization_session(self._model)
        if session is not None:
            return session.graph(index)
        return CircleGraph(self._model, index)

    def verify(self, *, raise_on_error: bool = True) -> VerificationReport:
        """Check internal Circle references and graph bookkeeping.

        This method does not execute the model or validate numerical accuracy or
        backend compatibility.
        """

        from tico.circle.verify import verify_document

        return verify_document(self, raise_on_error=raise_on_error)

    def summary(self) -> CircleModelSummary:
        """Return a structured model summary."""

        from tico.circle.inspect.summary import summarize_document

        return summarize_document(self)

    def __enter__(self) -> CircleDocument:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release_payloads()

    def __deepcopy__(self, memo: dict[int, Any]) -> CircleDocument:
        """Support deep copying while preserving the immutable source path."""

        return CircleDocument(copy.deepcopy(self._model, memo), source=self._source)

    def __repr__(self) -> str:
        """Return a concise representation for debugging."""

        source = str(self._source) if self._source is not None else "<memory>"
        return f"CircleDocument(source={source!r}, subgraphs={self.subgraph_count})"

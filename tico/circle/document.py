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

from tico.circle.graph import as_list, CircleGraph
from tico.circle.io import (
    BinaryDestination,
    BinarySource,
    load_model,
    model_from_bytes,
    model_to_bytes,
    save_model,
)

if TYPE_CHECKING:
    from tico.circle.inspect.summary import CircleModelSummary
    from tico.circle.verify import VerificationReport


class CircleDocument:
    """Own a mutable Circle Object API model and its optional source path."""

    def __init__(self, model: Any, *, source: str | os.PathLike[str] | None = None):
        if model is None:
            raise TypeError("CircleDocument requires a Circle Object API model.")
        if not hasattr(model, "subgraphs"):
            raise TypeError("Circle model must expose a subgraphs field.")
        self._model = model
        self._source = Path(source) if source is not None else None

    @property
    def model(self) -> Any:
        """Return the mutable generated Circle Object API model."""

        return self._model

    @property
    def source(self) -> Path | None:
        """Return the source path when the document was loaded from a file."""

        return self._source

    @property
    def subgraph_count(self) -> int:
        """Return the number of subgraphs in the model."""

        return len(as_list(self._model.subgraphs))

    @classmethod
    def load(cls, source: BinarySource) -> CircleDocument:
        """Load a Circle document from a path, standard input, or binary stream."""

        source_path: str | os.PathLike[str] | None = None
        if isinstance(source, (str, os.PathLike)) and os.fspath(source) != "-":
            source_path = source
        return cls(load_model(source), source=source_path)

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
        """Save the document to a path, standard output, or binary stream."""

        save_model(self._model, destination, atomic=atomic)

    def clone(self) -> CircleDocument:
        """Return a deep, independently mutable copy of the document."""

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
        """Build a graph index for a subgraph."""

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

    def __deepcopy__(self, memo: dict[int, Any]) -> CircleDocument:
        """Support deep copying while preserving the immutable source path."""

        return CircleDocument(copy.deepcopy(self._model, memo), source=self._source)

    def __repr__(self) -> str:
        """Return a concise representation for debugging."""

        source = str(self._source) if self._source is not None else "<memory>"
        return f"CircleDocument(source={source!r}, subgraphs={self.subgraph_count})"

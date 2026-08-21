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

import logging
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from tico.circle.document import CircleDocument

if TYPE_CHECKING:
    from tico.circle.graph import CircleGraph
    from tico.circle.mutation import CircleMutationTransaction
    from tico.circle.session import CircleOptimizationSession


@dataclass
class CirclePassContext:
    """Provide shared services and state to Circle graph passes."""

    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("tico.circle.passes")
    )
    verify_after_each_pass: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    _sessions: dict[int, tuple[Any, CircleOptimizationSession]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def session(self, document: CircleDocument) -> CircleOptimizationSession:
        """Return the model-scoped optimization session for this pass context."""

        from tico.circle.session import optimization_session_for

        model = document.model
        key = id(model)
        cached = self._sessions.get(key)
        if cached is not None and cached[0] is model:
            return cached[1]
        session = optimization_session_for(document)
        self._sessions[key] = (model, session)
        return session

    def graph(
        self,
        document: CircleDocument,
        subgraph_index: int = 0,
    ) -> CircleGraph:
        """Return a revision-checked producer/consumer graph index."""

        return self.session(document).graph(subgraph_index)

    def activate(
        self,
        document: CircleDocument,
    ) -> AbstractContextManager[CircleOptimizationSession]:
        """Activate shared session services for builders and rewrite helpers."""

        return self.session(document).activate()

    def mutation(
        self,
        document: CircleDocument,
        *,
        subgraph_index: int,
        plan: Any = None,
    ) -> CircleMutationTransaction:
        """Create an atomic local rewrite transaction."""

        return self.session(document).transaction(
            subgraph_index=subgraph_index,
            plan=plan,
        )


@dataclass(frozen=True)
class CirclePassResult:
    """Describe the observable result of one Circle graph pass."""

    modified: bool
    changes: int = 0
    diagnostics: tuple[str, ...] = ()


class CirclePass(ABC):
    """Define the interface for a transformation over a Circle document."""

    @property
    def name(self) -> str:
        """Return the stable pass name used by diagnostics and the CLI."""

        return self.__class__.__name__

    @abstractmethod
    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Mutate a Circle document and report whether it changed."""

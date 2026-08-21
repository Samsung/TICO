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

import weakref
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock
from typing import Any

from tico.circle.builder import ConstantPool
from tico.circle.graph import as_list, CircleGraph
from tico.circle.value import TensorValueCodec


@dataclass(frozen=True)
class CircleSessionRevision:
    """Capture model and subgraph revisions at one optimization boundary."""

    model: int
    subgraphs: tuple[int, ...]


@dataclass(frozen=True)
class CircleOptimizationStatistics:
    """Report session-level cache and mutation activity."""

    graph_cache_hits: int
    graph_cache_misses: int
    committed_mutations: int
    invalidations: int


_ACTIVE_SESSION: ContextVar[CircleOptimizationSession | None] = ContextVar(
    "tico_circle_active_optimization_session",
    default=None,
)
_SESSION_REGISTRY_LOCK = RLock()
_SESSION_REGISTRY: dict[
    int,
    tuple[
        weakref.ReferenceType[Any] | None,
        Any | None,
        weakref.ReferenceType[CircleOptimizationSession],
    ],
] = {}


class CircleOptimizationSession:
    """Own reusable analysis and construction services for one mutable Circle model.

    A session is model-scoped rather than pass-scoped. It keeps producer/consumer
    indexes until the corresponding subgraph revision changes and provides shared
    constant pools to builders created by independent passes.
    """

    def __init__(self, document_or_model: Any) -> None:
        """Bind one mutable model and initialize empty revision-aware caches."""

        self.model = getattr(document_or_model, "model", document_or_model)
        if self.model is None or not hasattr(self.model, "subgraphs"):
            raise TypeError("CircleOptimizationSession requires a Circle model.")
        self._model_revision = 0
        self._subgraph_revisions = [
            0 for _ in as_list(getattr(self.model, "subgraphs", None))
        ]
        self._graph_cache: dict[int, tuple[int, CircleGraph]] = {}
        self._constant_pools: dict[
            tuple[int, int],
            tuple[Any, Any, TensorValueCodec, ConstantPool],
        ] = {}
        self._tracked_constant_pools: dict[int, ConstantPool] = {}
        self._default_codec: TensorValueCodec | None = None
        self._graph_cache_hits = 0
        self._graph_cache_misses = 0
        self._committed_mutations = 0
        self._invalidations = 0
        self._lock = RLock()

    @property
    def revision(self) -> CircleSessionRevision:
        """Return a comparable token for detecting unreported pass mutation."""

        with self._lock:
            self._resize_revisions()
            return CircleSessionRevision(
                self._model_revision,
                tuple(self._subgraph_revisions),
            )

    @property
    def statistics(self) -> CircleOptimizationStatistics:
        """Return immutable cache and mutation counters."""

        with self._lock:
            return CircleOptimizationStatistics(
                graph_cache_hits=self._graph_cache_hits,
                graph_cache_misses=self._graph_cache_misses,
                committed_mutations=self._committed_mutations,
                invalidations=self._invalidations,
            )

    @contextmanager
    def activate(self) -> Iterator[CircleOptimizationSession]:
        """Expose shared services to builders and helpers in this context."""

        token = _ACTIVE_SESSION.set(self)
        try:
            yield self
        finally:
            _ACTIVE_SESSION.reset(token)

    def graph(self, subgraph_index: int = 0) -> CircleGraph:
        """Return a producer/consumer index valid for the current subgraph revision."""

        index = int(subgraph_index)
        with self._lock:
            self._resize_revisions()
            if index < 0 or index >= len(self._subgraph_revisions):
                raise IndexError(
                    f"Subgraph index {index} is outside 0.."
                    f"{len(self._subgraph_revisions) - 1}."
                )
            revision = self._subgraph_revisions[index]
            cached = self._graph_cache.get(index)
            if cached is not None and cached[0] == revision:
                self._graph_cache_hits += 1
                return cached[1]
            graph = CircleGraph(self.model, index)
            self._graph_cache[index] = (revision, graph)
            self._graph_cache_misses += 1
            return graph

    def existing_constant_pool(
        self,
        *,
        codec: TensorValueCodec,
        object_factory: Any = None,
    ) -> ConstantPool | None:
        """Return a compatible canonical pool without constructing one."""

        registry = getattr(codec, "registry", codec)
        key = (id(registry), id(object_factory))
        with self._lock:
            existing = self._constant_pools.get(key)
            if existing is None:
                return None
            expected_registry, expected_factory, _codec, pool = existing
            if expected_registry is registry and expected_factory is object_factory:
                self._tracked_constant_pools[id(pool)] = pool
                return pool
            return None

    def constant_pool(
        self,
        *,
        codec: TensorValueCodec | None = None,
        object_factory: Any = None,
    ) -> ConstantPool:
        """Return one shared pool instance for a codec/factory service pair."""

        with self._lock:
            if codec is None:
                if self._default_codec is None:
                    self._default_codec = TensorValueCodec()
                codec = self._default_codec
            registry = getattr(codec, "registry", codec)
            key = (id(registry), id(object_factory))
            existing = self.existing_constant_pool(
                codec=codec,
                object_factory=object_factory,
            )
            if existing is not None:
                existing.synchronize()
                return existing
            pool = ConstantPool(
                self.model,
                codec=codec,
                object_factory=object_factory,
            )
            self._constant_pools[key] = (
                registry,
                object_factory,
                codec,
                pool,
            )
            self._tracked_constant_pools[id(pool)] = pool
            return pool

    def register_constant_pool(self, pool: ConstantPool) -> ConstantPool:
        """Return the canonical pool for one codec registry and object factory."""

        if pool.model is not self.model:
            raise ValueError("constant pool must belong to this session's model.")
        registry = getattr(pool.codec, "registry", pool.codec)
        key = (id(registry), id(pool.object_factory))
        with self._lock:
            existing = self._constant_pools.get(key)
            if existing is not None:
                expected_registry, expected_factory, _codec, canonical = existing
                if (
                    expected_registry is registry
                    and expected_factory is pool.object_factory
                ):
                    self._tracked_constant_pools[id(canonical)] = canonical
                    return canonical
            self._constant_pools[key] = (
                registry,
                pool.object_factory,
                pool.codec,
                pool,
            )
            self._tracked_constant_pools[id(pool)] = pool
            return pool

    def mark_modified(
        self,
        subgraph_indices: Iterable[int] | None = None,
        *,
        touched_tensors: Mapping[int, Iterable[int]] | None = None,
        rebuild_constant_pools: bool = False,
    ) -> None:
        """Advance revisions, invalidate graph indexes, and refresh constants."""

        with self._lock:
            self._resize_revisions()
            indices = self._normalize_indices(subgraph_indices)
            self._model_revision += 1
            for index in indices:
                self._subgraph_revisions[index] += 1
                self._graph_cache.pop(index, None)
            self._committed_mutations += 1
            self._invalidations += len(indices)

        self.synchronize_constant_pools(
            force=rebuild_constant_pools,
            touched_tensors=touched_tensors,
        )

    def invalidate(
        self,
        subgraph_indices: Iterable[int] | None = None,
        *,
        rebuild_constant_pools: bool = False,
    ) -> None:
        """Drop cached analyses without recording a committed model mutation."""

        with self._lock:
            self._resize_revisions()
            indices = self._normalize_indices(subgraph_indices)
            for index in indices:
                self._graph_cache.pop(index, None)
            self._invalidations += len(indices)
        self.synchronize_constant_pools(force=bool(rebuild_constant_pools))

    def synchronize_constant_pools(
        self,
        *,
        force: bool = False,
        touched_tensors: Mapping[int, Iterable[int]] | None = None,
    ) -> None:
        """Synchronize every pool already materialized by this session."""

        with self._lock:
            pools = tuple(self._tracked_constant_pools.values())
        touched = tuple((touched_tensors or {}).items())
        for pool in pools:
            if force or not touched:
                pool.synchronize(force=force)
                continue
            for subgraph_index, tensor_indices in touched:
                pool.synchronize(
                    subgraph_index=int(subgraph_index),
                    tensor_indices=tuple(int(index) for index in tensor_indices),
                )

    def after_pass(
        self,
        *,
        modified: bool,
        revision_before: CircleSessionRevision,
    ) -> None:
        """Account for legacy passes that mutate without session-aware helpers."""

        revision_after = self.revision
        if modified and revision_after == revision_before:
            self.mark_modified(rebuild_constant_pools=True)
        elif revision_after != revision_before:
            self.synchronize_constant_pools()

    def transaction(
        self,
        *,
        subgraph_index: int,
        plan: Any = None,
    ) -> Any:
        """Create one atomic mutation scope, importing lazily to avoid cycles."""

        from tico.circle.mutation import CircleMutationTransaction

        return CircleMutationTransaction(
            self,
            subgraph_index=int(subgraph_index),
            plan=plan,
        )

    def _resize_revisions(self) -> None:
        count = len(as_list(getattr(self.model, "subgraphs", None)))
        current = len(self._subgraph_revisions)
        if count > current:
            self._subgraph_revisions.extend(0 for _ in range(count - current))
        elif count < current:
            del self._subgraph_revisions[count:]
            self._graph_cache = {
                index: cached
                for index, cached in self._graph_cache.items()
                if index < count
            }

    def _normalize_indices(
        self,
        subgraph_indices: Iterable[int] | None,
    ) -> tuple[int, ...]:
        indices = (
            tuple(range(len(self._subgraph_revisions)))
            if subgraph_indices is None
            else tuple(dict.fromkeys(int(index) for index in subgraph_indices))
        )
        for index in indices:
            if index < 0 or index >= len(self._subgraph_revisions):
                raise IndexError(
                    f"Subgraph index {index} is outside 0.."
                    f"{len(self._subgraph_revisions) - 1}."
                )
        return indices


def optimization_session_for(document_or_model: Any) -> CircleOptimizationSession:
    """Return the live optimization session associated with one model identity.

    The registry keeps only a weak reference to the session. Pass managers, builders,
    and callers that retain the returned object own its lifetime, so completed model
    compilations do not accumulate in process-global state.
    """

    model = getattr(document_or_model, "model", document_or_model)
    active = _ACTIVE_SESSION.get()
    if active is not None and active.model is model:
        return active

    model_id = id(model)
    with _SESSION_REGISTRY_LOCK:
        existing = _SESSION_REGISTRY.get(model_id)
        if existing is not None:
            model_reference, strong_model, session_reference = existing
            current_model = (
                model_reference() if model_reference is not None else strong_model
            )
            session = session_reference()
            if current_model is model and session is not None:
                return session
            _SESSION_REGISTRY.pop(model_id, None)

        session = CircleOptimizationSession(model)
        try:
            model_reference = weakref.ref(model)
            strong_model = None
        except TypeError:
            model_reference = None
            strong_model = model
        session_reference = weakref.ref(
            session,
            lambda reference, key=model_id: _remove_session(key, reference),
        )
        _SESSION_REGISTRY[model_id] = (
            model_reference,
            strong_model,
            session_reference,
        )
        return session


def existing_optimization_session(
    document_or_model: Any,
) -> CircleOptimizationSession | None:
    """Return an existing live session without allocating one."""

    model = getattr(document_or_model, "model", document_or_model)
    active = _ACTIVE_SESSION.get()
    if active is not None and active.model is model:
        return active
    with _SESSION_REGISTRY_LOCK:
        existing = _SESSION_REGISTRY.get(id(model))
        if existing is None:
            return None
        model_reference, strong_model, session_reference = existing
        current_model = (
            model_reference() if model_reference is not None else strong_model
        )
        session = session_reference()
        if current_model is model and session is not None:
            return session
        _SESSION_REGISTRY.pop(id(model), None)
        return None


def active_optimization_session(
    model: Any | None = None,
) -> CircleOptimizationSession | None:
    """Return the context-local session, optionally requiring one model identity."""

    session = _ACTIVE_SESSION.get()
    if session is None:
        return None
    return session if model is None or session.model is model else None


def _remove_session(
    model_id: int,
    expected: weakref.ReferenceType[CircleOptimizationSession] | None = None,
) -> None:
    with _SESSION_REGISTRY_LOCK:
        existing = _SESSION_REGISTRY.get(model_id)
        if existing is None:
            return
        if expected is not None and existing[2] is not expected:
            return
        _SESSION_REGISTRY.pop(model_id, None)


__all__ = [
    "CircleOptimizationSession",
    "CircleOptimizationStatistics",
    "CircleSessionRevision",
    "active_optimization_session",
    "existing_optimization_session",
    "optimization_session_for",
]

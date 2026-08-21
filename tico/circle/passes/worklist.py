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
from dataclasses import dataclass


@dataclass(frozen=True)
class CircleRuleWorkItem:
    """Request a forward rule scan from one subgraph operator index."""

    subgraph_index: int
    start_operator_index: int


class CircleRuleWorklist:
    """Schedule deterministic local scans plus a final global validation sweep."""

    def __init__(self, subgraph_count: int) -> None:
        """Seed one full ordered scan over every current subgraph."""

        count = int(subgraph_count)
        if count < 0:
            raise ValueError("subgraph_count must not be negative.")
        self._order: deque[int] = deque()
        self._starts: dict[int, int] = {}
        self._validation_required = False
        self._schedule_all(count)

    @property
    def pending(self) -> bool:
        """Return whether at least one local scan is queued."""

        return bool(self._order)

    def schedule(
        self,
        subgraph_index: int,
        start_operator_index: int = 0,
        *,
        front: bool = False,
    ) -> None:
        """Queue one subgraph, merging duplicate requests at the earliest index."""

        subgraph = int(subgraph_index)
        start = int(start_operator_index)
        if subgraph < 0:
            raise ValueError("subgraph_index must not be negative.")
        if start < 0:
            raise ValueError("start_operator_index must not be negative.")

        previous = self._starts.get(subgraph)
        if previous is not None:
            self._starts[subgraph] = min(previous, start)
            if front:
                self._order.remove(subgraph)
                self._order.appendleft(subgraph)
            return

        self._starts[subgraph] = start
        if front:
            self._order.appendleft(subgraph)
        else:
            self._order.append(subgraph)

    def pop(self) -> CircleRuleWorkItem | None:
        """Pop the next deterministic scan request, if one exists."""

        if not self._order:
            return None
        subgraph = self._order.popleft()
        start = self._starts.pop(subgraph)
        return CircleRuleWorkItem(subgraph, start)

    def mark_modified(self) -> None:
        """Require one full ordered sweep after all local work becomes stable."""

        self._validation_required = True

    def refill_for_validation(self, subgraph_count: int) -> bool:
        """Seed a global sweep when local work changed the document."""

        if self._order:
            return True
        if not self._validation_required:
            return False
        self._validation_required = False
        self._schedule_all(int(subgraph_count))
        return bool(self._order)

    def _schedule_all(self, subgraph_count: int) -> None:
        """Queue all subgraphs in stable index order from their first operator."""

        if subgraph_count < 0:
            raise ValueError("subgraph_count must not be negative.")
        for subgraph_index in range(subgraph_count):
            self.schedule(subgraph_index)


__all__ = ["CircleRuleWorkItem", "CircleRuleWorklist"]

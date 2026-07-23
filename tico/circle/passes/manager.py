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
from typing import Iterable

from tico.circle.document import CircleDocument
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult


class CirclePassStrategy(str, Enum):
    """Control how a pass pipeline reaches a fixed point."""

    ONCE = "once"
    UNTIL_NO_CHANGE = "until-no-change"
    RESTART = "restart"


@dataclass(frozen=True)
class CirclePassExecution:
    """Record one pass invocation in pipeline order."""

    pass_name: str
    result: CirclePassResult


@dataclass(frozen=True)
class CirclePassManagerResult:
    """Collect all pass invocations performed by a pass manager."""

    executions: tuple[CirclePassExecution, ...]

    @property
    def modified(self) -> bool:
        """Return whether any pass invocation changed the document."""

        return any(execution.result.modified for execution in self.executions)

    @property
    def changes(self) -> int:
        """Return the sum of pass-reported change counts."""

        return sum(execution.result.changes for execution in self.executions)


class CirclePassManager:
    """Run Circle passes with optional fixed-point scheduling and verification."""

    def __init__(
        self,
        passes: Iterable[CirclePass],
        *,
        strategy: CirclePassStrategy = CirclePassStrategy.ONCE,
        maximum_steps: int = 1000,
    ):
        self.passes = tuple(passes)
        self.strategy = strategy
        self.maximum_steps = maximum_steps
        if maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive.")

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext | None = None,
    ) -> CirclePassManagerResult:
        """Run configured passes and return an execution record."""

        context = context or CirclePassContext()
        executions: list[CirclePassExecution] = []
        if not self.passes:
            return CirclePassManagerResult(())

        step = 0
        pass_index = 0
        round_modified = False
        while True:
            if step >= self.maximum_steps:
                raise RuntimeError(
                    f"Circle pass pipeline exceeded {self.maximum_steps} steps; "
                    "a non-converging pass sequence is suspected."
                )
            circle_pass = self.passes[pass_index]
            context.logger.debug("Running Circle pass %s", circle_pass.name)
            result = circle_pass.run(document, context)
            executions.append(CirclePassExecution(circle_pass.name, result))
            step += 1

            if context.verify_after_each_pass:
                document.verify(raise_on_error=True)

            if self.strategy is CirclePassStrategy.ONCE:
                pass_index += 1
                if pass_index == len(self.passes):
                    break
                continue

            if self.strategy is CirclePassStrategy.RESTART:
                if result.modified:
                    pass_index = 0
                else:
                    pass_index += 1
                    if pass_index == len(self.passes):
                        break
                continue

            round_modified = round_modified or result.modified
            pass_index += 1
            if pass_index == len(self.passes):
                if not round_modified:
                    break
                pass_index = 0
                round_modified = False

        return CirclePassManagerResult(tuple(executions))

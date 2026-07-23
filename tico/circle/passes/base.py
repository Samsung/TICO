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
from dataclasses import dataclass, field
from typing import Any

from tico.circle.document import CircleDocument


@dataclass
class CirclePassContext:
    """Provide shared services and state to Circle graph passes."""

    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("tico.circle.passes")
    )
    verify_after_each_pass: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


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

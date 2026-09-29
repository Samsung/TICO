# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

import os
from typing import Literal

from tico.interpreter.backends import (
    DEFAULT_RUNTIME,
    resolve_runtime_name,
    RUNTIME_CHOICES,
)

Runtime = Literal["reference", "circle-interpreter", "onert"]

assert set(RUNTIME_CHOICES) == {"reference", "circle-interpreter", "onert"}


def selected_runtime() -> Runtime:
    """Return the runtime chosen through ``CCEX_RUNTIME`` (default: reference)."""

    name = resolve_runtime_name(os.environ.get("CCEX_RUNTIME") or DEFAULT_RUNTIME)
    return name  # type: ignore[return-value]

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

from __future__ import annotations

from pathlib import Path
from typing import Any

from tico.interpreter import infer
from tico.interpreter.backends import resolve_runtime_name


class CircleModel:
    """Serialized Circle model that can be saved, loaded, and executed.

    ``runtime`` selects how ``__call__`` executes the model. The default,
    ``"reference"``, is the built-in NumPy/PyTorch reference runtime. The names
    ``"circle-interpreter"`` and ``"onert"`` select optional external runtimes
    that must be installed separately; they are never chosen implicitly.
    """

    def __init__(self, circle_binary: bytes, *, runtime: str | None = None):
        self.circle_binary = circle_binary
        self.runtime = resolve_runtime_name(runtime)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return infer.infer_with_runtime(self.circle_binary, self.runtime, args, kwargs)

    @staticmethod
    def load(circle_path: str, *, runtime: str | None = None) -> CircleModel:
        with open(circle_path, "rb") as f:
            buf = bytes(f.read())
        return CircleModel(buf, runtime=runtime)

    def save(self, circle_path: str | Path) -> None:
        with open(circle_path, "wb") as f:
            f.write(self.circle_binary)

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

from typing import Any

import numpy as np

from tico.interpreter.backends import run_circle
from tico.utils.signature import ModelInputSpec


def infer_with_runtime(
    circle_binary: bytes,
    runtime: str | None,
    args: tuple,
    kwargs: dict,
) -> Any:
    """Bind user arguments to the Circle interface and execute with ``runtime``.

    A model with one output returns a single ``numpy.ndarray``; a model with
    several outputs returns a list in the serialized output order.
    """

    input_spec = ModelInputSpec(circle_binary)
    user_inputs = input_spec.bind(args, kwargs, check=True)

    output: list[np.ndarray] = run_circle(circle_binary, user_inputs, runtime=runtime)

    if len(output) == 1:
        return output[0]
    else:
        return output


def infer(circle_binary: bytes, *args: Any, **kwargs: Any) -> Any:
    """Execute a Circle model with the default (reference) runtime."""

    return infer_with_runtime(circle_binary, None, args, kwargs)

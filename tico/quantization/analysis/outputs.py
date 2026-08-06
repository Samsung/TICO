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

"""Model-output normalization for model-independent numerical analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, TypeAlias

import torch


NormalizedOutputs: TypeAlias = dict[str, torch.Tensor]
OutputAdapter: TypeAlias = Callable[[Any], Mapping[str, torch.Tensor]]


def normalize_outputs(
    outputs: Any,
    *,
    output_names: Sequence[str] | None = None,
) -> NormalizedOutputs:
    """Convert common model return values into a named tensor mapping."""
    if isinstance(outputs, torch.Tensor):
        names = tuple(output_names or ("output",))
        if len(names) != 1:
            raise ValueError(
                f"A Tensor result requires one output name, but received {len(names)}."
            )
        return {names[0]: outputs}

    if isinstance(outputs, Mapping):
        normalized: NormalizedOutputs = {}
        for name, value in outputs.items():
            if not isinstance(name, str):
                raise TypeError("Output mapping keys must be strings.")
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Output {name!r} must be a Tensor, got {type(value).__name__}."
                )
            normalized[name] = value
        if not normalized:
            raise ValueError("The output mapping must not be empty.")
        if output_names is not None:
            names = tuple(output_names)
            if set(normalized) != set(names):
                raise ValueError(
                    "Output mapping keys do not match the configured output names: "
                    f"{tuple(normalized)} != {names}."
                )
            return {name: normalized[name] for name in names}
        return normalized

    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        values = tuple(outputs)
        if not values:
            raise ValueError("The model output sequence must not be empty.")
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError(
                "Every value in the model output sequence must be a Tensor."
            )
        names = tuple(
            output_names or (f"output_{index}" for index in range(len(values)))
        )
        if len(names) != len(values):
            raise ValueError(
                f"Expected {len(names)} outputs from configured names, "
                f"got {len(values)}."
            )
        return dict(zip(names, values))

    raise TypeError(
        "Model outputs must be a Tensor, a tensor sequence, or a string-to-Tensor "
        f"mapping; received {type(outputs).__name__}."
    )


def make_output_adapter(
    output_names: Sequence[str] | None = None,
) -> OutputAdapter:
    """Create an adapter that applies ``normalize_outputs`` with fixed names."""
    names = None if output_names is None else tuple(output_names)

    def adapter(outputs: Any) -> NormalizedOutputs:
        return normalize_outputs(outputs, output_names=names)

    return adapter

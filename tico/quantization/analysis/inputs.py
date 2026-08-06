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

"""Model invocation helpers shared by quantization analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TypeAlias

import torch
from torch import nn


@dataclass(frozen=True)
class ModelInvocation:
    """Store positional and keyword arguments for one model invocation."""

    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)


ModelInput: TypeAlias = (
    torch.Tensor | tuple[Any, ...] | Mapping[str, Any] | ModelInvocation
)


def invoke_model(model: nn.Module, sample: ModelInput) -> Any:
    """Invoke ``model`` from a common calibration/evaluation sample format."""
    if isinstance(sample, ModelInvocation):
        return model(*sample.args, **dict(sample.kwargs))
    if isinstance(sample, torch.Tensor):
        return model(sample)
    if isinstance(sample, tuple):
        return model(*sample)
    if isinstance(sample, Mapping):
        return model(**dict(sample))
    raise TypeError(
        "A model sample must be a Tensor, tuple, mapping, or ModelInvocation; "
        f"received {type(sample).__name__}."
    )

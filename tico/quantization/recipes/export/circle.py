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

from pathlib import Path
from typing import Any

import torch

import tico
from tico.utils.utils import SuppressWarning


def export_full_circle(
    *,
    model: Any,
    example_input: Any,
    output_dir: str | Path,
    name: str = "model.q.circle",
    strict: bool = False,
) -> Path:
    """Best-effort full-model Circle export for PTQ/fake-quant models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name

    model = model.eval().cpu()

    export_model = model
    if hasattr(model, "wrapped") and hasattr(model.wrapped, "as_export_module"):
        # LLM wrappers expose export adapters for static prefill/decode paths.
        export_model = model.wrapped.as_export_module("prefill").eval()

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    if isinstance(example_input, torch.Tensor):
        args = (example_input.cpu(),)
        kwargs = {}
    elif isinstance(example_input, tuple):
        args = tuple(
            x.cpu() if isinstance(x, torch.Tensor) else x for x in example_input
        )
        kwargs = {}
    elif isinstance(example_input, dict):
        args = ()
        kwargs = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in example_input.items()
        }
    else:
        args = (example_input,)
        kwargs = {}

    with torch.no_grad(), SuppressWarning(UserWarning, ".*"):
        cm = tico.convert(export_model, args, kwargs=kwargs, strict=strict)
    cm.save(path)
    print(f"Saved Circle model to {path.resolve()}")
    return path

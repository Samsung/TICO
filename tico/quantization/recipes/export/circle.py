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

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

import tico
from tico.quantization.wrapq.utils.static_export import prepared_static_export
from tico.utils.utils import SuppressWarning


def convert_and_save_circle(
    module: torch.nn.Module,
    example_inputs: tuple[Any, ...],
    save_path: Path,
    *,
    kwargs: dict[str, Any] | None = None,
    dynamic_shapes: Any | None = None,
    strict: bool = False,
) -> None:
    """Convert one prepared export stage to Circle and save it."""
    print(f"Saving {save_path.name} to {save_path.resolve()}")
    with torch.no_grad(), SuppressWarning(UserWarning, ".*"):
        circle_model = tico.convert(
            module.eval(),
            example_inputs,
            kwargs=kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
        )
    circle_model.save(save_path)


def export_full_circle(
    *,
    model: Any,
    example_input: Any,
    output_dir: str | Path,
    name: str = "model.q.circle",
    strict: bool = False,
    max_seq_len: int | None = None,
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

    has_static_owner = any(
        callable(getattr(module, "_build_static_export_inputs", None))
        for module in export_model.modules()
    )
    if has_static_owner and max_seq_len is None:
        raise ValueError("Full text export requires an explicit max_seq_len profile.")
    preparation = (
        prepared_static_export(export_model, max_seq_len)
        if max_seq_len is not None else nullcontext()
    )
    with preparation, torch.no_grad(), SuppressWarning(UserWarning, ".*"):
        cm = tico.convert(export_model, args, kwargs=kwargs, strict=strict)
    cm.save(path)
    print(f"Saved Circle model to {path.resolve()}")
    return path

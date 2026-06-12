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

"""Utility helpers shared by wrapper smoke cases and the runner."""

import copy
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 2025, AI systems accelerated hardware-software co-design at scale.",
    "Quantization smoke tests should be short and deterministic.",
    "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    ...",
]


def smoke_section(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the nested debug.wrapper_smoke section if present."""
    debug = cfg.get("debug", {}) if isinstance(cfg, Mapping) else {}
    if isinstance(debug, Mapping):
        section = debug.get("wrapper_smoke", {})
        if isinstance(section, Mapping):
            return section
    return cfg


def cfg_get(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Read a dotted key from a nested mapping."""
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def runtime_device(cfg: Mapping[str, Any]) -> torch.device:
    """Return the configured runtime device."""
    value = cfg_get(cfg, "runtime.device", None)
    if value is None:
        value = cfg_get(smoke_section(cfg), "runtime.device", "cpu")
    return torch.device(str(value))


def calibration_iters(cfg: Mapping[str, Any], default: int = 3) -> int:
    """Return the number of calibration iterations for a smoke case."""
    value = cfg_get(smoke_section(cfg), "calibration_iters", default)
    return int(value)


def clone_module(module: torch.nn.Module) -> torch.nn.Module:
    """Return a detached deep copy of a module in eval mode."""
    return copy.deepcopy(module).eval()


def first_tensor(value: Any) -> torch.Tensor:
    """Return the first tensor found in a nested output object."""
    if isinstance(value, torch.Tensor):
        return value
    for attr in ("logits", "last_hidden_state", "pooler_output"):
        if hasattr(value, attr):
            tensor = getattr(value, attr)
            if isinstance(tensor, torch.Tensor):
                return tensor
    if isinstance(value, Mapping):
        for item in value.values():
            try:
                return first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"No tensor output found in object of type {type(value)!r}.")


def detach_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    """Detach a tensor and move it to CPU float32 for metrics and plotting."""
    if tensor.dtype == torch.bool:
        tensor = tensor.float()
    return tensor.detach().cpu().float()


def finite_all(tensor: torch.Tensor) -> bool:
    """Return whether every element in the tensor is finite."""
    return bool(torch.isfinite(tensor).all().item())


def safe_peir(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Compute PEIR using the existing metric helper with a local fallback."""
    try:
        from tico.quantization.evaluation.metric import compute_peir

        value = compute_peir(reference, candidate)
        if isinstance(value, torch.Tensor):
            return float(value.item())
        return float(value)
    except Exception:
        denom = reference.abs().max().item()
        denom = denom if denom != 0.0 else 1.0
        return float((candidate - reference).abs().max().item() / denom)


@contextmanager
def suppress_tico_warnings() -> Iterator[None]:
    """Suppress noisy warnings during Circle conversion when the helper is available."""
    try:
        from tico.utils.utils import SuppressWarning

        suppressor = SuppressWarning(UserWarning, ".*")
    except Exception:
        suppressor = None

    if suppressor is not None:
        with suppressor:
            yield
    else:
        yield


def ensure_output_dir(path: str | Path | None) -> Path:
    """Create and return the output directory used by a smoke run."""
    output_dir = Path(path or "./out/wrapper_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def format_metric(value: float | None) -> str:
    """Format a numeric metric for CLI output."""
    if value is None:
        return "n/a"
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"

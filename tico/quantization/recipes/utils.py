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

import dataclasses
import random
from pathlib import Path
from typing import Any, Mapping

import torch

from tico.quantization.config.specs import affine, mx, QuantSpec
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme


TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def torch_dtype_from_name(name: str | torch.dtype | None) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    if name is None:
        return torch.float32
    key = str(name).lower()
    if key not in TORCH_DTYPE_MAP:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return TORCH_DTYPE_MAP[key]


def wrapq_dtype_from_name(
    value: str | int | DType | None, *, default: DType | None = None
) -> DType:
    if isinstance(value, DType):
        return value
    if value is None:
        if default is None:
            raise ValueError("DType value is required.")
        return default
    if isinstance(value, int):
        # Historical convention: bit-width shorthand for unsigned weights.
        return DType.uint(value)

    raw = str(value).strip().lower()
    if raw.startswith("int"):
        return DType.int(int(raw[3:]))
    if raw.startswith("uint"):
        return DType.uint(int(raw[4:]))
    raise ValueError(f"Unsupported WrapQ dtype: {value}")


def qscheme_from_name(
    value: str | QScheme | None, *, default: QScheme | None = None
) -> QScheme:
    if isinstance(value, QScheme):
        return value
    if value is None:
        if default is None:
            raise ValueError("QScheme value is required.")
        return default
    key = str(value).lower()
    mapping = {
        "per_tensor_asymm": QScheme.PER_TENSOR_ASYMM,
        "per_tensor_symm": QScheme.PER_TENSOR_SYMM,
        "per_channel_asymm": QScheme.PER_CHANNEL_ASYMM,
        "per_channel_symm": QScheme.PER_CHANNEL_SYMM,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported qscheme: {value}")
    return mapping[key]


def quant_spec_from_config(
    value: Any,
    *,
    default: QuantSpec | None = None,
) -> QuantSpec | None:
    """Parse a recipe quantization spec into a QuantSpec.

    Supported forms:
      - ``None``: returns ``default``.
      - ``QuantSpec``: returned as-is.
      - scalar dtype/bit-width: affine spec, e.g. ``"int16"`` or ``4``.
      - mapping with ``kind: affine``: affine spec with ``dtype`` and optional
        ``qscheme``.
      - mapping with ``kind: mx``: MX spec with ``elem_format`` and MX kwargs.

    Examples:
        ``activation: int16``
        ``linear_weight: uint4``
        ``activation: {kind: mx, elem_format: fp8_e4m3, axis: -1}``
    """
    if value is None:
        return default
    if isinstance(value, QuantSpec):
        return value

    if isinstance(value, Mapping):
        kind = str(value.get("kind", value.get("type", "affine"))).strip().lower()

        if kind == "mx":
            return mx(
                str(value.get("elem_format", "fp8_e4m3")),
                axis=int(value.get("axis", -1)),
                shared_exp_method=str(value.get("shared_exp_method", "max")),
                round=str(value.get("round", "nearest")),
            )

        if kind == "affine":
            if "dtype" not in value:
                raise ValueError("Affine quant spec mapping requires a 'dtype' field.")
            qscheme = (
                qscheme_from_name(value.get("qscheme"))
                if value.get("qscheme") is not None
                else None
            )
            return affine(
                wrapq_dtype_from_name(value["dtype"]),
                qscheme=qscheme,
            )

        raise ValueError(f"Unsupported quant spec kind: {kind!r}")

    return affine(wrapq_dtype_from_name(value))


def quant_specs_equivalent(left: Any, right: Any) -> bool:
    """Return True if two recipe spec values resolve to the same QuantSpec."""
    if left is None or right is None:
        return left is right
    return quant_spec_from_config(left) == quant_spec_from_config(right)


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(value: Any, device: str | torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    return value


def filter_dataclass_kwargs(
    cls: type[Any], mapping: Mapping[str, Any]
) -> dict[str, Any]:
    valid = {field.name for field in dataclasses.fields(cls)}
    return {k: v for k, v in mapping.items() if k in valid}


def stage_payload(stage_cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in dict(stage_cfg).items() if k not in {"name", "enabled"}}


def ensure_output_dir(path: str | Path | None) -> Path:
    out = Path(path or "./out")
    out.mkdir(parents=True, exist_ok=True)
    return out

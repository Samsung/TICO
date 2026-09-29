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

from typing import Optional

import torch

from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme


TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
    "double": torch.float64,
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


def dtype_is_unsigned(dtype: DType) -> bool:
    """
    Return True when the dtype is unsigned.
    """
    return not dtype.signed


def auto_qscheme_for(dtype: DType, obs_name: Optional[str] = None) -> QScheme:
    """
    Choose the default qscheme associated with a dtype and observer name.

    Default policy:
      - signed dtype    -> symmetric per-tensor
      - unsigned dtype  -> asymmetric per-tensor
      - unsigned weight -> asymmetric per-channel
    """
    if dtype_is_unsigned(dtype):
        if obs_name == "weight":
            return QScheme.PER_CHANNEL_ASYMM
        return QScheme.PER_TENSOR_ASYMM
    return QScheme.PER_TENSOR_SYMM

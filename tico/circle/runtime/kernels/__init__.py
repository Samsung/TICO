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

"""Reference kernels for Circle builtin operators."""

from __future__ import annotations

from functools import lru_cache

from tico.circle.runtime.kernels.base import (
    ExecutionMode,
    Kernel,
    KernelContext,
    KernelRegistry,
    optional_builtin_code,
)
from tico.circle.runtime.kernels.elementwise import register_elementwise_kernels
from tico.circle.runtime.kernels.nn import register_nn_kernels
from tico.circle.runtime.kernels.quantize import register_quantize_kernels
from tico.circle.runtime.kernels.reduction import register_reduction_kernels
from tico.circle.runtime.kernels.shape_ops import register_shape_kernels

# Operators that move or select values without changing them. In NATIVE mode
# they may carry quantized integer tensors through unchanged; in FAKE_QUANTIZE
# mode they pass dequantized values through without a requantization step, which
# matches ONE's ConvertToFakeQuantizedModelPass treatment of these operators.
VALUE_PRESERVING_BUILTIN_NAMES = (
    "RESHAPE",
    "TRANSPOSE",
    "SQUEEZE",
    "EXPAND_DIMS",
    "BROADCAST_TO",
    "CONCATENATION",
    "SPLIT",
    "SPLIT_V",
    "SLICE",
    "STRIDED_SLICE",
    "GATHER",
    "GATHER_ND",
    "SELECT",
    "SELECT_V2",
    "PAD",
    "PADV2",
    "CAST",
    "DEPTH_TO_SPACE",
    "SPACE_TO_DEPTH",
    "PACK",
    "UNPACK",
    "TILE",
)

# Operators whose outputs are never quantized activations (indices, shapes).
INDEX_BUILTIN_NAMES = ("ARG_MAX", "ARG_MIN", "SHAPE")

# Operators that convert between quantized and float representations.
CONVERSION_BUILTIN_NAMES = ("QUANTIZE", "DEQUANTIZE")


def _codes(names: tuple[str, ...]) -> frozenset[int]:
    codes = (optional_builtin_code(name) for name in names)
    return frozenset(code for code in codes if code is not None)


@lru_cache(maxsize=1)
def value_preserving_builtin_codes() -> frozenset[int]:
    """Return builtin codes of value-preserving data-movement operators."""

    return _codes(VALUE_PRESERVING_BUILTIN_NAMES)


@lru_cache(maxsize=1)
def index_builtin_codes() -> frozenset[int]:
    """Return builtin codes of operators that produce indices or shapes."""

    return _codes(INDEX_BUILTIN_NAMES)


@lru_cache(maxsize=1)
def conversion_builtin_codes() -> frozenset[int]:
    """Return builtin codes of QUANTIZE and DEQUANTIZE."""

    return _codes(CONVERSION_BUILTIN_NAMES)


def default_kernel_registry() -> KernelRegistry:
    """Create a registry holding every builtin kernel of the reference runtime."""

    registry = KernelRegistry()
    register_elementwise_kernels(registry)
    register_shape_kernels(registry)
    register_reduction_kernels(registry)
    register_nn_kernels(registry)
    register_quantize_kernels(registry)
    return registry


__all__ = [
    "CONVERSION_BUILTIN_NAMES",
    "INDEX_BUILTIN_NAMES",
    "VALUE_PRESERVING_BUILTIN_NAMES",
    "ExecutionMode",
    "Kernel",
    "KernelContext",
    "KernelRegistry",
    "conversion_builtin_codes",
    "default_kernel_registry",
    "index_builtin_codes",
    "value_preserving_builtin_codes",
]

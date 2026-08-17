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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tico.circle.analysis import TensorContract
from tico.circle.builder import CircleBuilder
from tico.circle.value import TensorValue, TensorValueCodec

from test.unit_test.circle.infrastructure_fixture import (
    fake_object_factory,
    FLOAT32,
    INT32,
    make_registry,
)

ADD = 1
CAST = 2
EXPAND_DIMS = 3
PACK = 4
PADV2 = 5
PAD = 6
SPLIT_V = 7
SPLIT = 8
SQUEEZE = 9
STRIDED_SLICE = 10
TRANSPOSE = 11
RESHAPE = 12
SLICE = 13
ABS = 14
MUL = 15
MEAN = 16

BUILTIN_CODES = {
    "ADD": ADD,
    "CAST": CAST,
    "EXPAND_DIMS": EXPAND_DIMS,
    "PACK": PACK,
    "PADV2": PADV2,
    "PAD": PAD,
    "SPLIT_V": SPLIT_V,
    "SPLIT": SPLIT,
    "SQUEEZE": SQUEEZE,
    "STRIDED_SLICE": STRIDED_SLICE,
    "TRANSPOSE": TRANSPOSE,
    "RESHAPE": RESHAPE,
    "SLICE": SLICE,
    "ABS": ABS,
    "MUL": MUL,
    "MEAN": MEAN,
}
BUILTIN_OPTIONS_TYPES = {
    "ReshapeOptions": 1,
    "PadOptions": 2,
    "SplitOptions": 3,
}
TENSOR_TYPES = {"INT32": INT32}


@dataclass
class ReshapeOptions:
    """Provide the static target shape used by RESHAPE tests."""

    newShape: list[int] = field(default_factory=list)


@dataclass
class PadOptions:
    """Provide the empty options table used by PAD tests."""


@dataclass
class SplitOptions:
    """Provide the output count used by SPLIT and SPLIT_V tests."""

    numSplits: int = 1


@dataclass
class PackOptions:
    """Provide axis and input count used by PACK tests."""

    axis: int = 0
    valuesCount: int = 1


@dataclass
class SqueezeOptions:
    """Provide dimensions removed by SQUEEZE tests."""

    squeezeDims: list[int] = field(default_factory=list)


@dataclass
class StridedSliceOptions:
    """Provide mask fields used by STRIDED_SLICE tests."""

    beginMask: int = 0
    endMask: int = 0
    ellipsisMask: int = 0
    newAxisMask: int = 0
    shrinkAxisMask: int = 0


@dataclass
class BinaryOptions:
    """Provide fused-activation fields used by ADD and MUL tests."""

    fusedActivationFunction: int = 0
    potScaleInt16: bool = False


@dataclass
class CastOptions:
    """Provide source and target tensor types used by CAST tests."""

    inDataType: int = FLOAT32
    outDataType: int = FLOAT32


@dataclass
class ReducerOptions:
    """Provide keep-dim behavior used by MEAN tests."""

    keepDims: bool = True


def optimization_object_factory(table_name: str) -> Any:
    """Create core fake tables and optimization-specific option tables."""

    options = {
        "ReshapeOptions": ReshapeOptions,
        "PadOptions": PadOptions,
        "SplitOptions": SplitOptions,
    }
    factory = options.get(table_name)
    if factory is not None:
        return factory()
    return fake_object_factory(table_name)


def make_codec() -> TensorValueCodec:
    """Create the schema-independent codec used by optimization tests."""

    return TensorValueCodec(make_registry())


def make_builder(document, codec: TensorValueCodec) -> CircleBuilder:
    """Create a CircleBuilder using fake Object API tables."""

    return CircleBuilder(
        document,
        codec=codec,
        object_factory=optimization_object_factory,
    )


def add_i32(builder: CircleBuilder, name: str, values: Any) -> int:
    """Add one INT32 constant tensor."""

    return builder.add_constant(
        name,
        TensorValue.from_values(
            INT32,
            np.asarray(values, dtype=np.int32),
            dtype=np.int32,
        ),
    )


def add_f32(builder: CircleBuilder, name: str, values: Any) -> int:
    """Add one FLOAT32 constant tensor."""

    return builder.add_constant(
        name,
        TensorValue.from_values(
            FLOAT32,
            np.asarray(values, dtype=np.float32),
            dtype=np.float32,
        ),
    )


def static_contract(shape: tuple[int, ...], tensor_type: int = FLOAT32):
    """Create one static dense tensor contract."""

    return TensorContract(
        tensor_type=tensor_type,
        shape=shape,
        shape_signature=shape,
    )


def pass_kwargs(codec: TensorValueCodec) -> dict[str, Any]:
    """Return schema-independent constructor arguments shared by PR 3 passes."""

    return {
        "builtin_codes": BUILTIN_CODES,
        "builtin_options_types": BUILTIN_OPTIONS_TYPES,
        "tensor_types": TENSOR_TYPES,
        "codec": codec,
        "object_factory": optimization_object_factory,
    }

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
from tico.circle.value import (
    TensorQuantization,
    TensorTypeSpec,
    TensorValue,
    TensorValueCodec,
)

from test.unit_test.circle.infrastructure_fixture import (
    fake_object_factory,
    FLOAT32,
    INT32,
    INT8,
    make_registry,
)

FLOAT16 = 1
INT64 = 4

CUSTOM = 100
ADD = 101
MUL = 102
BATCH_MATMUL = 103
SPLIT_V = 104
MAX_POOL_2D = 105
FULLY_CONNECTED = 106
RESHAPE = 107
RELU = 108
RELU_N1_TO_1 = 109
RELU6 = 110
TANH = 111
GELU = 112
SLICE = 113
TRANSPOSE_CONV = 114
DEQUANTIZE = 115
DEPTHWISE_CONV_2D = 116
DENSIFY = 117
SPARSE_TO_DENSE = 118

ADD_OPTIONS = 200
MUL_OPTIONS = 201
BATCH_MATMUL_OPTIONS = 202
SPLIT_V_OPTIONS = 203
POOL_2D_OPTIONS = 204
FULLY_CONNECTED_OPTIONS = 205
RESHAPE_OPTIONS = 206
GELU_OPTIONS = 207
SLICE_OPTIONS = 208
TRANSPOSE_CONV_OPTIONS = 209

ACTIVATION_NONE = 0
ACTIVATION_RELU = 1
ACTIVATION_RELU_N1_TO_1 = 2
ACTIVATION_RELU6 = 3
ACTIVATION_TANH = 4

PADDING_SAME = 0
PADDING_VALID = 1

BUILTIN_CODES = {
    "CUSTOM": CUSTOM,
    "ADD": ADD,
    "MUL": MUL,
    "BATCH_MATMUL": BATCH_MATMUL,
    "SPLIT_V": SPLIT_V,
    "MAX_POOL_2D": MAX_POOL_2D,
    "FULLY_CONNECTED": FULLY_CONNECTED,
    "RESHAPE": RESHAPE,
    "RELU": RELU,
    "RELU_N1_TO_1": RELU_N1_TO_1,
    "RELU6": RELU6,
    "TANH": TANH,
    "GELU": GELU,
    "SLICE": SLICE,
    "TRANSPOSE_CONV": TRANSPOSE_CONV,
    "DEQUANTIZE": DEQUANTIZE,
    "DEPTHWISE_CONV_2D": DEPTHWISE_CONV_2D,
    "DENSIFY": DENSIFY,
    "SPARSE_TO_DENSE": SPARSE_TO_DENSE,
}

BUILTIN_OPTIONS_TYPES = {
    "AddOptions": ADD_OPTIONS,
    "MulOptions": MUL_OPTIONS,
    "BatchMatMulOptions": BATCH_MATMUL_OPTIONS,
    "SplitVOptions": SPLIT_V_OPTIONS,
    "Pool2DOptions": POOL_2D_OPTIONS,
    "FullyConnectedOptions": FULLY_CONNECTED_OPTIONS,
    "ReshapeOptions": RESHAPE_OPTIONS,
    "GeluOptions": GELU_OPTIONS,
    "SliceOptions": SLICE_OPTIONS,
    "TransposeConvOptions": TRANSPOSE_CONV_OPTIONS,
}

TENSOR_TYPES = {
    "FLOAT16": FLOAT16,
    "FLOAT32": FLOAT32,
    "INT8": INT8,
    "INT32": INT32,
    "INT64": INT64,
}

PADDING_VALUES = {"SAME": PADDING_SAME, "VALID": PADDING_VALID}
ACTIVATION_VALUES = {
    "NONE": ACTIVATION_NONE,
    "RELU": ACTIVATION_RELU,
    "RELU_N1_TO_1": ACTIVATION_RELU_N1_TO_1,
    "RELU6": ACTIVATION_RELU6,
    "TANH": ACTIVATION_TANH,
}


@dataclass
class BinaryOptions:
    """Provide fused activation fields for ADD and MUL compatibility tests."""

    fusedActivationFunction: int = ACTIVATION_NONE
    potScaleInt16: bool = False


@dataclass
class BatchMatMulOptions:
    """Provide adjoint fields for BATCH_MATMUL compatibility tests."""

    adjointLhs: bool = False
    adjointRhs: bool = False
    asymmetricQuantizeInputs: bool = False


@dataclass
class SplitVOptions:
    """Provide the serialized output count for SPLIT_V tests."""

    numSplits: int = 1


@dataclass
class Pool2DOptions:
    """Provide static pooling geometry for compatibility tests."""

    padding: int = PADDING_VALID
    strideH: int = 1
    strideW: int = 1
    filterHeight: int = 1
    filterWidth: int = 1
    fusedActivationFunction: int = ACTIVATION_NONE


@dataclass
class FullyConnectedOptions:
    """Provide default FullyConnected option fields for compatibility tests."""

    fusedActivationFunction: int = ACTIVATION_NONE
    weightsFormat: int = 0
    keepNumDims: bool = False
    asymmetricQuantizeInputs: bool = False


@dataclass
class ReshapeOptions:
    """Provide the static target shape used by legalization tests."""

    newShape: list[int] = field(default_factory=list)


@dataclass
class GeluOptions:
    """Provide exact or approximate GELU selection."""

    approximate: bool = False


@dataclass
class SliceOptions:
    """Provide the empty builtin options table used by SLICE."""


@dataclass
class TransposeConvOptions:
    """Provide static TransposeConv geometry and activation fields."""

    padding: int = PADDING_VALID
    strideH: int = 1
    strideW: int = 1
    fusedActivationFunction: int = ACTIVATION_NONE


@dataclass
class DepthwiseConv2DOptions:
    """Provide static depthwise-convolution geometry for heavy folding tests."""

    padding: int = PADDING_VALID
    strideH: int = 1
    strideW: int = 1
    depthMultiplier: int = 1
    dilationHFactor: int = 1
    dilationWFactor: int = 1
    fusedActivationFunction: int = ACTIVATION_NONE


@dataclass
class SparseIndexVector:
    """Provide one sparse-index union payload with a values vector."""

    values: list[int] = field(default_factory=list)


@dataclass
class DimensionMetadata:
    """Provide dense or sparse-CSR metadata for DENSIFY tests."""

    format: int = 0
    denseSize: int = 0
    arraySegments: Any = None
    arrayIndices: Any = None


@dataclass
class SparsityParameters:
    """Provide unblocked traversal metadata for one sparse constant."""

    traversalOrder: list[int] = field(default_factory=list)
    blockMap: list[int] = field(default_factory=list)
    dimMetadata: list[DimensionMetadata] = field(default_factory=list)


def compatibility_object_factory(table_name: str) -> Any:
    """Create fake Object API tables for heavy and compatibility tests."""

    options = {
        "AddOptions": BinaryOptions,
        "BatchMatMulOptions": BatchMatMulOptions,
        "FullyConnectedOptions": FullyConnectedOptions,
        "GeluOptions": GeluOptions,
        "MulOptions": BinaryOptions,
        "Pool2DOptions": Pool2DOptions,
        "ReshapeOptions": ReshapeOptions,
        "SliceOptions": SliceOptions,
        "SplitVOptions": SplitVOptions,
        "TransposeConvOptions": TransposeConvOptions,
    }
    factory = options.get(table_name)
    if factory is not None:
        return factory()
    return fake_object_factory(table_name)


def make_codec() -> TensorValueCodec:
    """Create a schema-independent codec including FLOAT16 and INT64."""

    registry = make_registry()
    registry.register(
        TensorTypeSpec(
            "FLOAT16",
            FLOAT16,
            np.dtype(np.float16),
            np.dtype(np.float16),
            16,
        )
    )
    registry.register(
        TensorTypeSpec(
            "INT64",
            INT64,
            np.dtype(np.int64),
            np.dtype(np.int64),
            64,
            signed=True,
        )
    )
    return TensorValueCodec(registry)


def make_builder(document, codec: TensorValueCodec) -> CircleBuilder:
    """Create a CircleBuilder using the compatibility fake factory."""

    return CircleBuilder(
        document,
        codec=codec,
        object_factory=compatibility_object_factory,
    )


def add_constant(
    builder: CircleBuilder,
    name: str,
    values: Any,
    tensor_type: int,
    dtype: Any,
    *,
    quantization: TensorQuantization | None = None,
) -> int:
    """Add one typed inline constant and return its tensor index."""

    return builder.add_constant(
        name,
        TensorValue.from_values(
            tensor_type,
            np.asarray(values, dtype=dtype),
            dtype=dtype,
            quantization=quantization,
        ),
    )


def static_contract(
    shape: tuple[int, ...],
    tensor_type: int = FLOAT32,
    *,
    quantization: TensorQuantization | None = None,
) -> TensorContract:
    """Create one static dense immutable tensor contract."""

    return TensorContract(
        tensor_type=tensor_type,
        shape=shape,
        shape_signature=shape,
        quantization=quantization,
    )


def compatibility_pass_kwargs(codec: TensorValueCodec) -> dict[str, Any]:
    """Return schema-independent constructor arguments for compatibility passes."""

    return {
        "builtin_codes": BUILTIN_CODES,
        "builtin_options_types": BUILTIN_OPTIONS_TYPES,
        "tensor_types": TENSOR_TYPES,
        "activation_none": ACTIVATION_NONE,
        "codec": codec,
        "object_factory": compatibility_object_factory,
    }

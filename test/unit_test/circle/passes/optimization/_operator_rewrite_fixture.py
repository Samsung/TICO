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

from dataclasses import dataclass
from typing import Any

from test.unit_test.circle.infrastructure_fixture import FLOAT32, INT32
from test.unit_test.circle.passes.optimization._fixture import (
    ABS,
    ADD,
    MEAN,
    MUL,
    optimization_object_factory,
    TRANSPOSE,
)

DIV = 30
SQRT = 31
RSQRT = 32
RELU = 33
RELU6 = 34
RELU_N1_TO_1 = 35
MINIMUM = 36
MAXIMUM = 37
PRELU = 38
GELU = 39
CUSTOM = 40
INSTANCE_NORM = 41
SUB = 42
CONV_2D = 43
CONCATENATION = 44
TRANSPOSE_CONV = 45

ACTIVATION_NONE = 0
ACTIVATION_RELU = 100
ACTIVATION_RELU6 = 101
ACTIVATION_RELU_N1_TO_1 = 102

ADD_OPTIONS = 50
MUL_OPTIONS = 51
SUB_OPTIONS = 52
DIV_OPTIONS = 53
REDUCER_OPTIONS = 54
GELU_OPTIONS = 55
INSTANCE_NORM_OPTIONS = 56
CONV_2D_OPTIONS = 57

BUILTIN_CODES = {
    "ABS": ABS,
    "ADD": ADD,
    "CONCATENATION": CONCATENATION,
    "CONV_2D": CONV_2D,
    "CUSTOM": CUSTOM,
    "DIV": DIV,
    "GELU": GELU,
    "INSTANCE_NORM": INSTANCE_NORM,
    "MAXIMUM": MAXIMUM,
    "MEAN": MEAN,
    "MINIMUM": MINIMUM,
    "MUL": MUL,
    "PRELU": PRELU,
    "RELU": RELU,
    "RELU6": RELU6,
    "RELU_N1_TO_1": RELU_N1_TO_1,
    "RSQRT": RSQRT,
    "SQRT": SQRT,
    "SUB": SUB,
    "TRANSPOSE": TRANSPOSE,
    "TRANSPOSE_CONV": TRANSPOSE_CONV,
}

BUILTIN_OPTIONS_TYPES = {
    "AddOptions": ADD_OPTIONS,
    "Conv2DOptions": CONV_2D_OPTIONS,
    "DivOptions": DIV_OPTIONS,
    "GeluOptions": GELU_OPTIONS,
    "InstanceNormOptions": INSTANCE_NORM_OPTIONS,
    "MulOptions": MUL_OPTIONS,
    "ReducerOptions": REDUCER_OPTIONS,
    "SubOptions": SUB_OPTIONS,
}

TENSOR_TYPES = {"FLOAT32": FLOAT32, "INT32": INT32, "INT64": 4}


@dataclass
class BinaryOptions:
    """Provide binary fused-activation fields for operator-rewrite tests."""

    fusedActivationFunction: int = ACTIVATION_NONE
    potScaleInt16: bool = False


@dataclass
class ReducerOptions:
    """Provide reduction keep-dim behavior for operator-rewrite tests."""

    keepDims: bool = False


@dataclass
class GeluOptions:
    """Provide exact or approximate GELU selection for operator-rewrite tests."""

    approximate: bool = False


@dataclass
class InstanceNormOptions:
    """Provide instance-normalization epsilon and activation fields."""

    epsilon: float = 1e-5
    fusedActivationFunction: int = ACTIVATION_NONE


@dataclass
class ConvOptions:
    """Provide a producer fused-activation field for activation tests."""

    fusedActivationFunction: int = ACTIVATION_NONE


def operator_rewrite_object_factory(table_name: str) -> Any:
    """Create fake generated tables used by operator-rewrite tests."""

    options = {
        "AddOptions": BinaryOptions,
        "Conv2DOptions": ConvOptions,
        "DivOptions": BinaryOptions,
        "GeluOptions": GeluOptions,
        "InstanceNormOptions": InstanceNormOptions,
        "MulOptions": BinaryOptions,
        "ReducerOptions": ReducerOptions,
        "SubOptions": BinaryOptions,
    }
    factory = options.get(table_name)
    if factory is not None:
        return factory()
    return optimization_object_factory(table_name)

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

from tico.circle.document import CircleDocument
from tico.circle.value import TensorTypeRegistry, TensorTypeSpec

FLOAT32 = 0
INT32 = 2
UINT8 = 3
INT8 = 9
INT4 = 17
UINT4 = 19


@dataclass
class FakeBuffer:
    """Provide generated Buffer fields used by infrastructure tests."""

    data: Any = None
    offset: int = 0
    size: int = 0


@dataclass
class FakeQuantizationParameters:
    """Provide generated affine quantization fields used by tests."""

    scale: list[float] = field(default_factory=list)
    zeroPoint: list[int] = field(default_factory=list)
    min: list[float] = field(default_factory=list)
    max: list[float] = field(default_factory=list)
    quantizedDimension: int = 0
    detailsType: int = 0
    details: Any = None


@dataclass
class FakeTensor:
    """Provide generated Tensor fields used by infrastructure tests."""

    name: str = ""
    buffer: int = 0
    shape: list[int] = field(default_factory=list)
    shapeSignature: list[int] | None = None
    type: int = FLOAT32
    quantization: Any = None
    isVariable: bool = False
    sparsity: Any = None
    hasRank: bool = False
    variantTensors: Any = None


@dataclass
class FakeOperator:
    """Provide generated Operator fields used by infrastructure tests."""

    opcodeIndex: int = 0
    inputs: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    intermediates: list[int] = field(default_factory=list)
    mutatingVariableInputs: list[bool] = field(default_factory=list)
    builtinOptionsType: int = 0
    builtinOptions: Any = None
    builtinOptions2Type: int = 0
    builtinOptions2: Any = None
    customOptionsFormat: int = 0
    customOptions: Any = None
    largeCustomOptionsOffset: int = 0
    largeCustomOptionsSize: int = 0


@dataclass
class FakeOperatorCode:
    """Provide generated OperatorCode fields used by infrastructure tests."""

    builtinCode: int = 0
    customCode: str | None = None
    deprecatedBuiltinCode: int = 0
    version: int = 1


@dataclass
class FakeTensorMap:
    """Provide generated TensorMap fields used by infrastructure tests."""

    name: str = ""
    tensorIndex: int = 0


@dataclass
class FakeSignatureDef:
    """Provide generated SignatureDef fields used by infrastructure tests."""

    signatureKey: str = "serving_default"
    subgraphIndex: int = 0
    inputs: list[FakeTensorMap] = field(default_factory=list)
    outputs: list[FakeTensorMap] = field(default_factory=list)


@dataclass
class FakeSubGraph:
    """Provide generated SubGraph fields used by infrastructure tests."""

    name: str = "main"
    tensors: list[FakeTensor] = field(default_factory=list)
    inputs: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    operators: list[FakeOperator] = field(default_factory=list)


@dataclass
class FakeModel:
    """Provide generated Model fields used by infrastructure tests."""

    subgraphs: list[FakeSubGraph] = field(default_factory=list)
    buffers: list[FakeBuffer] = field(default_factory=lambda: [FakeBuffer()])
    operatorCodes: list[FakeOperatorCode] = field(default_factory=list)
    signatureDefs: list[FakeSignatureDef] = field(default_factory=list)
    metadataBuffer: list[int] = field(default_factory=list)
    metadata: list[Any] = field(default_factory=list)
    version: int = 0
    description: str = "infrastructure-fixture"


@dataclass
class FakeDetails:
    """Provide an arbitrary nested quantization-details object."""

    blockSize: int = 32
    axes: list[int] = field(default_factory=lambda: [0, 1])


@dataclass
class FakeOptions:
    """Provide a simple builtin-options table for snapshot tests."""

    value: int = 0


def fake_object_factory(table_name: str) -> Any:
    """Create lightweight generated-table substitutes by schema table name."""

    factories = {
        "Buffer": FakeBuffer,
        "Operator": FakeOperator,
        "OperatorCode": FakeOperatorCode,
        "QuantizationParameters": FakeQuantizationParameters,
        "Tensor": FakeTensor,
    }
    try:
        return factories[table_name]()
    except KeyError as error:
        raise KeyError(f"Unsupported fake Circle table: {table_name!r}.") from error


def make_registry() -> TensorTypeRegistry:
    """Create a schema-independent registry covering test tensor types."""

    return TensorTypeRegistry(
        (
            TensorTypeSpec(
                "FLOAT32",
                FLOAT32,
                np.dtype(np.float32),
                np.dtype(np.float32),
                32,
            ),
            TensorTypeSpec(
                "INT32",
                INT32,
                np.dtype(np.int32),
                np.dtype(np.int32),
                32,
                signed=True,
            ),
            TensorTypeSpec(
                "UINT8",
                UINT8,
                np.dtype(np.uint8),
                np.dtype(np.uint8),
                8,
                signed=False,
            ),
            TensorTypeSpec(
                "INT8",
                INT8,
                np.dtype(np.int8),
                np.dtype(np.int8),
                8,
                signed=True,
            ),
            TensorTypeSpec(
                "INT4",
                INT4,
                np.dtype(np.int8),
                np.dtype(np.uint8),
                4,
                signed=True,
                packed=True,
            ),
            TensorTypeSpec(
                "UINT4",
                UINT4,
                np.dtype(np.uint8),
                np.dtype(np.uint8),
                4,
                signed=False,
                packed=True,
            ),
        )
    )


def make_empty_document(*, subgraph_count: int = 1) -> CircleDocument:
    """Create an empty model with reserved buffer zero and requested subgraphs."""

    model = FakeModel(
        subgraphs=[
            FakeSubGraph(name=f"subgraph_{index}") for index in range(subgraph_count)
        ]
    )
    return CircleDocument(model)


def add_runtime_tensor(
    document: CircleDocument,
    *,
    subgraph_index: int,
    name: str,
    shape: list[int],
    tensor_type: int = FLOAT32,
) -> int:
    """Append a non-constant tensor to a fixture subgraph."""

    tensor = FakeTensor(
        name=name,
        shape=list(shape),
        shapeSignature=list(shape),
        type=tensor_type,
        buffer=0,
    )
    subgraph = document.subgraph(subgraph_index)
    subgraph.tensors.append(tensor)
    return len(subgraph.tensors) - 1

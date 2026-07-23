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

from tico.circle.document import CircleDocument


@dataclass
class FakeBuffer:
    """Provide the buffer fields used by Circle artifact tests."""

    data: bytes | None = None
    offset: int = 0
    size: int = 0


@dataclass
class FakeTensor:
    """Provide the tensor fields used by Circle artifact tests."""

    name: str
    buffer: int = 0
    shape: list[int] = field(default_factory=list)
    shapeSignature: list[int] | None = None
    type: int = 0
    quantization: Any = None
    isVariable: bool = False


@dataclass
class FakeOperator:
    """Provide the operator fields used by Circle artifact tests."""

    opcodeIndex: int
    inputs: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    intermediates: list[int] | None = None
    builtinOptions: Any = None
    builtinOptions2: Any = None


@dataclass
class FakeOperatorCode:
    """Provide the operator-code fields used by Circle artifact tests."""

    builtinCode: int
    customCode: str | None = None
    deprecatedBuiltinCode: int = 0
    version: int = 1


@dataclass
class FakeSubGraph:
    """Provide the subgraph fields used by Circle artifact tests."""

    tensors: list[FakeTensor]
    inputs: list[int]
    outputs: list[int]
    operators: list[FakeOperator]
    name: str = "subgraph"


@dataclass
class FakeTensorMap:
    """Provide the signature tensor-map fields used by tests."""

    name: str
    tensorIndex: int


@dataclass
class FakeSignatureDef:
    """Provide the signature fields used by Circle artifact tests."""

    signatureKey: str
    subgraphIndex: int
    inputs: list[FakeTensorMap]
    outputs: list[FakeTensorMap]


@dataclass
class FakeMetadata:
    """Provide the metadata fields used by Circle artifact tests."""

    name: str
    buffer: int


@dataclass
class FakeModel:
    """Provide the model fields used by Circle artifact tests."""

    subgraphs: list[FakeSubGraph]
    buffers: list[FakeBuffer]
    operatorCodes: list[FakeOperatorCode]
    signatureDefs: list[FakeSignatureDef] = field(default_factory=list)
    metadataBuffer: list[int] = field(default_factory=list)
    metadata: list[FakeMetadata] = field(default_factory=list)
    version: int = 0
    description: str = "fixture"


def make_test_document() -> CircleDocument:
    """Create a two-subgraph model with dead objects and a shared weight buffer."""

    buffers = [
        FakeBuffer(),
        FakeBuffer(data=b"shared-weight"),
        FakeBuffer(data=b"dead-constant"),
        FakeBuffer(),
    ]
    operator_codes = [
        FakeOperatorCode(builtinCode=0),
        FakeOperatorCode(builtinCode=1),
        FakeOperatorCode(builtinCode=2),
    ]

    primary = FakeSubGraph(
        name="primary",
        tensors=[
            FakeTensor("x", buffer=0, shape=[1]),
            FakeTensor("shared_weight", buffer=1, shape=[1]),
            FakeTensor("add_out", buffer=0, shape=[1]),
            FakeTensor("dead_const", buffer=2, shape=[1]),
            FakeTensor("dead_out", buffer=0, shape=[1]),
            FakeTensor("output", buffer=0, shape=[1]),
            FakeTensor("orphan", buffer=3, shape=[1]),
        ],
        inputs=[0],
        outputs=[5],
        operators=[
            FakeOperator(opcodeIndex=0, inputs=[0, 1], outputs=[2]),
            FakeOperator(opcodeIndex=1, inputs=[2, 1], outputs=[5]),
            FakeOperator(opcodeIndex=0, inputs=[3, 1], outputs=[4]),
        ],
    )
    secondary = FakeSubGraph(
        name="secondary",
        tensors=[
            FakeTensor("y", buffer=0, shape=[1]),
            FakeTensor("shared_weight_secondary", buffer=1, shape=[1]),
            FakeTensor("secondary_output", buffer=0, shape=[1]),
        ],
        inputs=[0],
        outputs=[2],
        operators=[
            FakeOperator(opcodeIndex=1, inputs=[0, 1], outputs=[2]),
        ],
    )
    signatures = [
        FakeSignatureDef(
            signatureKey="primary",
            subgraphIndex=0,
            inputs=[FakeTensorMap("x", 0)],
            outputs=[FakeTensorMap("output", 5)],
        ),
        FakeSignatureDef(
            signatureKey="secondary",
            subgraphIndex=1,
            inputs=[FakeTensorMap("y", 0)],
            outputs=[FakeTensorMap("secondary_output", 2)],
        ),
    ]
    model = FakeModel(
        subgraphs=[primary, secondary],
        buffers=buffers,
        operatorCodes=operator_codes,
        signatureDefs=signatures,
        metadataBuffer=[0],
    )
    return CircleDocument(model)

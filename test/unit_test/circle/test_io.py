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

import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tico.circle._schema import decode_text
from tico.circle.document import CircleDocument
from tico.circle.errors import CircleIOError
from tico.circle.io import (
    model_from_bytes,
    model_to_bytes,
    read_circle_bytes,
    write_circle_bytes,
)


class FakeAccessor:
    """Provide a minimal root accessor for deserialization tests."""

    @staticmethod
    def GetRootAsModel(data, offset):
        return (bytes(data), offset)


class FakeObjectType:
    """Provide a minimal Object API unpacker for deserialization tests."""

    @staticmethod
    def InitFromObj(root):
        return {"root": root}


class FakeBuilder:
    """Provide the FlatBuffers builder methods used by serialization tests."""

    def __init__(self, initial_size):
        self.initial_size = initial_size
        self.finished = None

    def Finish(self, offset, identifier):
        self.finished = (offset, identifier)

    def Output(self):
        return b"serialized"


class FakeFlatbuffers:
    """Expose the fake builder through a module-like object."""

    Builder = FakeBuilder


class FakePackableModel:
    """Provide a minimal Object API Pack implementation."""

    def Pack(self, builder):
        self.builder = builder
        return 7


class CircleIOTest(unittest.TestCase):
    def test_deserialize_uses_generated_object_api(self):
        with patch(
            "tico.circle.io.accessor_api_type", return_value=FakeAccessor
        ), patch("tico.circle.io.object_api_type", return_value=FakeObjectType):
            model = model_from_bytes(b"\x08\x00\x00\x00CIR0model")

        self.assertEqual(model, {"root": (b"\x08\x00\x00\x00CIR0model", 0)})

    def test_deserialize_rejects_missing_file_identifier(self):
        with self.assertRaisesRegex(CircleIOError, "CIR0"):
            model_from_bytes(b"not-a-circle")

    def test_serialize_uses_circle_file_identifier(self):
        model = FakePackableModel()
        with patch("tico.circle.io._load_flatbuffers", return_value=FakeFlatbuffers):
            data = model_to_bytes(model)

        self.assertEqual(data, b"serialized")
        self.assertEqual(model.builder.finished, (7, b"CIR0"))

    def test_binary_stream_and_atomic_path_io(self):
        stream = io.BytesIO()
        write_circle_bytes(b"circle", stream)
        stream.seek(0)
        self.assertEqual(read_circle_bytes(stream), b"circle")

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.circle"
            write_circle_bytes(b"circle", path)
            self.assertEqual(read_circle_bytes(path), b"circle")


@unittest.skipUnless(
    importlib.util.find_spec("circle_schema") is not None
    and importlib.util.find_spec("flatbuffers") is not None,
    "circle-schema and flatbuffers are required for the integration round trip",
)
class CircleSchemaRoundTripTest(unittest.TestCase):
    def test_minimal_model_round_trip(self):
        from circle_schema import circle

        model = circle.Model.ModelT()
        model.version = 0
        model.description = "round-trip"
        model.operatorCodes = []
        model.subgraphs = []
        model.buffers = [circle.Buffer.BufferT()]
        model.signatureDefs = []
        model.metadataBuffer = []
        model.metadata = []

        document = CircleDocument(model)
        restored = CircleDocument.from_bytes(document.to_bytes())

        self.assertEqual(decode_text(restored.model.description), "round-trip")

    def test_metadata_buffer_vector_is_remapped_after_round_trip(self):
        import numpy as np
        from circle_schema import circle

        from tico.circle.rewrite import compact_model

        model = circle.Model.ModelT()
        model.version = 0
        model.description = "metadata-remap"
        model.buffers = [
            circle.Buffer.BufferT(),
            circle.Buffer.BufferT(),
            circle.Buffer.BufferT(),
        ]
        model.buffers[1].data = np.array([1], dtype=np.uint8)
        model.buffers[2].data = np.array([2], dtype=np.uint8)
        model.operatorCodes = []

        tensor = circle.Tensor.TensorT()
        tensor.name = "passthrough"
        tensor.shape = [1]
        tensor.shapeSignature = [1]
        tensor.type = circle.TensorType.TensorType.FLOAT32
        tensor.buffer = 0

        subgraph = circle.SubGraph.SubGraphT()
        subgraph.name = "main"
        subgraph.tensors = [tensor]
        subgraph.inputs = [0]
        subgraph.outputs = [0]
        subgraph.operators = []
        model.subgraphs = [subgraph]
        model.signatureDefs = []
        model.metadataBuffer = [2]
        model.metadata = []

        restored = CircleDocument.from_bytes(CircleDocument(model).to_bytes())
        stats = compact_model(restored.model)

        self.assertTrue(stats.modified)
        self.assertEqual(len(restored.model.buffers), 2)
        self.assertEqual([int(index) for index in restored.model.metadataBuffer], [1])
        self.assertTrue(restored.verify(raise_on_error=False).ok)

    def test_serialized_generated_vectors_can_be_extracted(self):
        import numpy as np
        from circle_schema import circle

        from tico.circle.inspect import summarize_document
        from tico.circle.operations import extract_by_operator_indices

        model = circle.Model.ModelT()
        model.version = 0
        model.description = "generated-vectors"
        model.buffers = [
            circle.Buffer.BufferT(),
            circle.Buffer.BufferT(),
            circle.Buffer.BufferT(),
        ]
        model.buffers[1].data = np.array([0, 0, 128, 63], dtype=np.uint8)
        model.buffers[2].data = np.array([0, 0, 0, 64], dtype=np.uint8)

        operator_code = circle.OperatorCode.OperatorCodeT()
        operator_code.builtinCode = circle.BuiltinOperator.BuiltinOperator.ADD
        operator_code.deprecatedBuiltinCode = operator_code.builtinCode
        model.operatorCodes = [operator_code]

        subgraph = circle.SubGraph.SubGraphT()
        subgraph.name = "main"
        subgraph.inputs = [0]
        subgraph.outputs = [4]

        def tensor(name, buffer_index=0):
            value = circle.Tensor.TensorT()
            value.name = name
            value.shape = [1]
            value.shapeSignature = [1]
            value.buffer = buffer_index
            value.type = circle.TensorType.TensorType.FLOAT32
            return value

        subgraph.tensors = [
            tensor("x"),
            tensor("weight", 1),
            tensor("selected_output"),
            tensor("dead_weight", 2),
            tensor("model_output"),
            tensor("dead_output"),
        ]

        def operator(inputs, outputs):
            value = circle.Operator.OperatorT()
            value.opcodeIndex = 0
            value.inputs = inputs
            value.outputs = outputs
            return value

        subgraph.operators = [
            operator([0, 1], [2]),
            operator([2, 1], [4]),
            operator([0, 3], [5]),
        ]
        model.subgraphs = [subgraph]
        model.signatureDefs = []
        model.metadataBuffer = []
        model.metadata = []

        restored = CircleDocument.from_bytes(CircleDocument(model).to_bytes())
        summary = summarize_document(restored)
        self.assertEqual(summary.subgraphs[0].inputs, 1)
        self.assertEqual(summary.subgraphs[0].outputs, 1)
        self.assertEqual(restored.graph().inputs, (0,))

        result = extract_by_operator_indices(restored, (0,))
        self.assertEqual(result.source_boundary.inputs, (0,))
        self.assertEqual(result.source_boundary.outputs, (2,))
        self.assertEqual(result.boundary.inputs, (0,))
        self.assertEqual(result.boundary.outputs, (2,))
        self.assertEqual(len(result.document.model.buffers), 2)

        reloaded = CircleDocument.from_bytes(result.document.to_bytes())
        self.assertTrue(reloaded.verify(raise_on_error=False).ok)
        self.assertEqual(reloaded.graph().inputs, (0,))
        self.assertEqual(reloaded.graph().outputs, (2,))

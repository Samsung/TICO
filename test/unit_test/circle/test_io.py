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

import gc
import importlib.util
import io
import os
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tico.circle._schema import decode_text
from tico.circle.document import CircleDocument
from tico.circle.errors import CircleIOError
from tico.circle.io import (
    CirclePayloadMapping,
    load_model_mapped,
    map_circle_file,
    model_from_bytes,
    model_to_bytes,
    read_circle_bytes,
    save_model,
    write_circle_bytes,
)
from tico.serialize import circle_binary as binary


class FakeAccessor:
    """Provide a minimal root accessor for deserialization tests."""

    @staticmethod
    def GetRootAsModel(data, offset):
        return FakeAccessor(bytes(data), offset)

    def __init__(self, data, offset):
        self.source = (data, offset)

    def BuffersLength(self):
        return 0


class FakeObjectType:
    """Provide a minimal Object API unpacker for deserialization tests."""

    @staticmethod
    def InitFromObj(root):
        return {"root": root.source}


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
        with patch(
            "tico.serialize.circle_binary._load_flatbuffers",
            return_value=FakeFlatbuffers,
        ):
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


def _fake_layout(header=b"header", payloads=(b"abc", bytes(range(40)))):
    views = tuple(memoryview(payload) for payload in payloads)
    offsets, _ = binary._payload_layout(len(header), [len(p) for p in payloads])
    return binary.CircleBinaryLayout(bytearray(header), views, offsets)


class CircleStreamingSaveTest(unittest.TestCase):
    """save_model streams a planned layout; it never joins the complete binary."""

    def test_stream_destination_receives_layout_without_joining(self):
        layout = _fake_layout()
        stream = io.BytesIO()
        with patch("tico.circle.io.plan_circle_layout", return_value=layout), patch(
            "tico.serialize.circle_binary._join_payloads"
        ) as join:
            save_model(SimpleNamespace(Pack=lambda builder: 0), stream)
        join.assert_not_called()
        self.assertEqual(stream.getvalue(), layout.to_bytes())

    def test_file_save_uses_the_streaming_inline_budget(self):
        layout = _fake_layout()
        with patch(
            "tico.circle.io.plan_circle_layout", return_value=layout
        ) as plan, tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "model.circle"
            save_model(SimpleNamespace(Pack=lambda builder: 0), path)
            self.assertEqual(path.read_bytes(), layout.to_bytes())
            save_model(SimpleNamespace(Pack=lambda builder: 0), path, atomic=False)
            self.assertEqual(path.read_bytes(), layout.to_bytes())
        for call in plan.call_args_list:
            self.assertEqual(call.kwargs["inline_budget"], 1 << 30)

    def test_stdout_destination_writes_binary_to_the_buffer(self):
        layout = _fake_layout()
        buffer = io.BytesIO()
        with patch("tico.circle.io.plan_circle_layout", return_value=layout), patch(
            "tico.circle.io.sys.stdout", SimpleNamespace(buffer=buffer)
        ):
            save_model(SimpleNamespace(Pack=lambda builder: 0), "-")
        self.assertEqual(buffer.getvalue(), layout.to_bytes())

    def test_atomic_failure_preserves_destination_and_removes_temporary(self):
        layout = _fake_layout()
        failing = SimpleNamespace(
            header=layout.header,
            payloads=layout.payloads,
            offsets=layout.offsets,
            write_to=None,
        )

        def write_to(stream, **_kwargs):
            stream.write(b"partial")
            raise OSError("disk full")

        failing.write_to = write_to
        with patch(
            "tico.circle.io.plan_circle_layout", return_value=failing
        ), tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.circle"
            path.write_bytes(b"previous")
            with self.assertRaisesRegex(CircleIOError, "Failed to write"):
                save_model(SimpleNamespace(Pack=lambda builder: 0), path)
            self.assertEqual(path.read_bytes(), b"previous")
            self.assertEqual(sorted(os.listdir(temporary)), ["model.circle"])

    def test_non_os_failure_still_removes_temporary_file(self):
        def write_to(stream, **_kwargs):
            raise RuntimeError("unexpected")

        failing = SimpleNamespace(write_to=write_to)
        with patch(
            "tico.circle.io.plan_circle_layout", return_value=failing
        ), tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.circle"
            with self.assertRaises(RuntimeError):
                save_model(SimpleNamespace(Pack=lambda builder: 0), path)
            self.assertEqual(os.listdir(temporary), [])

    def test_serialization_errors_become_io_errors_before_writing(self):
        with patch(
            "tico.circle.io.plan_circle_layout",
            side_effect=binary.CircleSerializationError("bad"),
        ), tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.circle"
            with self.assertRaisesRegex(CircleIOError, "Failed to serialize"):
                save_model(SimpleNamespace(Pack=lambda builder: 0), path)
            self.assertFalse(path.exists())
        with self.assertRaises(TypeError):
            save_model(object(), io.BytesIO())


class CirclePayloadMappingTest(unittest.TestCase):
    """Cover the mapped loader and the mapping owner's lifetime rules."""

    def _write(self, directory, data):
        path = Path(directory) / "model.circle"
        path.write_bytes(data)
        return path

    def test_map_rejects_empty_and_missing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(temporary, b"")
            with self.assertRaisesRegex(CircleIOError, "empty"):
                map_circle_file(path)
            with self.assertRaisesRegex(CircleIOError, "Failed to map"):
                map_circle_file(Path(temporary) / "missing.circle")

    def test_release_unmaps_when_no_view_is_alive(self):
        with tempfile.TemporaryDirectory() as temporary:
            mapping = map_circle_file(self._write(temporary, b"\x08\x00\x00\x00CIR0!"))
            self.assertEqual(mapping.size, 9)
            self.assertFalse(mapping.released)
            mapping.release()
            self.assertTrue(mapping.released)
            self.assertTrue(mapping.closed)
            mapping.release()  # idempotent
            with self.assertRaises(ValueError):
                mapping.buffer()

    def test_release_defers_unmapping_until_the_last_view_is_dropped(self):
        with tempfile.TemporaryDirectory() as temporary:
            mapping = map_circle_file(self._write(temporary, bytes(range(32))))
            view = np.frombuffer(mapping.buffer(), dtype=np.uint8, count=8, offset=16)
            mapping.release()
            self.assertTrue(mapping.released)
            self.assertFalse(mapping.closed)
            self.assertEqual(view.tobytes(), bytes(range(16, 24)))
            del view
            gc.collect()
            self.assertTrue(mapping.closed)

    def test_mapped_loader_copies_header_only_and_views_the_mapping(self):
        source = bytes(4) + b"CIR0" + bytes(24) + b"payload" * 100
        calls = []
        empty = SimpleNamespace(Offset=lambda: 0, Size=lambda: 0, DataLength=lambda: 0)
        external = SimpleNamespace(
            Offset=lambda: 32, Size=lambda: len(source) - 32, DataLength=lambda: 0
        )
        root = SimpleNamespace(
            BuffersLength=lambda: 2, Buffers=lambda i: (empty, external)[i]
        )

        def get_root(data, offset):
            calls.append(data)
            return root

        restored = SimpleNamespace(
            buffers=[
                SimpleNamespace(data=None, offset=0, size=0),
                SimpleNamespace(data=None, offset=32, size=len(source) - 32),
            ],
            subgraphs=[],
        )
        accessor = SimpleNamespace(GetRootAsModel=get_root)
        object_type = SimpleNamespace(InitFromObj=lambda value: restored)
        with tempfile.TemporaryDirectory() as temporary, patch(
            "tico.circle.io.accessor_api_type", return_value=accessor
        ), patch("tico.circle.io.object_api_type", return_value=object_type):
            path = self._write(temporary, source)
            model, mapping = load_model_mapped(path)
            self.assertIs(calls[0], mapping.buffer())
            self.assertIsInstance(calls[1], bytearray)
            self.assertEqual(len(calls[1]), 32)
            payload = model.buffers[1].data
            self.assertFalse(payload.flags.writeable)
            self.assertEqual(payload.tobytes(), source[32:])
            self.assertEqual((model.buffers[1].offset, model.buffers[1].size), (0, 0))
            mapping_ref = weakref.ref(mapping.buffer())
            mapping.release()
            self.assertFalse(mapping.closed)
            # The recorded parser argument is the mapping itself; drop it too.
            del payload, model, restored, calls[:]
            gc.collect()
            self.assertTrue(mapping.closed)
            self.assertIsNone(mapping_ref())

    def test_mapped_loader_releases_mapping_when_parsing_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(temporary, b"not-a-circle-file")
            recorded = []
            original = CirclePayloadMapping.__init__

            def record(self, file_path, mapping):
                original(self, file_path, mapping)
                recorded.append(weakref.ref(mapping))

            with patch.object(CirclePayloadMapping, "__init__", record):
                with self.assertRaisesRegex(CircleIOError, "CIR0"):
                    load_model_mapped(path)
            gc.collect()
            self.assertEqual([ref() for ref in recorded], [None])

    def test_same_file_detects_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(temporary, b"\x08\x00\x00\x00CIR0!")
            alias = Path(temporary) / "alias.circle"
            os.symlink(path, alias)
            with map_circle_file(path) as mapping:
                self.assertTrue(mapping.same_file(alias))
                self.assertTrue(mapping.same_file(os.fspath(path)))
                self.assertFalse(mapping.same_file(Path(temporary) / "other"))


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

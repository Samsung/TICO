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

"""Generated-schema round trips and ordinary converter API regression tests."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import io
import os
import struct
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import tico.circle.io as circle_io
from tico.circle.document import CircleDocument
from tico.circle.errors import CircleIOError
from tico.circle.io import model_from_bytes, model_to_bytes
from tico.serialize import circle_binary as binary

HAS_SCHEMA = (
    importlib.util.find_spec("flatbuffers") is not None
    and importlib.util.find_spec("circle_schema") is not None
)


def make_model(*payloads):
    from circle_schema import circle

    model = circle.Model.ModelT()
    model.version = 0
    model.description = "large-circle-regression"
    model.operatorCodes = []
    model.subgraphs = []
    model.buffers = [circle.Buffer.BufferT()]
    for data in payloads:
        value = circle.Buffer.BufferT()
        value.data = np.asarray(data, dtype=np.uint8)
        model.buffers.append(value)
    model.metadataBuffer = []
    model.metadata = []
    model.signatureDefs = []
    return model


def payloads_from_bytes(data):
    from circle_schema import circle

    root = circle.Model.Model.GetRootAsModel(data, 0)
    result = []
    for index in range(1, root.BuffersLength()):
        value = root.Buffers(index)
        if value.Offset() > 1:
            result.append(data[value.Offset() : value.Offset() + value.Size()])
        elif value.DataLength():
            result.append(value.DataAsNumpy().tobytes())
        else:
            result.append(b"")
    return result


def make_chain_model(*payloads, tensor_type=None):
    """Return ``x -> ADD(w0) -> ADD(w1) -> ...`` with one constant per operator.

    Extracting operator ``i`` keeps exactly buffer ``i + 1``; every other constant
    is discarded, which is what the ownership tests need to observe.
    """

    from circle_schema import circle

    model = make_model(*payloads)
    code = circle.OperatorCode.OperatorCodeT()
    code.builtinCode = circle.BuiltinOperator.BuiltinOperator.ADD
    code.deprecatedBuiltinCode = code.builtinCode
    model.operatorCodes = [code]
    if tensor_type is None:
        tensor_type = circle.TensorType.TensorType.FLOAT32

    def tensor(name, buffer_index=0):
        value = circle.Tensor.TensorT()
        value.name = name
        value.shape = [1]
        value.type = tensor_type
        value.buffer = buffer_index
        return value

    graph = circle.SubGraph.SubGraphT()
    graph.name = "main"
    tensors = [tensor("x")]
    operators = []
    previous = 0
    for index in range(len(payloads)):
        tensors.append(tensor(f"w{index}", index + 1))
        tensors.append(tensor(f"y{index}"))
        operator = circle.Operator.OperatorT()
        operator.opcodeIndex = 0
        operator.inputs = [previous, len(tensors) - 2]
        operator.outputs = [len(tensors) - 1]
        operators.append(operator)
        previous = len(tensors) - 1
    graph.tensors = tensors
    graph.inputs = [0]
    graph.outputs = [previous]
    graph.operators = operators
    model.subgraphs = [graph]
    return model


def digest(data):
    return hashlib.sha256(data).hexdigest()


class _RecordingMappings:
    """Record weak references to every mapping the code under test creates."""

    def __init__(self):
        self.mappings = []
        self.owners = []

    def __enter__(self):
        original = circle_io.CirclePayloadMapping.__init__
        recorder = self

        def record(self, path, mapping):
            original(self, path, mapping)
            recorder.mappings.append(weakref.ref(mapping))
            recorder.owners.append(self)

        self._patch = patch.object(circle_io.CirclePayloadMapping, "__init__", record)
        self._patch.start()
        return self

    def __exit__(self, *exc_info):
        self._patch.stop()

    @property
    def alive(self):
        return [reference() is not None for reference in self.mappings]


class CircleExternalIOUnitTest(unittest.TestCase):
    def test_deserialization_copies_only_header_and_restores_payload(self):
        source = bytes(4) + b"CIR0" + bytes(24) + b"payload" * 1000
        calls = []
        empty = SimpleNamespace(Offset=lambda: 0, Size=lambda: 0, DataLength=lambda: 0)
        external = SimpleNamespace(
            Offset=lambda: 32,
            Size=lambda: len(source) - 32,
            DataLength=lambda: 0,
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
        with patch("tico.circle.io.accessor_api_type", return_value=accessor), patch(
            "tico.circle.io.object_api_type", return_value=object_type
        ):
            actual = model_from_bytes(source)
        self.assertIs(calls[0], source)
        self.assertIsInstance(calls[1], bytearray)
        self.assertEqual(len(calls[1]), 32)
        self.assertIs(actual.buffers[1].data.base, source)
        self.assertEqual(actual.buffers[1].data.tobytes(), source[32:])
        self.assertEqual((actual.buffers[1].offset, actual.buffers[1].size), (0, 0))

    def test_invalid_external_ranges_become_circle_io_errors(self):
        value = SimpleNamespace(
            Offset=lambda: 64, Size=lambda: 16, DataLength=lambda: 0
        )
        empty = SimpleNamespace(Offset=lambda: 0, Size=lambda: 0, DataLength=lambda: 0)
        root = SimpleNamespace(
            BuffersLength=lambda: 2, Buffers=lambda i: (empty, value)[i]
        )
        accessor = SimpleNamespace(GetRootAsModel=lambda data, offset: root)
        with patch("tico.circle.io.accessor_api_type", return_value=accessor):
            with self.assertRaisesRegex(CircleIOError, "outside"):
                model_from_bytes(bytes(4) + b"CIR0" + bytes(8))


@unittest.skipUnless(HAS_SCHEMA, "circle-schema and flatbuffers are required")
class LargeCircleSchemaRoundTripTest(unittest.TestCase):
    def test_small_output_is_byte_identical_to_the_original_packer(self):
        import flatbuffers

        document = make_model(
            np.arange(37, dtype=np.uint8), np.array([255], dtype=np.uint8)
        )
        builder = flatbuffers.Builder(1024)
        builder.Finish(document.Pack(builder), b"CIR0")
        self.assertEqual(
            binary.serialize_circle_model(document), bytes(builder.Output())
        )

    def test_aggregate_payloads_use_aligned_offsets_and_full_bytes(self):
        from circle_schema import circle

        first = np.arange(2301, dtype=np.uint8)
        second = np.arange(2407, dtype=np.uint8)[::-1]
        document = make_model(first, second, np.array([42], dtype=np.uint8))
        with patch.object(binary, "_FLATBUFFER_LIMIT", 4096):
            data = binary.serialize_circle_model(document)
        self.assertGreater(len(data), 4096)
        self.assertEqual(
            payloads_from_bytes(data), [first.tobytes(), second.tobytes(), b"*"]
        )
        root = circle.Model.Model.GetRootAsModel(data, 0)
        self.assertEqual(root.Buffers(0).Offset(), 0)
        self.assertEqual(root.Buffers(0).DataLength(), 0)
        self.assertEqual(root.Buffers(3).Offset(), 0)
        for index in (1, 2):
            value = root.Buffers(index)
            self.assertEqual(value.DataLength(), 0)
            self.assertEqual(value.Offset() % 16, 0)
            self.assertGreater(value.Offset(), 1)
        self.assertIs(document.buffers[1].data, first)
        self.assertEqual(document.buffers[1].offset, 0)

    def test_edit_repack_recomputes_offsets_and_keeps_payloads(self):
        from circle_schema import circle

        values = np.arange(5001, dtype=np.uint8)
        with patch.object(binary, "_FLATBUFFER_LIMIT", 4096):
            original = binary.serialize_circle_model(make_model(values))
            document = CircleDocument.from_bytes(original)
            self.assertIs(document.model.buffers[1].data.base, original)
            self.assertFalse(document.model.buffers[1].data.flags.writeable)
            document.model.description = "a longer description " * 17
            stream = io.BytesIO()
            document.save(stream)
            saved = stream.getvalue()
        old = circle.Model.Model.GetRootAsModel(original, 0).Buffers(1)
        new = circle.Model.Model.GetRootAsModel(saved, 0).Buffers(1)
        self.assertNotEqual(old.Offset(), new.Offset())
        self.assertEqual(payloads_from_bytes(saved), [values.tobytes()])
        self.assertEqual(CircleDocument.from_bytes(saved).model.buffers[1].offset, 0)

    def test_smaller_edited_document_can_return_to_inline_layout(self):
        from circle_schema import circle

        with patch.object(binary, "_FLATBUFFER_LIMIT", 4096):
            original = binary.serialize_circle_model(
                make_model(np.arange(5001, dtype=np.uint8))
            )
            document = CircleDocument.from_bytes(original)
            document.model.buffers[1].data = np.array([1, 2, 3], dtype=np.uint8)
            saved = document.to_bytes()
        self.assertEqual(
            circle.Model.Model.GetRootAsModel(saved, 0).Buffers(1).Offset(), 0
        )
        self.assertEqual(payloads_from_bytes(saved), [b"\x01\x02\x03"])

    def test_truncated_or_placeholder_offsets_rejected(self):
        from circle_schema import circle

        with patch.object(binary, "_FLATBUFFER_LIMIT", 4096):
            original = binary.serialize_circle_model(
                make_model(np.arange(5001, dtype=np.uint8))
            )
        with self.assertRaisesRegex(CircleIOError, "outside"):
            model_from_bytes(original[:-1])
        corrupted = bytearray(original)
        table = circle.Model.Model.GetRootAsModel(corrupted, 0).Buffers(1)._tab
        struct.pack_into("<Q", corrupted, table.Pos + table.Offset(6), 1)
        with self.assertRaisesRegex(CircleIOError, "incomplete"):
            model_from_bytes(bytes(corrupted))

    def test_metadata_limit_failure_does_not_mutate_input(self):
        values = np.arange(5001, dtype=np.uint8)
        document = make_model(values)
        document.description = "x" * 5000
        with patch.object(binary, "_FLATBUFFER_LIMIT", 4096):
            with self.assertRaisesRegex(
                binary.CircleSerializationError, "Split the graph"
            ):
                binary.serialize_circle_model(document)
        self.assertIs(document.buffers[1].data, values)
        self.assertEqual((document.buffers[1].offset, document.buffers[1].size), (0, 0))

    def test_packed_uint4_data_and_quantization_metadata_survive(self):
        from circle_schema import circle

        packed = np.arange(5001, dtype=np.uint8)
        document = make_model(packed)
        tensor = circle.Tensor.TensorT()
        tensor.name = "packed_weight"
        tensor.shape = [len(packed) * 2]
        tensor.type = circle.TensorType.TensorType.UINT4
        tensor.buffer = 1
        tensor.quantization = circle.QuantizationParameters.QuantizationParametersT()
        tensor.quantization.scale = [0.125]
        tensor.quantization.zeroPoint = [7]
        graph = circle.SubGraph.SubGraphT()
        graph.tensors = [tensor]
        graph.inputs = []
        graph.outputs = [0]
        graph.operators = []
        document.subgraphs = [graph]
        with patch.object(binary, "_FLATBUFFER_LIMIT", 4096):
            saved = model_to_bytes(
                model_from_bytes(binary.serialize_circle_model(document))
            )
        restored = model_from_bytes(saved)
        self.assertEqual(payloads_from_bytes(saved), [packed.tobytes()])
        actual = restored.subgraphs[0].tensors[0]
        self.assertEqual(actual.type, tensor.type)
        np.testing.assert_array_equal(actual.quantization.scale, [0.125])
        np.testing.assert_array_equal(actual.quantization.zeroPoint, [7])

    def test_repeated_serialization_is_deterministic(self):
        document = make_model(np.arange(5001, dtype=np.uint8))
        with patch.object(binary, "_FLATBUFFER_LIMIT", 4096):
            self.assertEqual(
                binary.serialize_circle_model(document),
                binary.serialize_circle_model(document),
            )


@unittest.skipUnless(HAS_SCHEMA, "circle-schema and flatbuffers are required")
class LargeCircleExtractionRoundTripTest(unittest.TestCase):
    """Extract from appended-layout files, save by streaming, and reload.

    A reduced private FlatBuffer budget forces the appended layout with small
    arrays; nothing here allocates gigabytes.
    """

    PAYLOADS = (
        np.arange(5001, dtype=np.uint8),
        np.arange(7003, dtype=np.uint8)[::-1].copy(),
        np.arange(3001, dtype=np.uint8),
    )

    def setUp(self):
        self._limit = patch.object(binary, "_FLATBUFFER_LIMIT", 4096)
        self._limit.start()
        self.addCleanup(self._limit.stop)
        # Keep CLI diagnostics out of the test runner's output.
        quiet = patch.object(sys, "stderr", io.StringIO())
        quiet.start()
        self.addCleanup(quiet.stop)
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.directory = Path(self._temporary.name)
        self.source = self.directory / "source.circle"
        CircleDocument(make_chain_model(*self.PAYLOADS)).save(self.source)
        self.source_digest = digest(self.source.read_bytes())

    def _reload(self, path):
        document = CircleDocument.load(path)
        self.assertTrue(document.verify(raise_on_error=False).ok)
        return document

    def test_public_extraction_from_mapped_file_is_detached_and_reloadable(self):
        from tico.circle.operations import extract_by_operator_indices

        with _RecordingMappings() as recorded:
            with CircleDocument.load_mapped(self.source) as document:
                self.assertIs(document.payload_mapping, recorded.owners[0])
                self.assertFalse(document.model.buffers[1].data.flags.writeable)
                result = extract_by_operator_indices(document, (1,))
            # The mapping is released; the detached result keeps working.
            self.assertIsNone(document.payload_mapping)
            del document
            gc.collect()
            self.assertEqual(recorded.alive, [False])
        retained = result.document.model.buffers[1].data
        self.assertIsNone(retained.base)
        self.assertEqual(retained.tobytes(), self.PAYLOADS[1].tobytes())
        output = self.directory / "public.circle"
        result.document.save(output)
        reloaded = self._reload(output)
        self.assertEqual(
            payloads_from_bytes(output.read_bytes()), [self.PAYLOADS[1].tobytes()]
        )
        self.assertEqual(
            [decode(t.name) for t in reloaded.subgraph(0).tensors], ["y0", "w1", "y1"]
        )
        self.assertEqual(digest(self.source.read_bytes()), self.source_digest)

    def test_cli_extract_borrows_payloads_and_streams_without_joining(self):
        from tico.circle import _buffer
        from tico.circle.cli.main import main

        output = self.directory / "cli.circle"
        with _RecordingMappings() as recorded, patch.object(
            _buffer, "owned_payload_copy"
        ) as copy_payload, patch.object(binary, "_join_payloads") as join:
            self.assertEqual(
                main(["extract", str(self.source), "--ops", "1", "-o", str(output)]), 0
            )
        copy_payload.assert_not_called()
        join.assert_not_called()
        self.assertEqual(recorded.alive, [False])
        reloaded = self._reload(output)
        self.assertEqual(
            payloads_from_bytes(output.read_bytes()), [self.PAYLOADS[1].tobytes()]
        )
        self.assertEqual(len(reloaded.model.buffers), 2)
        # The appended layout was re-planned with a fresh offset.
        from circle_schema import circle

        root = circle.Model.Model.GetRootAsModel(output.read_bytes(), 0)
        self.assertGreater(root.Buffers(1).Offset(), 1)
        self.assertEqual(root.Buffers(1).Offset() % 16, 0)
        self.assertEqual(digest(self.source.read_bytes()), self.source_digest)

    def test_cli_extract_keeping_most_payloads_streams_a_large_output(self):
        from tico.circle.cli.main import main

        output = self.directory / "large.circle"
        with patch.object(binary, "_join_payloads") as join:
            self.assertEqual(
                main(["extract", str(self.source), "--ops", "0-2", "-o", str(output)]),
                0,
            )
        join.assert_not_called()
        self.assertGreater(output.stat().st_size, 4096)
        self.assertEqual(
            payloads_from_bytes(output.read_bytes()),
            [payload.tobytes() for payload in self.PAYLOADS],
        )
        self._reload(output)

    def test_cli_extract_onto_the_mapped_source_file_is_safe(self):
        from tico.circle.cli.main import main

        expected = self.PAYLOADS[2].tobytes()
        with _RecordingMappings() as recorded:
            self.assertEqual(
                main(
                    ["extract", str(self.source), "--ops", "2", "-o", str(self.source)]
                ),
                0,
            )
        self.assertEqual(recorded.alive, [False])
        self.assertEqual(payloads_from_bytes(self.source.read_bytes()), [expected])
        self._reload(self.source)
        self.assertEqual(sorted(os.listdir(self.directory)), ["source.circle"])

    def test_cli_extract_via_symlink_alias_onto_the_source(self):
        from tico.circle.cli.main import main

        alias = self.directory / "alias.circle"
        os.symlink(self.source, alias)
        self.assertEqual(
            main(["extract", str(alias), "--ops", "0", "-o", str(self.source)]), 0
        )
        self.assertEqual(
            payloads_from_bytes(self.source.read_bytes()), [self.PAYLOADS[0].tobytes()]
        )

    def test_cli_extract_writes_binary_to_stdout(self):
        from tico.circle.cli.main import main

        buffer = io.BytesIO()
        errors = io.StringIO()
        with patch.object(sys, "stdout", SimpleNamespace(buffer=buffer)), patch.object(
            sys, "stderr", errors
        ):
            self.assertEqual(
                main(["extract", str(self.source), "--ops", "0", "-o", "-"]), 0
            )
        self.assertEqual(buffer.getvalue()[4:8], b"CIR0")
        self.assertEqual(
            payloads_from_bytes(buffer.getvalue()), [self.PAYLOADS[0].tobytes()]
        )
        self.assertIn("Extracted operators [0]", errors.getvalue())
        self.assertTrue(
            CircleDocument.from_bytes(buffer.getvalue()).verify(raise_on_error=False).ok
        )

    def test_cli_extract_from_stdin_uses_the_eager_loader(self):
        from tico.circle.cli.main import main

        output = self.directory / "stdin.circle"
        with _RecordingMappings() as recorded, patch.object(
            sys, "stdin", SimpleNamespace(buffer=io.BytesIO(self.source.read_bytes()))
        ):
            self.assertEqual(main(["extract", "-", "--ops", "1", "-o", str(output)]), 0)
        self.assertEqual(recorded.mappings, [])
        self.assertEqual(
            payloads_from_bytes(output.read_bytes()), [self.PAYLOADS[1].tobytes()]
        )

    def test_repeated_cli_extractions_do_not_accumulate_mappings(self):
        from tico.circle.cli.main import main

        gc.disable()
        self.addCleanup(gc.enable)
        with _RecordingMappings() as recorded:
            for index in range(3):
                output = self.directory / f"repeat_{index}.circle"
                self.assertEqual(
                    main(
                        [
                            "extract",
                            str(self.source),
                            "--ops",
                            str(index),
                            "-o",
                            str(output),
                        ]
                    ),
                    0,
                )
                self.assertEqual(recorded.alive, [False] * (index + 1))

    def test_cli_failure_releases_the_mapping_and_writes_nothing(self):
        from tico.circle.cli.main import main

        output = self.directory / "never.circle"
        errors = io.StringIO()
        gc.disable()
        self.addCleanup(gc.enable)
        with _RecordingMappings() as recorded, patch.object(sys, "stderr", errors):
            self.assertEqual(
                main(["extract", str(self.source), "--ops", "42", "-o", str(output)]), 1
            )
        self.assertEqual(recorded.alive, [False])
        self.assertFalse(output.exists())
        self.assertIn("error:", errors.getvalue())

    def test_non_atomic_save_onto_mapped_source_is_refused(self):
        from tico.circle.operations import extract_by_operator_indices, PayloadOwnership

        with CircleDocument.load_mapped(self.source) as document:
            result = extract_by_operator_indices(
                document, (0,), payload_ownership=PayloadOwnership.BORROWED
            )
            borrowed = CircleDocument(
                result.document.model, payload_mapping=document.payload_mapping
            )
            with self.assertRaisesRegex(CircleIOError, "Refusing to overwrite"):
                borrowed.save(self.source, atomic=False)
            with self.assertRaisesRegex(CircleIOError, "Refusing to overwrite"):
                document.save(self.source, atomic=False)
        self.assertEqual(digest(self.source.read_bytes()), self.source_digest)

    def test_packed_uint4_payload_and_quantization_survive_extraction(self):
        from circle_schema import circle

        from tico.circle.cli.main import main

        packed = np.arange(5001, dtype=np.uint8)
        model = make_chain_model(packed, tensor_type=circle.TensorType.TensorType.UINT4)
        weight = model.subgraphs[0].tensors[1]
        weight.shape = [len(packed) * 2]
        weight.quantization = circle.QuantizationParameters.QuantizationParametersT()
        weight.quantization.scale = [0.125]
        weight.quantization.zeroPoint = [7]
        source = self.directory / "uint4.circle"
        CircleDocument(model).save(source)
        output = self.directory / "uint4.extracted.circle"
        self.assertEqual(
            main(["extract", str(source), "--ops", "0", "-o", str(output)]), 0
        )
        restored = self._reload(output)
        self.assertEqual(payloads_from_bytes(output.read_bytes()), [packed.tobytes()])
        actual = restored.subgraph(0).tensors[1]
        self.assertEqual(actual.type, circle.TensorType.TensorType.UINT4)
        np.testing.assert_array_equal(actual.quantization.scale, [0.125])
        np.testing.assert_array_equal(actual.quantization.zeroPoint, [7])

    def test_corrupted_external_range_is_still_rejected_when_mapped(self):
        corrupted = bytearray(self.source.read_bytes())
        from circle_schema import circle

        table = circle.Model.Model.GetRootAsModel(corrupted, 0).Buffers(1)._tab
        struct.pack_into("<Q", corrupted, table.Pos + table.Offset(6), 1)
        broken = self.directory / "broken.circle"
        broken.write_bytes(corrupted)
        with _RecordingMappings() as recorded:
            with self.assertRaisesRegex(CircleIOError, "incomplete"):
                CircleDocument.load_mapped(broken)
            truncated = self.directory / "truncated.circle"
            truncated.write_bytes(self.source.read_bytes()[:-1])
            with self.assertRaisesRegex(CircleIOError, "outside"):
                CircleDocument.load_mapped(truncated)
        gc.collect()
        self.assertEqual(recorded.alive, [False, False])

    def test_small_inline_document_save_is_byte_identical_to_to_bytes(self):
        self._limit.stop()
        self.addCleanup(self._limit.start)
        document = CircleDocument(make_chain_model(*self.PAYLOADS))
        output = self.directory / "inline.circle"
        document.save(output)
        self.assertEqual(output.read_bytes(), document.to_bytes())
        from circle_schema import circle

        root = circle.Model.Model.GetRootAsModel(output.read_bytes(), 0)
        self.assertEqual(root.Buffers(1).Offset(), 0)
        self.assertGreater(root.Buffers(1).DataLength(), 0)


def decode(value):
    return value.decode() if isinstance(value, bytes) else value


@unittest.skipUnless(HAS_SCHEMA, "circle-schema and flatbuffers are required")
class LargeCircleConversionTest(unittest.TestCase):
    @staticmethod
    def _module_and_inputs():
        import torch

        module = torch.nn.Embedding(257, 16).eval()
        with torch.no_grad():
            module.weight.copy_(
                torch.arange(257 * 16, dtype=torch.float32).reshape(257, 16)
            )
        return module, (torch.tensor([[2, 200]], dtype=torch.long),)

    def test_public_module_exported_program_and_pt2_apis_with_o1(self):
        import tico
        import torch
        from tico.circle.export import optimize_for_export
        from tico.utils.model import CircleModel

        for entry in ("module", "exported_program", "pt2"):
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as temporary:
                module, args = self._module_and_inputs()
                expected = module.weight.detach().numpy().tobytes()
                with patch.object(binary, "_FLATBUFFER_LIMIT", 4096), patch(
                    "tico.circle.export.optimize_for_export", wraps=optimize_for_export
                ) as optimize:
                    if entry == "module":
                        result = tico.convert(module, args)
                    else:
                        ep = torch.export.export(module, args)
                        if entry == "exported_program":
                            result = tico.convert_from_exported_program(ep)
                        else:
                            pt2 = Path(temporary) / "embedding.pt2"
                            torch.export.save(ep, pt2)
                            result = tico.convert_from_pt2(pt2)
                    optimize.assert_called_once()
                    path = Path(temporary) / "embedding.circle"
                    result.save(path)
                    document = CircleDocument.load(path)
                    self.assertTrue(document.verify(raise_on_error=False).ok)
                    document.save(Path(temporary) / "repacked.circle")
                self.assertIsInstance(result.circle_binary, bytes)
                self.assertGreater(len(result.circle_binary), 4096)
                self.assertIn(expected, payloads_from_bytes(result.circle_binary))
                self.assertEqual(
                    CircleModel.load(str(path)).circle_binary, result.circle_binary
                )

    @unittest.skipUnless(
        os.environ.get("TICO_RUN_ONE_LARGE_CIRCLE") == "1", "opt-in ONE inference"
    )
    def test_small_external_embedding_executes_with_one(self):
        import tico
        import torch

        module, args = self._module_and_inputs()
        with patch.object(binary, "_FLATBUFFER_LIMIT", 4096):
            result = tico.convert(module, args)
        actual = result(*args)
        if isinstance(actual, (tuple, list)):
            actual = actual[0]
        torch.testing.assert_close(torch.as_tensor(actual), module(*args))


@unittest.skipUnless(
    HAS_SCHEMA and os.environ.get("TICO_RUN_LARGE_CIRCLE") == "1",
    "opt-in real >2 GiB serialization; requires a high-memory 64-bit host",
)
class LargeCircleRealSizeTest(unittest.TestCase):
    def test_real_binary_larger_than_two_gib(self):
        # A sparse, file-backed source avoids initializing a second huge array.
        # The complete output bytes still require more than 2 GiB of RAM.
        size = 2**31 + 17
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "payload.bin"
            with source.open("wb") as stream:
                stream.truncate(size)
            values = np.memmap(source, dtype=np.uint8, mode="r+", shape=(size,))
            values[:16] = np.arange(16, dtype=np.uint8)
            values[-16:] = np.arange(16, dtype=np.uint8)[::-1]
            values.flush()
            document = make_model(values)
            data = binary.serialize_circle_model(document)
            self.assertGreater(len(data), 2**31)
            restored = model_from_bytes(data)
            actual = restored.buffers[1].data
            self.assertEqual(actual.nbytes, size)
            self.assertEqual(
                hashlib.sha256(actual).digest(), hashlib.sha256(values).digest()
            )
            destination = Path(temporary) / "large.circle"
            with destination.open("wb") as stream:
                stream.write(data)
            self.assertEqual(destination.stat().st_size, len(data))
            del document, restored, actual, values

    @staticmethod
    def _sparse_payload(directory, name, size):
        """Create a file-backed uint8 array with recognizable edges; never in RAM."""

        source = Path(directory) / name
        with source.open("wb") as stream:
            stream.truncate(size)
        values = np.memmap(source, dtype=np.uint8, mode="r+", shape=(size,))
        values[:16] = np.arange(16, dtype=np.uint8)
        values[-16:] = np.arange(16, dtype=np.uint8)[::-1]
        values.flush()
        return values

    def _extract_round_trip(self, sizes, keep):
        """Save a chain model by streaming, extract via the CLI, verify digests.

        This checks the file-extraction round trip only. It does not execute the
        model on any backend.
        """

        from tico.circle.cli.main import main

        with tempfile.TemporaryDirectory() as temporary:
            payloads = [
                self._sparse_payload(temporary, f"payload_{index}.bin", size)
                for index, size in enumerate(sizes)
            ]
            expected = [hashlib.sha256(values).digest() for values in payloads]
            source = Path(temporary) / "source.circle"
            CircleDocument(make_chain_model(*payloads)).save(source)
            self.assertGreater(source.stat().st_size, 2**31)
            del payloads
            gc.collect()

            output = Path(temporary) / "extracted.circle"
            spec = ",".join(str(index) for index in keep)
            with _RecordingMappings() as recorded:
                self.assertEqual(
                    main(["extract", str(source), "--ops", spec, "-o", str(output)]),
                    0,
                )
            self.assertEqual(recorded.alive, [False])
            with CircleDocument.load_mapped(output) as restored:
                self.assertTrue(restored.verify(raise_on_error=False).ok)
                buffers = restored.model.buffers
                self.assertEqual(len(buffers), len(keep) + 1)
                for position, index in enumerate(keep):
                    actual = buffers[position + 1].data
                    self.assertEqual(actual.nbytes, sizes[index])
                    self.assertEqual(hashlib.sha256(actual).digest(), expected[index])
            return output.stat().st_size

    def test_extract_from_aggregate_larger_than_two_gib_keeps_a_small_region(self):
        # Three constants that only together exceed the FlatBuffer limit.
        size = 2**30 - 2**20
        saved = self._extract_round_trip([size, size, size], keep=(1,))
        self.assertLess(saved, 2**31)

    def test_extract_from_single_buffer_larger_than_two_gib(self):
        # One constant that alone exceeds the limit, plus a small neighbour that
        # is discarded; the retained output is itself larger than 2 GiB.
        saved = self._extract_round_trip([2**31 + 17, 4096], keep=(0,))
        self.assertGreater(saved, 2**31)


if __name__ == "__main__":
    unittest.main()

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

"""Size/layout policy tests; no multi-gigabyte allocations are needed."""

from __future__ import annotations

import struct
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from tico.serialize import circle_binary as binary


def buffer(data=None, *, offset=0, size=0):
    return SimpleNamespace(data=data, offset=offset, size=size)


def model(*payloads):
    return SimpleNamespace(
        buffers=[buffer(), *(buffer(data) for data in payloads)],
        subgraphs=[],
        Pack=Mock(return_value=7),
    )


class AccessorBuffer:
    def __init__(self, offset=0, size=0, length=0):
        self.offset, self.size, self.length = offset, size, length

    def Offset(self):
        return self.offset

    def Size(self):
        return self.size

    def DataLength(self):
        return self.length


class AccessorRoot:
    def __init__(self, *buffers):
        self.buffers = buffers

    def BuffersLength(self):
        return len(self.buffers)

    def Buffers(self, index):
        return self.buffers[index]


class CircleBinaryLayoutTest(unittest.TestCase):
    def test_offsets_cross_two_and_four_gib_without_allocation(self):
        offsets, end = binary._payload_layout(257, [2**31 + 3, 2**31 + 5, 7])
        self.assertEqual(offsets[0], 272)
        self.assertGreater(offsets[1], 2**31)
        self.assertGreater(offsets[2], 2**32)
        self.assertTrue(all(offset % 16 == 0 for offset in offsets))
        self.assertEqual(end, offsets[-1] + 7)

    def test_alignment_boundaries(self):
        for header in range(65):
            for size in (1, 2, 15, 16, 17, 31, 32, 33):
                with self.subTest(header=header, size=size):
                    offsets, end = binary._payload_layout(header, [size, 3])
                    self.assertEqual(offsets[0], (header + 15) // 16 * 16)
                    self.assertEqual(offsets[1], (offsets[0] + size + 15) // 16 * 16)
                    self.assertEqual(end, offsets[1] + 3)

    def test_invalid_layout_rejected(self):
        for header, sizes in ((-1, [1]), (16, [0]), (16, [-1])):
            with self.subTest(header=header, sizes=sizes):
                with self.assertRaises(binary.CircleSerializationError):
                    binary._payload_layout(header, sizes)

    def test_uint64_overflow_rejected(self):
        with self.assertRaisesRegex(binary.CircleSerializationError, "uint64"):
            binary._payload_layout(16, [2**64 - 1])

    def test_address_space_overflow_rejected(self):
        with patch.object(binary.sys, "maxsize", 100):
            with self.assertRaisesRegex(
                binary.CircleSerializationError, "address space"
            ):
                binary._payload_layout(16, [100])

    def test_join_includes_header_padding_and_all_payloads(self):
        header = bytearray(b"header")
        data = [memoryview(b"abc"), memoryview(b"12345")]
        offsets, total = binary._payload_layout(len(header), [3, 5])
        result = binary._join_payloads(header, data, offsets)
        self.assertIsInstance(result, bytes)
        self.assertEqual(len(result), total)
        self.assertEqual(result, b"header" + bytes(10) + b"abc" + bytes(13) + b"12345")

    def test_join_rejects_overlaps_and_missing_offsets(self):
        with self.assertRaisesRegex(binary.CircleSerializationError, "overlap"):
            binary._join_payloads(bytearray(8), [memoryview(b"ab")], [4])
        with self.assertRaisesRegex(binary.CircleSerializationError, "one offset"):
            binary._join_payloads(bytearray(), [memoryview(b"ab")], [])

    def test_contiguous_payload_is_not_copied(self):
        values = np.arange(17, dtype=np.uint8)
        view = binary._byte_view(values)
        values[3] = 200
        self.assertEqual(view[3], 200)
        self.assertEqual(binary._buffer_size(values), 17)

    def test_noncontiguous_payload_has_logical_byte_order(self):
        values = np.arange(32, dtype=np.uint8)[::-2]
        self.assertEqual(bytes(binary._byte_view(values)), values.tobytes())
        self.assertEqual(binary._buffer_size(values), values.size)

    def test_python_sequences_are_validated(self):
        self.assertEqual(bytes(binary._byte_view([0, 127, 255])), b"\0\x7f\xff")
        with self.assertRaises(ValueError):
            binary._byte_view([256])

    def test_non_byte_arrays_rejected(self):
        for values in (np.zeros(4, dtype=np.float32), np.zeros((2, 2), dtype=np.uint8)):
            with self.subTest(shape=values.shape, dtype=values.dtype):
                with self.assertRaisesRegex(
                    binary.CircleSerializationError, "uint8 vector"
                ):
                    binary._buffer_size(values)

    def test_non_byte_memoryview_is_rejected_consistently(self):
        data = memoryview(np.arange(4, dtype=np.int32))
        for function in (binary._buffer_size, binary._byte_view):
            with self.assertRaisesRegex(
                binary.CircleSerializationError, "unsigned bytes"
            ):
                function(data)


class RecordingStream:
    """Collect writes and optionally accept only ``limit`` bytes per call."""

    def __init__(self, limit=None, fail_after=None):
        self.chunks = []
        self.limit = limit
        self.fail_after = fail_after

    def write(self, view):
        if self.fail_after is not None and self.written >= self.fail_after:
            raise OSError("disk full")
        data = bytes(view)
        if self.limit is not None:
            data = data[: self.limit]
        self.chunks.append(data)
        return len(data)

    @property
    def written(self):
        return sum(len(chunk) for chunk in self.chunks)

    def getvalue(self):
        return b"".join(self.chunks)


class CircleBinaryLayoutStreamingTest(unittest.TestCase):
    def _layout(self):
        header = bytearray(b"header")
        payloads = (memoryview(b"abc"), memoryview(bytes(range(40))))
        offsets, _ = binary._payload_layout(len(header), [3, 40])
        return binary.CircleBinaryLayout(header, payloads, offsets)

    def test_layout_size_and_bytes_match_join(self):
        layout = self._layout()
        expected = binary._join_payloads(layout.header, layout.payloads, layout.offsets)
        self.assertFalse(layout.inline)
        self.assertEqual(layout.size, len(expected))
        self.assertEqual(layout.to_bytes(), expected)

    def test_inline_layout_to_bytes_is_the_header(self):
        layout = binary.CircleBinaryLayout(bytearray(b"flat"), (), ())
        self.assertTrue(layout.inline)
        self.assertEqual(layout.size, 4)
        self.assertEqual(layout.to_bytes(), b"flat")
        self.assertIsInstance(layout.to_bytes(), bytes)

    def test_write_to_streams_header_padding_and_bounded_chunks(self):
        layout = self._layout()
        stream = RecordingStream()
        with patch.object(binary, "_join_payloads") as join:
            written = layout.write_to(stream, chunk_size=16)
        join.assert_not_called()
        self.assertEqual(written, layout.size)
        self.assertEqual(stream.getvalue(), layout.to_bytes())
        self.assertLessEqual(max(len(chunk) for chunk in stream.chunks), 16)
        # Padding is emitted separately and never exceeds the alignment.
        self.assertIn(bytes(10), stream.chunks)

    def test_write_to_honors_partial_writes(self):
        layout = self._layout()
        stream = RecordingStream(limit=5)
        layout.write_to(stream, chunk_size=64)
        self.assertEqual(stream.getvalue(), layout.to_bytes())

    def test_write_to_rejects_streams_that_make_no_progress(self):
        layout = self._layout()
        for result in (0, None, 999):
            with self.subTest(result=result):
                stream = SimpleNamespace(write=lambda view, result=result: result)
                with self.assertRaises(OSError):
                    layout.write_to(stream)

    def test_write_failure_propagates_after_partial_output(self):
        layout = self._layout()
        stream = RecordingStream(fail_after=8)
        with self.assertRaisesRegex(OSError, "disk full"):
            layout.write_to(stream, chunk_size=4)
        self.assertLess(stream.written, layout.size)

    def test_invalid_chunk_size_and_layouts_rejected(self):
        layout = self._layout()
        with self.assertRaises(ValueError):
            layout.write_to(RecordingStream(), chunk_size=0)
        with self.assertRaisesRegex(binary.CircleSerializationError, "overlap"):
            binary.CircleBinaryLayout(bytearray(8), (memoryview(b"ab"),), (4,))
        with self.assertRaisesRegex(binary.CircleSerializationError, "one offset"):
            binary.CircleBinaryLayout(bytearray(), (memoryview(b"ab"),), ())


class CircleBinaryInlineBudgetTest(unittest.TestCase):
    """The streaming save path may lower the inline threshold, never raise it."""

    @staticmethod
    def _schema_stub():
        return CircleBinaryPolicyTest._schema_stub()

    def test_budget_below_payload_total_skips_inline_pack(self):
        document = model(b"a" * 300, b"b" * 300)
        with patch.object(
            binary, "_pack_flatbuffer", return_value=bytearray(100)
        ) as pack, patch.dict(sys.modules, {"circle_schema": self._schema_stub()}):
            layout = binary.plan_circle_layout(document, inline_budget=500)
        self.assertEqual(pack.call_count, 1)
        self.assertIsNone(pack.call_args.args[0].buffers[1].data)
        self.assertFalse(layout.inline)
        self.assertEqual([view.nbytes for view in layout.payloads], [300, 300])
        self.assertTrue(all(offset % 16 == 0 for offset in layout.offsets))

    def test_budget_above_payload_total_keeps_inline_layout(self):
        document = model(b"a" * 300, b"b" * 300)
        with patch.object(
            binary, "_pack_flatbuffer", return_value=bytearray(b"inline")
        ) as pack:
            layout = binary.plan_circle_layout(document, inline_budget=600)
        pack.assert_called_once_with(document, binary._FLATBUFFER_LIMIT)
        self.assertTrue(layout.inline)
        self.assertEqual(layout.to_bytes(), b"inline")

    def test_budget_cannot_exceed_the_flatbuffer_threshold(self):
        document = model(b"a" * 700, b"b" * 700)
        with patch.object(binary, "_FLATBUFFER_LIMIT", 1024), patch.object(
            binary, "_pack_flatbuffer", return_value=bytearray(64)
        ) as pack, patch.dict(sys.modules, {"circle_schema": self._schema_stub()}):
            layout = binary.plan_circle_layout(document, inline_budget=10**9)
        self.assertEqual(pack.call_count, 1)
        self.assertFalse(layout.inline)

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            binary.plan_circle_layout(model(b"abc"), inline_budget=-1)


class CircleBinaryRangeTest(unittest.TestCase):
    def test_inline_buffers_have_no_external_ranges(self):
        root = AccessorRoot(AccessorBuffer(), AccessorBuffer(length=16))
        self.assertEqual(binary.external_buffer_ranges(root, 512), ())

    def test_valid_range_and_exact_end_of_file(self):
        root = AccessorRoot(AccessorBuffer(), AccessorBuffer(32, 16))
        self.assertEqual(
            binary.external_buffer_ranges(root, 48),
            (binary.ExternalBufferRange(1, 32, 16),),
        )

    def test_out_of_bounds_and_incomplete_ranges_rejected(self):
        for offset, size in ((0, 5), (1, 5), (16, 0), (4, 8), (65, 1), (60, 5)):
            with self.subTest(offset=offset, size=size):
                root = AccessorRoot(AccessorBuffer(), AccessorBuffer(offset, size))
                with self.assertRaises(binary.CircleSerializationError):
                    binary.external_buffer_ranges(root, 64)

    def test_buffer_zero_must_not_be_external(self):
        with self.assertRaisesRegex(binary.CircleSerializationError, "buffer 0"):
            binary.external_buffer_ranges(AccessorRoot(AccessorBuffer(16, 2)), 32)

    def test_ambiguous_inline_and_external_data_rejected(self):
        root = AccessorRoot(AccessorBuffer(), AccessorBuffer(32, 16, 16))
        with self.assertRaisesRegex(binary.CircleSerializationError, "both inline"):
            binary.external_buffer_ranges(root, 64)

    def test_wide_file_offsets_do_not_wrap(self):
        offset = 2**32 + 16
        root = AccessorRoot(AccessorBuffer(), AccessorBuffer(offset, 32))
        self.assertEqual(
            binary.external_buffer_ranges(root, offset + 32)[0].offset, offset
        )
        with self.assertRaises(binary.CircleSerializationError):
            binary.external_buffer_ranges(root, 1024)

    def test_restore_retains_read_only_zero_copy_payload(self):
        source = bytes(32) + b"constant-data"
        document = model()
        document.buffers.append(buffer(offset=32, size=13))
        binary.restore_external_buffers(
            document, source, (binary.ExternalBufferRange(1, 32, 13),)
        )
        value = document.buffers[1]
        self.assertEqual(value.data.tobytes(), b"constant-data")
        self.assertIs(value.data.base, source)
        self.assertFalse(value.data.flags.writeable)
        self.assertEqual((value.offset, value.size), (0, 0))

    def test_direct_repack_of_unresolved_offsets_rejected(self):
        document = model(b"data")
        document.buffers[1].offset = 16
        document.buffers[1].size = 4
        with self.assertRaisesRegex(binary.CircleSerializationError, "unresolved"):
            binary.serialize_circle_model(document)

    def test_external_custom_options_fail_explicitly(self):
        document = model()
        document.subgraphs = [
            SimpleNamespace(
                operators=[
                    SimpleNamespace(
                        largeCustomOptionsOffset=64, largeCustomOptionsSize=8
                    )
                ]
            )
        ]
        with self.assertRaisesRegex(binary.CircleSerializationError, "custom operator"):
            binary.serialize_circle_model(document)


class CircleBinaryPolicyTest(unittest.TestCase):
    """Mock the wire packer to isolate size policy and non-mutation guarantees."""

    @staticmethod
    def _schema_stub():
        # This deliberately is not a FlatBuffer parser. Generated-schema/wire
        # compatibility is tested separately in test_large_circle_roundtrip.py.
        def get_root(header, offset):
            def get_buffer(index):
                return SimpleNamespace(
                    _tab=SimpleNamespace(
                        Pos=index * 16,
                        Offset=lambda slot: 8 if slot == 6 else 0,
                    )
                )

            return SimpleNamespace(Buffers=get_buffer)

        return SimpleNamespace(
            circle=SimpleNamespace(
                Model=SimpleNamespace(Model=SimpleNamespace(GetRootAsModel=get_root))
            )
        )

    def test_small_model_keeps_complete_inline_bytes(self):
        document = model(b"abc")
        with patch.object(
            binary, "_pack_flatbuffer", return_value=bytearray(b"complete")
        ) as pack:
            result = binary.serialize_circle_model(document)
        self.assertEqual(result, b"complete")
        pack.assert_called_once_with(document, binary._FLATBUFFER_LIMIT)

    def test_aggregate_size_triggers_external_layout(self):
        # Both constants fit separately; only their aggregate exceeds the budget.
        document = model(b"a" * 700, b"b" * 700)
        header = bytearray(100)
        with patch.object(binary, "_FLATBUFFER_LIMIT", 1024), patch.object(
            binary, "_pack_flatbuffer", return_value=header
        ) as pack, patch.dict(sys.modules, {"circle_schema": self._schema_stub()}):
            result = binary.serialize_circle_model(document)
        projected = pack.call_args.args[0]
        self.assertIsNot(projected, document)
        self.assertIs(projected.buffers[0], document.buffers[0])
        self.assertIsNone(projected.buffers[1].data)
        self.assertEqual(projected.buffers[1].size, 700)
        self.assertEqual(document.buffers[1].data, b"a" * 700)
        self.assertEqual(document.buffers[1].offset, 0)
        self.assertEqual(pack.call_count, 1)
        first = struct.unpack_from("<Q", result, 24)[0]
        second = struct.unpack_from("<Q", result, 40)[0]
        self.assertEqual(result[first : first + 700], b"a" * 700)
        self.assertEqual(result[second : second + 700], b"b" * 700)
        self.assertGreater(len(result), 1024)
        self.assertEqual(first % 16, 0)
        self.assertEqual(second % 16, 0)

    def test_actual_pack_overflow_uses_external_layout(self):
        document = model(b"abcd")
        with patch.object(
            binary,
            "_pack_flatbuffer",
            side_effect=[
                binary._FlatbufferTooLarge("metadata overhead"),
                bytearray(64),
            ],
        ) as pack, patch.dict(sys.modules, {"circle_schema": self._schema_stub()}):
            result = binary.serialize_circle_model(document)
        self.assertEqual(pack.call_count, 2)
        self.assertTrue(result.endswith(b"abcd"))

    def test_ordinary_errors_are_not_a_size_fallback(self):
        for error in (ValueError("bad tensor"), MemoryError("no RAM")):
            with self.subTest(error=type(error).__name__):
                with patch.object(
                    binary, "_pack_flatbuffer", side_effect=error
                ) as pack:
                    with self.assertRaises(type(error)):
                        binary.serialize_circle_model(model(b"abcd"))
                self.assertEqual(pack.call_count, 1)

    def test_metadata_too_large_is_not_swallowed(self):
        document = model(b"abc")
        with patch.object(
            binary, "_pack_flatbuffer", side_effect=binary._FlatbufferTooLarge
        ):
            with self.assertRaisesRegex(
                binary.CircleSerializationError, "Split the graph"
            ):
                binary.serialize_circle_model(document)
        self.assertEqual(document.buffers[1].data, b"abc")
        self.assertEqual(document.buffers[1].offset, 0)

    def test_nonempty_sentinel_rejected(self):
        document = model()
        document.buffers[0].data = b"x"
        with self.assertRaisesRegex(binary.CircleSerializationError, "buffer 0"):
            binary.serialize_circle_model(document)

    def test_one_byte_and_empty_buffers_remain_inline(self):
        document = model(b"z", b"", b"abc")
        with patch.object(
            binary,
            "_pack_flatbuffer",
            side_effect=[binary._FlatbufferTooLarge(), bytearray(96)],
        ) as pack, patch.dict(sys.modules, {"circle_schema": self._schema_stub()}):
            binary.serialize_circle_model(document)
        projected = pack.call_args.args[0]
        self.assertEqual(projected.buffers[1].data, b"z")
        self.assertEqual(projected.buffers[2].data, b"")
        self.assertIsNone(projected.buffers[3].data)

    def test_no_relocatable_data_reports_metadata_error(self):
        with patch.object(
            binary, "_pack_flatbuffer", side_effect=binary._FlatbufferTooLarge
        ):
            with self.assertRaisesRegex(
                binary.CircleSerializationError, "no relocatable"
            ):
                binary.serialize_circle_model(model(b"x"))

    def test_invalid_model_rejected(self):
        with self.assertRaises(TypeError):
            binary.serialize_circle_model(None)

    def test_size_guard_runs_before_vector_copy(self):
        class Builder:
            def __init__(self, initial_size):
                self.prep_called = False

            def Finish(self, offset, identifier):
                pass

            def Offset(self):
                return 0

            def Prep(self, size, additional):
                self.prep_called = True

        runtime = SimpleNamespace(
            Builder=Builder, builder=SimpleNamespace(BuilderSizeError=RuntimeError)
        )
        copied = []

        def pack(builder):
            builder.Prep(4, 2**31)
            copied.append(True)

        document = SimpleNamespace(Pack=pack)
        with patch.object(binary, "_load_flatbuffers", return_value=runtime):
            with self.assertRaises(binary._FlatbufferTooLarge):
                binary._pack_flatbuffer(document, 1024)
        self.assertEqual(copied, [])


if __name__ == "__main__":
    unittest.main()

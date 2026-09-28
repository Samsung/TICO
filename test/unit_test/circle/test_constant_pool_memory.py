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

"""Regression tests for exact, bounded-memory Circle constant indexing."""

from __future__ import annotations

import gc
import importlib.util
import struct
import tracemalloc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tico.circle import _buffer
from tico.circle.analysis import TensorContract
from tico.circle.builder import CircleBuilder, ConstantKey, ConstantPool
from tico.circle.document import CircleDocument
from tico.circle.graph import CircleGraph
from tico.circle.session import CircleOptimizationSession
from tico.circle.value import (
    TensorQuantization,
    TensorTypeRegistry,
    TensorTypeSpec,
    TensorValue,
    TensorValueCodec,
)


def make_codec():
    """Use an explicit registry so core pool tests need no generated schema."""

    return TensorValueCodec(
        TensorTypeRegistry(
            [
                TensorTypeSpec("FLOAT32", 0, np.float32, np.float32, 32),
                TensorTypeSpec("INT32", 2, np.int32, np.int32, 32, True),
                TensorTypeSpec("UINT8", 3, np.uint8, np.uint8, 8, False),
                TensorTypeSpec("UINT4", 1004, np.uint8, np.uint8, 4, False, True),
            ]
        )
    )


def object_factory(name):
    """Create mutable schema-like tables, as other Circle unit fixtures do."""

    if name == "Buffer":
        return SimpleNamespace(data=None, offset=0, size=0)
    return SimpleNamespace()


def make_tensor(index, size, **overrides):
    tensor = SimpleNamespace(
        name=f"tensor_{index}",
        buffer=index,
        type=3,
        shape=[size],
        shapeSignature=None,
        isVariable=False,
        quantization=None,
    )
    vars(tensor).update(overrides)
    return tensor


def make_model(*payloads, aliases=1, subgraphs=1):
    buffers = [object_factory("Buffer")]
    for payload in payloads:
        buffers.append(SimpleNamespace(data=payload, offset=0, size=0))
    graphs = []
    for _ in range(subgraphs):
        tensors = []
        for index, payload in enumerate(payloads, 1):
            for alias in range(aliases):
                size = 0 if payload is None else len(payload)
                tensors.append(make_tensor(index, size, name=f"w{index}_{alias}"))
        graphs.append(
            SimpleNamespace(
                tensors=tensors, inputs=[], outputs=[], operators=[], name="main"
            )
        )
    return SimpleNamespace(
        buffers=buffers,
        subgraphs=graphs,
        operatorCodes=[],
        signatureDefs=[],
        metadata=[],
        metadataBuffer=[],
        description="memory regression",
        version=0,
    )


def make_pool(model):
    return ConstantPool(model, codec=make_codec(), object_factory=object_factory)


class PayloadViewTest(unittest.TestCase):
    def test_contiguous_array_is_borrowed_and_not_frozen(self):
        array = np.arange(17, dtype=np.uint8)
        view = _buffer.inline_payload_view(SimpleNamespace(data=array))
        self.assertIsNotNone(view)
        self.assertTrue(np.shares_memory(array, np.frombuffer(view, np.uint8)))
        self.assertTrue(array.flags.writeable)
        array[5] = 99
        self.assertEqual(view[5], 99)

    def test_read_only_payload_remains_read_only(self):
        array = np.frombuffer(b"payload", dtype=np.uint8)
        view = _buffer.inline_payload_view(SimpleNamespace(data=array))
        self.assertTrue(view.readonly)
        self.assertEqual(bytes(view), b"payload")

    def test_supported_storage_representations(self):
        for source in (b"abc", bytearray(b"abc"), memoryview(b"abc"), [97, 98, 99]):
            with self.subTest(storage=type(source).__name__):
                view = _buffer.inline_payload_view(SimpleNamespace(data=source))
                self.assertEqual(bytes(view), b"abc")

    def test_noncontiguous_array_preserves_logical_order(self):
        values = np.arange(31, dtype=np.uint8)[::-3]
        view = _buffer.inline_payload_view(SimpleNamespace(data=values))
        self.assertEqual(bytes(view), values.tobytes())

    def test_noncontiguous_memoryview_preserves_logical_order(self):
        values = memoryview(b"0123456789")[::2]
        view = _buffer.inline_payload_view(SimpleNamespace(data=values))
        self.assertEqual(bytes(view), b"02468")

    def test_absent_and_unresolved_payloads_are_not_indexed(self):
        for buffer in (
            SimpleNamespace(data=None),
            SimpleNamespace(data=b"x", offset=64),
            SimpleNamespace(data=None, size=100),
            SimpleNamespace(data=["bad"]),
        ):
            self.assertIsNone(_buffer.inline_payload_view(buffer))

    def test_empty_payload_is_distinct_from_absent(self):
        view = _buffer.inline_payload_view(SimpleNamespace(data=b""))
        self.assertEqual(bytes(view), b"")
        self.assertIsNone(_buffer.inline_payload_view(SimpleNamespace(data=None)))

    def test_digest_uses_length_and_actual_byte_content(self):
        first = _buffer.payload_fingerprint(memoryview(b"abc"))
        self.assertEqual(first[0], 3)
        self.assertEqual(len(first[1]), 32)
        self.assertEqual(
            first, _buffer.payload_fingerprint(memoryview(bytearray(b"abc")))
        )
        self.assertNotEqual(first, _buffer.payload_fingerprint(memoryview(b"abd")))

    def test_exact_equality_checks_every_chunk_and_length(self):
        for mismatch in (0, 15, 16, 31, 32, 96):
            left = bytearray(97)
            right = bytearray(left)
            right[mismatch] = 1
            with patch.object(_buffer, "_COMPARE_CHUNK_BYTES", 16):
                self.assertFalse(
                    _buffer.payloads_equal(memoryview(left), memoryview(right))
                )
                self.assertTrue(
                    _buffer.payloads_equal(memoryview(left), memoryview(left))
                )
        self.assertFalse(_buffer.payloads_equal(memoryview(b"x"), memoryview(b"xx")))
        self.assertTrue(_buffer.payloads_equal(memoryview(b""), memoryview(b"")))

    def test_equality_is_bitwise_not_floating_point_equality(self):
        positive_zero = memoryview(struct.pack("<I", 0))
        negative_zero = memoryview(struct.pack("<I", 0x80000000))
        nan_a = memoryview(struct.pack("<I", 0x7FC00001))
        nan_b = memoryview(struct.pack("<I", 0x7FC00002))
        self.assertFalse(_buffer.payloads_equal(positive_zero, negative_zero))
        self.assertTrue(_buffer.payloads_equal(nan_a, nan_a))
        self.assertFalse(_buffer.payloads_equal(nan_a, nan_b))


class ConstantPoolMemoryTest(unittest.TestCase):
    def test_initial_index_has_no_payload_sized_allocations(self):
        source = np.full(16 * 1024 * 1024, 7, dtype=np.uint8)
        model = make_model(source)
        make_pool(make_model(np.zeros(1, dtype=np.uint8)))  # Warm lazy imports.
        gc.collect()
        tracemalloc.start()
        try:
            pool = make_pool(model)
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(pool.statistics["buffers"], 1)
        self.assertEqual(pool.statistics["tensors"], 1)
        self.assertLess(current, 1024 * 1024)
        self.assertLess(peak, 2 * 1024 * 1024)
        self.assertIs(model.buffers[1].data, source)

    def test_exact_duplicate_comparison_uses_bounded_scratch(self):
        left = np.full(8 * 1024 * 1024, 3, dtype=np.uint8)
        right = left.copy()
        model = make_model(left, right)
        tracemalloc.start()
        try:
            pool = make_pool(model)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(pool.statistics["buffers"], 1)
        self.assertLess(peak, 2 * 1024 * 1024)

    def test_shared_tensor_aliases_do_not_rehash_weight(self):
        model = make_model(np.arange(32, dtype=np.uint8), aliases=32, subgraphs=2)
        with patch(
            "tico.circle.builder.payload_fingerprint",
            wraps=_buffer.payload_fingerprint,
        ) as digest:
            pool = make_pool(model)
        self.assertEqual(digest.call_count, 1)
        self.assertEqual(pool.statistics["tensors"], 2)

    def test_digest_collision_does_not_deduplicate_different_content(self):
        with patch(
            "tico.circle.builder.payload_fingerprint",
            return_value=(4, b"collision"),
        ):
            pool = make_pool(make_model())
            first = pool.intern_buffer(b"aaaa")
            second = pool.intern_buffer(b"bbbb")
            self.assertNotEqual(first, second)
            self.assertEqual(pool.intern_buffer(b"aaaa"), first)
            self.assertEqual(pool.intern_buffer(b"bbbb"), second)
            self.assertEqual(pool.statistics["buffers"], 2)

    def test_digest_collision_does_not_merge_tensors(self):
        with patch(
            "tico.circle.builder.payload_fingerprint",
            return_value=(2, b"collision"),
        ):
            model = make_model(np.array([1, 2], np.uint8), np.array([3, 4], np.uint8))
            pool = make_pool(model)
            for expected, values in enumerate(([1, 2], [3, 4])):
                value = TensorValue(3, (2,), np.array(values, np.uint8))
                index = pool.intern_constant(
                    subgraph_index=0, name="lookup", value=value
                )
                self.assertEqual(index, expected)

    def test_existing_duplicates_choose_first_buffer_and_first_tensor(self):
        model = make_model(np.array([1, 2], np.uint8), np.array([1, 2], np.uint8))
        pool = make_pool(model)
        self.assertEqual(pool.statistics["buffers"], 1)
        value = TensorValue(3, (2,), np.array([1, 2], np.uint8))
        self.assertEqual(
            pool.intern_constant(subgraph_index=0, name="again", value=value),
            0,
        )
        self.assertEqual(pool.intern_buffer(b"\x01\x02"), 1)

    def test_explicit_duplicates_remain_available_for_reuse(self):
        pool = make_pool(make_model())
        first = pool.add_buffer(b"payload", deduplicate=False)
        second = pool.add_buffer(b"payload", deduplicate=False)
        self.assertNotEqual(first, second)
        self.assertEqual(pool.intern_buffer(b"payload"), first)
        self.assertEqual(pool.statistics["buffers"], 1)

    def test_added_buffer_owns_independent_writable_storage(self):
        model = make_model()
        pool = make_pool(model)
        payload = bytearray(b"abc")
        index = pool.add_buffer(payload)
        payload[0] = 0
        self.assertEqual(bytes(model.buffers[index].data), b"abc")
        self.assertTrue(model.buffers[index].data.flags.writeable)

    def test_tensor_contracts_and_subgraph_namespaces_are_preserved(self):
        model = make_model(subgraphs=2)
        pool = make_pool(model)
        value = TensorValue(3, (2,), np.array([1, 2], np.uint8))
        first = pool.intern_constant(subgraph_index=0, name="a", value=value)
        repeat = pool.intern_constant(subgraph_index=0, name="b", value=value)
        other_graph = pool.intern_constant(subgraph_index=1, name="c", value=value)
        matrix = TensorValue(3, (1, 2), value.data.reshape(1, 2))
        different_shape = pool.intern_constant(subgraph_index=0, name="d", value=matrix)
        self.assertEqual(first, repeat)
        self.assertEqual(other_graph, 0)
        self.assertNotEqual(first, different_shape)
        self.assertEqual(len(model.buffers), 2)

    def test_quantization_contracts_are_not_merged(self):
        pool = make_pool(make_model())
        indices = []
        for scale in (0.5, 1.0):
            value = TensorValue(
                3,
                (2,),
                np.array([1, 2], np.uint8),
                TensorQuantization(scale=(scale,), zero_point=(0,)),
            )
            indices.append(
                pool.intern_constant(subgraph_index=0, name="q", value=value)
            )
        self.assertNotEqual(*indices)
        self.assertEqual(pool.statistics["buffers"], 1)

    def test_packed_uint4_reuses_exact_encoded_bytes(self):
        model = make_model()
        pool = make_pool(model)
        value = TensorValue(1004, (3,), np.array([0, 15, 7], np.uint8))
        index = pool.intern_constant(subgraph_index=0, name="packed", value=value)
        repeat = pool.intern_constant(subgraph_index=0, name="same", value=value)
        self.assertEqual(index, repeat)
        self.assertEqual(bytes(model.buffers[1].data), bytes([0xF0, 0x07]))
        pool.synchronize(force=True)
        self.assertEqual(
            pool.intern_constant(subgraph_index=0, name="after", value=value),
            index,
        )

    def test_external_buffer_and_absent_data_are_excluded(self):
        model = make_model(np.array([1], np.uint8), None)
        model.buffers[1].offset = 64
        model.buffers[1].size = 1
        pool = make_pool(model)
        self.assertEqual(pool.statistics["buffers"], 0)
        self.assertEqual(pool.statistics["tensors"], 0)
        self.assertEqual(pool.intern_buffer(b"\x01"), 3)

    def test_variable_tensors_are_not_interned_as_immutable(self):
        model = make_model(np.array([1], np.uint8))
        model.subgraphs[0].tensors[0].isVariable = True
        pool = make_pool(model)
        value = TensorValue(3, (1,), np.array([1], np.uint8))
        self.assertEqual(
            pool.intern_constant(subgraph_index=0, name="const", value=value),
            1,
        )

    def test_empty_payload_is_reused_but_not_buffer_zero(self):
        pool = make_pool(make_model(np.empty(0, np.uint8)))
        self.assertEqual(pool.intern_buffer(b""), 1)

    def test_append_synchronization_does_not_rehash_existing_buffer(self):
        model = make_model(np.arange(17, dtype=np.uint8))
        pool = make_pool(model)
        model.buffers.append(
            SimpleNamespace(data=np.array([99], np.uint8), offset=0, size=0)
        )
        model.subgraphs[0].tensors.append(make_tensor(2, 1))
        with patch(
            "tico.circle.builder.payload_fingerprint",
            wraps=_buffer.payload_fingerprint,
        ) as digest:
            pool.synchronize()
        self.assertEqual(digest.call_count, 1)
        self.assertEqual(pool.statistics["buffers"], 2)
        self.assertEqual(pool.statistics["rebuilds"], 0)

    def test_touched_tensor_restores_equivalent_sibling(self):
        model = make_model(np.array([1, 2], np.uint8), aliases=2)
        pool = make_pool(model)
        model.subgraphs[0].tensors[0].shape = [1, 2]
        pool.synchronize(subgraph_index=0, tensor_indices=(0,))
        value = TensorValue(3, (2,), np.array([1, 2], np.uint8))
        self.assertEqual(
            pool.intern_constant(subgraph_index=0, name="reuse", value=value),
            1,
        )

    def test_force_rebuild_after_in_place_payload_change(self):
        model = make_model(np.array([1, 2], np.uint8))
        pool = make_pool(model)
        model.buffers[1].data[:] = [3, 4]
        pool.synchronize(force=True)
        self.assertEqual(pool.intern_buffer(bytes([3, 4])), 1)
        self.assertNotEqual(pool.intern_buffer(bytes([1, 2])), 1)

    def test_force_rebuild_does_not_retain_replaced_array(self):
        model = make_model(np.full(1024, 1, np.uint8))
        old = weakref.ref(model.buffers[1].data)
        pool = make_pool(model)
        model.buffers[1].data = np.full(1024, 2, np.uint8)
        pool.synchronize(force=True)
        gc.collect()
        self.assertIsNone(old())
        self.assertEqual(pool.intern_buffer(bytes([2]) * 1024), 1)

    def test_compaction_rebuilds_canonical_indices(self):
        model = make_model(np.array([1], np.uint8), np.array([2], np.uint8))
        pool = make_pool(model)
        del model.buffers[1]
        model.subgraphs[0].tensors.pop(0)
        model.subgraphs[0].tensors[0].buffer = 1
        pool.synchronize()
        self.assertEqual(pool.intern_buffer(bytes([2])), 1)
        self.assertEqual(pool.statistics["rebuilds"], 1)

    def test_active_session_delegates_without_duplicate_indexing(self):
        model = make_model(np.arange(1024, dtype=np.uint8))
        codec = make_codec()
        session = CircleOptimizationSession(model)
        with session.activate():
            first = ConstantPool(model, codec=codec, object_factory=object_factory)
            with patch(
                "tico.circle.builder.payload_fingerprint",
                wraps=_buffer.payload_fingerprint,
            ) as digest:
                second = ConstantPool(model, codec=codec, object_factory=object_factory)
            self.assertEqual(digest.call_count, 0)
            self.assertEqual(first.statistics, second.statistics)

    def test_watched_payload_commit_refreshes_all_tensor_aliases(self):
        model = make_model(np.array([1, 2], np.uint8), aliases=2)
        session = CircleOptimizationSession(model)
        pool = session.constant_pool(codec=make_codec(), object_factory=object_factory)
        with session.transaction(subgraph_index=0) as transaction:
            transaction.watch_buffer(1)
            model.buffers[1].data[:] = [3, 4]
            transaction.commit()
        self.assertEqual(pool.statistics["rebuilds"], 1)
        value = TensorValue(3, (2,), np.array([3, 4], np.uint8))
        self.assertEqual(
            pool.intern_constant(subgraph_index=0, name="new", value=value),
            0,
        )
        self.assertNotEqual(pool.intern_buffer(bytes([1, 2])), 1)

    def test_watched_payload_rollback_restores_pool(self):
        model = make_model(np.array([1, 2], np.uint8))
        session = CircleOptimizationSession(model)
        pool = session.constant_pool(codec=make_codec(), object_factory=object_factory)
        with session.transaction(subgraph_index=0) as transaction:
            transaction.watch_buffer(1)
            model.buffers[1].data[:] = [3, 4]
        self.assertEqual(pool.intern_buffer(bytes([1, 2])), 1)
        self.assertNotEqual(pool.intern_buffer(bytes([3, 4])), 1)

    def test_public_constant_key_keeps_bytes_contract(self):
        contract = TensorContract(tensor_type=3, shape=(2,))
        first = ConstantKey(contract, b"ab")
        self.assertEqual(first, ConstantKey(contract, b"ab"))
        self.assertNotEqual(first, ConstantKey(contract, b"ac"))
        self.assertEqual(hash(first), hash(ConstantKey(contract, b"ab")))


class ConstantFoldPreflightMemoryTest(unittest.TestCase):
    def test_budget_preflight_borrows_payload_and_rejects_without_copy(self):
        from tico.circle.passes.optimization.fold.constant_subgraph import (
            _FoldBudgetState,
            _required_input_payloads,
            ConstantFoldPolicy,
        )

        array = np.full(16 * 1024 * 1024, 1, np.uint8)
        model = make_model(array)
        graph = CircleGraph(model, 0)
        budget = _FoldBudgetState(ConstantFoldPolicy(maximum_input_bytes=1024))
        tracemalloc.start()
        try:
            payloads = _required_input_payloads(model, graph, (0,), (0,))
            reason = budget.rejection_reason(
                input_bytes=sum(len(value) for value in payloads.values()),
                output_bytes=1,
                compute_cost=1,
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertIn("constant inputs", reason)
        self.assertIsInstance(payloads[0], memoryview)
        self.assertTrue(np.shares_memory(array, np.frombuffer(payloads[0], np.uint8)))
        self.assertLess(peak, 1024 * 1024)

    def test_late_runtime_input_does_not_copy_earlier_large_constant(self):
        from tico.circle.passes.optimization.fold.constant_subgraph import (
            _required_input_payloads,
        )

        model = make_model(np.full(16 * 1024 * 1024, 1, np.uint8))
        model.subgraphs[0].tensors.append(make_tensor(0, 1, name="input"))
        model.subgraphs[0].inputs = [1]
        graph = CircleGraph(model, 0)
        tracemalloc.start()
        try:
            result = _required_input_payloads(model, graph, (0, 1), (0, 1))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertIsNone(result)
        self.assertLess(peak, 1024 * 1024)

    def test_repeated_tensor_inputs_keep_original_budget_accounting(self):
        from tico.circle.passes.optimization.fold.constant_subgraph import (
            _required_input_payloads,
        )

        model = make_model(np.arange(7, dtype=np.uint8))
        payloads = _required_input_payloads(
            model, CircleGraph(model, 0), (0, 0), (0, 1)
        )
        self.assertEqual(sum(len(value) for value in payloads.values()), 7)


HAS_SCHEMA = (
    importlib.util.find_spec("circle_schema") is not None
    and importlib.util.find_spec("flatbuffers") is not None
)


@unittest.skipUnless(HAS_SCHEMA, "circle-schema and flatbuffers are required")
class ConstantPoolGeneratedSchemaTest(unittest.TestCase):
    def test_resolved_external_payload_stays_borrowed_while_indexing(self):
        from circle_schema import circle
        from tico.serialize import circle_binary

        model = circle.Model.ModelT()
        model.buffers = [circle.Buffer.BufferT(), circle.Buffer.BufferT()]
        model.buffers[1].data = np.full(16 * 1024 * 1024, 5, np.uint8)
        tensor = circle.Tensor.TensorT()
        tensor.name = "weight"
        tensor.type = circle.TensorType.TensorType.UINT8
        tensor.shape = [model.buffers[1].data.size]
        tensor.buffer = 1
        graph = circle.SubGraph.SubGraphT()
        graph.tensors = [tensor]
        graph.inputs = []
        graph.outputs = [0]
        graph.operators = []
        model.subgraphs = [graph]
        model.operatorCodes = []
        with patch.object(circle_binary, "_FLATBUFFER_LIMIT", 4096):
            original = CircleDocument(model).to_bytes()
        document = CircleDocument.from_bytes(original)
        array = document.model.buffers[1].data
        self.assertFalse(array.flags.writeable)
        tracemalloc.start()
        try:
            pool = ConstantPool(document.model)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 2 * 1024 * 1024)
        self.assertEqual(pool.statistics["buffers"], 1)
        self.assertIs(document.model.buffers[1].data, array)
        with patch.object(circle_binary, "_FLATBUFFER_LIMIT", 4096):
            restored = CircleDocument.from_bytes(document.to_bytes())
        np.testing.assert_array_equal(restored.model.buffers[1].data, array)

    def test_default_o1_keeps_dynamic_gather_payload_intact(self):
        from circle_schema import circle
        from tico.circle.export import optimize_for_export
        from tico.serialize import circle_binary

        model = circle.Model.ModelT()
        model.buffers = [circle.Buffer.BufferT(), circle.Buffer.BufferT()]
        weights = np.arange(1024, dtype=np.float32).reshape(256, 4)
        model.buffers[1].data = weights.reshape(-1).view(np.uint8)
        tensors = []
        for name, dtype, shape, signature, buffer in (
            ("weight", circle.TensorType.TensorType.FLOAT32, [256, 4], None, 1),
            ("ids", circle.TensorType.TensorType.INT32, [1], [-1], 0),
            ("output", circle.TensorType.TensorType.FLOAT32, [1, 4], [-1, 4], 0),
        ):
            tensor = circle.Tensor.TensorT()
            tensor.name, tensor.type, tensor.shape = name, dtype, shape
            tensor.shapeSignature, tensor.buffer = signature, buffer
            tensors.append(tensor)
        graph = circle.SubGraph.SubGraphT()
        graph.tensors, graph.inputs, graph.outputs = tensors, [1], [2]
        opcode = circle.OperatorCode.OperatorCodeT()
        opcode.builtinCode = circle.BuiltinOperator.BuiltinOperator.GATHER
        opcode.deprecatedBuiltinCode = opcode.builtinCode
        op = circle.Operator.OperatorT()
        op.opcodeIndex, op.inputs, op.outputs = 0, [0, 1], [2]
        op.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.GatherOptions
        op.builtinOptions = circle.GatherOptions.GatherOptionsT()
        graph.operators = [op]
        model.subgraphs, model.operatorCodes = [graph], [opcode]
        with patch.object(circle_binary, "_FLATBUFFER_LIMIT", 2048):
            original = CircleDocument(model).to_bytes()
            result = optimize_for_export(original)
        restored = CircleDocument.from_bytes(result)
        self.assertTrue(restored.verify(raise_on_error=False).ok)
        weight_tensor = next(
            t
            for t in restored.model.subgraphs[0].tensors
            if t.name in ("weight", b"weight")
        )
        payload = restored.model.buffers[weight_tensor.buffer].data
        np.testing.assert_array_equal(payload, weights.reshape(-1).view(np.uint8))


if __name__ == "__main__":
    unittest.main()

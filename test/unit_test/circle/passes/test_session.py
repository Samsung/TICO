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

import copy
import gc
import unittest
import weakref

import numpy as np

from tico.circle import (
    CircleBuilder,
    ConstantPool,
    TensorContract,
    TensorValue,
    TensorValueCodec,
)
from tico.circle.passes import (
    CirclePass,
    CirclePassContext,
    CirclePassManager,
    CirclePassResult,
    optimization_session_for,
)
from tico.circle.passes.cleanup import CompactIndicesPass

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    fake_object_factory,
    FLOAT32,
    make_empty_document,
    make_registry,
)


class _LegacyInputMutationPass(CirclePass):
    """Mutate directly to exercise pass-manager fallback invalidation."""

    def run(self, document, context):
        del context
        document.subgraph().inputs = []
        return CirclePassResult(modified=True, changes=1)


class _FailingLegacyInputMutationPass(CirclePass):
    """Mutate directly and fail before reporting a pass result."""

    def run(self, document, context):
        del context
        document.subgraph().inputs = []
        raise RuntimeError("legacy failure")


class CircleOptimizationSessionTest(unittest.TestCase):
    """Check revision-aware graph caching and model-shared constant indexes."""

    def test_graph_cache_reuses_one_index_until_transaction_commit(self) -> None:
        """Rebuild producer/consumer state only after a committed mutation."""

        document = make_empty_document()
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1],
        )
        document.subgraph().inputs = [source]
        session = optimization_session_for(document)

        first = session.graph(0)
        second = session.graph(0)
        self.assertIs(first, second)

        with session.transaction(subgraph_index=0) as mutation:
            mutation.watch_subgraph_field("inputs")
            document.subgraph().inputs = []
            mutation.commit()

        third = session.graph(0)
        self.assertIsNot(first, third)
        self.assertEqual(third.inputs, ())
        self.assertGreaterEqual(session.statistics.graph_cache_hits, 1)
        self.assertGreaterEqual(session.statistics.graph_cache_misses, 2)

    def test_document_graph_uses_the_active_session_cache(self) -> None:
        """Let legacy passes using document.graph participate in cache reuse."""

        document = make_empty_document()
        session = optimization_session_for(document)

        with session.activate():
            first = document.graph(0)
            second = document.graph(0)

        self.assertIs(first, second)
        self.assertGreaterEqual(session.statistics.graph_cache_hits, 1)

    def test_builders_share_constant_pool_and_reuse_semantic_constant(self) -> None:
        """Reuse indexes and tensors across independent builders."""

        document = make_empty_document()
        registry = make_registry()
        first_codec = TensorValueCodec(registry)
        second_codec = TensorValueCodec(registry)
        first_builder = CircleBuilder(
            document,
            codec=first_codec,
            object_factory=fake_object_factory,
        )
        second_builder = CircleBuilder(
            document,
            codec=second_codec,
            object_factory=fake_object_factory,
        )
        value = TensorValue.from_values(
            FLOAT32,
            np.asarray([1.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )

        first = first_builder.add_constant("first", value)
        second = second_builder.add_constant("second", value)

        self.assertIs(first_builder.constant_pool, second_builder.constant_pool)
        self.assertEqual(first, second)
        self.assertEqual(len(document.subgraph().tensors), 1)
        self.assertEqual(len(document.model.buffers), 2)
        statistics = first_builder.constant_pool.statistics
        self.assertEqual(statistics["tensors"], 1)
        self.assertEqual(statistics["buffers"], 1)

    def test_direct_pool_delegates_to_the_session_canonical_pool(self) -> None:
        """Share indexes with passes that construct ConstantPool directly."""

        document = make_empty_document()
        registry = make_registry()
        builder = CircleBuilder(
            document,
            codec=TensorValueCodec(registry),
            object_factory=fake_object_factory,
        )
        direct = ConstantPool(
            document.model,
            codec=TensorValueCodec(registry),
            object_factory=fake_object_factory,
        )
        value = TensorValue.from_values(
            FLOAT32,
            np.asarray([4.0], dtype=np.float32),
            dtype=np.float32,
        )

        first = builder.add_constant("builder", value)
        second = direct.intern_constant(
            subgraph_index=0,
            name="direct",
            value=value,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(document.subgraph().tensors), 1)
        self.assertEqual(direct.statistics, builder.constant_pool.statistics)

    def test_pool_refresh_recovers_an_equivalent_sibling_constant(self) -> None:
        """Keep semantic deduplication when the indexed representative changes."""

        document = make_empty_document()
        codec = TensorValueCodec(make_registry())
        builder = CircleBuilder(
            document,
            codec=codec,
            object_factory=fake_object_factory,
        )
        value = TensorValue.from_values(
            FLOAT32,
            np.asarray([3.0], dtype=np.float32),
            dtype=np.float32,
        )
        first = builder.add_constant("first", value)
        duplicate = copy.deepcopy(document.subgraph().tensors[first])
        duplicate.name = "duplicate"
        document.subgraph().tensors.append(duplicate)
        session = optimization_session_for(document)
        session.mark_modified((0,))

        with session.transaction(subgraph_index=0) as mutation:
            mutation.watch_tensor(first)
            document.subgraph().tensors[first].buffer = 0
            mutation.commit()

        reused = builder.add_constant("reused", value)
        self.assertEqual(reused, 1)
        self.assertEqual(len(document.subgraph().tensors), 2)

    def test_pool_rebuilds_after_transaction_rollback(self) -> None:
        """Never return a stale index after constants are rolled back."""

        document = make_empty_document()
        codec = TensorValueCodec(make_registry())
        builder = CircleBuilder(
            document,
            codec=codec,
            object_factory=fake_object_factory,
        )
        first_value = TensorValue.from_values(
            FLOAT32,
            np.asarray([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        second_value = TensorValue.from_values(
            FLOAT32,
            np.asarray([2.0], dtype=np.float32),
            dtype=np.float32,
        )
        builder.add_constant("first", first_value)
        session = optimization_session_for(document)

        with self.assertRaisesRegex(RuntimeError, "abort"):
            with session.transaction(subgraph_index=0):
                builder.add_constant("temporary", second_value)
                raise RuntimeError("abort")

        self.assertEqual(len(document.subgraph().tensors), 1)
        self.assertEqual(len(document.model.buffers), 2)
        restored = builder.add_constant("restored", second_value)
        self.assertEqual(restored, 1)
        self.assertEqual(len(document.subgraph().tensors), 2)
        self.assertEqual(len(document.model.buffers), 3)

    def test_compaction_rebuilds_pool_and_invalidates_cached_graph(self) -> None:
        """Refresh remapped tensor and buffer indexes after one-shot compaction."""

        document = make_empty_document()
        codec = TensorValueCodec(make_registry())
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1],
        )
        builder = CircleBuilder(
            document,
            codec=codec,
            object_factory=fake_object_factory,
        )
        used_value = TensorValue.from_values(
            FLOAT32,
            np.asarray([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        dead_value = TensorValue.from_values(
            FLOAT32,
            np.asarray([2.0], dtype=np.float32),
            dtype=np.float32,
        )
        used = builder.add_constant("used", used_value)
        builder.add_constant("dead", dead_value)
        output = builder.add_operator(
            101,
            inputs=(source, used),
            output_contracts=(
                TensorContract(
                    tensor_type=FLOAT32,
                    shape=(1,),
                    shape_signature=(1,),
                ),
            ),
            output_names=("output",),
        )[0]
        document.subgraph().inputs = [source]
        document.subgraph().outputs = [output]
        session = optimization_session_for(document)
        cached = session.graph(0)

        result = CompactIndicesPass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertIsNot(session.graph(0), cached)
        restored = builder.add_constant("restored", dead_value)
        self.assertLess(restored, len(document.subgraph().tensors))
        self.assertEqual(
            codec.decode_tensor(
                document.model,
                subgraph_index=0,
                tensor_index=restored,
            ).data.tolist(),
            [2.0],
        )

    def test_manager_invalidates_cache_for_legacy_direct_mutation(self) -> None:
        """Detect modified passes that did not call session-aware helpers."""

        document = make_empty_document()
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1],
        )
        document.subgraph().inputs = [source]
        context = CirclePassContext(verify_after_each_pass=False)
        session = context.session(document)
        cached = session.graph(0)

        CirclePassManager([_LegacyInputMutationPass()]).run(document, context)

        updated = session.graph(0)
        self.assertIsNot(updated, cached)
        self.assertEqual(updated.inputs, ())

    def test_registry_does_not_keep_an_unowned_session_alive(self) -> None:
        """Avoid accumulating completed model sessions in process-global state."""

        document = make_empty_document()
        session = optimization_session_for(document)
        reference = weakref.ref(session)

        del session
        gc.collect()

        self.assertIsNone(reference())

    def test_manager_invalidates_cache_when_a_legacy_pass_fails(self) -> None:
        """Do not retain stale graph state after an untracked partial mutation."""

        document = make_empty_document()
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1],
        )
        document.subgraph().inputs = [source]
        context = CirclePassContext(verify_after_each_pass=False)
        session = context.session(document)
        cached = session.graph(0)

        with self.assertRaisesRegex(RuntimeError, "legacy failure"):
            CirclePassManager([_FailingLegacyInputMutationPass()]).run(
                document,
                context,
            )

        updated = session.graph(0)
        self.assertIsNot(updated, cached)
        self.assertEqual(updated.inputs, ())

    def test_context_returns_same_model_scoped_session(self) -> None:
        """Let independently created pass contexts reuse one model's analyses."""

        document = make_empty_document()
        first = CirclePassContext().session(document)
        second = CirclePassContext().session(document)

        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()

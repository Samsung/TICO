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

import unittest

import numpy as np

from tico.circle.analysis import TensorContract
from tico.circle.builder import CircleBuilder, ConstantPool
from tico.circle.errors import CircleRewriteError
from tico.circle.value import TensorValue, TensorValueCodec

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    fake_object_factory,
    FakeBuffer,
    FakeOperator,
    FLOAT32,
    make_empty_document,
    make_registry,
)


class CircleBuilderTest(unittest.TestCase):
    """Check safe construction and deduplication across Circle index spaces."""

    def setUp(self):
        """Create a schema-independent codec for builder tests."""

        self.codec = TensorValueCodec(make_registry())

    def test_constant_pool_never_reuses_external_buffer_metadata(self):
        """Keep unresolved external payload references outside the inline pool."""

        document = make_empty_document()
        document.model.buffers.append(
            FakeBuffer(
                data=np.array([], dtype=np.uint8),
                offset=64,
                size=4,
            )
        )
        builder = CircleBuilder(
            document,
            codec=self.codec,
            object_factory=fake_object_factory,
        )
        empty = TensorValue(
            FLOAT32,
            (0,),
            np.array([], dtype=np.float32),
        )

        tensor_index = builder.add_constant("empty", empty)

        self.assertEqual(document.subgraph().tensors[tensor_index].buffer, 2)
        self.assertEqual(len(document.model.buffers), 3)

    def test_non_deduplicated_buffer_is_available_to_future_pool_lookups(self):
        """Register explicit duplicate storage without losing future reuse behavior."""

        document = make_empty_document()
        builder = CircleBuilder(
            document,
            codec=self.codec,
            object_factory=fake_object_factory,
        )
        first = builder.add_buffer(b"payload", deduplicate=False)
        second = builder.add_buffer(b"payload", deduplicate=False)
        reused = builder.add_buffer(b"payload")

        self.assertNotEqual(first, second)
        self.assertEqual(reused, first)
        self.assertEqual(len(document.model.buffers), 3)

    def test_constant_pool_deduplicates_tensor_and_buffer_in_one_subgraph(self):
        """Reuse one local tensor when value and complete contract are identical."""

        document = make_empty_document()
        builder = CircleBuilder(
            document,
            codec=self.codec,
            object_factory=fake_object_factory,
        )
        value = TensorValue(
            FLOAT32,
            (2,),
            np.array([1.0, 2.0], dtype=np.float32),
        )

        first = builder.add_constant("weight", value)
        second = builder.add_constant("other_name", value)

        self.assertEqual(first, second)
        self.assertEqual(len(document.subgraph().tensors), 1)
        self.assertEqual(len(document.model.buffers), 2)

    def test_builder_rejects_a_pool_from_another_model(self):
        """Prevent model-global buffer indexes from crossing model ownership."""

        first_document = make_empty_document()
        second_document = make_empty_document()
        pool = ConstantPool(
            first_document.model,
            codec=self.codec,
            object_factory=fake_object_factory,
        )

        with self.assertRaisesRegex(ValueError, "same Circle model"):
            CircleBuilder(second_document, constant_pool=pool)

    def test_reindexed_pool_recognizes_builder_created_constant(self):
        """Deduplicate constants after rebuilding indexes from generated defaults."""

        document = make_empty_document()
        value = TensorValue(
            FLOAT32,
            (1,),
            np.array([5.0], dtype=np.float32),
        )
        first_builder = CircleBuilder(
            document,
            codec=self.codec,
            object_factory=fake_object_factory,
        )
        first = first_builder.add_constant("first", value)
        rebuilt_pool = ConstantPool(
            document.model,
            codec=self.codec,
            object_factory=fake_object_factory,
        )
        second_builder = CircleBuilder(
            document,
            constant_pool=rebuilt_pool,
        )
        second = second_builder.add_constant("second", value)

        self.assertEqual(first, second)
        self.assertEqual(len(document.subgraph().tensors), 1)
        self.assertEqual(len(document.model.buffers), 2)

    def test_constant_pool_shares_buffer_but_not_tensor_between_subgraphs(self):
        """Respect subgraph-local tensor indices while sharing model-global bytes."""

        document = make_empty_document(subgraph_count=2)
        pool = ConstantPool(
            document.model,
            codec=self.codec,
            object_factory=fake_object_factory,
        )
        value = TensorValue(
            FLOAT32,
            (1,),
            np.array([3.0], dtype=np.float32),
        )
        first_builder = CircleBuilder(
            document,
            subgraph_index=0,
            codec=self.codec,
            object_factory=fake_object_factory,
            constant_pool=pool,
        )
        second_builder = CircleBuilder(
            document,
            subgraph_index=1,
            codec=self.codec,
            object_factory=fake_object_factory,
            constant_pool=pool,
        )

        first = first_builder.add_constant("first", value)
        second = second_builder.add_constant("second", value)

        self.assertEqual(first, 0)
        self.assertEqual(second, 0)
        self.assertEqual(len(document.model.buffers), 2)
        self.assertEqual(
            document.subgraph(0).tensors[0].buffer,
            document.subgraph(1).tensors[0].buffer,
        )

    def test_add_operator_supports_multiple_outputs_and_reuses_opcode(self):
        """Create all outputs and one shared operator-code record for repeated ops."""

        document = make_empty_document()
        input_index = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="input",
            shape=[1, 4],
        )
        builder = CircleBuilder(
            document,
            codec=self.codec,
            object_factory=fake_object_factory,
        )
        contract = TensorContract(
            tensor_type=FLOAT32,
            shape=(1, 2),
            shape_signature=(1, 2),
        )

        outputs = builder.add_operator(
            49,
            inputs=(input_index,),
            output_contracts=(contract, contract),
            output_names=("left", "right"),
        )
        builder.add_operator(
            49,
            inputs=(outputs[0],),
            output_contracts=(contract,),
            output_names=("left",),
        )

        self.assertEqual(outputs, (1, 2))
        self.assertEqual(document.subgraph().operators[0].outputs, [1, 2])
        self.assertEqual(document.subgraph().tensors[3].name, "left_0")
        self.assertEqual(len(document.model.operatorCodes), 1)

    def test_replace_operator_requires_preserved_output_indices(self):
        """Reject a replacement that silently changes an existing graph boundary."""

        document = make_empty_document()
        first = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="first",
            shape=[1],
        )
        second = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="second",
            shape=[1],
        )
        builder = CircleBuilder(
            document,
            codec=self.codec,
            object_factory=fake_object_factory,
        )
        opcode = builder.find_or_add_operator_code(0)
        document.subgraph().operators.append(
            FakeOperator(opcodeIndex=opcode, inputs=[first], outputs=[first])
        )
        replacement = FakeOperator(
            opcodeIndex=opcode,
            inputs=[first],
            outputs=[second],
        )

        with self.assertRaisesRegex(CircleRewriteError, "preserve output"):
            builder.replace_operator(0, replacement)

    def test_add_operator_rolls_back_output_tensors_after_failure(self):
        """Avoid partially created outputs when an operator input is invalid."""

        document = make_empty_document()
        builder = CircleBuilder(
            document,
            codec=self.codec,
            object_factory=fake_object_factory,
        )
        contract = TensorContract(tensor_type=FLOAT32, shape=(1,))

        with self.assertRaises(CircleRewriteError):
            builder.add_operator(
                0,
                inputs=(99,),
                output_contracts=(contract,),
            )
        self.assertEqual(document.subgraph().tensors, [])


if __name__ == "__main__":
    unittest.main()

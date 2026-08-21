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

import unittest
from dataclasses import dataclass

import numpy as np

from tico.circle import CircleBuilder, TensorContract, TensorValue, TensorValueCodec
from tico.circle._object import freeze_object
from tico.circle.mutation import current_mutation
from tico.circle.passes import (
    CirclePassContext,
    CircleRewriteRule,
    CircleRulePass,
    RewriteApplication,
    RewritePlan,
)
from tico.circle.rewrite import replace_tensor_uses

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    fake_object_factory,
    FakeSignatureDef,
    FakeTensorMap,
    FLOAT32,
    make_empty_document,
    make_registry,
)


@dataclass(frozen=True, kw_only=True)
class _ExternalFailingPlan(RewritePlan):
    """Exercise transaction discovery for plans declared outside tico.circle."""


class _FailingRewriteRule(CircleRewriteRule[_ExternalFailingPlan]):
    """Exercise allocation, rewiring, and tensor mutation before failure."""

    def __init__(self, codec: TensorValueCodec) -> None:
        self.codec = codec

    def match(self, document, graph, operator_index, context):
        del context
        if operator_index != 0:
            return None
        operator = graph.subgraph.operators[operator_index]
        return _ExternalFailingPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*operator.inputs, *operator.outputs),
        )

    def apply(self, document, plan, context):
        del context
        builder = CircleBuilder(
            document,
            subgraph_index=plan.subgraph_index,
            codec=self.codec,
            object_factory=fake_object_factory,
        )
        value = TensorValue.from_values(
            FLOAT32,
            np.asarray([9.0], dtype=np.float32),
            dtype=np.float32,
        )
        builder.add_constant("temporary", value)
        document.model.operatorCodes[plan.anchor.opcode_index].version = 99
        constant = plan.anchor.inputs[1]
        buffer_index = document.subgraph(plan.subgraph_index).tensors[constant].buffer
        mutation = current_mutation(
            model=document.model,
            subgraph_index=plan.subgraph_index,
        )
        assert mutation is not None
        mutation.watch_buffer(buffer_index)
        document.model.buffers[buffer_index].data[0] = 1
        output = plan.anchor.outputs[0]
        document.subgraph(plan.subgraph_index).tensors[output].name = "corrupted"
        replace_tensor_uses(
            document.model,
            subgraph_index=plan.subgraph_index,
            old_tensor_index=output,
            new_tensor_index=plan.anchor.inputs[0],
        )
        raise RuntimeError("intentional rewrite failure")


class CircleMutationTransactionTest(unittest.TestCase):
    """Check atomic rollback and automatic participation of rewrite helpers."""

    def _document(self):
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
        constant = builder.add_constant(
            "constant",
            TensorValue.from_values(
                FLOAT32,
                np.asarray([5.0], dtype=np.float32),
                dtype=np.float32,
            ),
        )
        output = builder.add_operator(
            101,
            inputs=(source, constant),
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
        document.model.signatureDefs = [
            FakeSignatureDef(
                inputs=[FakeTensorMap(name="source", tensorIndex=source)],
                outputs=[FakeTensorMap(name="output", tensorIndex=output)],
            )
        ]
        return document, codec

    def test_rule_pass_rolls_back_failed_apply_completely(self) -> None:
        """Restore graph objects, constants, interfaces, and signature mappings."""

        document, codec = self._document()
        before = freeze_object(document.model)
        session = CirclePassContext().session(document)
        cached = session.graph(0)

        with self.assertRaisesRegex(RuntimeError, "intentional rewrite failure"):
            CircleRulePass([_FailingRewriteRule(codec)]).run(
                document,
                CirclePassContext(verify_after_each_pass=False),
            )

        self.assertEqual(freeze_object(document.model), before)
        self.assertTrue(document.verify(raise_on_error=False).ok)
        self.assertIsNot(session.graph(0), cached)

    def test_uncommitted_manual_transaction_rolls_back(self) -> None:
        """Treat leaving a mutation scope without commit as an aborted rewrite."""

        document, _codec = self._document()
        before = freeze_object(document.model)
        session = CirclePassContext().session(document)

        with session.transaction(subgraph_index=0) as mutation:
            mutation.watch_operator(0)
            document.subgraph().operators[0].inputs = []

        self.assertEqual(freeze_object(document.model), before)

    def test_committed_transaction_advances_only_selected_subgraph(self) -> None:
        """Invalidate a local graph without discarding unrelated subgraph indexes."""

        document = make_empty_document(subgraph_count=2)
        session = CirclePassContext().session(document)
        first = session.graph(0)
        second = session.graph(1)

        with session.transaction(subgraph_index=0) as mutation:
            mutation.watch_subgraph_field("inputs")
            document.subgraph(0).inputs = []
            mutation.commit()

        self.assertIsNot(session.graph(0), first)
        self.assertIs(session.graph(1), second)


if __name__ == "__main__":
    unittest.main()

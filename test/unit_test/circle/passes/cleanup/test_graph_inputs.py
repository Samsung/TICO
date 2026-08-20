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
from types import SimpleNamespace

from tico.circle.document import CircleDocument
from tico.circle.passes.cleanup import prune_unused_graph_inputs

from test.unit_test.circle.fixture import (
    FakeBuffer,
    FakeModel,
    FakeOperator,
    FakeOperatorCode,
    FakeSignatureDef,
    FakeSubGraph,
    FakeTensor,
    FakeTensorMap,
)


class GraphInputPruningTest(unittest.TestCase):
    def test_removes_only_unbound_unused_inputs(self):
        subgraph = FakeSubGraph(
            tensors=[
                FakeTensor("used", shape=[1]),
                FakeTensor("unused", shape=[1]),
                FakeTensor("output", shape=[1]),
            ],
            inputs=[0, 1],
            outputs=[2],
            operators=[FakeOperator(opcodeIndex=0, inputs=[0], outputs=[2])],
        )
        document = CircleDocument(
            FakeModel(
                subgraphs=[subgraph],
                buffers=[FakeBuffer()],
                operatorCodes=[FakeOperatorCode(builtinCode=10)],
            )
        )

        result = prune_unused_graph_inputs(document)

        self.assertTrue(result.modified)
        self.assertEqual(result.changes, 1)
        self.assertEqual(subgraph.inputs, [0])

    def test_preserves_signature_bound_input(self):
        subgraph = FakeSubGraph(
            tensors=[
                FakeTensor("used", shape=[1]),
                FakeTensor("public_unused", shape=[1]),
                FakeTensor("output", shape=[1]),
            ],
            inputs=[0, 1],
            outputs=[2],
            operators=[FakeOperator(opcodeIndex=0, inputs=[0], outputs=[2])],
        )
        document = CircleDocument(
            FakeModel(
                subgraphs=[subgraph],
                buffers=[FakeBuffer()],
                operatorCodes=[FakeOperatorCode(builtinCode=10)],
                signatureDefs=[
                    FakeSignatureDef(
                        signatureKey="serving_default",
                        subgraphIndex=0,
                        inputs=[FakeTensorMap("public_unused", 1)],
                        outputs=[FakeTensorMap("output", 2)],
                    )
                ],
            )
        )

        result = prune_unused_graph_inputs(document)

        self.assertFalse(result.modified)
        self.assertEqual(subgraph.inputs, [0, 1])

    def test_preserves_complete_referenced_subgraph_interface(self):
        callee = FakeSubGraph(
            name="callee",
            tensors=[
                FakeTensor("used", shape=[1]),
                FakeTensor("caller_bound_unused", shape=[1]),
                FakeTensor("output", shape=[1]),
            ],
            inputs=[0, 1],
            outputs=[2],
            operators=[FakeOperator(opcodeIndex=0, inputs=[0], outputs=[2])],
        )
        caller = FakeSubGraph(
            name="caller",
            tensors=[FakeTensor("output", shape=[1])],
            inputs=[],
            outputs=[0],
            operators=[
                FakeOperator(
                    opcodeIndex=1,
                    inputs=[],
                    outputs=[0],
                    builtinOptions=SimpleNamespace(subgraphIndex=1),
                )
            ],
        )
        document = CircleDocument(
            FakeModel(
                subgraphs=[caller, callee],
                buffers=[FakeBuffer()],
                operatorCodes=[
                    FakeOperatorCode(builtinCode=10),
                    FakeOperatorCode(builtinCode=20),
                ],
            )
        )

        result = prune_unused_graph_inputs(document, (1,))

        self.assertFalse(result.modified)
        self.assertEqual(callee.inputs, [0, 1])


if __name__ == "__main__":
    unittest.main()

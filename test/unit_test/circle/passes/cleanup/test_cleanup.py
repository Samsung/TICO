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

from tico.circle.analysis import OperatorEffectAnalysis
from tico.circle.document import CircleDocument
from tico.circle.passes import CirclePassContext
from tico.circle.passes.cleanup import CompactIndicesPass, DeadCodeEliminationPass

from test.unit_test.circle.fixture import (
    FakeBuffer,
    FakeModel,
    FakeOperator,
    FakeOperatorCode,
    FakeSignatureDef,
    FakeSubGraph,
    FakeTensor,
    FakeTensorMap,
    make_test_document,
)


class CircleCleanupPassTest(unittest.TestCase):
    def test_dead_code_elimination_removes_unreachable_operator(self):
        document = make_test_document()

        result = DeadCodeEliminationPass(subgraph_indices=(0,)).run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(len(document.subgraph(0).operators), 2)
        self.assertEqual(len(document.subgraph(1).operators), 1)

    def test_compact_indices_removes_dead_objects_after_dce(self):
        document = make_test_document()
        context = CirclePassContext(verify_after_each_pass=False)
        DeadCodeEliminationPass(subgraph_indices=(0,)).run(document, context)

        result = CompactIndicesPass().run(document, context)

        self.assertTrue(result.modified)
        self.assertEqual(len(document.subgraph(0).tensors), 4)
        self.assertEqual(len(document.model.buffers), 2)
        self.assertEqual(len(document.model.operatorCodes), 2)
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_dce_preserves_effectful_operator_and_its_predecessor(self):
        """Keep a disconnected stateful root and the values required to execute it."""

        document = CircleDocument(
            FakeModel(
                subgraphs=[
                    FakeSubGraph(
                        tensors=[
                            FakeTensor("input", shape=[1]),
                            FakeTensor("middle", shape=[1]),
                            FakeTensor("state_result", shape=[1]),
                        ],
                        inputs=[0],
                        outputs=[],
                        operators=[
                            FakeOperator(opcodeIndex=0, inputs=[0], outputs=[1]),
                            FakeOperator(opcodeIndex=1, inputs=[1], outputs=[2]),
                        ],
                    )
                ],
                buffers=[FakeBuffer()],
                operatorCodes=[
                    FakeOperatorCode(builtinCode=10),
                    FakeOperatorCode(builtinCode=20),
                ],
            )
        )
        effects = OperatorEffectAnalysis(effectful_builtin_codes=(20,))

        result = DeadCodeEliminationPass(effect_analysis=effects).run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph(0).operators), 2)
        self.assertEqual(document.subgraph(0).inputs, [0])

    def test_dce_removes_dead_pure_graph_without_outputs(self):
        """A graph without outputs may still discard a disconnected pure chain."""

        document = CircleDocument(
            FakeModel(
                subgraphs=[
                    FakeSubGraph(
                        tensors=[
                            FakeTensor("input", shape=[1]),
                            FakeTensor("dead", shape=[1]),
                        ],
                        inputs=[0],
                        outputs=[],
                        operators=[
                            FakeOperator(opcodeIndex=0, inputs=[0], outputs=[1]),
                        ],
                    )
                ],
                buffers=[FakeBuffer()],
                operatorCodes=[FakeOperatorCode(builtinCode=10)],
            )
        )
        effects = OperatorEffectAnalysis()

        result = DeadCodeEliminationPass(effect_analysis=effects).run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(document.subgraph(0).operators, [])
        self.assertEqual(document.subgraph(0).inputs, [])

    def test_dce_preserves_unknown_custom_operator(self):
        """Treat custom implementations as effectful unless explicitly classified."""

        document = CircleDocument(
            FakeModel(
                subgraphs=[
                    FakeSubGraph(
                        tensors=[
                            FakeTensor("input", shape=[1]),
                            FakeTensor("custom_result", shape=[1]),
                        ],
                        inputs=[0],
                        outputs=[],
                        operators=[
                            FakeOperator(opcodeIndex=0, inputs=[0], outputs=[1]),
                        ],
                    )
                ],
                buffers=[FakeBuffer()],
                operatorCodes=[
                    FakeOperatorCode(builtinCode=99, customCode="StatefulCustom"),
                ],
            )
        )
        effects = OperatorEffectAnalysis(custom_builtin_code=99)

        result = DeadCodeEliminationPass(effect_analysis=effects).run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph(0).operators), 1)
        self.assertEqual(document.subgraph(0).inputs, [0])

    def test_dce_preserves_signature_bound_unused_input(self):
        """Do not invalidate a signature when optimization removes data dependence."""

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
                        inputs=[
                            FakeTensorMap("used", 0),
                            FakeTensorMap("public_unused", 1),
                        ],
                        outputs=[FakeTensorMap("output", 2)],
                    )
                ],
            )
        )

        result = DeadCodeEliminationPass(effect_analysis=OperatorEffectAnalysis()).run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(document.subgraph(0).inputs, [0, 1])
        self.assertTrue(document.verify(raise_on_error=False).ok)


if __name__ == "__main__":
    unittest.main()

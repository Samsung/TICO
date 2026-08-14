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

from tico.circle.errors import CircleRewriteError
from tico.circle.rewrite import (
    replace_operator_output_uses,
    replace_tensor_uses,
    replace_tensor_uses_many,
    TensorUseReplacement,
)

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    FakeOperator,
    FakeOperatorCode,
    FakeSignatureDef,
    FakeTensorMap,
    make_empty_document,
)


class CircleRewriteHelpersTest(unittest.TestCase):
    """Check simultaneous and multi-output tensor-use replacement helpers."""

    def test_legacy_single_replacement_delegates_without_behavior_change(self):
        """Preserve the established consumer, output, and signature behavior."""

        document = make_empty_document()
        for name in ("old", "new", "sink"):
            add_runtime_tensor(
                document,
                subgraph_index=0,
                name=name,
                shape=[1],
            )
        document.model.operatorCodes = [FakeOperatorCode(builtinCode=0)]
        document.subgraph().operators = [
            FakeOperator(opcodeIndex=0, inputs=[0], outputs=[2])
        ]
        document.subgraph().outputs = [0]
        document.model.signatureDefs = [
            FakeSignatureDef(outputs=[FakeTensorMap(name="old", tensorIndex=0)])
        ]

        stats = replace_tensor_uses(
            document.model,
            subgraph_index=0,
            old_tensor_index=0,
            new_tensor_index=1,
        )

        self.assertEqual(document.subgraph().operators[0].inputs, [1])
        self.assertEqual(document.subgraph().outputs, [1])
        self.assertEqual(document.model.signatureDefs[0].outputs[0].tensorIndex, 1)
        self.assertEqual(stats.remapped_references, 3)

    def test_self_mapping_does_not_conflict_with_real_replacement(self):
        """Ignore no-op entries before detecting duplicate mapping conflicts."""

        document = make_empty_document()
        for name in ("zero", "one"):
            add_runtime_tensor(
                document,
                subgraph_index=0,
                name=name,
                shape=[1],
            )
        document.model.operatorCodes = [FakeOperatorCode(builtinCode=0)]
        document.subgraph().operators = [
            FakeOperator(opcodeIndex=0, inputs=[0], outputs=[1])
        ]

        replace_tensor_uses_many(
            document.model,
            subgraph_index=0,
            replacements=((0, 0), (0, 1)),
        )
        self.assertEqual(document.subgraph().operators[0].inputs, [1])

    def test_many_replacements_do_not_cascade_through_chains(self):
        """Resolve every old reference exactly once against the original vector."""

        document = make_empty_document()
        for name in ("zero", "one", "two"):
            add_runtime_tensor(
                document,
                subgraph_index=0,
                name=name,
                shape=[1],
            )
        document.model.operatorCodes = [FakeOperatorCode(builtinCode=0)]
        document.subgraph().operators = [
            FakeOperator(opcodeIndex=0, inputs=[0, 1], outputs=[2])
        ]
        document.subgraph().outputs = [0, 1]

        stats = replace_tensor_uses_many(
            document.model,
            subgraph_index=0,
            replacements=((0, 1), (1, 2)),
        )

        self.assertEqual(document.subgraph().operators[0].inputs, [1, 2])
        self.assertEqual(document.subgraph().outputs, [1, 2])
        self.assertEqual(stats.remapped_references, 4)

    def test_many_replacements_support_cycles(self):
        """Swap two tensor references without applying one mapping twice."""

        document = make_empty_document()
        for name in ("zero", "one", "output"):
            add_runtime_tensor(
                document,
                subgraph_index=0,
                name=name,
                shape=[1],
            )
        document.model.operatorCodes = [FakeOperatorCode(builtinCode=0)]
        document.subgraph().operators = [
            FakeOperator(opcodeIndex=0, inputs=[0, 1], outputs=[2])
        ]

        replace_tensor_uses_many(
            document.model,
            subgraph_index=0,
            replacements=(
                TensorUseReplacement(0, 1),
                TensorUseReplacement(1, 0),
            ),
        )
        self.assertEqual(document.subgraph().operators[0].inputs, [1, 0])

    def test_multi_output_helper_updates_consumers_outputs_and_signatures(self):
        """Replace all outputs of one producer while retaining producer identity."""

        document = make_empty_document()
        for name in ("input", "old_a", "old_b", "new_a", "new_b", "sink"):
            add_runtime_tensor(
                document,
                subgraph_index=0,
                name=name,
                shape=[1],
            )
        document.model.operatorCodes = [FakeOperatorCode(builtinCode=0)]
        document.subgraph().operators = [
            FakeOperator(opcodeIndex=0, inputs=[0], outputs=[1, 2]),
            FakeOperator(opcodeIndex=0, inputs=[1, 2], outputs=[5]),
        ]
        document.subgraph().outputs = [1, 2]
        document.model.signatureDefs = [
            FakeSignatureDef(
                outputs=[
                    FakeTensorMap(name="a", tensorIndex=1),
                    FakeTensorMap(name="b", tensorIndex=2),
                ]
            )
        ]

        stats = replace_operator_output_uses(
            document.model,
            subgraph_index=0,
            operator_index=0,
            new_tensor_indices=(3, 4),
        )

        self.assertEqual(document.subgraph().operators[0].outputs, [1, 2])
        self.assertEqual(document.subgraph().operators[1].inputs, [3, 4])
        self.assertEqual(document.subgraph().outputs, [3, 4])
        self.assertEqual(
            [item.tensorIndex for item in document.model.signatureDefs[0].outputs],
            [3, 4],
        )
        self.assertEqual(stats.remapped_references, 6)

    def test_conflicting_replacements_are_rejected(self):
        """Reject ambiguous mappings for one original tensor index."""

        document = make_empty_document()
        for name in ("zero", "one", "two"):
            add_runtime_tensor(
                document,
                subgraph_index=0,
                name=name,
                shape=[1],
            )

        with self.assertRaisesRegex(CircleRewriteError, "conflicting"):
            replace_tensor_uses_many(
                document.model,
                subgraph_index=0,
                replacements=((0, 1), (0, 2)),
            )

    def test_multi_output_helper_requires_one_replacement_per_output(self):
        """Reject positional replacement lists with a different output arity."""

        document = make_empty_document()
        for name in ("input", "left", "right"):
            add_runtime_tensor(
                document,
                subgraph_index=0,
                name=name,
                shape=[1],
            )
        document.model.operatorCodes = [FakeOperatorCode(builtinCode=0)]
        document.subgraph().operators = [
            FakeOperator(opcodeIndex=0, inputs=[0], outputs=[1, 2])
        ]

        with self.assertRaisesRegex(CircleRewriteError, "2 outputs"):
            replace_operator_output_uses(
                document.model,
                subgraph_index=0,
                operator_index=0,
                new_tensor_indices=(0,),
            )


if __name__ == "__main__":
    unittest.main()

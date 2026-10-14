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
from typing import Any

import numpy as np

from tico.circle.analysis import OperatorEffectAnalysis, OperatorPurityAnalysis
from tico.circle.graph import CircleGraph

from test.unit_test.circle.fixture import (
    FakeBuffer,
    FakeModel,
    FakeOperator,
    FakeOperatorCode,
    FakeSubGraph,
    FakeTensor,
)


class SubgraphReferenceAnalysisTest(unittest.TestCase):
    """Keep CSE and DCE conservative for the same option-reference forms."""

    def setUp(self) -> None:
        self.operator = FakeOperator(opcodeIndex=0, inputs=[0], outputs=[1])
        self.graph = CircleGraph(
            FakeModel(
                subgraphs=[
                    FakeSubGraph(
                        tensors=[FakeTensor("input"), FakeTensor("output")],
                        inputs=[0],
                        outputs=[1],
                        operators=[self.operator],
                    )
                ],
                buffers=[FakeBuffer()],
                operatorCodes=[FakeOperatorCode(builtinCode=0)],
            )
        )
        self.purity = OperatorPurityAnalysis()
        self.effects = OperatorEffectAnalysis()

    def _assert_classification(self, *, pure: bool, effectful: bool) -> None:
        self.assertEqual(self.purity.is_pure(self.graph, 0, builtin_code=0), pure)
        self.assertEqual(self.effects.has_observable_effect(self.graph, 0), effectful)

    def test_absent_options_are_not_references(self) -> None:
        self._assert_classification(pure=True, effectful=False)

    def test_scalar_references_include_zero_negative_and_numpy_integers(self) -> None:
        """Classify references independently of whether their targets are valid."""

        for value in (0, -1, np.int64(1)):
            with self.subTest(value=value):
                self.operator.builtinOptions = SimpleNamespace(thenSubgraphIndex=value)
                self._assert_classification(pure=False, effectful=True)

    def test_reference_names_allow_call_options_and_underscores(self) -> None:
        for field in ("subgraph", "then_subgraph_index", "THEN_SUBGRAPH_INDEX"):
            with self.subTest(field=field):
                self.operator.builtinOptions = SimpleNamespace(**{field: 0})
                self._assert_classification(pure=False, effectful=True)

    def test_secondary_options_are_checked_after_empty_primary_options(self) -> None:
        self.operator.builtinOptions = SimpleNamespace(branchSubgraphIndices=[])
        self.operator.builtinOptions2 = SimpleNamespace(bodySubgraphIndex=0)

        self._assert_classification(pure=False, effectful=True)

    def test_nonempty_reference_vectors_are_effectful(self) -> None:
        for field in ("branchSubgraphIndices", "branch_subgraph_indices"):
            for value in ([0], np.array([0, 1], dtype=np.int32)):
                with self.subTest(field=field, value=value):
                    self.operator.builtinOptions = SimpleNamespace(**{field: value})
                    self._assert_classification(pure=False, effectful=True)

    def test_empty_reference_vectors_are_not_references(self) -> None:
        empty_values: tuple[Any, ...] = (None, [], np.array([], dtype=np.int32))
        for value in empty_values:
            with self.subTest(value=value):
                self.operator.builtinOptions = SimpleNamespace(
                    branchSubgraphIndices=value
                )
                self._assert_classification(pure=True, effectful=False)

    def test_empty_vector_does_not_hide_a_later_reference(self) -> None:
        self.operator.builtinOptions = SimpleNamespace(
            aSubgraphIndices=[], zSubgraphIndex=0
        )

        self._assert_classification(pure=False, effectful=True)

    def test_ignored_fields_do_not_create_references(self) -> None:
        cases = (
            ("thenSubgraphIndex", lambda: 0),
            ("_thenSubgraphIndex", 0),
            ("axis", 0),
            ("thenSubgraphIndex", 0.0),
            ("thenSubgraphIndex", "0"),
            ("thenSubgraphIndex", None),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                self.operator.builtinOptions = SimpleNamespace(**{field: value})
                self._assert_classification(pure=True, effectful=False)

    def test_unreadable_property_does_not_hide_other_references(self) -> None:
        class UnreadableOptions:
            @property
            def aSubgraphIndex(self):
                raise RuntimeError("unreadable option")

        self.operator.builtinOptions = UnreadableOptions()
        self._assert_classification(pure=True, effectful=False)

        setattr(self.operator.builtinOptions, "zSubgraphIndex", 0)
        self._assert_classification(pure=False, effectful=True)

    def test_noniterable_reference_vector_is_conservatively_effectful(self) -> None:
        for value in (0, np.array(0, dtype=np.int32)):
            with self.subTest(value=value):
                self.operator.builtinOptions = SimpleNamespace(
                    branchSubgraphIndices=value
                )
                self._assert_classification(pure=False, effectful=True)

    def test_invalid_vector_entry_value_error_propagates(self) -> None:
        self.operator.builtinOptions = SimpleNamespace(
            branchSubgraphIndices=["invalid"]
        )

        with self.assertRaisesRegex(ValueError, "invalid literal"):
            self.purity.is_pure(self.graph, 0, builtin_code=0)
        with self.assertRaisesRegex(ValueError, "invalid literal"):
            self.effects.has_observable_effect(self.graph, 0)


if __name__ == "__main__":
    unittest.main()

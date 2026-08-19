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

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    make_empty_document,
)
from test.unit_test.circle.passes.optimization._fixture import (
    ADD,
    make_builder,
    make_codec,
    static_contract,
)
from test.unit_test.circle.passes.optimization._operator_rewrite_fixture import CUSTOM

from tico.circle.analysis import OperatorPurityAnalysis


@dataclass
class BranchOptions:
    """Provide one control-flow subgraph reference for purity tests."""

    thenSubgraphIndex: int = 1


class OperatorPurityAnalysisTest(unittest.TestCase):
    """Check conservative operator-purity classification."""

    def setUp(self) -> None:
        """Create one schema-independent builder and source tensors."""

        self.document = make_empty_document(subgraph_count=2)
        self.builder = make_builder(self.document, make_codec())
        self.left = add_runtime_tensor(
            self.document,
            subgraph_index=0,
            name="left",
            shape=[2],
        )
        self.right = add_runtime_tensor(
            self.document,
            subgraph_index=0,
            name="right",
            shape=[2],
        )
        self.document.subgraph().inputs = [self.left, self.right]

    def _add(self, **kwargs) -> int:
        """Append one fixture operator and return its operator index."""

        self.builder.add_operator(
            ADD,
            inputs=(self.left, self.right),
            output_contracts=(static_contract((2,)),),
            output_names=("output",),
            **kwargs,
        )
        return len(self.document.subgraph().operators) - 1

    def test_accepts_plain_nonvariable_operator(self) -> None:
        """Classify a plain runtime operator as pure."""

        operator_index = self._add()
        analysis = OperatorPurityAnalysis(custom_builtin_code=CUSTOM)

        self.assertTrue(
            analysis.is_pure(
                self.document.graph(),
                operator_index,
                builtin_code=ADD,
            )
        )

    def test_rejects_known_impure_builtin(self) -> None:
        """Reject a builtin code explicitly classified as impure."""

        operator_index = self._add()
        analysis = OperatorPurityAnalysis(impure_builtin_codes=(ADD,))

        self.assertFalse(
            analysis.is_pure(
                self.document.graph(),
                operator_index,
                builtin_code=ADD,
            )
        )

    def test_rejects_mutating_input(self) -> None:
        """Reject an operator with an explicit mutating input marker."""

        operator_index = self._add(mutating_variable_inputs=(True, False))
        analysis = OperatorPurityAnalysis()

        self.assertFalse(
            analysis.is_pure(
                self.document.graph(),
                operator_index,
                builtin_code=ADD,
            )
        )

    def test_rejects_variable_tensor(self) -> None:
        """Reject an operator that reads a variable tensor."""

        self.document.subgraph().tensors[self.left].isVariable = True
        operator_index = self._add()
        analysis = OperatorPurityAnalysis()

        self.assertFalse(
            analysis.is_pure(
                self.document.graph(),
                operator_index,
                builtin_code=ADD,
            )
        )

    def test_rejects_custom_operator_by_default(self) -> None:
        """Reject custom operators unless the caller opts in explicitly."""

        operator_index = self._add()
        analysis = OperatorPurityAnalysis(custom_builtin_code=CUSTOM)

        self.assertFalse(
            analysis.is_pure(
                self.document.graph(),
                operator_index,
                builtin_code=CUSTOM,
            )
        )

    def test_rejects_subgraph_reference(self) -> None:
        """Reject an operator whose options refer to another subgraph."""

        operator_index = self._add(builtin_options=BranchOptions())
        analysis = OperatorPurityAnalysis()

        self.assertFalse(
            analysis.is_pure(
                self.document.graph(),
                operator_index,
                builtin_code=ADD,
            )
        )


if __name__ == "__main__":
    unittest.main()

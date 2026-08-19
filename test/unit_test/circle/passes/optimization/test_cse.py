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

from tico.circle.analysis import TensorContract
from tico.circle.passes.base import CirclePassContext
from tico.circle.passes.optimization import CommonSubexpressionEliminationPass
from tico.circle.value import TensorQuantization

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    INT8,
    make_empty_document,
)
from test.unit_test.circle.passes.optimization._fixture import (
    ABS,
    ADD,
    make_builder,
    make_codec,
    MUL,
    static_contract,
)
from test.unit_test.circle.passes.optimization._operator_rewrite_fixture import (
    BinaryOptions,
    BUILTIN_CODES,
)


class CommonSubexpressionEliminationPassTest(unittest.TestCase):
    """Check structural CSE, contracts, multi-output mapping, and idempotence."""

    def setUp(self) -> None:
        """Create one schema-independent document, builder, and pass context."""

        self.document = make_empty_document()
        self.builder = make_builder(self.document, make_codec())
        self.context = CirclePassContext(verify_after_each_pass=False)
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

    def _pass(self) -> CommonSubexpressionEliminationPass:
        """Create CSE with schema-independent builtin-code overrides."""

        return CommonSubexpressionEliminationPass(builtin_codes=BUILTIN_CODES)

    def _binary(
        self,
        code: int,
        left: int,
        right: int,
        name: str,
        *,
        contract: TensorContract | None = None,
        activation: int = 0,
    ) -> int:
        """Append one binary fixture and return its output tensor index."""

        return self.builder.add_operator(
            code,
            inputs=(left, right),
            output_contracts=(contract or static_contract((2,)),),
            output_names=(name,),
            builtin_options=BinaryOptions(
                fusedActivationFunction=activation,
            ),
        )[0]

    def test_reuses_single_output_expression(self) -> None:
        """Replace a duplicate output use with the first expression output."""

        canonical = self._binary(ADD, self.left, self.right, "canonical")
        duplicate = self._binary(ADD, self.left, self.right, "duplicate")
        sink = self._binary(MUL, duplicate, self.right, "sink")
        self.document.subgraph().outputs = [sink]

        result = self._pass().run(self.document, self.context)

        self.assertTrue(result.modified)
        self.assertEqual(result.changes, 1)
        self.assertEqual(self.document.subgraph().operators[-1].inputs[0], canonical)

    def test_reuses_chained_duplicates_in_one_run(self) -> None:
        """Canonicalize duplicate inputs while scanning a duplicate expression chain."""

        first_add = self._binary(ADD, self.left, self.right, "first_add")
        second_add = self._binary(ADD, self.left, self.right, "second_add")
        first_mul = self._binary(MUL, first_add, self.right, "first_mul")
        second_mul = self._binary(MUL, second_add, self.right, "second_mul")
        sink = self._binary(ADD, second_mul, self.right, "sink")
        self.document.subgraph().outputs = [sink]

        result = self._pass().run(self.document, self.context)

        self.assertEqual(result.changes, 2)
        self.assertEqual(self.document.subgraph().operators[3].inputs[0], first_add)
        self.assertEqual(self.document.subgraph().operators[-1].inputs[0], first_mul)

        second_result = self._pass().run(self.document, self.context)
        self.assertFalse(second_result.modified)

    def test_replaces_multi_output_uses_pairwise(self) -> None:
        """Map every duplicate output to the corresponding canonical output."""

        canonical = self.builder.add_operator(
            ABS,
            inputs=(self.left,),
            output_contracts=(static_contract((2,)), static_contract((2,))),
            output_names=("canonical_0", "canonical_1"),
        )
        duplicate = self.builder.add_operator(
            ABS,
            inputs=(self.left,),
            output_contracts=(static_contract((2,)), static_contract((2,))),
            output_names=("duplicate_0", "duplicate_1"),
        )
        first_sink = self._binary(ADD, duplicate[0], self.right, "first_sink")
        second_sink = self._binary(ADD, duplicate[1], self.right, "second_sink")
        self.document.subgraph().outputs = [first_sink, second_sink]

        result = self._pass().run(self.document, self.context)

        self.assertEqual(result.changes, 1)
        operators = self.document.subgraph().operators
        self.assertEqual(operators[-2].inputs[0], canonical[0])
        self.assertEqual(operators[-1].inputs[0], canonical[1])

    def test_keeps_different_builtin_options(self) -> None:
        """Do not merge operators whose serialized builtin options differ."""

        first = self._binary(
            ADD,
            self.left,
            self.right,
            "first",
            activation=0,
        )
        second = self._binary(
            ADD,
            self.left,
            self.right,
            "second",
            activation=77,
        )
        first_sink = self._binary(MUL, first, self.right, "first_sink")
        second_sink = self._binary(MUL, second, self.right, "second_sink")
        self.document.subgraph().outputs = [first_sink, second_sink]

        result = self._pass().run(self.document, self.context)

        self.assertFalse(result.modified)

    def test_keeps_different_output_quantization(self) -> None:
        """Do not merge outputs whose complete tensor contracts differ."""

        first_contract = TensorContract(
            tensor_type=INT8,
            shape=(2,),
            shape_signature=(2,),
            quantization=TensorQuantization(scale=(0.5,), zero_point=(0,)),
        )
        second_contract = TensorContract(
            tensor_type=INT8,
            shape=(2,),
            shape_signature=(2,),
            quantization=TensorQuantization(scale=(0.25,), zero_point=(0,)),
        )
        first = self._binary(
            ADD,
            self.left,
            self.right,
            "first",
            contract=first_contract,
        )
        second = self._binary(
            ADD,
            self.left,
            self.right,
            "second",
            contract=second_contract,
        )
        first_sink = self._binary(MUL, first, self.right, "first_sink")
        second_sink = self._binary(MUL, second, self.right, "second_sink")
        self.document.subgraph().outputs = [first_sink, second_sink]

        result = self._pass().run(self.document, self.context)

        self.assertFalse(result.modified)

    def test_rejects_operator_code_with_custom_identifier(self) -> None:
        """Keep custom-coded expressions even when their builtin code looks pure."""

        first = self.builder.add_operator(
            ADD,
            inputs=(self.left, self.right),
            output_contracts=(static_contract((2,)),),
            output_names=("first",),
            custom_code="vendor.operation",
        )[0]
        second = self.builder.add_operator(
            ADD,
            inputs=(self.left, self.right),
            output_contracts=(static_contract((2,)),),
            output_names=("second",),
            custom_code="vendor.operation",
        )[0]
        first_sink = self._binary(MUL, first, self.right, "first_sink")
        second_sink = self._binary(MUL, second, self.right, "second_sink")
        self.document.subgraph().outputs = [first_sink, second_sink]

        result = self._pass().run(self.document, self.context)

        self.assertFalse(result.modified)

    def test_preserves_duplicate_graph_output_identity(self) -> None:
        """Keep a duplicate producer when its tensor is a graph output."""

        self._binary(ADD, self.left, self.right, "canonical")
        duplicate = self._binary(ADD, self.left, self.right, "public_output")
        self.document.subgraph().outputs = [duplicate]

        result = self._pass().run(self.document, self.context)

        self.assertFalse(result.modified)
        self.assertEqual(self.document.subgraph().outputs, [duplicate])
        self.assertEqual(
            self.document.subgraph().tensors[duplicate].name,
            "public_output",
        )


if __name__ == "__main__":
    unittest.main()

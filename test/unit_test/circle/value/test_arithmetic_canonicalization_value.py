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

import importlib.util
import unittest
from typing import Any

import numpy as np

_HAS_GENERATED_SCHEMA = (
    importlib.util.find_spec("circle_schema") is not None
    and importlib.util.find_spec("flatbuffers") is not None
)

if _HAS_GENERATED_SCHEMA:
    from circle_schema import circle

    from tico.circle.passes import (
        CirclePassContext,
        CirclePassManager,
        SimplifyArithmeticPass,
    )
    from tico.circle.passes.cleanup import DeadCodeEliminationPass

    from test.support.circle.builder import CircleModelBuilder
    from test.support.circle.evaluator import CircleReferenceEvaluator
    from test.support.circle.value_test import CircleValueTestCase
else:
    circle = None
    SimplifyArithmeticPass = None
    CirclePassContext = None
    CirclePassManager = None
    DeadCodeEliminationPass = None
    CircleModelBuilder = object
    CircleReferenceEvaluator = object
    CircleValueTestCase = unittest.TestCase


class _ArithmeticCircleModelBuilder(CircleModelBuilder):
    """Extend value-test fixtures with a FLOAT32 DIV builder."""

    def div(self, lhs: int, rhs: int, *, name: str) -> int:
        """Add an activation-free DIV operator and return its output tensor."""

        options = circle.DivOptions.DivOptionsT()
        options.fusedActivationFunction = self._activation_none()
        return self._binary_operator(
            self._builtin_operator("DIV"),
            lhs,
            rhs,
            name=name,
            options_type=self._builtin_options("DivOptions"),
            options=options,
        )


class _ArithmeticReferenceEvaluator(CircleReferenceEvaluator):
    """Add the DIV kernel required by arithmetic canonicalization tests."""

    def __init__(self) -> None:
        """Register one conservative activation-free DIV handler."""

        super().__init__()
        code = int(circle.BuiltinOperator.BuiltinOperator.DIV)
        self._handlers[code] = self._evaluate_div

    def _evaluate_div(
        self,
        operator: Any,
        inputs: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, ...]:
        """Evaluate a two-input DIV operator."""

        self._require_no_fused_activation(operator)
        self._require_input_count("DIV", inputs, 2)
        return (np.divide(inputs[0], inputs[1]),)


@unittest.skipUnless(
    _HAS_GENERATED_SCHEMA,
    "circle-schema and flatbuffers are required for Circle value tests",
)
class ArithmeticCanonicalizationValueTest(CircleValueTestCase):
    """Check generated Circle values after arithmetic canonicalization."""

    def setUp(self) -> None:
        """Use the value evaluator extended with DIV support."""

        super().setUp()
        self.evaluator = _ArithmeticReferenceEvaluator()

    def test_mul_div_chain_is_canonicalized_without_changing_values(self) -> None:
        """Rewrite `(x * 2) / 4` to `x * 0.5` and preserve output values."""

        builder = _ArithmeticCircleModelBuilder(
            description="arithmetic-canonicalization-value-test"
        )
        source_tensor = builder.input("input", [2])
        two = builder.const_f32("two", np.float32(2.0))
        product = builder.mul(source_tensor, two, name="product")
        four = builder.const_f32("four", np.float32(4.0))
        output = builder.div(product, four, name="output")
        builder.set_outputs(output)
        source = builder.build()
        input_value = np.array([4.0, 10.0], dtype=np.float32)
        expected = np.array([2.0, 5.0], dtype=np.float32)

        result = self.assert_pass_preserves_value(
            source,
            (input_value,),
            lambda document: CirclePassManager(
                [SimplifyArithmeticPass(), DeadCodeEliminationPass()]
            ).run(
                document,
                CirclePassContext(verify_after_each_pass=True),
            ),
            expected_outputs=(expected,),
        )

        self.assertEqual(len(result.document.subgraph().operators), 1)
        operator = result.document.subgraph().operators[0]
        operator_code = result.document.model.operatorCodes[operator.opcodeIndex]
        self.assertEqual(
            int(operator_code.builtinCode),
            int(circle.BuiltinOperator.BuiltinOperator.MUL),
        )


if __name__ == "__main__":
    unittest.main()

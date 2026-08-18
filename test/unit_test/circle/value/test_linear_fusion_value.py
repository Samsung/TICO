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

from collections.abc import Sequence
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
        FuseLinearOpsPass,
    )
    from tico.circle.passes.cleanup import DeadCodeEliminationPass

    from test.support.circle.builder import CircleModelBuilder
    from test.support.circle.evaluator import CircleReferenceEvaluator
    from test.support.circle.value_test import CircleValueTestCase
else:
    circle = None
    CirclePassContext = None
    CirclePassManager = None
    FuseLinearOpsPass = None
    DeadCodeEliminationPass = None
    CircleModelBuilder = object
    CircleReferenceEvaluator = object
    CircleValueTestCase = unittest.TestCase


class _LinearCircleModelBuilder(CircleModelBuilder):
    """Extend value-test fixtures with a minimal FLOAT32 FullyConnected builder."""

    def fully_connected(
        self,
        source: int,
        weight: int,
        bias: int,
        *,
        name: str,
    ) -> int:
        """Add one activation-free three-input FULLY_CONNECTED operator."""

        source_tensor = self._tensor(source)
        weight_tensor = self._tensor(weight)
        bias_tensor = self._tensor(bias)
        source_shape = tuple(int(value) for value in source_tensor.shape)
        weight_shape = tuple(int(value) for value in weight_tensor.shape)
        bias_shape = tuple(int(value) for value in bias_tensor.shape)
        if (
            len(source_shape) != 2
            or len(weight_shape) != 2
            or source_shape[-1] != weight_shape[-1]
            or bias_shape != (weight_shape[0],)
        ):
            raise ValueError("Invalid FullyConnected fixture tensor shapes.")

        output = self._add_tensor(
            name,
            (source_shape[0], weight_shape[0]),
            dtype=np.float32,
            buffer_index=0,
        )
        options = circle.FullyConnectedOptions.FullyConnectedOptionsT()
        options.fusedActivationFunction = self._activation_none()
        options.weightsFormat = 0
        options.keepNumDims = False
        options.asymmetricQuantizeInputs = False
        self._append_operator(
            self._builtin_operator("FULLY_CONNECTED"),
            inputs=[source, weight, bias],
            outputs=[output],
            options_type=self._builtin_options("FullyConnectedOptions"),
            options=options,
        )
        return output


class _LinearReferenceEvaluator(CircleReferenceEvaluator):
    """Add the FullyConnected kernel required by linear-fusion value tests."""

    def __init__(self) -> None:
        """Register one conservative FLOAT32 FullyConnected handler."""

        super().__init__()
        code = int(circle.BuiltinOperator.BuiltinOperator.FULLY_CONNECTED)
        self._handlers[code] = self._evaluate_fully_connected

    def _evaluate_fully_connected(
        self,
        operator: Any,
        inputs: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, ...]:
        """Evaluate a three-input activation-free FullyConnected operator."""

        self._require_no_fused_activation(operator)
        if len(inputs) != 3:
            raise ValueError(
                "FULLY_CONNECTED value tests require data, weight, and bias."
            )
        source, weight, bias = inputs
        if source.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
            raise ValueError("Unsupported FullyConnected value-test ranks.")
        return (np.matmul(source, weight.T) + bias,)


@unittest.skipUnless(
    _HAS_GENERATED_SCHEMA,
    "circle-schema and flatbuffers are required for Circle value tests",
)
class CircleLinearFusionValueTest(CircleValueTestCase):
    """Check generated Circle round trips and values after linear fusion."""

    def setUp(self) -> None:
        """Use the value evaluator extended with FullyConnected support."""

        super().setUp()
        self.evaluator = _LinearReferenceEvaluator()

    def test_post_fc_add_fuses_without_changing_values(self) -> None:
        """Fold a channel ADD into FC bias and remove old operators with DCE."""

        builder = _LinearCircleModelBuilder(description="post-fc-add-fusion-value-test")
        source_tensor = builder.input("input", [1, 2])
        weight = builder.const_f32(
            "weight",
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )
        bias = builder.const_f32(
            "bias",
            np.array([0.5, -0.5], dtype=np.float32),
        )
        fc = builder.fully_connected(
            source_tensor,
            weight,
            bias,
            name="fc_output",
        )
        offset = builder.const_f32(
            "offset",
            np.array([1.0, 2.0], dtype=np.float32),
        )
        output = builder.add(fc, offset, name="output")
        builder.set_outputs(output)
        source = builder.build()
        input_value = np.array([[1.0, 2.0]], dtype=np.float32)
        expected = np.array([[6.5, 12.5]], dtype=np.float32)

        result = self.assert_pass_preserves_value(
            source,
            (input_value,),
            lambda document: CirclePassManager(
                [FuseLinearOpsPass(), DeadCodeEliminationPass()]
            ).run(
                document,
                CirclePassContext(verify_after_each_pass=True),
            ),
            expected_outputs=(expected,),
        )

        operators: Sequence[Any] = result.document.subgraph().operators
        self.assertEqual(len(operators), 1)
        operator_code = result.document.model.operatorCodes[operators[0].opcodeIndex]
        self.assertEqual(
            int(operator_code.builtinCode),
            int(circle.BuiltinOperator.BuiltinOperator.FULLY_CONNECTED),
        )


if __name__ == "__main__":
    unittest.main()

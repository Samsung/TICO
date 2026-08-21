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

import numpy as np

from tico.circle.graph import OPTIONAL_TENSOR_INDEX
from tico.circle.passes import CirclePassContext, LegalizeDynamicFullyConnectedPass
from tico.circle.passes.optimization._utils import operator_builtin_code

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    FLOAT32,
    make_empty_document,
)
from test.unit_test.circle.passes.optimization._compatibility_fixture import (
    ACTIVATION_RELU,
    ACTIVATION_VALUES,
    ADD,
    add_constant,
    BATCH_MATMUL,
    BUILTIN_CODES,
    BUILTIN_OPTIONS_TYPES,
    compatibility_object_factory,
    FULLY_CONNECTED,
    FULLY_CONNECTED_OPTIONS,
    FullyConnectedOptions,
    make_builder,
    make_codec,
    RESHAPE,
    static_contract,
    TENSOR_TYPES,
)


class DynamicFullyConnectedLegalizationTest(unittest.TestCase):
    """Check conservative lowering of dynamic-weight FLOAT32 FC operators."""

    def setUp(self) -> None:
        """Create schema-independent value and Object API services."""

        self.codec = make_codec()

    def _pass(self) -> LegalizeDynamicFullyConnectedPass:
        """Create the legalization with explicit enum values."""

        return LegalizeDynamicFullyConnectedPass(
            builtin_codes=BUILTIN_CODES,
            builtin_options_types=BUILTIN_OPTIONS_TYPES,
            tensor_types=TENSOR_TYPES,
            activation_values=ACTIVATION_VALUES,
            activation_none=0,
            codec=self.codec,
            object_factory=compatibility_object_factory,
        )

    def test_dynamic_weight_fc_lowers_to_bmm_reshape_and_bias_add(self) -> None:
        """Preserve the rank-2 FC output contract after batched multiplication."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 2, 3],
        )
        weights = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="weights",
            shape=[4, 3],
        )
        bias = add_constant(builder, "bias", [1.0, 2.0, 3.0, 4.0], FLOAT32, np.float32)
        output = builder.add_operator(
            FULLY_CONNECTED,
            inputs=(source, weights, bias),
            output_contracts=(static_contract((4, 4)),),
            output_names=("fc",),
            builtin_options_type=FULLY_CONNECTED_OPTIONS,
            builtin_options=FullyConnectedOptions(
                fusedActivationFunction=ACTIVATION_RELU,
                keepNumDims=False,
            ),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        operators = document.subgraph().operators
        self.assertEqual(
            [operator_builtin_code(document.model, operator) for operator in operators],
            [BATCH_MATMUL, RESHAPE, ADD],
        )
        self.assertFalse(operators[0].builtinOptions.adjointLhs)
        self.assertTrue(operators[0].builtinOptions.adjointRhs)
        self.assertEqual(operators[-1].outputs, [output])
        self.assertEqual(
            operators[-1].builtinOptions.fusedActivationFunction,
            ACTIVATION_RELU,
        )

    def test_constant_weight_fc_is_not_a_dynamic_legalization_candidate(self) -> None:
        """Leave ordinary constant-weight FC operators to optimization and folding."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1, 3],
        )
        weights = add_constant(
            builder,
            "weights",
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            FLOAT32,
            np.float32,
        )
        output = builder.add_operator(
            FULLY_CONNECTED,
            inputs=(source, weights, OPTIONAL_TENSOR_INDEX),
            output_contracts=(static_contract((1, 2)),),
            output_names=("fc",),
            builtin_options_type=FULLY_CONNECTED_OPTIONS,
            builtin_options=FullyConnectedOptions(),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 1)
        self.assertEqual(
            operator_builtin_code(document.model, document.subgraph().operators[0]),
            FULLY_CONNECTED,
        )


if __name__ == "__main__":
    unittest.main()

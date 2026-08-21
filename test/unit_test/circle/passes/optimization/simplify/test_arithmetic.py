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

from tico.circle.passes.base import CirclePassContext
from tico.circle.passes.cleanup import DeadCodeEliminationPass
from tico.circle.passes.optimization._utils import operator_builtin_code
from tico.circle.passes.optimization.policy import FloatingPointRewritePolicy
from tico.circle.passes.optimization.simplify.arithmetic import (
    ArithmeticCanonicalizationPolicy,
    SimplifyArithmeticPass,
)

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    make_empty_document,
)
from test.unit_test.circle.passes.optimization._fixture import (
    add_f32,
    make_builder,
    make_codec,
    static_contract,
)
from test.unit_test.circle.passes.optimization._operator_rewrite_fixture import (
    ACTIVATION_NONE,
    BinaryOptions,
    BUILTIN_CODES,
    BUILTIN_OPTIONS_TYPES,
    DIV,
    DIV_OPTIONS,
    MUL,
    MUL_OPTIONS,
    operator_rewrite_object_factory,
    RSQRT,
    SQRT,
    TENSOR_TYPES,
)


class SimplifyArithmeticPassTest(unittest.TestCase):
    """Check scalar MUL/DIV canonicalization and RSQRT transformation."""

    def setUp(self) -> None:
        """Create one schema-independent codec and pass context."""

        self.codec = make_codec()
        self.context = CirclePassContext(verify_after_each_pass=False)

    def _pass(
        self,
        *,
        policy: ArithmeticCanonicalizationPolicy | None = None,
    ) -> SimplifyArithmeticPass:
        """Create the arithmetic pass with fake schema identities."""

        return SimplifyArithmeticPass(
            builtin_codes=BUILTIN_CODES,
            builtin_options_types=BUILTIN_OPTIONS_TYPES,
            tensor_types=TENSOR_TYPES,
            activation_none=ACTIVATION_NONE,
            codec=self.codec,
            object_factory=operator_rewrite_object_factory,
            policy=policy,
        )

    def _binary(
        self,
        builder,
        code: int,
        left: int,
        right: int,
        shape,
        name: str,
        *,
        activation: int = ACTIVATION_NONE,
    ) -> int:
        """Append one MUL or DIV fixture."""

        options_type = MUL_OPTIONS if code == MUL else DIV_OPTIONS
        return builder.add_operator(
            code,
            inputs=(left, right),
            output_contracts=(static_contract(tuple(shape)),),
            output_names=(name,),
            builtin_options_type=options_type,
            builtin_options=BinaryOptions(
                fusedActivationFunction=activation,
            ),
        )[0]

    def _scalar(self, document, tensor_index: int) -> float:
        """Decode one scalar constant from the fixture model."""

        value = self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=tensor_index,
        )
        return float(value.data.reshape(-1)[0])

    def test_combines_multiply_then_divide(self) -> None:
        """Rewrite (x * 2) / 4 as x * 0.5 and leave old MUL for DCE."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2],
        )
        two = add_f32(builder, "two", 2.0)
        product = self._binary(builder, MUL, source, two, (2,), "product")
        four = add_f32(builder, "four", 4.0)
        output = self._binary(builder, DIV, product, four, (2,), "output")
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertTrue(result.modified)
        anchor = document.subgraph().operators[-1]
        self.assertEqual(operator_builtin_code(document.model, anchor), MUL)
        self.assertEqual(anchor.inputs[0], source)
        self.assertAlmostEqual(self._scalar(document, anchor.inputs[1]), 0.5)
        self.assertEqual(len(document.subgraph().operators), 2)
        DeadCodeEliminationPass().run(document, self.context)
        self.assertEqual(len(document.subgraph().operators), 1)

    def test_combines_constant_divided_by_product(self) -> None:
        """Rewrite 8 / (x * 2) as 4 / x."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2],
        )
        two = add_f32(builder, "two", 2.0)
        product = self._binary(builder, MUL, source, two, (2,), "product")
        eight = add_f32(builder, "eight", 8.0)
        output = self._binary(builder, DIV, eight, product, (2,), "output")
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertTrue(result.modified)
        anchor = document.subgraph().operators[-1]
        self.assertEqual(operator_builtin_code(document.model, anchor), DIV)
        self.assertAlmostEqual(self._scalar(document, anchor.inputs[0]), 4.0)
        self.assertEqual(anchor.inputs[1], source)

    def test_rewrites_sqrt_denominator_in_place(self) -> None:
        """Rewrite x / SQRT(y) as x * RSQRT(y) without adding an operator."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        numerator = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="numerator",
            shape=[2],
        )
        denominator = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="denominator",
            shape=[2],
        )
        sqrt = builder.add_operator(
            SQRT,
            inputs=(denominator,),
            output_contracts=(static_contract((2,)),),
            output_names=("sqrt",),
        )[0]
        output = self._binary(
            builder,
            DIV,
            numerator,
            sqrt,
            (2,),
            "output",
            activation=77,
        )
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertTrue(result.modified)
        producer, anchor = document.subgraph().operators
        self.assertEqual(operator_builtin_code(document.model, producer), RSQRT)
        self.assertEqual(producer.inputs, [denominator])
        self.assertEqual(operator_builtin_code(document.model, anchor), MUL)
        self.assertEqual(anchor.inputs, [numerator, sqrt])
        self.assertEqual(anchor.builtinOptions.fusedActivationFunction, 77)

    def test_keeps_sqrt_when_its_output_has_another_consumer(self) -> None:
        """Preserve SQRT semantics when its result has observable fanout."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        numerator = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="numerator",
            shape=[2],
        )
        denominator = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="denominator",
            shape=[2],
        )
        sqrt = builder.add_operator(
            SQRT,
            inputs=(denominator,),
            output_contracts=(static_contract((2,)),),
            output_names=("sqrt",),
        )[0]
        output = self._binary(builder, DIV, numerator, sqrt, (2,), "output")
        extra = self._binary(builder, MUL, sqrt, numerator, (2,), "extra")
        document.subgraph().outputs = [output, extra]

        result = self._pass().run(document, self.context)

        self.assertFalse(result.modified)

    def test_skips_zero_scalar_divisor(self) -> None:
        """Avoid folding scalar division whose denominator is zero."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1],
        )
        two = add_f32(builder, "two", 2.0)
        product = self._binary(builder, MUL, source, two, (1,), "product")
        zero = add_f32(builder, "zero", 0.0)
        output = self._binary(builder, DIV, product, zero, (1,), "output")
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertFalse(result.modified)

    def test_strict_policy_disables_arithmetic_reassociation(self) -> None:
        """Leave scalar MUL/DIV chains unchanged in strict mode."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1],
        )
        two = add_f32(builder, "two", 2.0)
        product = self._binary(builder, MUL, source, two, (1,), "product")
        four = add_f32(builder, "four", 4.0)
        output = self._binary(builder, DIV, product, four, (1,), "output")
        document.subgraph().outputs = [output]
        policy = ArithmeticCanonicalizationPolicy(
            floating_point_policy=FloatingPointRewritePolicy.STRICT
        )

        result = self._pass(policy=policy).run(document, self.context)

        self.assertFalse(result.modified)


if __name__ == "__main__":
    unittest.main()

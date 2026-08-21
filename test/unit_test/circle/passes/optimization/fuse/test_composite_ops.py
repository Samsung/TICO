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
from tico.circle.passes.optimization.fuse.composite import (
    CompositeFusionPolicy,
    FuseCompositeOpsPass,
)
from tico.circle.passes.optimization.policy import FloatingPointRewritePolicy

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    FLOAT32,
    make_empty_document,
)
from test.unit_test.circle.passes.optimization._fixture import (
    add_f32,
    add_i32,
    make_builder,
    make_codec,
    static_contract,
)
from test.unit_test.circle.passes.optimization._operator_rewrite_fixture import (
    ABS,
    ACTIVATION_NONE,
    ACTIVATION_RELU,
    ACTIVATION_RELU6,
    ACTIVATION_RELU_N1_TO_1,
    ADD,
    ADD_OPTIONS,
    BinaryOptions,
    BUILTIN_CODES,
    BUILTIN_OPTIONS_TYPES,
    CONV_2D,
    CONV_2D_OPTIONS,
    ConvOptions,
    CUSTOM,
    DIV,
    DIV_OPTIONS,
    GELU,
    INSTANCE_NORM,
    MAXIMUM,
    MEAN,
    MINIMUM,
    MUL,
    MUL_OPTIONS,
    operator_rewrite_object_factory,
    PRELU,
    REDUCER_OPTIONS,
    ReducerOptions,
    RELU,
    RELU6,
    RSQRT,
    SQRT,
    SUB,
    SUB_OPTIONS,
    TENSOR_TYPES,
)


class FuseCompositeOpsPassTest(unittest.TestCase):
    """Check composite recognition and conservative rejection behavior."""

    def setUp(self) -> None:
        """Create one schema-independent codec and pass context."""

        self.codec = make_codec()
        self.context = CirclePassContext(verify_after_each_pass=False)

    def _pass(
        self,
        *,
        policy: CompositeFusionPolicy | None = None,
    ) -> FuseCompositeOpsPass:
        """Create the composite pass with fake schema identities."""

        return FuseCompositeOpsPass(
            builtin_codes=BUILTIN_CODES,
            builtin_options_types=BUILTIN_OPTIONS_TYPES,
            tensor_types=TENSOR_TYPES,
            activation_none=ACTIVATION_NONE,
            activation_relu=ACTIVATION_RELU,
            activation_relu6=ACTIVATION_RELU6,
            activation_relu_n1_to_1=ACTIVATION_RELU_N1_TO_1,
            codec=self.codec,
            object_factory=operator_rewrite_object_factory,
            policy=policy,
        )

    def _op(
        self,
        builder,
        code: int,
        inputs,
        shape,
        name: str,
        *,
        options_type: int | None = None,
        options=None,
        custom_code: str | None = None,
    ) -> int:
        """Append one fixture operator with a static FLOAT32 output."""

        return builder.add_operator(
            code,
            inputs=tuple(inputs),
            output_contracts=(static_contract(tuple(shape)),),
            output_names=(name,),
            custom_code=custom_code,
            builtin_options_type=options_type,
            builtin_options=options,
        )[0]

    def _binary(self, builder, code, left, right, shape, name):
        """Append one binary operator with no fused activation."""

        option_type = {
            ADD: ADD_OPTIONS,
            MUL: MUL_OPTIONS,
            SUB: SUB_OPTIONS,
            DIV: DIV_OPTIONS,
        }.get(code)
        options = BinaryOptions() if option_type is not None else None
        return self._op(
            builder,
            code,
            (left, right),
            shape,
            name,
            options_type=option_type,
            options=options,
        )

    def test_fuses_standalone_relu_into_single_consumer_producer(self) -> None:
        """Move RELU into producer options while leaving old producer for DCE."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1, 2],
        )
        weight = add_f32(builder, "weight", [1.0, 2.0])
        conv = self._op(
            builder,
            CONV_2D,
            (source, weight),
            (1, 2),
            "conv",
            options_type=CONV_2D_OPTIONS,
            options=ConvOptions(),
        )
        output = self._op(builder, RELU, (conv,), (1, 2), "output")
        document.subgraph().inputs = [source]
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertTrue(result.modified)
        anchor = document.subgraph().operators[-1]
        self.assertEqual(operator_builtin_code(document.model, anchor), CONV_2D)
        self.assertEqual(anchor.outputs, [output])
        self.assertEqual(
            anchor.builtinOptions.fusedActivationFunction,
            ACTIVATION_RELU,
        )
        self.assertEqual(len(document.subgraph().operators), 2)
        dce = DeadCodeEliminationPass().run(document, self.context)
        self.assertTrue(dce.modified)
        self.assertEqual(len(document.subgraph().operators), 1)

    def test_does_not_fuse_activation_when_producer_output_has_fanout(self) -> None:
        """Preserve a standalone activation when its pre-activation is reused."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1, 2],
        )
        weight = add_f32(builder, "weight", [1.0, 2.0])
        conv = self._op(
            builder,
            CONV_2D,
            (source, weight),
            (1, 2),
            "conv",
            options_type=CONV_2D_OPTIONS,
            options=ConvOptions(),
        )
        output = self._op(builder, RELU, (conv,), (1, 2), "output")
        other = self._binary(builder, ADD, conv, weight, (1, 2), "other")
        document.subgraph().outputs = [output, other]

        result = self._pass().run(document, self.context)

        self.assertFalse(result.modified)

    def test_recognizes_minimum_maximum_relu6(self) -> None:
        """Replace MAXIMUM(MINIMUM(x, 6), 0) with RELU6."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1, 2],
        )
        six = add_f32(builder, "six", 6.0)
        zero = add_f32(builder, "zero", 0.0)
        minimum = self._op(
            builder,
            MINIMUM,
            (source, six),
            (1, 2),
            "minimum",
        )
        output = self._op(
            builder,
            MAXIMUM,
            (zero, minimum),
            (1, 2),
            "output",
        )
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertTrue(result.modified)
        anchor = document.subgraph().operators[-1]
        self.assertEqual(operator_builtin_code(document.model, anchor), RELU6)
        self.assertEqual(anchor.inputs, [source])

    def test_recognizes_reciprocal_sqrt(self) -> None:
        """Replace DIV(1, SQRT(x)) with RSQRT(x)."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2],
        )
        sqrt = self._op(builder, SQRT, (source,), (2,), "sqrt")
        one = add_f32(builder, "one", 1.0)
        output = self._binary(builder, DIV, one, sqrt, (2,), "output")
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertTrue(result.modified)
        anchor = document.subgraph().operators[-1]
        self.assertEqual(operator_builtin_code(document.model, anchor), RSQRT)
        self.assertEqual(anchor.inputs, [source])

    def test_recognizes_prelu_decomposition(self) -> None:
        """Replace the canonical ABS/SUB/MUL/RELU tree with PRELU."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1, 2],
        )
        absolute = self._op(builder, ABS, (source,), (1, 2), "absolute")
        negative = self._binary(
            builder,
            SUB,
            source,
            absolute,
            (1, 2),
            "negative",
        )
        alpha = add_f32(builder, "alpha", [0.1, 0.2])
        scaled = self._binary(
            builder,
            MUL,
            negative,
            alpha,
            (1, 2),
            "scaled",
        )
        half = add_f32(builder, "half", 0.5)
        half_scaled = self._binary(
            builder,
            MUL,
            half,
            scaled,
            (1, 2),
            "half_scaled",
        )
        positive = self._op(builder, RELU, (source,), (1, 2), "positive")
        output = self._binary(
            builder,
            ADD,
            positive,
            half_scaled,
            (1, 2),
            "output",
        )
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertTrue(result.modified)
        anchor = document.subgraph().operators[-1]
        self.assertEqual(operator_builtin_code(document.model, anchor), PRELU)
        self.assertEqual(anchor.inputs, [source, alpha])

    def test_recognizes_exact_erf_gelu(self) -> None:
        """Replace the exact ERF multiplication tree with GELU."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2],
        )
        sqrt_half = add_f32(builder, "sqrt_half", np.sqrt(0.5))
        erf_input = self._binary(
            builder,
            MUL,
            source,
            sqrt_half,
            (2,),
            "erf_input",
        )
        erf = self._op(
            builder,
            CUSTOM,
            (erf_input,),
            (2,),
            "erf",
            custom_code="Erf",
        )
        one = add_f32(builder, "one", 1.0)
        shifted = self._binary(builder, ADD, erf, one, (2,), "shifted")
        core = self._binary(builder, MUL, source, shifted, (2,), "core")
        half = add_f32(builder, "half", 0.5)
        output = self._binary(builder, MUL, core, half, (2,), "output")
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertTrue(result.modified)
        anchor = document.subgraph().operators[-1]
        self.assertEqual(operator_builtin_code(document.model, anchor), GELU)
        self.assertEqual(anchor.inputs, [source])
        self.assertFalse(anchor.builtinOptions.approximate)

    def test_recognizes_rank_four_instance_norm(self) -> None:
        """Replace a canonical NHWC instance-normalization decomposition."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1, 2, 2, 3],
        )
        axes = add_i32(builder, "axes", [1, 2])
        mean = self._op(
            builder,
            MEAN,
            (source, axes),
            (1, 1, 1, 3),
            "mean",
            options_type=REDUCER_OPTIONS,
            options=ReducerOptions(keepDims=True),
        )
        centered = self._binary(
            builder,
            SUB,
            source,
            mean,
            (1, 2, 2, 3),
            "centered",
        )
        square = self._binary(
            builder,
            MUL,
            centered,
            centered,
            (1, 2, 2, 3),
            "square",
        )
        variance = self._op(
            builder,
            MEAN,
            (square, axes),
            (1, 1, 1, 3),
            "variance",
            options_type=REDUCER_OPTIONS,
            options=ReducerOptions(keepDims=True),
        )
        epsilon = add_f32(builder, "epsilon", 1e-5)
        variance_eps = self._binary(
            builder,
            ADD,
            variance,
            epsilon,
            (1, 1, 1, 3),
            "variance_eps",
        )
        inverse_std = self._op(
            builder,
            RSQRT,
            (variance_eps,),
            (1, 1, 1, 3),
            "inverse_std",
        )
        normalized = self._binary(
            builder,
            MUL,
            centered,
            inverse_std,
            (1, 2, 2, 3),
            "normalized",
        )
        gamma = add_f32(builder, "gamma", [1.0, 1.5, 2.0])
        scaled = self._binary(
            builder,
            MUL,
            normalized,
            gamma,
            (1, 2, 2, 3),
            "scaled",
        )
        beta = add_f32(builder, "beta", [0.1, 0.2, 0.3])
        output = self._binary(
            builder,
            ADD,
            scaled,
            beta,
            (1, 2, 2, 3),
            "output",
        )
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertTrue(result.modified)
        anchor = document.subgraph().operators[-1]
        self.assertEqual(
            operator_builtin_code(document.model, anchor),
            INSTANCE_NORM,
        )
        self.assertEqual(anchor.inputs, [source, gamma, beta])
        self.assertAlmostEqual(anchor.builtinOptions.epsilon, 1e-5)

    def test_strict_policy_disables_algebraic_composite_recognition(self) -> None:
        """Leave RELU6 decompositions unchanged in strict floating-point mode."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1],
        )
        six = add_f32(builder, "six", 6.0)
        zero = add_f32(builder, "zero", 0.0)
        minimum = self._op(builder, MINIMUM, (source, six), (1,), "minimum")
        output = self._op(
            builder,
            MAXIMUM,
            (minimum, zero),
            (1,),
            "output",
        )
        document.subgraph().outputs = [output]
        policy = CompositeFusionPolicy(
            floating_point_policy=FloatingPointRewritePolicy.STRICT
        )

        result = self._pass(policy=policy).run(document, self.context)

        self.assertFalse(result.modified)


if __name__ == "__main__":
    unittest.main()

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
from tico.circle.passes.optimization.fusion.reduction_ops import (
    ReductionSimplificationPolicy,
    SimplifyReductionOpsPass,
)
from tico.circle.passes.optimization.policy import FloatingPointRewritePolicy

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    make_empty_document,
)
from test.unit_test.circle.passes.optimization._fixture import (
    add_i32,
    make_builder,
    make_codec,
    static_contract,
)
from test.unit_test.circle.passes.optimization._operator_rewrite_fixture import (
    BUILTIN_CODES,
    BUILTIN_OPTIONS_TYPES,
    MEAN,
    operator_rewrite_object_factory,
    REDUCER_OPTIONS,
    ReducerOptions,
    TENSOR_TYPES,
    TRANSPOSE,
)


class SimplifyReductionOpsPassTest(unittest.TestCase):
    """Check MEAN composition and transpose elimination."""

    def setUp(self) -> None:
        """Create one schema-independent codec and pass context."""

        self.codec = make_codec()
        self.context = CirclePassContext(verify_after_each_pass=False)

    def _pass(
        self,
        *,
        policy: ReductionSimplificationPolicy | None = None,
    ) -> SimplifyReductionOpsPass:
        """Create the reduction pass with fake schema identities."""

        return SimplifyReductionOpsPass(
            builtin_codes=BUILTIN_CODES,
            builtin_options_types=BUILTIN_OPTIONS_TYPES,
            tensor_types=TENSOR_TYPES,
            codec=self.codec,
            object_factory=operator_rewrite_object_factory,
            policy=policy,
        )

    def _mean(
        self,
        builder,
        source: int,
        axes: int,
        output_shape,
        name: str,
        *,
        keep_dims: bool,
    ) -> int:
        """Append one static MEAN fixture."""

        return builder.add_operator(
            MEAN,
            inputs=(source, axes),
            output_contracts=(static_contract(tuple(output_shape)),),
            output_names=(name,),
            builtin_options_type=REDUCER_OPTIONS,
            builtin_options=ReducerOptions(keepDims=keep_dims),
        )[0]

    def _decoded_axes(self, document, operator) -> tuple[int, ...]:
        """Decode the axis tensor consumed by one rewritten MEAN."""

        value = self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=operator.inputs[1],
        )
        return tuple(int(item) for item in value.data.reshape(-1))

    def test_merges_consecutive_mean_without_keep_dims(self) -> None:
        """Map the second reduced axis through the rank-reduced intermediate."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3, 4],
        )
        first_axes = add_i32(builder, "first_axes", [1])
        first = self._mean(
            builder,
            source,
            first_axes,
            (2, 4),
            "first",
            keep_dims=False,
        )
        second_axes = add_i32(builder, "second_axes", [1])
        output = self._mean(
            builder,
            first,
            second_axes,
            (2,),
            "output",
            keep_dims=False,
        )
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertTrue(result.modified)
        anchor = document.subgraph().operators[-1]
        self.assertEqual(operator_builtin_code(document.model, anchor), MEAN)
        self.assertEqual(anchor.inputs[0], source)
        self.assertEqual(self._decoded_axes(document, anchor), (1, 2))
        self.assertEqual(len(document.subgraph().operators), 2)
        DeadCodeEliminationPass().run(document, self.context)
        self.assertEqual(len(document.subgraph().operators), 1)

    def test_merges_consecutive_mean_with_keep_dims(self) -> None:
        """Union axis sets when both reductions retain reduced dimensions."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3, 4],
        )
        first_axes = add_i32(builder, "first_axes", [1])
        first = self._mean(
            builder,
            source,
            first_axes,
            (2, 1, 4),
            "first",
            keep_dims=True,
        )
        second_axes = add_i32(builder, "second_axes", [2])
        output = self._mean(
            builder,
            first,
            second_axes,
            (2, 1, 1),
            "output",
            keep_dims=True,
        )
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertTrue(result.modified)
        anchor = document.subgraph().operators[-1]
        self.assertEqual(anchor.inputs[0], source)
        self.assertEqual(self._decoded_axes(document, anchor), (1, 2))

    def test_removes_transpose_before_mean_when_survivors_are_ordered(self) -> None:
        """Remap reduced axes to source order without changing output layout."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3, 4],
        )
        permutation = add_i32(builder, "permutation", [1, 0, 2])
        transposed = builder.add_operator(
            TRANSPOSE,
            inputs=(source, permutation),
            output_contracts=(static_contract((3, 2, 4)),),
            output_names=("transposed",),
        )[0]
        axes = add_i32(builder, "axes", [0])
        output = self._mean(
            builder,
            transposed,
            axes,
            (2, 4),
            "output",
            keep_dims=False,
        )
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertTrue(result.modified)
        anchor = document.subgraph().operators[-1]
        self.assertEqual(anchor.inputs[0], source)
        self.assertEqual(self._decoded_axes(document, anchor), (1,))

    def test_keeps_transpose_when_surviving_axis_order_changes(self) -> None:
        """Reject a transpose whose unreduced axes remain permuted."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3, 4],
        )
        permutation = add_i32(builder, "permutation", [2, 0, 1])
        transposed = builder.add_operator(
            TRANSPOSE,
            inputs=(source, permutation),
            output_contracts=(static_contract((4, 2, 3)),),
            output_names=("transposed",),
        )[0]
        axes = add_i32(builder, "axes", [1])
        output = self._mean(
            builder,
            transposed,
            axes,
            (4, 3),
            "output",
            keep_dims=False,
        )
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertFalse(result.modified)

    def test_strict_policy_disables_reduction_reassociation(self) -> None:
        """Leave consecutive MEAN operators unchanged in strict mode."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3, 4],
        )
        first_axes = add_i32(builder, "first_axes", [0])
        first = self._mean(
            builder,
            source,
            first_axes,
            (3, 4),
            "first",
            keep_dims=False,
        )
        second_axes = add_i32(builder, "second_axes", [0])
        output = self._mean(
            builder,
            first,
            second_axes,
            (4,),
            "output",
            keep_dims=False,
        )
        document.subgraph().outputs = [output]
        policy = ReductionSimplificationPolicy(
            floating_point_policy=FloatingPointRewritePolicy.STRICT
        )

        result = self._pass(policy=policy).run(document, self.context)

        self.assertFalse(result.modified)


if __name__ == "__main__":
    unittest.main()

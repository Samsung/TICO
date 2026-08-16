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

import numpy as np

from tico.circle.passes import CirclePassContext, SimplifyViewOpsPass
from tico.circle.passes.cleanup import DeadCodeEliminationPass
from tico.circle.passes.optimization._utils import operator_builtin_code

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    FakeSignatureDef,
    FakeTensorMap,
    make_empty_document,
)
from test.unit_test.circle.passes.optimization._fixture import (
    ABS,
    add_f32,
    add_i32,
    BUILTIN_CODES,
    BUILTIN_OPTIONS_TYPES,
    make_builder,
    make_codec,
    MEAN,
    MUL,
    optimization_object_factory,
    ReducerOptions,
    RESHAPE,
    ReshapeOptions,
    static_contract,
    TENSOR_TYPES,
    TRANSPOSE,
)


class SimplifyViewOpsTest(unittest.TestCase):
    """Check identity removal, view composition, and safe RESHAPE motion."""

    def setUp(self) -> None:
        """Create a schema-independent constant codec for each test."""

        self.codec = make_codec()

    def _pass(self) -> SimplifyViewOpsPass:
        """Create the view pass with fake schema identities."""

        return SimplifyViewOpsPass(
            builtin_codes=BUILTIN_CODES,
            builtin_options_types=BUILTIN_OPTIONS_TYPES,
            tensor_types=TENSOR_TYPES,
            codec=self.codec,
            object_factory=optimization_object_factory,
        )

    def _reshape(self, builder, source, shape, name):
        """Append one static RESHAPE and return its output tensor index."""

        shape_input = add_i32(builder, f"{name}_shape", shape)
        return builder.add_operator(
            RESHAPE,
            inputs=(source, shape_input),
            output_contracts=(static_contract(tuple(shape)),),
            output_names=(name,),
            builtin_options=ReshapeOptions(list(shape)),
        )[0]

    def _transpose(self, builder, source, input_shape, permutation, name):
        """Append one static TRANSPOSE and return its output tensor index."""

        perm_input = add_i32(builder, f"{name}_perm", permutation)
        output_shape = tuple(input_shape[axis] for axis in permutation)
        return builder.add_operator(
            TRANSPOSE,
            inputs=(source, perm_input),
            output_contracts=(static_contract(output_shape),),
            output_names=(name,),
        )[0]

    def test_identity_reshape_and_transpose_are_rewired_before_dce(self) -> None:
        """Bypass identity views while leaving operator deletion to external DCE."""

        for code in (RESHAPE, TRANSPOSE):
            with self.subTest(code=code):
                document = make_empty_document()
                builder = make_builder(document, self.codec)
                source = add_runtime_tensor(
                    document,
                    subgraph_index=0,
                    name="source",
                    shape=[2, 3],
                )
                document.subgraph().inputs = [source]
                if code == RESHAPE:
                    output = self._reshape(builder, source, [2, 3], "output")
                else:
                    output = self._transpose(
                        builder,
                        source,
                        (2, 3),
                        [0, 1],
                        "output",
                    )
                document.subgraph().outputs = [output]

                result = self._pass().run(
                    document,
                    CirclePassContext(verify_after_each_pass=False),
                )

                self.assertTrue(result.modified)
                self.assertEqual(len(document.subgraph().operators), 1)
                self.assertEqual(document.subgraph().outputs, [source])

                cleanup = DeadCodeEliminationPass().run(
                    document,
                    CirclePassContext(verify_after_each_pass=False),
                )

                self.assertTrue(cleanup.modified)
                self.assertEqual(document.subgraph().operators, [])

    def test_consecutive_reshape_chain_is_composed_before_dce(self) -> None:
        """Reconnect the final RESHAPE and leave the first one for external DCE."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3],
        )
        document.subgraph().inputs = [source]
        first = self._reshape(builder, source, [3, 2], "first")
        second = self._reshape(builder, first, [1, 6], "second")
        document.subgraph().outputs = [second]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(len(document.subgraph().operators), 2)
        operator = document.subgraph().operators[1]
        self.assertEqual(operator_builtin_code(document.model, operator), RESHAPE)
        self.assertEqual(operator.inputs[0], source)

        cleanup = DeadCodeEliminationPass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(cleanup.modified)
        self.assertEqual(len(document.subgraph().operators), 1)
        self.assertIs(document.subgraph().operators[0], operator)

    def test_malformed_reshape_chain_is_preserved(self) -> None:
        """Keep a chain when the first RESHAPE metadata disagrees with its target."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3],
        )
        document.subgraph().inputs = [source]
        first = self._reshape(builder, source, [3, 2], "first")
        document.subgraph().tensors[first].shape = [2, 3]
        document.subgraph().tensors[first].shapeSignature = [2, 3]
        second = self._reshape(builder, first, [1, 6], "second")
        document.subgraph().outputs = [second]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 2)

    def test_inverse_transpose_pair_is_rewired_before_dce(self) -> None:
        """Bypass inverse transposes and leave both dead operators for DCE."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3],
        )
        document.subgraph().inputs = [source]
        first = self._transpose(builder, source, (2, 3), [1, 0], "first")
        second = self._transpose(builder, first, (3, 2), [1, 0], "second")
        document.subgraph().outputs = [second]
        document.model.signatureDefs = [
            FakeSignatureDef(
                signatureKey="serving_default",
                subgraphIndex=0,
                inputs=[FakeTensorMap(name="source", tensorIndex=source)],
                outputs=[FakeTensorMap(name="output", tensorIndex=second)],
            )
        ]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(len(document.subgraph().operators), 2)
        self.assertEqual(document.subgraph().outputs, [source])
        self.assertEqual(
            document.model.signatureDefs[0].outputs[0].tensorIndex,
            source,
        )

        cleanup = DeadCodeEliminationPass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(cleanup.modified)
        self.assertEqual(document.subgraph().operators, [])

    def test_dead_identity_view_without_uses_is_left_for_dce(self) -> None:
        """Avoid repeatedly matching an already dead identity view."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3],
        )
        document.subgraph().inputs = [source]
        self._reshape(builder, source, [2, 3], "dead_identity")
        document.subgraph().outputs = [source]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 1)

    def test_inverse_pair_preserves_shared_first_transpose_after_dce(self) -> None:
        """Keep a first TRANSPOSE that still feeds another live consumer."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3],
        )
        document.subgraph().inputs = [source]
        first = self._transpose(builder, source, (2, 3), [1, 0], "first")
        second = self._transpose(builder, first, (3, 2), [1, 0], "second")
        fanout = builder.add_operator(
            ABS,
            inputs=(first,),
            output_contracts=(static_contract((3, 2)),),
            output_names=("fanout",),
        )[0]
        document.subgraph().outputs = [second, fanout]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(document.subgraph().outputs, [source, fanout])
        self.assertEqual(len(document.subgraph().operators), 3)

        cleanup = DeadCodeEliminationPass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(cleanup.modified)
        self.assertEqual(
            [
                operator_builtin_code(document.model, operator)
                for operator in document.subgraph().operators
            ],
            [TRANSPOSE, ABS],
        )

    def test_malformed_transpose_chain_is_preserved(self) -> None:
        """Keep a chain whose intermediate tensor shape violates its permutation."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3],
        )
        document.subgraph().inputs = [source]
        first = self._transpose(builder, source, (2, 3), [1, 0], "first")
        document.subgraph().tensors[first].shape = [2, 3]
        document.subgraph().tensors[first].shapeSignature = [2, 3]
        second = self._transpose(builder, first, (2, 3), [1, 0], "second")
        document.subgraph().outputs = [second]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 2)

    def test_nonidentity_transpose_chain_is_composed_before_dce(self) -> None:
        """Rewrite the second TRANSPOSE and leave the first one for external DCE."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3, 4],
        )
        document.subgraph().inputs = [source]
        first = self._transpose(
            builder,
            source,
            (2, 3, 4),
            [1, 2, 0],
            "first",
        )
        second = self._transpose(
            builder,
            first,
            (3, 4, 2),
            [1, 2, 0],
            "second",
        )
        document.subgraph().outputs = [second]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(len(document.subgraph().operators), 2)
        operator = document.subgraph().operators[1]
        self.assertEqual(operator_builtin_code(document.model, operator), TRANSPOSE)
        self.assertEqual(operator.inputs[0], source)
        permutation = self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=operator.inputs[1],
        )
        np.testing.assert_array_equal(
            permutation.data,
            np.array([2, 0, 1], dtype=np.int32),
        )

        cleanup = DeadCodeEliminationPass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(cleanup.modified)
        self.assertEqual(document.subgraph().operators, [operator])

    def test_reshape_moves_after_unary_elementwise(self) -> None:
        """Swap RESHAPE and ABS while preserving intermediate tensor ownership."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3],
        )
        document.subgraph().inputs = [source]
        reshaped = self._reshape(builder, source, [3, 2], "reshaped")
        output = builder.add_operator(
            ABS,
            inputs=(reshaped,),
            output_contracts=(static_contract((3, 2)),),
            output_names=("output",),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        operators = document.subgraph().operators
        self.assertEqual(
            [operator_builtin_code(document.model, op) for op in operators],
            [ABS, RESHAPE],
        )
        self.assertEqual(operators[0].inputs, [source])
        self.assertEqual(operators[0].outputs, [reshaped])
        self.assertEqual(operators[1].inputs[0], reshaped)
        self.assertEqual(document.subgraph().tensors[reshaped].shape, [2, 3])

    def test_reshape_moves_after_scalar_binary_elementwise(self) -> None:
        """Move RESHAPE after scalar MUL without changing scalar broadcasting."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3],
        )
        document.subgraph().inputs = [source]
        reshaped = self._reshape(builder, source, [3, 2], "reshaped")
        scalar = add_f32(builder, "scalar", 2.0)
        output = builder.add_operator(
            MUL,
            inputs=(reshaped, scalar),
            output_contracts=(static_contract((3, 2)),),
            output_names=("output",),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        operators = document.subgraph().operators
        self.assertEqual(
            [operator_builtin_code(document.model, op) for op in operators],
            [MUL, RESHAPE],
        )
        self.assertEqual(operators[0].inputs, [source, scalar])

    def test_reshape_moves_after_compatible_keep_dims_mean(self) -> None:
        """Move RESHAPE after MEAN when dimensions through its axis are unchanged."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3, 4, 5],
        )
        document.subgraph().inputs = [source]
        reshaped = self._reshape(builder, source, [2, 3, 10, 2], "reshaped")
        axis = add_i32(builder, "axis", [1])
        output = builder.add_operator(
            MEAN,
            inputs=(reshaped, axis),
            output_contracts=(static_contract((2, 1, 10, 2)),),
            output_names=("output",),
            builtin_options=ReducerOptions(keepDims=True),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        operators = document.subgraph().operators
        self.assertEqual(
            [operator_builtin_code(document.model, op) for op in operators],
            [MEAN, RESHAPE],
        )
        self.assertEqual(operators[0].inputs, [source, axis])
        self.assertEqual(document.subgraph().tensors[reshaped].shape, [2, 1, 4, 5])
        target = self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=operators[1].inputs[1],
        )
        np.testing.assert_array_equal(
            target.data,
            np.array([2, 1, 10, 2], dtype=np.int32),
        )

    def test_elementwise_motion_requires_valid_reshape_target(self) -> None:
        """Keep RESHAPE before elementwise when target metadata is inconsistent."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3],
        )
        document.subgraph().inputs = [source]
        reshaped = self._reshape(builder, source, [3, 2], "reshaped")
        reshape_operator = document.subgraph().operators[0]
        reshape_operator.builtinOptions.newShape = [2, 3]
        output = builder.add_operator(
            ABS,
            inputs=(reshaped,),
            output_contracts=(static_contract((3, 2)),),
            output_names=("output",),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(
            [
                operator_builtin_code(document.model, operator)
                for operator in document.subgraph().operators
            ],
            [RESHAPE, ABS],
        )

    def test_elementwise_motion_requires_single_consumer(self) -> None:
        """Keep RESHAPE in place when its output feeds more than one consumer."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3],
        )
        document.subgraph().inputs = [source]
        reshaped = self._reshape(builder, source, [3, 2], "reshaped")
        first = builder.add_operator(
            ABS,
            inputs=(reshaped,),
            output_contracts=(static_contract((3, 2)),),
            output_names=("first",),
        )[0]
        second = builder.add_operator(
            ABS,
            inputs=(reshaped,),
            output_contracts=(static_contract((3, 2)),),
            output_names=("second",),
        )[0]
        document.subgraph().outputs = [first, second]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(
            [
                operator_builtin_code(document.model, operator)
                for operator in document.subgraph().operators
            ],
            [RESHAPE, ABS, ABS],
        )


if __name__ == "__main__":
    unittest.main()

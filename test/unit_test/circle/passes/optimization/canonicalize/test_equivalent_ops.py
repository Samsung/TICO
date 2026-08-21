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

from tico.circle.passes import CanonicalizeEquivalentOpsPass, CirclePassContext
from tico.circle.passes.optimization._utils import operator_builtin_code
from tico.circle.value import TensorQuantization, TensorValueCodec

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    FLOAT32,
    make_empty_document,
)
from test.unit_test.circle.passes.optimization._fixture import (
    add_f32,
    add_i32,
    BUILTIN_CODES,
    BUILTIN_OPTIONS_TYPES,
    EXPAND_DIMS,
    make_builder,
    make_codec,
    optimization_object_factory,
    PACK,
    PackOptions,
    PAD,
    PADV2,
    pass_kwargs,
    RESHAPE,
    SPLIT,
    SPLIT_V,
    SplitOptions,
    SQUEEZE,
    SqueezeOptions,
    static_contract,
    STRIDED_SLICE,
    StridedSliceOptions,
    TENSOR_TYPES,
    TRANSPOSE,
)


class _FailReshapeOptionsFactory:
    """Fail after a RESHAPE shape constant has been allocated."""

    def __call__(self, table_name: str):
        """Delegate table creation except for the replacement options table."""

        if table_name == "ReshapeOptions":
            raise RuntimeError("synthetic RESHAPE options failure")
        return optimization_object_factory(table_name)


class CanonicalizeEquivalentOpsTest(unittest.TestCase):
    """Check equivalent-operator canonicalization and conservative rejections."""

    def setUp(self) -> None:
        """Create a schema-independent constant codec for each test."""

        self.codec = make_codec()

    def _pass(self) -> CanonicalizeEquivalentOpsPass:
        """Create the pass with fake schema identities and Object API tables."""

        return CanonicalizeEquivalentOpsPass(**pass_kwargs(self.codec))

    def _assert_single_reshape(self, document, expected_shape) -> None:
        """Assert that a document contains one static two-input RESHAPE."""

        operators = document.subgraph().operators
        self.assertEqual(len(operators), 1)
        operator = operators[0]
        self.assertEqual(operator_builtin_code(document.model, operator), RESHAPE)
        self.assertEqual(len(operator.inputs), 2)
        shape = self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=operator.inputs[1],
        )
        np.testing.assert_array_equal(
            shape.data,
            np.asarray(expected_shape, dtype=np.int32),
        )
        self.assertEqual(operator.builtinOptions.newShape, list(expected_shape))

    def test_failed_replacement_rolls_back_allocated_constants(self) -> None:
        """Restore model sequences when replacement construction raises."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3],
        )
        document.subgraph().inputs = [source]
        axis = add_i32(builder, "axis", 1)
        output = builder.add_operator(
            EXPAND_DIMS,
            inputs=(source, axis),
            output_contracts=(static_contract((2, 1, 3)),),
            output_names=("output",),
        )[0]
        document.subgraph().outputs = [output]
        original_operator = document.subgraph().operators[0]
        original_counts = (
            len(document.model.buffers),
            len(document.model.operatorCodes),
            len(document.subgraph().tensors),
            len(document.subgraph().operators),
        )

        folding_pass = CanonicalizeEquivalentOpsPass(
            builtin_codes=BUILTIN_CODES,
            builtin_options_types=BUILTIN_OPTIONS_TYPES,
            tensor_types=TENSOR_TYPES,
            codec=self.codec,
            object_factory=_FailReshapeOptionsFactory(),
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic RESHAPE options failure"):
            folding_pass.run(
                document,
                CirclePassContext(verify_after_each_pass=False),
            )

        self.assertEqual(
            (
                len(document.model.buffers),
                len(document.model.operatorCodes),
                len(document.subgraph().tensors),
                len(document.subgraph().operators),
            ),
            original_counts,
        )
        self.assertIs(document.subgraph().operators[0], original_operator)
        self.assertEqual(document.subgraph().outputs, [output])

    def test_rank_only_operators_canonicalize_to_reshape(self) -> None:
        """Convert EXPAND_DIMS, one-input PACK, SQUEEZE, and view transpose."""

        cases = (
            (
                EXPAND_DIMS,
                (2, 3),
                (2, 1, 3),
                lambda builder: (add_i32(builder, "axis", [1]),),
                None,
            ),
            (
                PACK,
                (2, 3),
                (2, 3, 1),
                lambda builder: (),
                PackOptions(axis=2, valuesCount=1),
            ),
            (
                SQUEEZE,
                (1, 2, 1, 3),
                (2, 3),
                lambda builder: (),
                SqueezeOptions([0, 2]),
            ),
            (
                TRANSPOSE,
                (1, 2, 3),
                (2, 3, 1),
                lambda builder: (add_i32(builder, "perm", [1, 2, 0]),),
                None,
            ),
        )
        for source_code, input_shape, output_shape, extra_inputs, options in cases:
            with self.subTest(source_code=source_code):
                document = make_empty_document()
                builder = make_builder(document, self.codec)
                source = add_runtime_tensor(
                    document,
                    subgraph_index=0,
                    name="source",
                    shape=list(input_shape),
                )
                document.subgraph().inputs = [source]
                inputs = (source, *extra_inputs(builder))
                output = builder.add_operator(
                    source_code,
                    inputs=inputs,
                    output_contracts=(static_contract(output_shape),),
                    output_names=("output",),
                    builtin_options=options,
                )[0]
                document.subgraph().outputs = [output]

                result = self._pass().run(
                    document,
                    CirclePassContext(verify_after_each_pass=False),
                )

                self.assertTrue(result.modified)
                self._assert_single_reshape(document, output_shape)

    def test_view_only_strided_slice_canonicalizes_to_reshape(self) -> None:
        """Convert unit-axis shrink and insertion without slicing payload data."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1, 2],
        )
        document.subgraph().inputs = [source]
        begin = add_i32(builder, "begin", [0, 0, 0])
        end = add_i32(builder, "end", [1, 2, 0])
        strides = add_i32(builder, "strides", [1, 1, 1])
        output = builder.add_operator(
            STRIDED_SLICE,
            inputs=(source, begin, end, strides),
            output_contracts=(static_contract((2, 1)),),
            output_names=("output",),
            builtin_options=StridedSliceOptions(
                beginMask=0b010,
                endMask=0b010,
                newAxisMask=0b100,
                shrinkAxisMask=0b001,
            ),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self._assert_single_reshape(document, (2, 1))

    def test_zero_padv2_canonicalizes_but_nonzero_value_does_not(self) -> None:
        """Replace PADV2 only when the explicit padding value is exact zero."""

        for padding_value, expected_modified in ((0.0, True), (1.0, False)):
            with self.subTest(padding_value=padding_value):
                document = make_empty_document()
                builder = make_builder(document, self.codec)
                source = add_runtime_tensor(
                    document,
                    subgraph_index=0,
                    name="source",
                    shape=[2],
                )
                document.subgraph().inputs = [source]
                paddings = add_i32(builder, "paddings", [[1, 1]])
                value = add_f32(builder, "value", padding_value)
                output = builder.add_operator(
                    PADV2,
                    inputs=(source, paddings, value),
                    output_contracts=(static_contract((4,)),),
                    output_names=("output",),
                )[0]
                document.subgraph().outputs = [output]

                result = self._pass().run(
                    document,
                    CirclePassContext(verify_after_each_pass=False),
                )

                self.assertEqual(result.modified, expected_modified)
                operator = document.subgraph().operators[0]
                expected_code = PAD if expected_modified else PADV2
                self.assertEqual(
                    operator_builtin_code(document.model, operator),
                    expected_code,
                )
                if expected_modified:
                    self.assertEqual(operator.inputs, [source, paddings])

    def test_equal_split_v_canonicalizes_but_unequal_sizes_do_not(self) -> None:
        """Replace SPLIT_V only after inferred sizes resolve to equal partitions."""

        for sizes, expected_modified in (([3, -1], True), ([2, 4], False)):
            with self.subTest(sizes=sizes):
                document = make_empty_document()
                builder = make_builder(document, self.codec)
                source = add_runtime_tensor(
                    document,
                    subgraph_index=0,
                    name="source",
                    shape=[6],
                )
                document.subgraph().inputs = [source]
                size_tensor = add_i32(builder, "sizes", sizes)
                axis = add_i32(builder, "axis", 0)
                outputs = builder.add_operator(
                    SPLIT_V,
                    inputs=(source, size_tensor, axis),
                    output_contracts=(
                        static_contract((sizes[0],)),
                        static_contract((6 - sizes[0],)),
                    ),
                    output_names=("left", "right"),
                    builtin_options=SplitOptions(numSplits=2),
                )
                document.subgraph().outputs = list(outputs)

                result = self._pass().run(
                    document,
                    CirclePassContext(verify_after_each_pass=False),
                )

                self.assertEqual(result.modified, expected_modified)
                operator = document.subgraph().operators[0]
                expected_code = SPLIT if expected_modified else SPLIT_V
                self.assertEqual(
                    operator_builtin_code(document.model, operator),
                    expected_code,
                )
                if expected_modified:
                    self.assertEqual(operator.inputs, [axis, source])
                    self.assertEqual(operator.builtinOptions.numSplits, 2)

    def test_per_axis_quantized_view_is_kept_conservative(self) -> None:
        """Keep rank-changing views when per-axis qparam remapping is required."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        quantization = TensorQuantization(
            scale=(0.25, 0.5),
            zero_point=(0, 0),
            quantized_dimension=1,
        )
        source_contract = static_contract((1, 2))
        source_contract = source_contract.__class__(
            tensor_type=source_contract.tensor_type,
            shape=source_contract.shape,
            shape_signature=source_contract.shape_signature,
            quantization=quantization,
        )
        output_contract = source_contract.__class__(
            tensor_type=source_contract.tensor_type,
            shape=(1, 1, 2),
            shape_signature=(1, 1, 2),
            quantization=quantization,
        )
        source = builder.add_tensor("source", source_contract)
        document.subgraph().inputs = [source]
        axis = add_i32(builder, "axis", 1)
        output = builder.add_operator(
            EXPAND_DIMS,
            inputs=(source, axis),
            output_contracts=(output_contract,),
            output_names=("output",),
        )[0]
        document.subgraph().outputs = [output]

        result = CanonicalizeEquivalentOpsPass(
            builtin_codes=BUILTIN_CODES,
            builtin_options_types=BUILTIN_OPTIONS_TYPES,
            tensor_types=TENSOR_TYPES,
            codec=self.codec,
            object_factory=optimization_object_factory,
        ).run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(
            operator_builtin_code(document.model, document.subgraph().operators[0]),
            EXPAND_DIMS,
        )

    def test_identity_patterns_are_left_to_identity_elimination(self) -> None:
        """Do not canonicalize patterns owned by identity-removal passes."""

        cases = (
            (
                TRANSPOSE,
                (2, 3),
                lambda builder: (add_i32(builder, "perm", [0, 1]),),
                None,
            ),
            (
                STRIDED_SLICE,
                (2, 3),
                lambda builder: (
                    add_i32(builder, "begin", [0, 0]),
                    add_i32(builder, "end", [2, 3]),
                    add_i32(builder, "strides", [1, 1]),
                ),
                StridedSliceOptions(),
            ),
            (
                SPLIT_V,
                (4,),
                lambda builder: (
                    add_i32(builder, "size", [-1]),
                    add_i32(builder, "axis", 0),
                ),
                SplitOptions(numSplits=1),
            ),
        )
        for source_code, shape, extra_inputs, options in cases:
            with self.subTest(source_code=source_code):
                document = make_empty_document()
                builder = make_builder(document, self.codec)
                source = add_runtime_tensor(
                    document,
                    subgraph_index=0,
                    name="source",
                    shape=list(shape),
                )
                document.subgraph().inputs = [source]
                inputs = (source, *extra_inputs(builder))
                output = builder.add_operator(
                    source_code,
                    inputs=inputs,
                    output_contracts=(static_contract(shape),),
                    output_names=("output",),
                    builtin_options=options,
                )[0]
                document.subgraph().outputs = [output]

                result = self._pass().run(
                    document,
                    CirclePassContext(verify_after_each_pass=False),
                )

                self.assertFalse(result.modified)
                self.assertEqual(len(document.subgraph().operators), 1)


if __name__ == "__main__":
    unittest.main()

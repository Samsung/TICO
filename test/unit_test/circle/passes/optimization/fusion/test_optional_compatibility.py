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

from math import sqrt

import numpy as np

from tico.circle.passes import (
    CirclePassContext,
    FuseLegacyFCGeluFCPass,
    FuseTransposeConvSlicePass,
)
from tico.circle.passes.optimization._utils import operator_builtin_code

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    FLOAT32,
    INT32,
    make_empty_document,
)
from test.unit_test.circle.passes.optimization._compatibility_fixture import (
    ADD,
    add_constant,
    ADD_OPTIONS,
    BUILTIN_CODES,
    BUILTIN_OPTIONS_TYPES,
    compatibility_object_factory,
    CUSTOM,
    FULLY_CONNECTED,
    FULLY_CONNECTED_OPTIONS,
    FullyConnectedOptions,
    GELU,
    make_builder,
    make_codec,
    MUL,
    MUL_OPTIONS,
    PADDING_SAME,
    PADDING_VALID,
    PADDING_VALUES,
    SLICE,
    SLICE_OPTIONS,
    SliceOptions,
    static_contract,
    TENSOR_TYPES,
    TRANSPOSE_CONV,
    TRANSPOSE_CONV_OPTIONS,
    TransposeConvOptions,
)


class OptionalCompatibilityFusionTest(unittest.TestCase):
    """Check opt-in graph fusions that are intentionally absent from default O1."""

    def setUp(self) -> None:
        """Create schema-independent value services."""

        self.codec = make_codec()

    def test_transpose_conv_slice_folds_spatial_crop_into_padding(self) -> None:
        """Replace a representable crop with a new SAME TransposeConv."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1, 2, 2, 1],
        )
        output_shape = add_constant(
            builder,
            "pre_shape",
            [1, 5, 5, 1],
            INT32,
            np.int32,
        )
        filter_index = add_constant(
            builder,
            "filter",
            np.ones((1, 3, 3, 1), dtype=np.float32),
            FLOAT32,
            np.float32,
        )
        bias = add_constant(builder, "bias", [0.0], FLOAT32, np.float32)
        pre_output = builder.add_operator(
            TRANSPOSE_CONV,
            inputs=(output_shape, filter_index, source, bias),
            output_contracts=(static_contract((1, 5, 5, 1)),),
            output_names=("pre_crop",),
            builtin_options_type=TRANSPOSE_CONV_OPTIONS,
            builtin_options=TransposeConvOptions(
                padding=PADDING_VALID,
                strideH=2,
                strideW=2,
            ),
        )[0]
        begin = add_constant(
            builder,
            "begin",
            [0, 1, 1, 0],
            INT32,
            np.int32,
        )
        size = add_constant(
            builder,
            "size",
            [1, 3, 3, 1],
            INT32,
            np.int32,
        )
        final_output = builder.add_operator(
            SLICE,
            inputs=(pre_output, begin, size),
            output_contracts=(static_contract((1, 3, 3, 1)),),
            output_names=("cropped",),
            builtin_options_type=SLICE_OPTIONS,
            builtin_options=SliceOptions(),
        )[0]
        document.subgraph().outputs = [final_output]
        circle_pass = FuseTransposeConvSlicePass(
            builtin_codes=BUILTIN_CODES,
            builtin_options_types=BUILTIN_OPTIONS_TYPES,
            tensor_types=TENSOR_TYPES,
            padding_values=PADDING_VALUES,
            codec=self.codec,
            object_factory=compatibility_object_factory,
        )

        result = circle_pass.run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        replacement = document.subgraph().operators[1]
        self.assertEqual(
            operator_builtin_code(document.model, replacement),
            TRANSPOSE_CONV,
        )
        self.assertEqual(replacement.outputs, [final_output])
        self.assertEqual(replacement.builtinOptions.padding, PADDING_SAME)
        shape_value = self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=replacement.inputs[0],
        )
        np.testing.assert_array_equal(
            shape_value.data,
            np.asarray([1, 3, 3, 1], dtype=np.int32),
        )

    def test_transpose_conv_slice_preserves_an_exposed_pre_crop_output(self) -> None:
        """Skip fusion when the producer tensor is also a graph output."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1, 2, 2, 1],
        )
        output_shape = add_constant(
            builder,
            "pre_shape",
            [1, 5, 5, 1],
            INT32,
            np.int32,
        )
        filter_index = add_constant(
            builder,
            "filter",
            np.ones((1, 3, 3, 1), dtype=np.float32),
            FLOAT32,
            np.float32,
        )
        bias = add_constant(builder, "bias", [0.0], FLOAT32, np.float32)
        pre_output = builder.add_operator(
            TRANSPOSE_CONV,
            inputs=(output_shape, filter_index, source, bias),
            output_contracts=(static_contract((1, 5, 5, 1)),),
            output_names=("pre_crop",),
            builtin_options_type=TRANSPOSE_CONV_OPTIONS,
            builtin_options=TransposeConvOptions(
                padding=PADDING_VALID,
                strideH=2,
                strideW=2,
            ),
        )[0]
        begin = add_constant(
            builder,
            "begin",
            [0, 1, 1, 0],
            INT32,
            np.int32,
        )
        size = add_constant(
            builder,
            "size",
            [1, 3, 3, 1],
            INT32,
            np.int32,
        )
        final_output = builder.add_operator(
            SLICE,
            inputs=(pre_output, begin, size),
            output_contracts=(static_contract((1, 3, 3, 1)),),
            output_names=("cropped",),
            builtin_options_type=SLICE_OPTIONS,
            builtin_options=SliceOptions(),
        )[0]
        document.subgraph().outputs = [pre_output, final_output]
        circle_pass = FuseTransposeConvSlicePass(
            builtin_codes=BUILTIN_CODES,
            builtin_options_types=BUILTIN_OPTIONS_TYPES,
            tensor_types=TENSOR_TYPES,
            padding_values=PADDING_VALUES,
            codec=self.codec,
            object_factory=compatibility_object_factory,
        )

        result = circle_pass.run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(
            operator_builtin_code(document.model, document.subgraph().operators[1]),
            SLICE,
        )

    def test_legacy_fc_erf_pattern_becomes_exact_gelu_and_rescaled_fc(self) -> None:
        """Replace the legacy exact-GELU branch and double back-FC weights."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1, 2],
        )
        front_weight_values = np.asarray(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=np.float32,
        )
        front_bias_values = np.asarray([0.5, -0.5], dtype=np.float32)
        front_weights = add_constant(
            builder,
            "front_weights",
            front_weight_values,
            FLOAT32,
            np.float32,
        )
        front_bias = add_constant(
            builder,
            "front_bias",
            front_bias_values,
            FLOAT32,
            np.float32,
        )
        front_output = builder.add_operator(
            FULLY_CONNECTED,
            inputs=(source, front_weights, front_bias),
            output_contracts=(static_contract((1, 2)),),
            output_names=("front",),
            builtin_options_type=FULLY_CONNECTED_OPTIONS,
            builtin_options=FullyConnectedOptions(),
        )[0]
        factor = np.float32(sqrt(0.5))
        scaled_weights = add_constant(
            builder,
            "scaled_weights",
            front_weight_values * factor,
            FLOAT32,
            np.float32,
        )
        scaled_bias = add_constant(
            builder,
            "scaled_bias",
            front_bias_values * factor,
            FLOAT32,
            np.float32,
        )
        scaled_output = builder.add_operator(
            FULLY_CONNECTED,
            inputs=(source, scaled_weights, scaled_bias),
            output_contracts=(static_contract((1, 2)),),
            output_names=("scaled",),
            builtin_options_type=FULLY_CONNECTED_OPTIONS,
            builtin_options=FullyConnectedOptions(),
        )[0]
        erf_output = builder.add_operator(
            CUSTOM,
            custom_code="Erf",
            inputs=(scaled_output,),
            output_contracts=(static_contract((1, 2)),),
            output_names=("erf",),
        )[0]
        one = add_constant(builder, "one", 1.0, FLOAT32, np.float32)
        add_output = builder.add_operator(
            ADD,
            inputs=(erf_output, one),
            output_contracts=(static_contract((1, 2)),),
            output_names=("erf_plus_one",),
            builtin_options_type=ADD_OPTIONS,
            builtin_options=compatibility_object_factory("AddOptions"),
        )[0]
        mul_output = builder.add_operator(
            MUL,
            inputs=(front_output, add_output),
            output_contracts=(static_contract((1, 2)),),
            output_names=("gated",),
            builtin_options_type=MUL_OPTIONS,
            builtin_options=compatibility_object_factory("MulOptions"),
        )[0]
        back_weight_values = np.asarray([[2.0, -3.0]], dtype=np.float32)
        back_weights = add_constant(
            builder,
            "back_weights",
            back_weight_values,
            FLOAT32,
            np.float32,
        )
        back_bias = add_constant(builder, "back_bias", [0.25], FLOAT32, np.float32)
        final_output = builder.add_operator(
            FULLY_CONNECTED,
            inputs=(mul_output, back_weights, back_bias),
            output_contracts=(static_contract((1, 1)),),
            output_names=("output",),
            builtin_options_type=FULLY_CONNECTED_OPTIONS,
            builtin_options=FullyConnectedOptions(),
        )[0]
        document.subgraph().outputs = [final_output]
        circle_pass = FuseLegacyFCGeluFCPass(
            builtin_codes=BUILTIN_CODES,
            builtin_options_types=BUILTIN_OPTIONS_TYPES,
            tensor_types=TENSOR_TYPES,
            activation_none=0,
            codec=self.codec,
            object_factory=compatibility_object_factory,
        )

        result = circle_pass.run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        operators = document.subgraph().operators
        self.assertEqual(operator_builtin_code(document.model, operators[5]), GELU)
        self.assertFalse(operators[5].builtinOptions.approximate)
        self.assertEqual(
            operator_builtin_code(document.model, operators[6]),
            FULLY_CONNECTED,
        )
        self.assertEqual(operators[6].outputs, [final_output])
        rescaled = self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=operators[6].inputs[1],
        )
        np.testing.assert_array_equal(
            rescaled.data,
            back_weight_values * np.float32(2.0),
        )


if __name__ == "__main__":
    unittest.main()

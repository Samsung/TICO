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

from tico.circle.analysis import TensorContract
from tico.circle.passes import CirclePassContext, EliminateIdentityOpsPass
from tico.circle.passes.optimization._utils import operator_builtin_code
from tico.circle.value import TensorQuantization, TensorValue

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    FakeSignatureDef,
    FakeTensorMap,
    FLOAT32,
    INT32,
    INT8,
    make_empty_document,
)
from test.unit_test.circle.passes.optimization._fixture import (
    ADD,
    add_f32,
    add_i32,
    BinaryOptions,
    BUILTIN_CODES,
    CAST,
    CastOptions,
    make_builder,
    make_codec,
    optimization_object_factory,
    SLICE,
    SPLIT,
    SPLIT_V,
    SplitOptions,
    static_contract,
    STRIDED_SLICE,
    StridedSliceOptions,
)


class RemoveNoOpOperatorsTest(unittest.TestCase):
    """Check contract-preserving no-op elimination and rejection conditions."""

    def setUp(self) -> None:
        """Create a schema-independent codec for each test."""

        self.codec = make_codec()

    def _pass(self) -> EliminateIdentityOpsPass:
        """Create the no-op pass with fake schema identities."""

        return EliminateIdentityOpsPass(
            builtin_codes=BUILTIN_CODES,
            activation_none=0,
            codec=self.codec,
            object_factory=optimization_object_factory,
        )

    def test_add_zero_remaps_graph_and_signature_outputs(self) -> None:
        """Remove ADD and preserve graph and signature output bindings."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2],
        )
        document.subgraph().inputs = [source]
        zero = add_f32(builder, "zero", 0.0)
        output = builder.add_operator(
            ADD,
            inputs=(source, zero),
            output_contracts=(static_contract((2,)),),
            output_names=("output",),
            builtin_options=BinaryOptions(),
        )[0]
        document.subgraph().outputs = [output]
        document.model.signatureDefs = [
            FakeSignatureDef(outputs=[FakeTensorMap(name="output", tensorIndex=output)])
        ]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(document.subgraph().operators, [])
        self.assertEqual(document.subgraph().outputs, [source])
        self.assertEqual(
            document.model.signatureDefs[0].outputs[0].tensorIndex,
            source,
        )

    def test_quantized_add_zero_is_kept_conservative(self) -> None:
        """Keep quantized ADD because integer requantization may not be identity."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        quantization = TensorQuantization(
            scale=(0.25,),
            zero_point=(3,),
        )
        contract = TensorContract(
            tensor_type=INT8,
            shape=(2,),
            shape_signature=(2,),
            quantization=quantization,
        )
        source = builder.add_tensor("source", contract)
        document.subgraph().inputs = [source]
        zero = builder.add_constant(
            "zero",
            TensorValue.from_values(
                INT8,
                [3],
                dtype=np.int8,
                quantization=quantization,
            ),
        )
        output = builder.add_operator(
            ADD,
            inputs=(source, zero),
            output_contracts=(contract,),
            output_names=("output",),
            builtin_options=BinaryOptions(),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 1)

    def test_add_with_activation_or_nonzero_value_is_preserved(self) -> None:
        """Keep ADD when activation or arithmetic semantics remain observable."""

        cases = (
            (0.0, BinaryOptions(fusedActivationFunction=1)),
            (1.0, BinaryOptions()),
        )
        for value, options in cases:
            with self.subTest(value=value, options=options):
                document = make_empty_document()
                builder = make_builder(document, self.codec)
                source = add_runtime_tensor(
                    document,
                    subgraph_index=0,
                    name="source",
                    shape=[2],
                )
                document.subgraph().inputs = [source]
                constant = add_f32(builder, "constant", value)
                output = builder.add_operator(
                    ADD,
                    inputs=(source, constant),
                    output_contracts=(static_contract((2,)),),
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

    def test_same_type_cast_is_removed_but_real_cast_is_preserved(self) -> None:
        """Remove only CAST operations with identical complete contracts."""

        for output_type, expected_modified in ((FLOAT32, True), (INT32, False)):
            with self.subTest(output_type=output_type):
                document = make_empty_document()
                builder = make_builder(document, self.codec)
                source = add_runtime_tensor(
                    document,
                    subgraph_index=0,
                    name="source",
                    shape=[2],
                )
                document.subgraph().inputs = [source]
                output = builder.add_operator(
                    CAST,
                    inputs=(source,),
                    output_contracts=(static_contract((2,), output_type),),
                    output_names=("output",),
                    builtin_options=CastOptions(
                        inDataType=FLOAT32,
                        outDataType=output_type,
                    ),
                )[0]
                document.subgraph().outputs = [output]

                result = self._pass().run(
                    document,
                    CirclePassContext(verify_after_each_pass=False),
                )

                self.assertEqual(result.modified, expected_modified)
                self.assertEqual(
                    len(document.subgraph().operators),
                    0 if expected_modified else 1,
                )

    def test_dynamic_full_slice_requires_inferred_size(self) -> None:
        """Avoid treating a concrete placeholder size as a dynamic full slice."""

        for size, expected_modified in (([-1, 3], True), ([1, 3], False)):
            with self.subTest(size=size):
                document = make_empty_document()
                builder = make_builder(document, self.codec)
                contract = TensorContract(
                    tensor_type=FLOAT32,
                    shape=(1, 3),
                    shape_signature=(-1, 3),
                )
                source = builder.add_tensor("source", contract)
                document.subgraph().inputs = [source]
                begin = add_i32(builder, "begin", [0, 0])
                size_tensor = add_i32(builder, "size", size)
                output = builder.add_operator(
                    SLICE,
                    inputs=(source, begin, size_tensor),
                    output_contracts=(contract,),
                    output_names=("output",),
                )[0]
                document.subgraph().outputs = [output]

                result = self._pass().run(
                    document,
                    CirclePassContext(verify_after_each_pass=False),
                )

                self.assertEqual(result.modified, expected_modified)

    def test_full_slice_is_removed_but_partial_slice_is_preserved(self) -> None:
        """Recognize full-range SLICE using both concrete and inferred sizes."""

        for size, expected_modified in (([-1, 3], True), ([1, 3], False)):
            with self.subTest(size=size):
                document = make_empty_document()
                builder = make_builder(document, self.codec)
                source = add_runtime_tensor(
                    document,
                    subgraph_index=0,
                    name="source",
                    shape=[2, 3],
                )
                document.subgraph().inputs = [source]
                begin = add_i32(builder, "begin", [0, 0])
                size_tensor = add_i32(builder, "size", size)
                output_shape = (2, 3) if expected_modified else (1, 3)
                output = builder.add_operator(
                    SLICE,
                    inputs=(source, begin, size_tensor),
                    output_contracts=(static_contract(output_shape),),
                    output_names=("output",),
                )[0]
                document.subgraph().outputs = [output]

                result = self._pass().run(
                    document,
                    CirclePassContext(verify_after_each_pass=False),
                )

                self.assertEqual(result.modified, expected_modified)

    def test_identity_strided_slice_is_removed(self) -> None:
        """Remove a rank-preserving full STRIDED_SLICE."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3],
        )
        document.subgraph().inputs = [source]
        begin = add_i32(builder, "begin", [0, 0])
        end = add_i32(builder, "end", [2, 3])
        strides = add_i32(builder, "strides", [1, 1])
        output = builder.add_operator(
            STRIDED_SLICE,
            inputs=(source, begin, end, strides),
            output_contracts=(static_contract((2, 3)),),
            output_names=("output",),
            builtin_options=StridedSliceOptions(),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(document.subgraph().operators, [])

    def test_single_output_split_and_split_v_are_removed(self) -> None:
        """Remove one-output SPLIT and SPLIT_V forms that return the full input."""

        for code in (SPLIT, SPLIT_V):
            with self.subTest(code=code):
                document = make_empty_document()
                builder = make_builder(document, self.codec)
                source = add_runtime_tensor(
                    document,
                    subgraph_index=0,
                    name="source",
                    shape=[4],
                )
                document.subgraph().inputs = [source]
                axis = add_i32(builder, "axis", 0)
                inputs: tuple[int, ...]
                if code == SPLIT:
                    inputs = (axis, source)
                else:
                    size = add_i32(builder, "size", [-1])
                    inputs = (source, size, axis)
                output = builder.add_operator(
                    code,
                    inputs=inputs,
                    output_contracts=(static_contract((4,)),),
                    output_names=("output",),
                    builtin_options=SplitOptions(numSplits=1),
                )[0]
                document.subgraph().outputs = [output]

                result = self._pass().run(
                    document,
                    CirclePassContext(verify_after_each_pass=False),
                )

                self.assertTrue(result.modified)
                self.assertEqual(document.subgraph().operators, [])
                self.assertEqual(document.subgraph().outputs, [source])

    def test_multi_output_split_is_preserved(self) -> None:
        """Keep a real multi-output split operation."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[4],
        )
        document.subgraph().inputs = [source]
        axis = add_i32(builder, "axis", 0)
        outputs = builder.add_operator(
            SPLIT,
            inputs=(axis, source),
            output_contracts=(static_contract((2,)), static_contract((2,))),
            output_names=("left", "right"),
            builtin_options=SplitOptions(numSplits=2),
        )
        document.subgraph().outputs = list(outputs)

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(
            operator_builtin_code(document.model, document.subgraph().operators[0]),
            SPLIT,
        )


if __name__ == "__main__":
    unittest.main()

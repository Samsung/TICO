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

from tico.circle.passes import CirclePassContext
from tico.circle.passes.optimization.fold import (
    ConstantFoldingProfile,
    FoldConstantsPass,
)
from tico.circle.passes.optimization.fold.evaluators import (
    ConstantEvaluatorRegistry,
    register_heavy_constant_evaluators,
)
from tico.circle.value import TensorQuantization

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    FakeBuffer,
    FakeTensor,
    FLOAT32,
    INT32,
    INT8,
    make_empty_document,
)
from test.unit_test.circle.passes.optimization._compatibility_fixture import (
    add_constant,
    BUILTIN_CODES,
    compatibility_object_factory,
    DENSIFY,
    DEPTHWISE_CONV_2D,
    DepthwiseConv2DOptions,
    DEQUANTIZE,
    DimensionMetadata,
    FLOAT16,
    FULLY_CONNECTED,
    FullyConnectedOptions,
    INT64,
    make_builder,
    make_codec,
    PADDING_VALUES,
    SPARSE_TO_DENSE,
    SparseIndexVector,
    SparsityParameters,
    static_contract,
)


class HeavyConstantFoldingTest(unittest.TestCase):
    """Check expensive constant evaluators and sparse compatibility coverage."""

    def setUp(self) -> None:
        """Create an explicit heavy registry without generated schema imports."""

        self.codec = make_codec()
        self.registry = ConstantEvaluatorRegistry()
        register_heavy_constant_evaluators(
            self.registry,
            builtin_codes=BUILTIN_CODES,
            padding_values=PADDING_VALUES,
        )

    def _pass(self) -> FoldConstantsPass:
        """Create a heavy fold pass using fake Object API tables."""

        return FoldConstantsPass(
            profile=ConstantFoldingProfile.HEAVY,
            evaluator_registry=self.registry,
            codec=self.codec,
            object_factory=compatibility_object_factory,
        )

    def _run_and_decode(self, document, output_index):
        """Run the pass and decode one preserved output tensor."""

        document.subgraph().outputs = [output_index]
        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )
        value = self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=document.subgraph().outputs[0],
        )
        return result, value

    def test_dequantize_supports_per_axis_affine_parameters(self) -> None:
        """Apply serialized per-axis scale and zero-point vectors exactly."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        quantization = TensorQuantization(
            scale=(0.5, 1.0),
            zero_point=(0, -1),
            quantized_dimension=1,
        )
        source = add_constant(
            builder,
            "source",
            [[0, 1], [2, 3]],
            INT8,
            np.int8,
            quantization=quantization,
        )
        output = builder.add_operator(
            DEQUANTIZE,
            inputs=(source,),
            output_contracts=(static_contract((2, 2)),),
            output_names=("dequantized",),
        )[0]

        result, value = self._run_and_decode(document, output)

        self.assertTrue(result.modified)
        np.testing.assert_array_equal(
            value.data,
            np.asarray([[0.0, 2.0], [1.0, 4.0]], dtype=np.float32),
        )

    def test_dequantize_converts_float16_storage(self) -> None:
        """Fold FLOAT16-to-FLOAT32 conversion without affine metadata."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_constant(
            builder,
            "source",
            [1.5, -2.25],
            FLOAT16,
            np.float16,
        )
        output = builder.add_operator(
            DEQUANTIZE,
            inputs=(source,),
            output_contracts=(static_contract((2,)),),
            output_names=("converted",),
        )[0]

        result, value = self._run_and_decode(document, output)

        self.assertTrue(result.modified)
        np.testing.assert_array_equal(
            value.data,
            np.asarray([1.5, -2.25], dtype=np.float32),
        )

    def test_fully_connected_folds_constant_float32_inputs(self) -> None:
        """Evaluate default-format FC including its constant bias."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_constant(builder, "source", [[1.0, 2.0]], FLOAT32, np.float32)
        weights = add_constant(
            builder,
            "weights",
            [[3.0, 4.0], [5.0, 6.0]],
            FLOAT32,
            np.float32,
        )
        bias = add_constant(builder, "bias", [1.0, -1.0], FLOAT32, np.float32)
        output = builder.add_operator(
            FULLY_CONNECTED,
            inputs=(source, weights, bias),
            output_contracts=(static_contract((1, 2)),),
            output_names=("fc",),
            builtin_options=FullyConnectedOptions(),
        )[0]

        result, value = self._run_and_decode(document, output)

        self.assertTrue(result.modified)
        np.testing.assert_array_equal(
            value.data,
            np.asarray([[12.0, 16.0]], dtype=np.float32),
        )

    def test_depthwise_conv2d_folds_static_nhwc_float32(self) -> None:
        """Evaluate one depthwise filter with explicit bias and VALID padding."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_constant(
            builder,
            "source",
            [[[[1.0], [2.0]], [[3.0], [4.0]]]],
            FLOAT32,
            np.float32,
        )
        filter_index = add_constant(
            builder,
            "filter",
            [[[[2.0]]]],
            FLOAT32,
            np.float32,
        )
        bias = add_constant(builder, "bias", [1.0], FLOAT32, np.float32)
        output = builder.add_operator(
            DEPTHWISE_CONV_2D,
            inputs=(source, filter_index, bias),
            output_contracts=(static_contract((1, 2, 2, 1)),),
            output_names=("depthwise",),
            builtin_options=DepthwiseConv2DOptions(),
        )[0]

        result, value = self._run_and_decode(document, output)

        self.assertTrue(result.modified)
        np.testing.assert_array_equal(
            value.data,
            np.asarray([[[[3.0], [5.0]], [[7.0], [9.0]]]], dtype=np.float32),
        )

    def test_densify_expands_unblocked_sparse_csr_storage(self) -> None:
        """Decode traversal metadata and place sparse values at dense coordinates."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        payload = np.asarray([1.0, 2.0], dtype=np.float32).view(np.uint8).copy()
        document.model.buffers.append(FakeBuffer(data=payload))
        sparsity = SparsityParameters(
            traversalOrder=[0, 1],
            dimMetadata=[
                DimensionMetadata(format=0, denseSize=2),
                DimensionMetadata(
                    format=1,
                    arraySegments=SparseIndexVector([0, 1, 2]),
                    arrayIndices=SparseIndexVector([0, 2]),
                ),
            ],
        )
        sparse_tensor = FakeTensor(
            name="sparse",
            buffer=len(document.model.buffers) - 1,
            shape=[2, 3],
            shapeSignature=[2, 3],
            type=FLOAT32,
            sparsity=sparsity,
        )
        document.subgraph().tensors.append(sparse_tensor)
        sparse_index = len(document.subgraph().tensors) - 1
        output = builder.add_operator(
            DENSIFY,
            inputs=(sparse_index,),
            output_contracts=(static_contract((2, 3)),),
            output_names=("dense",),
        )[0]

        result, value = self._run_and_decode(document, output)

        self.assertTrue(result.modified)
        np.testing.assert_array_equal(
            value.data,
            np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float32),
        )

    def test_sparse_to_dense_folds_empty_indices_to_default_fill(self) -> None:
        """Materialize the restricted empty-indices default-value pattern."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        indices = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="indices",
            shape=[0],
            tensor_type=INT32,
        )
        output_shape = add_constant(
            builder,
            "output_shape",
            [2, 3],
            INT32,
            np.int32,
        )
        values = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="values",
            shape=[0],
            tensor_type=INT32,
        )
        default = add_constant(builder, "default", 7, INT32, np.int32)
        output = builder.add_operator(
            SPARSE_TO_DENSE,
            inputs=(indices, output_shape, values, default),
            output_contracts=(static_contract((2, 3), INT32),),
            output_names=("filled",),
        )[0]

        result, value = self._run_and_decode(document, output)

        self.assertTrue(result.modified)
        np.testing.assert_array_equal(
            value.data,
            np.full((2, 3), 7, dtype=np.int32),
        )

    def test_densify_does_not_fold_a_sparse_graph_input(self) -> None:
        """Preserve runtime input semantics even when sparse default bytes exist."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        payload = np.asarray([1.0], dtype=np.float32).view(np.uint8).copy()
        document.model.buffers.append(FakeBuffer(data=payload))
        sparsity = SparsityParameters(
            traversalOrder=[0],
            dimMetadata=[
                DimensionMetadata(
                    format=1,
                    arraySegments=SparseIndexVector([0, 1]),
                    arrayIndices=SparseIndexVector([1]),
                )
            ],
        )
        document.subgraph().tensors.append(
            FakeTensor(
                name="runtime_sparse",
                buffer=len(document.model.buffers) - 1,
                shape=[2],
                shapeSignature=[2],
                type=FLOAT32,
                sparsity=sparsity,
            )
        )
        sparse_index = len(document.subgraph().tensors) - 1
        document.subgraph().inputs = [sparse_index]
        output = builder.add_operator(
            DENSIFY,
            inputs=(sparse_index,),
            output_contracts=(static_contract((2,)),),
            output_names=("dense",),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 1)

    def test_sparse_to_dense_rejects_mismatched_index_and_shape_types(self) -> None:
        """Do not fold a graph that the builtin would reject as malformed."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        indices = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="indices",
            shape=[0],
            tensor_type=INT64,
        )
        output_shape = add_constant(
            builder,
            "output_shape",
            [2],
            INT32,
            np.int32,
        )
        values = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="values",
            shape=[0],
            tensor_type=INT32,
        )
        default = add_constant(builder, "default", 0, INT32, np.int32)
        output = builder.add_operator(
            SPARSE_TO_DENSE,
            inputs=(indices, output_shape, values, default),
            output_contracts=(static_contract((2,), INT32),),
            output_names=("filled",),
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

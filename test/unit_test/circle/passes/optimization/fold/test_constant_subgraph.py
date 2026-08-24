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

import operator
import unittest
from dataclasses import dataclass

import numpy as np

from tico.circle.analysis import TensorContract
from tico.circle.builder import CircleBuilder
from tico.circle.passes import CirclePassContext, FoldConstantsPass
from tico.circle.passes.optimization.fold import (
    ConstantEvaluation,
    ConstantEvaluator,
    ConstantEvaluatorRegistry,
    ConstantFoldPolicy,
)
from tico.circle.passes.optimization.fold.evaluators import (
    BinaryElementwiseEvaluator,
    CastEvaluator,
    GatherEvaluator,
    ReshapeEvaluator,
    ShapeEvaluator,
    SqueezeEvaluator,
)
from tico.circle.value import TensorQuantization, TensorValue, TensorValueCodec

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    fake_object_factory,
    FakeSignatureDef,
    FakeTensorMap,
    FLOAT32,
    INT32,
    INT8,
    make_empty_document,
    make_registry,
)

ADD = 10
MUL = 11
CAST = 12
RESHAPE = 13
SHAPE = 14
SQUEEZE = 15
GATHER = 16
UNSUPPORTED = 17
MULTI_OUTPUT = 18


@dataclass
class BinaryOptions:
    """Provide fused-activation and INT16 scaling fields for binary tests."""

    fusedActivationFunction: int = 0
    potScaleInt16: bool = False


@dataclass
class CastOptions:
    """Provide source and target tensor types for CAST tests."""

    inDataType: int = FLOAT32
    outDataType: int = INT32


@dataclass
class ReshapeOptions:
    """Provide the static fallback target used by RESHAPE."""

    newShape: list[int]


@dataclass
class ShapeOptions:
    """Provide the output tensor type used by SHAPE."""

    outType: int = INT32


@dataclass
class SqueezeOptions:
    """Provide dimensions removed by SQUEEZE."""

    squeezeDims: list[int]


@dataclass
class GatherOptions:
    """Provide axis and batch dimensions used by GATHER."""

    axis: int = 0
    batchDims: int = 0


class TwoOutputEvaluator(ConstantEvaluator):
    """Produce two outputs to validate generic multi-output replacement."""

    def constant_input_positions(self, context):
        """Require the synthetic source tensor to be constant."""

        return (0,)

    def evaluate(self, context):
        """Return source-plus-one and source-plus-two outputs."""

        source = context.input_value(0).data
        outputs = []
        for offset, contract in zip((1.0, 2.0), context.output_contracts):
            outputs.append(
                TensorValue(
                    tensor_type=contract.tensor_type,
                    shape=contract.shape,
                    data=source + np.float32(offset),
                    quantization=contract.quantization,
                )
            )
        return ConstantEvaluation(tuple(outputs))


class FailSecondBufferFactory:
    """Raise while creating the second output buffer for rollback tests."""

    def __init__(self) -> None:
        """Initialize the count of Buffer table creation requests."""

        self.buffer_requests = 0

    def __call__(self, table_name: str):
        """Delegate Object API creation and fail on the second Buffer request."""

        if table_name == "Buffer":
            self.buffer_requests += 1
            if self.buffer_requests == 2:
                raise RuntimeError("synthetic buffer allocation failure")
        return fake_object_factory(table_name)


class ConstantSubgraphFoldTest(unittest.TestCase):
    """Check first-stage constant evaluators, scheduling, budgets, and cleanup."""

    def setUp(self):
        """Create schema-independent value and evaluator registries."""

        self.codec = TensorValueCodec(make_registry())
        self.registry = ConstantEvaluatorRegistry(
            (
                (ADD, BinaryElementwiseEvaluator("ADD", operator.add)),
                (MUL, BinaryElementwiseEvaluator("MUL", operator.mul)),
                (CAST, CastEvaluator()),
                (RESHAPE, ReshapeEvaluator()),
                (SHAPE, ShapeEvaluator()),
                (SQUEEZE, SqueezeEvaluator()),
                (GATHER, GatherEvaluator()),
            )
        )

    def _pass(self, *, policy=None, registry=None):
        """Create a pass using fake Object API tables and explicit registries."""

        return FoldConstantsPass(
            policy=policy,
            evaluator_registry=registry or self.registry,
            codec=self.codec,
            object_factory=fake_object_factory,
        )

    def _builder(self, document):
        """Create a CircleBuilder sharing the test codec and fake factory."""

        return CircleBuilder(
            document,
            codec=self.codec,
            object_factory=fake_object_factory,
        )

    def _constant(self, builder, name, values, tensor_type, dtype):
        """Add one typed constant and return its tensor index."""

        return builder.add_constant(
            name,
            TensorValue.from_values(
                tensor_type,
                values,
                dtype=dtype,
            ),
        )

    def test_add_mul_chain_reaches_fixed_point(self):
        """Fold a newly constant successor after the predecessor is replaced."""

        document = make_empty_document()
        builder = self._builder(document)
        lhs = self._constant(builder, "lhs", [1.0, 2.0], FLOAT32, np.float32)
        rhs = self._constant(builder, "rhs", [3.0, 4.0], FLOAT32, np.float32)
        scale = self._constant(builder, "scale", 2.0, FLOAT32, np.float32)
        vector_contract = TensorContract(FLOAT32, (2,))
        added = builder.add_operator(
            ADD,
            inputs=(lhs, rhs),
            output_contracts=(vector_contract,),
            output_names=("added",),
            builtin_options=BinaryOptions(),
        )[0]
        multiplied = builder.add_operator(
            MUL,
            inputs=(added, scale),
            output_contracts=(vector_contract,),
            output_names=("multiplied",),
            builtin_options=BinaryOptions(),
        )[0]
        document.subgraph().outputs = [multiplied]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(len(document.subgraph().operators), 0)
        output = self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=document.subgraph().outputs[0],
        )
        np.testing.assert_array_equal(output.data, np.array([8.0, 12.0], np.float32))
        self.assertEqual(
            sum("CONSTANT_FOLDED" in diagnostic for diagnostic in result.diagnostics),
            2,
        )

    def test_cast_uses_truncation_and_rejects_overflow(self):
        """Fold representable float-to-int casts and keep unsafe casts unchanged."""

        document = make_empty_document()
        builder = self._builder(document)
        source = self._constant(
            builder,
            "source",
            [1.8, -2.2],
            FLOAT32,
            np.float32,
        )
        output = builder.add_operator(
            CAST,
            inputs=(source,),
            output_contracts=(TensorContract(INT32, (2,)),),
            output_names=("cast",),
            builtin_options=CastOptions(),
        )[0]
        document.subgraph().outputs = [output]
        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )
        self.assertTrue(result.modified)
        value = self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=document.subgraph().outputs[0],
        )
        np.testing.assert_array_equal(value.data, np.array([1, -2], np.int32))

        unsafe = make_empty_document()
        unsafe_builder = self._builder(unsafe)
        large = self._constant(
            unsafe_builder,
            "large",
            [1.0e20],
            FLOAT32,
            np.float32,
        )
        unsafe_output = unsafe_builder.add_operator(
            CAST,
            inputs=(large,),
            output_contracts=(TensorContract(INT32, (1,)),),
            output_names=("unsafe_cast",),
            builtin_options=CastOptions(),
        )[0]
        unsafe.subgraph().outputs = [unsafe_output]
        unsafe_result = self._pass().run(
            unsafe,
            CirclePassContext(verify_after_each_pass=False),
        )
        self.assertFalse(unsafe_result.modified)
        self.assertEqual(len(unsafe.subgraph().operators), 1)

        boundary = make_empty_document()
        boundary_builder = self._builder(boundary)
        rounded_above_maximum = self._constant(
            boundary_builder,
            "rounded_above_maximum",
            [np.float32(2**31)],
            FLOAT32,
            np.float32,
        )
        boundary_output = boundary_builder.add_operator(
            CAST,
            inputs=(rounded_above_maximum,),
            output_contracts=(TensorContract(INT32, (1,)),),
            output_names=("boundary_cast",),
            builtin_options=CastOptions(),
        )[0]
        boundary.subgraph().outputs = [boundary_output]

        boundary_result = self._pass().run(
            boundary,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(boundary_result.modified)
        self.assertEqual(len(boundary.subgraph().operators), 1)

    def test_reshape_and_squeeze_fold_as_dense_views(self):
        """Fold view-only shape operations while preserving logical values."""

        document = make_empty_document()
        builder = self._builder(document)
        source = self._constant(
            builder,
            "source",
            [[[1.0], [2.0]]],
            FLOAT32,
            np.float32,
        )
        squeezed = builder.add_operator(
            SQUEEZE,
            inputs=(source,),
            output_contracts=(TensorContract(FLOAT32, (2,)),),
            output_names=("squeezed",),
            builtin_options=SqueezeOptions([0, 2]),
        )[0]
        target = self._constant(
            builder,
            "target",
            [1, 2],
            INT32,
            np.int32,
        )
        reshaped = builder.add_operator(
            RESHAPE,
            inputs=(squeezed, target),
            output_contracts=(TensorContract(FLOAT32, (1, 2)),),
            output_names=("reshaped",),
            builtin_options=ReshapeOptions([1, 2]),
        )[0]
        document.subgraph().outputs = [reshaped]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(len(document.subgraph().operators), 0)
        value = self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=document.subgraph().outputs[0],
        )
        np.testing.assert_array_equal(value.data, np.array([[1.0, 2.0]], np.float32))

    def test_shape_folds_runtime_metadata_and_dce_removes_producer_chain(self):
        """Fold static SHAPE and remove the producer made dead by metadata folding."""

        document = make_empty_document()
        builder = self._builder(document)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3],
        )
        document.subgraph().inputs = [source]
        produced = builder.add_operator(
            UNSUPPORTED,
            inputs=(source,),
            output_contracts=(TensorContract(FLOAT32, (2, 3)),),
            output_names=("produced",),
        )[0]
        shape = builder.add_operator(
            SHAPE,
            inputs=(produced,),
            output_contracts=(TensorContract(INT32, (2,)),),
            output_names=("shape",),
            builtin_options=ShapeOptions(),
        )[0]
        document.subgraph().outputs = [shape]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(len(document.subgraph().operators), 0)
        self.assertEqual(document.subgraph().inputs, [])
        value = self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=document.subgraph().outputs[0],
        )
        np.testing.assert_array_equal(value.data, np.array([2, 3], np.int32))
        self.assertTrue(any("removed operators" in item for item in result.diagnostics))

    def test_shape_prunes_direct_input_after_removing_the_only_operator(self):
        """Prune a direct SHAPE input after folding leaves an empty graph."""

        document = make_empty_document()
        builder = self._builder(document)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3],
        )
        document.subgraph().inputs = [source]
        output = builder.add_operator(
            SHAPE,
            inputs=(source,),
            output_contracts=(TensorContract(INT32, (2,)),),
            output_names=("shape",),
            builtin_options=ShapeOptions(),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(document.subgraph().operators, [])
        self.assertEqual(document.subgraph().inputs, [])
        value = self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=document.subgraph().outputs[0],
        )
        np.testing.assert_array_equal(value.data, np.array([2, 3], np.int32))
        self.assertTrue(
            any("removed graph inputs" in item for item in result.diagnostics)
        )

    def test_shape_keeps_an_unused_input_referenced_by_a_signature(self):
        """Preserve signature validity when static SHAPE removes data dependence."""

        document = make_empty_document()
        builder = self._builder(document)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[2, 3],
        )
        document.subgraph().inputs = [source]
        output = builder.add_operator(
            SHAPE,
            inputs=(source,),
            output_contracts=(TensorContract(INT32, (2,)),),
            output_names=("shape",),
            builtin_options=ShapeOptions(),
        )[0]
        document.subgraph().outputs = [output]
        document.model.signatureDefs = [
            FakeSignatureDef(
                signatureKey="serving_default",
                subgraphIndex=0,
                inputs=[FakeTensorMap("source", source)],
                outputs=[FakeTensorMap("shape", output)],
            )
        ]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(document.subgraph().operators, [])
        self.assertEqual(document.subgraph().inputs, [source])
        self.assertEqual(
            document.model.signatureDefs[0].inputs[0].tensorIndex,
            source,
        )

    def test_shape_does_not_fold_dynamic_input_contract(self):
        """Keep SHAPE when a serialized dimension is dynamic."""

        document = make_empty_document()
        builder = self._builder(document)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1, 3],
        )
        document.subgraph().tensors[source].shapeSignature = [-1, 3]
        document.subgraph().inputs = [source]
        output = builder.add_operator(
            SHAPE,
            inputs=(source,),
            output_contracts=(TensorContract(INT32, (2,)),),
            output_names=("shape",),
            builtin_options=ShapeOptions(),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 1)

    def test_zero_element_output_is_not_materialized_as_an_empty_constant(self):
        """Keep operators whose empty payload cannot prove constant ownership."""

        document = make_empty_document()
        builder = self._builder(document)
        scalar = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="scalar",
            shape=[],
        )
        document.subgraph().inputs = [scalar]
        output = builder.add_operator(
            SHAPE,
            inputs=(scalar,),
            output_contracts=(TensorContract(INT32, (0,)),),
            output_names=("scalar_shape",),
            builtin_options=ShapeOptions(),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 1)

    def test_gather_folds_in_range_indices(self):
        """Fold an axis gather and preserve the serialized output shape."""

        document = make_empty_document()
        builder = self._builder(document)
        params = self._constant(
            builder,
            "params",
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            FLOAT32,
            np.float32,
        )
        indices = self._constant(
            builder,
            "indices",
            [2, 0],
            INT32,
            np.int32,
        )
        output = builder.add_operator(
            GATHER,
            inputs=(params, indices),
            output_contracts=(TensorContract(FLOAT32, (2, 2)),),
            output_names=("gathered",),
            builtin_options=GatherOptions(axis=0),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        value = self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=document.subgraph().outputs[0],
        )
        np.testing.assert_array_equal(
            value.data,
            np.array([[5.0, 6.0], [1.0, 2.0]], np.float32),
        )

    def test_nonzero_fused_activation_prevents_binary_folding(self):
        """Keep binary operators whose fused activation changes the result."""

        document = make_empty_document()
        builder = self._builder(document)
        lhs = self._constant(builder, "lhs", [-1.0], FLOAT32, np.float32)
        rhs = self._constant(builder, "rhs", [0.0], FLOAT32, np.float32)
        output = builder.add_operator(
            ADD,
            inputs=(lhs, rhs),
            output_contracts=(TensorContract(FLOAT32, (1,)),),
            output_names=("relu_add",),
            builtin_options=BinaryOptions(fusedActivationFunction=1),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 1)

    def test_integer_binary_rejects_fixed_width_overflow(self):
        """Avoid replacing integer kernels with NumPy wraparound behavior."""

        document = make_empty_document()
        builder = self._builder(document)
        lhs = self._constant(builder, "lhs", [127], INT8, np.int8)
        rhs = self._constant(builder, "rhs", [1], INT8, np.int8)
        output = builder.add_operator(
            ADD,
            inputs=(lhs, rhs),
            output_contracts=(TensorContract(INT8, (1,)),),
            output_names=("overflow",),
            builtin_options=BinaryOptions(),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 1)

    def test_graph_input_is_not_treated_as_constant_from_a_default_buffer(self):
        """Preserve runtime input semantics even when a tensor carries bytes."""

        document = make_empty_document()
        builder = self._builder(document)
        runtime_input = self._constant(
            builder,
            "runtime_input",
            [1.0],
            FLOAT32,
            np.float32,
        )
        rhs = self._constant(builder, "rhs", [2.0], FLOAT32, np.float32)
        document.subgraph().inputs = [runtime_input]
        output = builder.add_operator(
            ADD,
            inputs=(runtime_input, rhs),
            output_contracts=(TensorContract(FLOAT32, (1,)),),
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

    def test_output_budget_skips_candidate_without_mutation(self):
        """Report a budget skip and leave graph references unchanged."""

        document = make_empty_document()
        builder = self._builder(document)
        lhs = self._constant(builder, "lhs", [1.0, 2.0], FLOAT32, np.float32)
        rhs = self._constant(builder, "rhs", [3.0, 4.0], FLOAT32, np.float32)
        output = builder.add_operator(
            ADD,
            inputs=(lhs, rhs),
            output_contracts=(TensorContract(FLOAT32, (2,)),),
            output_names=("added",),
            builtin_options=BinaryOptions(),
        )[0]
        document.subgraph().outputs = [output]
        policy = ConstantFoldPolicy(maximum_output_bytes=4)

        result = self._pass(policy=policy).run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(document.subgraph().outputs, [output])
        self.assertEqual(len(document.subgraph().operators), 1)
        self.assertTrue(
            any("CONSTANT_FOLD_BUDGET" in item for item in result.diagnostics)
        )

    def test_total_output_budget_stops_a_constant_chain_conservatively(self):
        """Fold within the total budget and preserve the first over-budget operator."""

        document = make_empty_document()
        builder = self._builder(document)
        lhs = self._constant(builder, "lhs", [1.0, 2.0], FLOAT32, np.float32)
        rhs = self._constant(builder, "rhs", [3.0, 4.0], FLOAT32, np.float32)
        scale = self._constant(builder, "scale", 2.0, FLOAT32, np.float32)
        contract = TensorContract(FLOAT32, (2,))
        added = builder.add_operator(
            ADD,
            inputs=(lhs, rhs),
            output_contracts=(contract,),
            output_names=("added",),
            builtin_options=BinaryOptions(),
        )[0]
        output = builder.add_operator(
            MUL,
            inputs=(added, scale),
            output_contracts=(contract,),
            output_names=("output",),
            builtin_options=BinaryOptions(),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass(
            policy=ConstantFoldPolicy(maximum_total_output_bytes=8)
        ).run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(len(document.subgraph().operators), 1)
        self.assertEqual(document.subgraph().operators[0].inputs[0], added)
        folded_input = self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=added,
        )
        np.testing.assert_array_equal(
            folded_input.data,
            np.array([4.0, 6.0], np.float32),
        )
        self.assertTrue(
            any("CONSTANT_FOLD_BUDGET" in item for item in result.diagnostics)
        )

    def test_quantized_binary_is_skipped_but_quantized_reshape_is_folded(self):
        """Keep arithmetic conservative while allowing exact quantized views."""

        quantization = TensorQuantization(
            scale=(0.25,),
            zero_point=(0,),
            quantized_dimension=0,
        )
        document = make_empty_document()
        builder = self._builder(document)
        lhs = builder.add_constant(
            "lhs",
            TensorValue.from_values(
                INT8,
                [1, 2],
                dtype=np.int8,
                quantization=quantization,
            ),
        )
        rhs = builder.add_constant(
            "rhs",
            TensorValue.from_values(
                INT8,
                [3, 4],
                dtype=np.int8,
                quantization=quantization,
            ),
        )
        quantized_contract = TensorContract(
            INT8,
            (2,),
            quantization=quantization,
        )
        added = builder.add_operator(
            ADD,
            inputs=(lhs, rhs),
            output_contracts=(quantized_contract,),
            output_names=("quantized_add",),
            builtin_options=BinaryOptions(),
        )[0]
        target = self._constant(builder, "target", [1, 2], INT32, np.int32)
        reshaped = builder.add_operator(
            RESHAPE,
            inputs=(lhs, target),
            output_contracts=(TensorContract(INT8, (1, 2), quantization=quantization),),
            output_names=("quantized_reshape",),
            builtin_options=ReshapeOptions([1, 2]),
        )[0]
        document.subgraph().outputs = [added, reshaped]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(len(document.subgraph().operators), 1)
        self.assertEqual(document.subgraph().operators[0].outputs, [added])
        value = self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=reshaped,
        )
        self.assertEqual(value.quantization, quantization)
        np.testing.assert_array_equal(value.data, np.array([[1, 2]], np.int8))

    def test_custom_multi_output_evaluator_preserves_outputs_and_signature(self):
        """Materialize every output position without changing interface indices."""

        document = make_empty_document()
        builder = self._builder(document)
        source = self._constant(builder, "source", [2.0], FLOAT32, np.float32)
        contract = TensorContract(FLOAT32, (1,))
        first, second = builder.add_operator(
            MULTI_OUTPUT,
            inputs=(source,),
            output_contracts=(contract, contract),
            output_names=("first", "second"),
        )
        document.subgraph().outputs = [first, second]
        document.model.signatureDefs = [
            FakeSignatureDef(
                outputs=[
                    FakeTensorMap(name="first", tensorIndex=first),
                    FakeTensorMap(name="second", tensorIndex=second),
                ]
            )
        ]
        registry = self.registry.copy()
        registry.register(MULTI_OUTPUT, TwoOutputEvaluator())
        source_outputs = list(document.subgraph().outputs)
        source_signature_outputs = [
            mapping.tensorIndex for mapping in document.model.signatureDefs[0].outputs
        ]

        result = self._pass(registry=registry).run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(len(document.subgraph().operators), 0)
        self.assertEqual(document.subgraph().outputs, source_outputs)
        output_values = [
            self.codec.decode_tensor(
                document.model,
                subgraph_index=0,
                tensor_index=tensor_index,
            ).data.item()
            for tensor_index in document.subgraph().outputs
        ]
        self.assertEqual(output_values, [3.0, 4.0])
        self.assertEqual(
            [
                mapping.tensorIndex
                for mapping in document.model.signatureDefs[0].outputs
            ],
            document.subgraph().outputs,
        )
        self.assertEqual(source_signature_outputs, document.subgraph().outputs)

    def test_multi_output_buffer_failure_rolls_back_without_graph_mutation(self):
        """Rollback appended buffers when one multi-output allocation fails."""

        document = make_empty_document()
        builder = self._builder(document)
        source = self._constant(builder, "source", [2.0], FLOAT32, np.float32)
        contract = TensorContract(FLOAT32, (1,))
        first, second = builder.add_operator(
            MULTI_OUTPUT,
            inputs=(source,),
            output_contracts=(contract, contract),
            output_names=("first", "second"),
        )
        document.subgraph().outputs = [first, second]
        registry = self.registry.copy()
        registry.register(MULTI_OUTPUT, TwoOutputEvaluator())
        original_buffer_count = len(document.model.buffers)
        original_output_buffers = [
            document.subgraph().tensors[index].buffer
            for index in document.subgraph().outputs
        ]

        failing_factory = FailSecondBufferFactory()
        constant_fold = FoldConstantsPass(
            evaluator_registry=registry,
            codec=self.codec,
            object_factory=failing_factory,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "synthetic buffer allocation failure",
        ):
            constant_fold.run(
                document,
                CirclePassContext(verify_after_each_pass=False),
            )

        self.assertEqual(len(document.model.buffers), original_buffer_count)
        self.assertEqual(len(document.subgraph().operators), 1)
        self.assertEqual(
            [
                document.subgraph().tensors[index].buffer
                for index in document.subgraph().outputs
            ],
            original_output_buffers,
        )


if __name__ == "__main__":
    unittest.main()

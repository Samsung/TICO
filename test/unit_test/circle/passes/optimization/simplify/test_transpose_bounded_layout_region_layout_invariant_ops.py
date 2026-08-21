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

import struct
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from tico.circle.document import CircleDocument
from tico.circle.passes import (
    CirclePassContext,
    CirclePassManager,
    CirclePassStrategy,
    EliminateTransposeBoundedLayoutRegionPass,
    SimplifyViewOpsPass,
)
from tico.circle.passes.cleanup import CompactIndicesPass, DeadCodeEliminationPass
from tico.circle.passes.optimization.simplify import transpose_region_rules as rules
from tico.circle.passes.optimization.simplify._layout_utils import (
    _TRANSPOSE_BUILTIN_CODE,
)

from test.unit_test.circle.fixture import (
    FakeBuffer,
    FakeModel,
    FakeOperator,
    FakeOperatorCode,
    FakeSignatureDef,
    FakeSubGraph,
    FakeTensor,
    FakeTensorMap,
)


_FLOAT32_TENSOR_TYPE = 0
_INT32_TENSOR_TYPE = 2
_BOOL_TENSOR_TYPE = 6
_UINT8_TENSOR_TYPE = 3
_SOURCE_TO_REGION = (0, 3, 1, 2)
_REGION_TO_SOURCE = (0, 2, 3, 1)
_SOURCE_SHAPE = [1, 4, 5, 3]
_REGION_SHAPE = [1, 3, 4, 5]


@dataclass
class FakeQuantization:
    """Provide affine quantization fields used by layout-invariant tests."""

    scale: list[float]
    zeroPoint: list[int]
    quantizedDimension: int = 0


def _encoding(
    *,
    quantized: bool,
) -> tuple[int, FakeQuantization | None]:
    """Return a test tensor type and optional per-tensor qparam."""

    if not quantized:
        return _FLOAT32_TENSOR_TYPE, None
    return _UINT8_TENSOR_TYPE, FakeQuantization([0.125], [127])


def _tensor(
    name: str,
    shape: list[int],
    *,
    tensor_type: int,
    quantization: FakeQuantization | None,
) -> FakeTensor:
    """Create one fake runtime tensor with a complete shape signature."""

    return FakeTensor(
        name,
        shape=list(shape),
        shapeSignature=list(shape),
        type=tensor_type,
        quantization=quantization,
        buffer=0,
    )


def _permutation_buffer(values: tuple[int, ...]) -> FakeBuffer:
    """Create one inline INT32 permutation buffer."""

    return FakeBuffer(data=struct.pack(f"<{len(values)}i", *values))


def _document(
    tensors: list[FakeTensor],
    operators: list[FakeOperator],
    operator_codes: list[int],
    *,
    inputs: list[int],
    output: int,
) -> CircleDocument:
    """Create one single-subgraph Circle document for a region test."""

    subgraph = FakeSubGraph(
        name="main",
        tensors=tensors,
        inputs=inputs,
        outputs=[output],
        operators=operators,
    )
    model = FakeModel(
        subgraphs=[subgraph],
        buffers=[
            FakeBuffer(),
            _permutation_buffer(_SOURCE_TO_REGION),
            _permutation_buffer(_REGION_TO_SOURCE),
        ],
        operatorCodes=[FakeOperatorCode(builtinCode=code) for code in operator_codes],
        signatureDefs=[
            FakeSignatureDef(
                signatureKey="main",
                subgraphIndex=0,
                inputs=[
                    FakeTensorMap(f"input_{position}", tensor_index)
                    for position, tensor_index in enumerate(inputs)
                ],
                outputs=[FakeTensorMap("output", output)],
            )
        ],
    )
    return CircleDocument(model)


def _make_elementwise_document(
    builtin_code: int,
    *,
    input_count: int,
    quantized: bool = False,
) -> tuple[CircleDocument, int, int]:
    """Create one Transpose-bounded same-shape elementwise region."""

    tensor_type, qparam = _encoding(quantized=quantized)
    tensors: list[FakeTensor] = []
    source_indices: list[int] = []
    for position in range(input_count):
        source_indices.append(len(tensors))
        tensors.append(
            _tensor(
                f"source_{position}_nhwc",
                _SOURCE_SHAPE,
                tensor_type=tensor_type,
                quantization=qparam,
            )
        )

    source_to_region_index = len(tensors)
    tensors.append(
        FakeTensor(
            "to_nchw",
            buffer=1,
            shape=[4],
            type=_INT32_TENSOR_TYPE,
        )
    )

    region_input_indices: list[int] = []
    for position in range(input_count):
        region_input_indices.append(len(tensors))
        tensors.append(
            _tensor(
                f"region_input_{position}_nchw",
                _REGION_SHAPE,
                tensor_type=tensor_type,
                quantization=qparam,
            )
        )

    region_output_index = len(tensors)
    tensors.append(
        _tensor(
            "region_output_nchw",
            _REGION_SHAPE,
            tensor_type=tensor_type,
            quantization=qparam,
        )
    )
    region_to_source_index = len(tensors)
    tensors.append(
        FakeTensor(
            "to_nhwc",
            buffer=2,
            shape=[4],
            type=_INT32_TENSOR_TYPE,
        )
    )
    final_output_index = len(tensors)
    tensors.append(
        _tensor(
            "output_nhwc",
            _SOURCE_SHAPE,
            tensor_type=tensor_type,
            quantization=qparam,
        )
    )

    operators = [
        FakeOperator(
            opcodeIndex=0,
            inputs=[source_index, source_to_region_index],
            outputs=[region_input_index],
        )
        for source_index, region_input_index in zip(
            source_indices,
            region_input_indices,
        )
    ]
    target_operator_index = len(operators)
    operators.append(
        FakeOperator(
            opcodeIndex=1,
            inputs=region_input_indices,
            outputs=[region_output_index],
        )
    )
    operators.append(
        FakeOperator(
            opcodeIndex=0,
            inputs=[region_output_index, region_to_source_index],
            outputs=[final_output_index],
        )
    )

    return (
        _document(
            tensors,
            operators,
            [_TRANSPOSE_BUILTIN_CODE, builtin_code],
            inputs=source_indices,
            output=final_output_index,
        ),
        target_operator_index,
        region_output_index,
    )


def _make_mixed_binary_unary_document(
    *,
    quantized: bool,
) -> CircleDocument:
    """Create one MUL-to-RELU region bounded by three Transpose nodes."""

    tensor_type, qparam = _encoding(quantized=quantized)
    mul_code = rules._BINARY_LAYOUT_INVARIANT_BUILTIN_CODES["MUL"]
    relu_code = rules._UNARY_LAYOUT_INVARIANT_BUILTIN_CODES["RELU"]
    tensors = [
        _tensor(
            "lhs_nhwc",
            _SOURCE_SHAPE,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        _tensor(
            "rhs_nhwc",
            _SOURCE_SHAPE,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        FakeTensor("to_nchw", buffer=1, shape=[4], type=_INT32_TENSOR_TYPE),
        _tensor(
            "lhs_nchw",
            _REGION_SHAPE,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        _tensor(
            "rhs_nchw",
            _REGION_SHAPE,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        _tensor(
            "mul_nchw",
            _REGION_SHAPE,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        _tensor(
            "relu_nchw",
            _REGION_SHAPE,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        FakeTensor("to_nhwc", buffer=2, shape=[4], type=_INT32_TENSOR_TYPE),
        _tensor(
            "output_nhwc",
            _SOURCE_SHAPE,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
    ]
    operators = [
        FakeOperator(opcodeIndex=0, inputs=[0, 2], outputs=[3]),
        FakeOperator(opcodeIndex=0, inputs=[1, 2], outputs=[4]),
        FakeOperator(opcodeIndex=1, inputs=[3, 4], outputs=[5]),
        FakeOperator(opcodeIndex=2, inputs=[5], outputs=[6]),
        FakeOperator(opcodeIndex=0, inputs=[6, 7], outputs=[8]),
    ]
    return _document(
        tensors,
        operators,
        [_TRANSPOSE_BUILTIN_CODE, mul_code, relu_code],
        inputs=[0, 1],
        output=8,
    )


def _builtin_codes(document: CircleDocument) -> list[int]:
    """Return live operator builtin codes in graph order."""

    return [
        document.model.operatorCodes[operator.opcodeIndex].builtinCode
        for operator in document.subgraph().operators
    ]


def _rule_context(
    input_shapes: list[list[int]],
    output_shape: list[int],
) -> rules._RegionOpContext:
    """Create one minimal rule context for direct shape validation."""

    tensors = [SimpleNamespace(shape=shape) for shape in input_shapes]
    tensors.append(SimpleNamespace(shape=output_shape))
    operator = SimpleNamespace(
        inputs=list(range(len(input_shapes))),
        outputs=[len(input_shapes)],
    )
    graph = SimpleNamespace(
        subgraph=SimpleNamespace(tensors=tensors),
    )
    return rules._RegionOpContext(
        graph=graph,
        operator_index=0,
        operator=operator,
        source_to_region_permutation=_SOURCE_TO_REGION,
        region_to_source_permutation=_REGION_TO_SOURCE,
    )


class LayoutInvariantRuleRegistryTest(unittest.TestCase):
    """Verify layout-invariant operator family registration and validation."""

    def test_all_unary_builtins_use_unary_rule(self) -> None:
        """Resolve every registered unary builtin to the unary rule class."""

        for name, builtin_code in rules._UNARY_LAYOUT_INVARIANT_BUILTIN_CODES.items():
            with self.subTest(name=name):
                self.assertIsInstance(
                    rules._rule_for_builtin_code(builtin_code),
                    rules._SameShapeUnaryElementwiseRule,
                )

    def test_all_binary_builtins_use_binary_rule(self) -> None:
        """Resolve every registered binary builtin to the binary rule class."""

        for name, builtin_code in rules._BINARY_LAYOUT_INVARIANT_BUILTIN_CODES.items():
            with self.subTest(name=name):
                self.assertIsInstance(
                    rules._rule_for_builtin_code(builtin_code),
                    rules._SameShapeBinaryElementwiseRule,
                )

    def test_add_n_uses_variadic_rule(self) -> None:
        """Resolve ADD_N to the same-shape variadic elementwise rule."""

        rule = rules._rule_for_builtin_code(rules._ADD_N_BUILTIN_CODE)
        self.assertIsInstance(rule, rules._SameShapeVariadicElementwiseRule)
        operator = SimpleNamespace(inputs=[0, 1, 2])
        self.assertEqual(rule.data_input_positions(operator), (0, 1, 2))
        self.assertEqual(rule.data_output_positions(operator), (0,))

    def test_axis_sensitive_builtins_remain_unregistered(self) -> None:
        """Keep layout-sensitive and rank-changing operators as boundaries."""

        for name in ("PRELU", "RESHAPE", "SOFTMAX"):
            with self.subTest(name=name):
                builtin_code = rules._builtin_operator_value(name)
                self.assertIsNone(rules._rule_for_builtin_code(builtin_code))

    def test_unary_rule_rejects_shape_change(self) -> None:
        """Reject a unary operator whose output rank or shape changes."""

        builtin_code = rules._UNARY_LAYOUT_INVARIANT_BUILTIN_CODES["RELU"]
        rule = rules._rule_for_builtin_code(builtin_code)
        context = _rule_context([_REGION_SHAPE], [1, 3, 20])
        self.assertIsNone(rule.plan_rewrite(context))

    def test_binary_rule_rejects_broadcasting(self) -> None:
        """Reject a binary operator whose operands require broadcasting."""

        builtin_code = rules._BINARY_LAYOUT_INVARIANT_BUILTIN_CODES["MUL"]
        rule = rules._rule_for_builtin_code(builtin_code)
        context = _rule_context(
            [_REGION_SHAPE, [1, 3, 1, 1]],
            _REGION_SHAPE,
        )
        self.assertIsNone(rule.plan_rewrite(context))

    def test_variadic_rule_rejects_mismatched_input_shape(self) -> None:
        """Reject ADD_N when one input shape differs from the output shape."""

        rule = rules._rule_for_builtin_code(rules._ADD_N_BUILTIN_CODE)
        context = _rule_context(
            [_REGION_SHAPE, _REGION_SHAPE, [1, 3, 1, 1]],
            _REGION_SHAPE,
        )
        self.assertIsNone(rule.plan_rewrite(context))


class LayoutInvariantRegionPassTest(unittest.TestCase):
    """Verify bounded-region elimination for the new operator families."""

    def test_rewrites_unary_relu_region(self) -> None:
        """Rewrite one unary RELU region and bypass two Transpose nodes."""

        relu_code = rules._UNARY_LAYOUT_INVARIANT_BUILTIN_CODES["RELU"]
        document, operator_index, output_index = _make_elementwise_document(
            relu_code,
            input_count=1,
        )
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        self.assertEqual(result.changes, 2)
        self.assertEqual(document.subgraph().operators[operator_index].inputs, [0])
        self.assertEqual(document.subgraph().outputs, [output_index])
        self.assertEqual(
            document.subgraph().tensors[output_index].shape,
            _SOURCE_SHAPE,
        )
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_rewrites_per_tensor_quantized_mul_region(self) -> None:
        """Rewrite one UINT8 MUL region and preserve per-tensor qparams."""

        mul_code = rules._BINARY_LAYOUT_INVARIANT_BUILTIN_CODES["MUL"]
        document, operator_index, output_index = _make_elementwise_document(
            mul_code,
            input_count=2,
            quantized=True,
        )
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        self.assertEqual(result.changes, 3)
        self.assertEqual(
            document.subgraph().operators[operator_index].inputs,
            [0, 1],
        )
        output = document.subgraph().tensors[output_index]
        self.assertEqual(output.type, _UINT8_TENSOR_TYPE)
        self.assertEqual(output.quantization.scale, [0.125])
        self.assertEqual(output.quantization.zeroPoint, [127])
        self.assertEqual(output.quantization.quantizedDimension, 0)
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_rewrites_dtype_changing_dequantize_region(self) -> None:
        """Rewrite DEQUANTIZE while preserving distinct input and output types."""

        dequantize_code = rules._UNARY_LAYOUT_INVARIANT_BUILTIN_CODES["DEQUANTIZE"]
        document, operator_index, output_index = _make_elementwise_document(
            dequantize_code,
            input_count=1,
            quantized=True,
        )
        operator = document.subgraph().operators[operator_index]
        final_output_index = document.subgraph().outputs[0]
        document.subgraph().tensors[operator.outputs[0]].type = _FLOAT32_TENSOR_TYPE
        document.subgraph().tensors[operator.outputs[0]].quantization = None
        document.subgraph().tensors[final_output_index].type = _FLOAT32_TENSOR_TYPE
        document.subgraph().tensors[final_output_index].quantization = None

        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        self.assertEqual(document.subgraph().operators[operator_index].inputs, [0])
        output = document.subgraph().tensors[output_index]
        self.assertEqual(output.type, _FLOAT32_TENSOR_TYPE)
        self.assertIsNone(output.quantization)
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_rewrites_comparison_with_boolean_output(self) -> None:
        """Rewrite EQUAL while preserving its boolean output tensor type."""

        equal_code = rules._BINARY_LAYOUT_INVARIANT_BUILTIN_CODES["EQUAL"]
        document, operator_index, output_index = _make_elementwise_document(
            equal_code,
            input_count=2,
        )
        operator = document.subgraph().operators[operator_index]
        final_output_index = document.subgraph().outputs[0]
        document.subgraph().tensors[operator.outputs[0]].type = _BOOL_TENSOR_TYPE
        document.subgraph().tensors[operator.outputs[0]].quantization = None
        document.subgraph().tensors[final_output_index].type = _BOOL_TENSOR_TYPE
        document.subgraph().tensors[final_output_index].quantization = None

        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        self.assertEqual(
            document.subgraph().operators[operator_index].inputs,
            [0, 1],
        )
        self.assertEqual(
            document.subgraph().tensors[output_index].type,
            _BOOL_TENSOR_TYPE,
        )
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_rewrites_variadic_add_n_region(self) -> None:
        """Rewrite one three-input ADD_N region and bypass four Transposes."""

        document, operator_index, output_index = _make_elementwise_document(
            rules._ADD_N_BUILTIN_CODE,
            input_count=3,
        )
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        self.assertEqual(result.changes, 4)
        self.assertEqual(
            document.subgraph().operators[operator_index].inputs,
            [0, 1, 2],
        )
        self.assertEqual(document.subgraph().outputs, [output_index])
        self.assertEqual(
            document.subgraph().tensors[output_index].shape,
            _SOURCE_SHAPE,
        )
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_restart_pipeline_removes_mixed_region_transposes(self) -> None:
        """Remove all boundary Transposes around a MUL-to-RELU component."""

        document = _make_mixed_binary_unary_document(quantized=True)
        pipeline = CirclePassManager(
            [
                EliminateTransposeBoundedLayoutRegionPass(),
                SimplifyViewOpsPass(),
                DeadCodeEliminationPass(),
                CompactIndicesPass(),
            ],
            strategy=CirclePassStrategy.RESTART,
        )
        result = pipeline.run(document, CirclePassContext())

        self.assertTrue(result.modified)
        self.assertEqual(
            _builtin_codes(document),
            [
                rules._BINARY_LAYOUT_INVARIANT_BUILTIN_CODES["MUL"],
                rules._UNARY_LAYOUT_INVARIANT_BUILTIN_CODES["RELU"],
            ],
        )
        self.assertEqual(document.subgraph().outputs, [3])
        self.assertEqual(document.subgraph().tensors[3].shape, _SOURCE_SHAPE)
        self.assertTrue(document.verify(raise_on_error=False).ok)


if __name__ == "__main__":
    unittest.main()

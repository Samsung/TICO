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
_UINT8_TENSOR_TYPE = 3
_SOURCE_TO_REGION = (0, 3, 1, 2)
_REGION_TO_SOURCE = (0, 2, 3, 1)


@dataclass
class FakeQuantization:
    """Provide affine quantization fields used by multi-output tests."""

    scale: list[float]
    zeroPoint: list[int]
    quantizedDimension: int = 0


@dataclass(frozen=True)
class _MultiOutputFixture:
    """Describe key tensor and operator indices in one synthetic split graph."""

    document: CircleDocument
    target_operator_index: int
    axis_tensor_index: int
    size_splits_tensor_index: int | None
    region_output_indices: tuple[int, ...]
    final_output_indices: tuple[int, ...]
    used_output_positions: tuple[int, ...]


def _encoding(
    *,
    quantized: bool,
) -> tuple[int, FakeQuantization | None]:
    """Return a data type and optional per-tensor quantization metadata."""

    if not quantized:
        return _FLOAT32_TENSOR_TYPE, None
    return _UINT8_TENSOR_TYPE, FakeQuantization([0.125], [127])


def _tensor(
    name: str,
    shape: tuple[int, ...] | list[int],
    *,
    tensor_type: int,
    quantization: FakeQuantization | None,
    buffer: int = 0,
) -> FakeTensor:
    """Create one fake tensor with a complete static shape signature."""

    shape_list = list(shape)
    return FakeTensor(
        name,
        shape=shape_list,
        shapeSignature=list(shape_list),
        type=tensor_type,
        quantization=quantization,
        buffer=buffer,
    )


def _i32_buffer(values: tuple[int, ...]) -> FakeBuffer:
    """Create one little-endian inline INT32 buffer."""

    return FakeBuffer(data=struct.pack(f"<{len(values)}i", *values))


def _decode_i32_buffer(buffer: FakeBuffer) -> tuple[int, ...]:
    """Decode one fake inline buffer into INT32 values."""

    data = buffer.data or b""
    return tuple(
        struct.unpack_from("<i", data, offset)[0] for offset in range(0, len(data), 4)
    )


def _builtin_codes(document: CircleDocument) -> list[int]:
    """Return live operator builtin codes in graph order."""

    return [
        document.model.operatorCodes[operator.opcodeIndex].builtinCode
        for operator in document.subgraph().operators
    ]


def _source_shape(region_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Convert one region-layout shape into source-layout order."""

    return tuple(region_shape[index] for index in _REGION_TO_SOURCE)


def _region_shape(source_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Convert one source-layout shape into region-layout order."""

    return tuple(source_shape[index] for index in _SOURCE_TO_REGION)


def _make_model(
    *,
    tensors: list[FakeTensor],
    operators: list[FakeOperator],
    buffers: list[FakeBuffer],
    target_builtin_code: int,
    graph_inputs: list[int],
    graph_outputs: list[int],
) -> CircleDocument:
    """Create one single-subgraph model for a split-region test."""

    subgraph = FakeSubGraph(
        name="main",
        tensors=tensors,
        inputs=graph_inputs,
        outputs=graph_outputs,
        operators=operators,
    )
    model = FakeModel(
        subgraphs=[subgraph],
        buffers=buffers,
        operatorCodes=[
            FakeOperatorCode(builtinCode=_TRANSPOSE_BUILTIN_CODE),
            FakeOperatorCode(builtinCode=target_builtin_code),
        ],
        signatureDefs=[
            FakeSignatureDef(
                signatureKey="main",
                subgraphIndex=0,
                inputs=[
                    FakeTensorMap(f"input_{position}", tensor_index)
                    for position, tensor_index in enumerate(graph_inputs)
                ],
                outputs=[
                    FakeTensorMap(f"output_{position}", tensor_index)
                    for position, tensor_index in enumerate(graph_outputs)
                ],
            )
        ],
    )
    return CircleDocument(model)


def _append_output_boundaries(
    *,
    tensors: list[FakeTensor],
    operators: list[FakeOperator],
    buffers: list[FakeBuffer],
    region_output_indices: tuple[int, ...],
    used_output_positions: tuple[int, ...],
    tensor_type: int,
    quantization: FakeQuantization | None,
) -> tuple[int, ...]:
    """Append inverse Transpose nodes for selected region outputs."""

    inverse_buffer_index = len(buffers)
    buffers.append(_i32_buffer(_REGION_TO_SOURCE))
    inverse_tensor_index = len(tensors)
    tensors.append(
        FakeTensor(
            "to_source_layout",
            buffer=inverse_buffer_index,
            shape=[4],
            type=_INT32_TENSOR_TYPE,
        )
    )

    final_output_indices: list[int] = []
    for output_position in used_output_positions:
        region_tensor_index = region_output_indices[output_position]
        final_tensor_index = len(tensors)
        region_output_shape = tuple(tensors[region_tensor_index].shape)
        tensors.append(
            _tensor(
                f"output_{output_position}_nhwc",
                _source_shape(region_output_shape),
                tensor_type=tensor_type,
                quantization=quantization,
            )
        )
        operators.append(
            FakeOperator(
                opcodeIndex=0,
                inputs=[region_tensor_index, inverse_tensor_index],
                outputs=[final_tensor_index],
            )
        )
        final_output_indices.append(final_tensor_index)
    return tuple(final_output_indices)


def _make_split_document(
    *,
    source_shape: tuple[int, ...] = (1, 2, 3, 6),
    axis: int = 1,
    num_outputs: int = 3,
    option_num_splits: int | None = None,
    output_axis_sizes: tuple[int, ...] | None = None,
    used_output_positions: tuple[int, ...] | None = None,
    quantized: bool = False,
    constant_axis: bool = True,
) -> _MultiOutputFixture:
    """Create one equal-size SPLIT region with configurable static metadata."""

    tensor_type, qparam = _encoding(quantized=quantized)
    region_input_shape = _region_shape(source_shape)
    normalized_axis = axis + 4 if axis < 0 else axis
    if output_axis_sizes is None:
        split_size = region_input_shape[normalized_axis] // num_outputs
        output_axis_sizes = (split_size,) * num_outputs
    if used_output_positions is None:
        used_output_positions = tuple(range(num_outputs))
    if option_num_splits is None:
        option_num_splits = num_outputs

    buffers = [FakeBuffer(), _i32_buffer(_SOURCE_TO_REGION)]
    tensors = [
        _tensor(
            "input_nhwc",
            source_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        FakeTensor("to_region_layout", buffer=1, shape=[4], type=_INT32_TENSOR_TYPE),
        _tensor(
            "input_nchw",
            region_input_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
    ]
    graph_inputs = [0]
    if constant_axis:
        axis_buffer_index = len(buffers)
        buffers.append(_i32_buffer((axis,)))
        axis_tensor_index = len(tensors)
        tensors.append(
            FakeTensor(
                "axis",
                buffer=axis_buffer_index,
                shape=[],
                shapeSignature=[],
                type=_INT32_TENSOR_TYPE,
            )
        )
    else:
        axis_tensor_index = len(tensors)
        tensors.append(
            FakeTensor(
                "axis",
                buffer=0,
                shape=[],
                shapeSignature=[],
                type=_INT32_TENSOR_TYPE,
            )
        )
        graph_inputs.append(axis_tensor_index)

    region_output_indices: list[int] = []
    for position, axis_size in enumerate(output_axis_sizes):
        shape = list(region_input_shape)
        shape[normalized_axis] = axis_size
        region_output_indices.append(len(tensors))
        tensors.append(
            _tensor(
                f"split_{position}_nchw",
                tuple(shape),
                tensor_type=tensor_type,
                quantization=qparam,
            )
        )

    operators = [
        FakeOperator(opcodeIndex=0, inputs=[0, 1], outputs=[2]),
        FakeOperator(
            opcodeIndex=1,
            inputs=[axis_tensor_index, 2],
            outputs=region_output_indices,
            builtinOptions=SimpleNamespace(numSplits=option_num_splits),
        ),
    ]
    final_output_indices = _append_output_boundaries(
        tensors=tensors,
        operators=operators,
        buffers=buffers,
        region_output_indices=tuple(region_output_indices),
        used_output_positions=used_output_positions,
        tensor_type=tensor_type,
        quantization=qparam,
    )
    document = _make_model(
        tensors=tensors,
        operators=operators,
        buffers=buffers,
        target_builtin_code=rules._SPLIT_BUILTIN_CODE,
        graph_inputs=graph_inputs,
        graph_outputs=list(final_output_indices),
    )
    return _MultiOutputFixture(
        document=document,
        target_operator_index=1,
        axis_tensor_index=axis_tensor_index,
        size_splits_tensor_index=None,
        region_output_indices=tuple(region_output_indices),
        final_output_indices=final_output_indices,
        used_output_positions=used_output_positions,
    )


def _resolve_sizes_for_fixture(
    split_sizes: tuple[int, ...],
    input_size: int,
) -> tuple[int, ...]:
    """Resolve one optional -1 size for constructing expected output tensors."""

    values = list(split_sizes)
    if values.count(-1) == 1:
        inferred_index = values.index(-1)
        known_sum = sum(value for value in values if value >= 0)
        values[inferred_index] = input_size - known_sum
    return tuple(values)


def _make_split_v_document(
    *,
    source_shape: tuple[int, ...] = (1, 2, 3, 6),
    axis: int = 1,
    split_sizes: tuple[int, ...] = (2, -1, 1),
    option_num_splits: int | None = None,
    output_axis_sizes: tuple[int, ...] | None = None,
    used_output_positions: tuple[int, ...] | None = None,
    quantized: bool = False,
    constant_axis: bool = True,
    constant_sizes: bool = True,
) -> _MultiOutputFixture:
    """Create one static SPLIT_V region with configurable output sizes."""

    tensor_type, qparam = _encoding(quantized=quantized)
    region_input_shape = _region_shape(source_shape)
    normalized_axis = axis + 4 if axis < 0 else axis
    num_outputs = len(split_sizes)
    if output_axis_sizes is None:
        output_axis_sizes = _resolve_sizes_for_fixture(
            split_sizes,
            region_input_shape[normalized_axis],
        )
    if used_output_positions is None:
        used_output_positions = tuple(range(num_outputs))
    if option_num_splits is None:
        option_num_splits = num_outputs

    buffers = [FakeBuffer(), _i32_buffer(_SOURCE_TO_REGION)]
    tensors = [
        _tensor(
            "input_nhwc",
            source_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        FakeTensor("to_region_layout", buffer=1, shape=[4], type=_INT32_TENSOR_TYPE),
        _tensor(
            "input_nchw",
            region_input_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
    ]
    graph_inputs = [0]

    size_splits_tensor_index = len(tensors)
    if constant_sizes:
        size_buffer_index = len(buffers)
        buffers.append(_i32_buffer(split_sizes))
        tensors.append(
            FakeTensor(
                "size_splits",
                buffer=size_buffer_index,
                shape=[num_outputs],
                shapeSignature=[num_outputs],
                type=_INT32_TENSOR_TYPE,
            )
        )
    else:
        tensors.append(
            FakeTensor(
                "size_splits",
                buffer=0,
                shape=[num_outputs],
                shapeSignature=[num_outputs],
                type=_INT32_TENSOR_TYPE,
            )
        )
        graph_inputs.append(size_splits_tensor_index)

    axis_tensor_index = len(tensors)
    if constant_axis:
        axis_buffer_index = len(buffers)
        buffers.append(_i32_buffer((axis,)))
        tensors.append(
            FakeTensor(
                "axis",
                buffer=axis_buffer_index,
                shape=[],
                shapeSignature=[],
                type=_INT32_TENSOR_TYPE,
            )
        )
    else:
        tensors.append(
            FakeTensor(
                "axis",
                buffer=0,
                shape=[],
                shapeSignature=[],
                type=_INT32_TENSOR_TYPE,
            )
        )
        graph_inputs.append(axis_tensor_index)

    region_output_indices: list[int] = []
    for position, axis_size in enumerate(output_axis_sizes):
        shape = list(region_input_shape)
        shape[normalized_axis] = axis_size
        region_output_indices.append(len(tensors))
        tensors.append(
            _tensor(
                f"split_v_{position}_nchw",
                tuple(shape),
                tensor_type=tensor_type,
                quantization=qparam,
            )
        )

    operators = [
        FakeOperator(opcodeIndex=0, inputs=[0, 1], outputs=[2]),
        FakeOperator(
            opcodeIndex=1,
            inputs=[2, size_splits_tensor_index, axis_tensor_index],
            outputs=region_output_indices,
            builtinOptions=SimpleNamespace(numSplits=option_num_splits),
        ),
    ]
    final_output_indices = _append_output_boundaries(
        tensors=tensors,
        operators=operators,
        buffers=buffers,
        region_output_indices=tuple(region_output_indices),
        used_output_positions=used_output_positions,
        tensor_type=tensor_type,
        quantization=qparam,
    )
    document = _make_model(
        tensors=tensors,
        operators=operators,
        buffers=buffers,
        target_builtin_code=rules._SPLIT_V_BUILTIN_CODE,
        graph_inputs=graph_inputs,
        graph_outputs=list(final_output_indices),
    )
    return _MultiOutputFixture(
        document=document,
        target_operator_index=1,
        axis_tensor_index=axis_tensor_index,
        size_splits_tensor_index=size_splits_tensor_index,
        region_output_indices=tuple(region_output_indices),
        final_output_indices=final_output_indices,
        used_output_positions=used_output_positions,
    )


def _axis_values(fixture: _MultiOutputFixture) -> tuple[int, ...]:
    """Return the current axis constant values used by the target operator."""

    operator = fixture.document.subgraph().operators[fixture.target_operator_index]
    axis_position = 0 if fixture.size_splits_tensor_index is None else 2
    axis_tensor = fixture.document.subgraph().tensors[operator.inputs[axis_position]]
    return _decode_i32_buffer(fixture.document.model.buffers[axis_tensor.buffer])


class MultiOutputRegionRuleRegistryTest(unittest.TestCase):
    """Verify registration and data-position metadata for split operators."""

    def test_split_rules_are_registered(self) -> None:
        """Resolve SPLIT and SPLIT_V to their dedicated multi-output rules."""

        self.assertIsInstance(
            rules._rule_for_builtin_code(rules._SPLIT_BUILTIN_CODE),
            rules._SplitRule,
        )
        self.assertIsInstance(
            rules._rule_for_builtin_code(rules._SPLIT_V_BUILTIN_CODE),
            rules._SplitVRule,
        )

    def test_split_rule_data_positions_follow_circle_contract(self) -> None:
        """Check data inputs and every dynamic output position for SPLIT."""

        operator = SimpleNamespace(inputs=[0, 1], outputs=[2, 3, 4])
        rule = rules._rule_for_builtin_code(rules._SPLIT_BUILTIN_CODE)
        self.assertEqual(rule.data_input_positions(operator), (1,))
        self.assertEqual(rule.data_output_positions(operator), (0, 1, 2))

    def test_split_v_rule_data_positions_follow_circle_contract(self) -> None:
        """Check data inputs and every dynamic output position for SPLIT_V."""

        operator = SimpleNamespace(inputs=[0, 1, 2], outputs=[3, 4, 5])
        rule = rules._rule_for_builtin_code(rules._SPLIT_V_BUILTIN_CODE)
        self.assertEqual(rule.data_input_positions(operator), (0,))
        self.assertEqual(rule.data_output_positions(operator), (0, 1, 2))


class MultiOutputRegionPassTest(unittest.TestCase):
    """Verify bounded-region elimination for SPLIT and SPLIT_V."""

    def test_rewrites_split_and_remaps_channel_axis(self) -> None:
        """Move a three-output channel SPLIT from NCHW into NHWC."""

        fixture = _make_split_document()
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            fixture.document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        self.assertEqual(result.changes, 4)
        operator = fixture.document.subgraph().operators[fixture.target_operator_index]
        self.assertEqual(operator.inputs[1], 0)
        self.assertEqual(_axis_values(fixture), (3,))
        self.assertEqual(
            fixture.document.subgraph().outputs,
            list(fixture.region_output_indices),
        )
        for tensor_index in fixture.region_output_indices:
            self.assertEqual(
                fixture.document.subgraph().tensors[tensor_index].shape,
                [1, 2, 3, 2],
            )
        self.assertTrue(fixture.document.verify(raise_on_error=False).ok)

    def test_rewrites_negative_split_axis_and_preserves_uint8_qparams(self) -> None:
        """Normalize a negative axis while preserving per-tensor activation qparams."""

        fixture = _make_split_document(axis=-3, quantized=True)
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            fixture.document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        self.assertEqual(_axis_values(fixture), (3,))
        for tensor_index in fixture.region_output_indices:
            tensor = fixture.document.subgraph().tensors[tensor_index]
            self.assertEqual(tensor.type, _UINT8_TENSOR_TYPE)
            self.assertEqual(tensor.quantization.scale, [0.125])
            self.assertEqual(tensor.quantization.zeroPoint, [127])

    def test_updates_an_unused_split_output_to_source_layout(self) -> None:
        """Rewrite used outputs and update an unconsumed output tensor shape."""

        fixture = _make_split_document(used_output_positions=(0, 2))
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            fixture.document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        self.assertEqual(result.changes, 3)
        self.assertEqual(
            fixture.document.subgraph().tensors[fixture.region_output_indices[1]].shape,
            [1, 2, 3, 2],
        )
        self.assertEqual(
            fixture.document.subgraph().outputs,
            [fixture.region_output_indices[0], fixture.region_output_indices[2]],
        )

    def test_rejects_uneven_split_dimension(self) -> None:
        """Reject SPLIT when the selected dimension is not evenly divisible."""

        fixture = _make_split_document(
            source_shape=(1, 2, 3, 5),
            num_outputs=2,
            output_axis_sizes=(2, 3),
        )
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            fixture.document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)
        self.assertEqual(
            fixture.document.subgraph().outputs,
            list(fixture.final_output_indices),
        )

    def test_rejects_split_num_splits_mismatch(self) -> None:
        """Reject SPLIT when builtin options disagree with output count."""

        fixture = _make_split_document(option_num_splits=2)
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            fixture.document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)

    def test_rejects_nonconstant_split_axis(self) -> None:
        """Reject SPLIT when its axis cannot be remapped statically."""

        fixture = _make_split_document(constant_axis=False)
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            fixture.document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)

    def test_rewrites_split_v_and_preserves_size_splits(self) -> None:
        """Move SPLIT_V to NHWC while preserving its ordered split-size vector."""

        fixture = _make_split_v_document()
        size_tensor = fixture.document.subgraph().tensors[
            fixture.size_splits_tensor_index
        ]
        old_size_payload = fixture.document.model.buffers[size_tensor.buffer].data

        result = EliminateTransposeBoundedLayoutRegionPass().run(
            fixture.document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        self.assertEqual(result.changes, 4)
        operator = fixture.document.subgraph().operators[fixture.target_operator_index]
        self.assertEqual(operator.inputs[0], 0)
        self.assertEqual(operator.inputs[1], fixture.size_splits_tensor_index)
        self.assertEqual(_axis_values(fixture), (3,))
        self.assertEqual(
            fixture.document.model.buffers[size_tensor.buffer].data,
            old_size_payload,
        )
        self.assertEqual(
            [
                fixture.document.subgraph().tensors[index].shape
                for index in fixture.region_output_indices
            ],
            [[1, 2, 3, 2], [1, 2, 3, 3], [1, 2, 3, 1]],
        )
        self.assertTrue(fixture.document.verify(raise_on_error=False).ok)

    def test_rewrites_negative_split_v_axis_with_uint8_outputs(self) -> None:
        """Normalize SPLIT_V axis and preserve per-tensor UINT8 output metadata."""

        fixture = _make_split_v_document(axis=-3, quantized=True)
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            fixture.document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        self.assertEqual(_axis_values(fixture), (3,))
        for tensor_index in fixture.region_output_indices:
            tensor = fixture.document.subgraph().tensors[tensor_index]
            self.assertEqual(tensor.type, _UINT8_TENSOR_TYPE)
            self.assertEqual(tensor.quantization.quantizedDimension, 0)

    def test_rejects_multiple_inferred_split_v_sizes(self) -> None:
        """Reject SPLIT_V when more than one output size is inferred."""

        fixture = _make_split_v_document(
            split_sizes=(2, -1, -1),
            output_axis_sizes=(2, 2, 2),
        )
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            fixture.document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)

    def test_rejects_split_v_size_sum_mismatch(self) -> None:
        """Reject SPLIT_V when explicit sizes do not cover the input dimension."""

        fixture = _make_split_v_document(
            split_sizes=(2, 2, 1),
            output_axis_sizes=(2, 2, 1),
        )
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            fixture.document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)

    def test_rejects_split_v_num_splits_mismatch(self) -> None:
        """Reject SPLIT_V when numSplits differs from the output vector length."""

        fixture = _make_split_v_document(option_num_splits=2)
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            fixture.document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)

    def test_rejects_nonconstant_split_v_sizes(self) -> None:
        """Reject SPLIT_V when size_splits cannot be validated statically."""

        fixture = _make_split_v_document(constant_sizes=False)
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            fixture.document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)

    def test_restart_pipeline_removes_all_split_v_transposes(self) -> None:
        """Keep one multi-output operator after restart scheduling and cleanup."""

        fixture = _make_split_v_document(quantized=True)
        pipeline = CirclePassManager(
            [
                EliminateTransposeBoundedLayoutRegionPass(),
                SimplifyViewOpsPass(),
                DeadCodeEliminationPass(),
                CompactIndicesPass(),
            ],
            strategy=CirclePassStrategy.RESTART,
        )
        result = pipeline.run(fixture.document, CirclePassContext())

        self.assertTrue(result.modified)
        self.assertEqual(
            _builtin_codes(fixture.document),
            [rules._SPLIT_V_BUILTIN_CODE],
        )
        self.assertEqual(len(fixture.document.subgraph().outputs), 3)
        self.assertEqual(
            [
                fixture.document.subgraph().tensors[index].shape
                for index in fixture.document.subgraph().outputs
            ],
            [[1, 2, 3, 2], [1, 2, 3, 3], [1, 2, 3, 1]],
        )
        self.assertTrue(fixture.document.verify(raise_on_error=False).ok)


if __name__ == "__main__":
    unittest.main()

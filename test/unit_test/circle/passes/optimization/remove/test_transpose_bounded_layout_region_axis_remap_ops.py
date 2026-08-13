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
from typing import Any

from tico.circle.document import CircleDocument
from tico.circle.passes import (
    CirclePassContext,
    CirclePassManager,
    CirclePassStrategy,
    EliminateTransposeBoundedLayoutRegionPass,
    RemoveRedundantLayoutOpsPass,
)
from tico.circle.passes.cleanup import CompactIndicesPass, DeadCodeEliminationPass
from tico.circle.passes.optimization.remove import (
    transpose_bounded_layout_region_rules as rules,
)
from tico.circle.passes.optimization.remove.layout_ops import _TRANSPOSE_BUILTIN_CODE

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
    """Provide affine quantization fields used by axis-remap tests."""

    scale: list[float]
    zeroPoint: list[int]
    quantizedDimension: int = 0


@dataclass(frozen=True)
class _ConstantSpec:
    """Describe one constant tensor used by a synthetic Circle operator."""

    name: str
    shape: tuple[int, ...]
    tensor_type: int
    payload: bytes


def _encoding(
    *,
    quantized: bool,
) -> tuple[int, FakeQuantization | None]:
    """Return a data tensor type and optional per-tensor qparam."""

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


def _i32_constant(name: str, values: tuple[int, ...]) -> _ConstantSpec:
    """Create one flat INT32 constant specification."""

    return _ConstantSpec(
        name=name,
        shape=(len(values),),
        tensor_type=_INT32_TENSOR_TYPE,
        payload=struct.pack(f"<{len(values)}i", *values),
    )


def _padding_constant(
    name: str,
    rows: tuple[tuple[int, int], ...],
) -> _ConstantSpec:
    """Create one rank-by-two INT32 padding constant specification."""

    values = tuple(value for row in rows for value in row)
    return _ConstantSpec(
        name=name,
        shape=(len(rows), 2),
        tensor_type=_INT32_TENSOR_TYPE,
        payload=struct.pack(f"<{len(values)}i", *values),
    )


def _float_scalar(name: str, value: float) -> _ConstantSpec:
    """Create one scalar FLOAT32 constant specification."""

    return _ConstantSpec(
        name=name,
        shape=(),
        tensor_type=_FLOAT32_TENSOR_TYPE,
        payload=struct.pack("<f", value),
    )


def _permutation_buffer(values: tuple[int, ...]) -> FakeBuffer:
    """Create one inline INT32 permutation buffer."""

    return FakeBuffer(data=struct.pack(f"<{len(values)}i", *values))


def _decode_i32_buffer(buffer: FakeBuffer) -> tuple[int, ...]:
    """Decode one fake inline buffer as little-endian INT32 values."""

    data = buffer.data or b""
    return tuple(
        struct.unpack_from("<i", data, offset)[0] for offset in range(0, len(data), 4)
    )


def _document(
    *,
    tensors: list[FakeTensor],
    operators: list[FakeOperator],
    buffers: list[FakeBuffer],
    operator_codes: list[int],
    inputs: list[int],
    output: int,
) -> CircleDocument:
    """Create one single-subgraph Circle document for an axis-remap test."""

    subgraph = FakeSubGraph(
        name="main",
        tensors=tensors,
        inputs=inputs,
        outputs=[output],
        operators=operators,
    )
    model = FakeModel(
        subgraphs=[subgraph],
        buffers=buffers,
        operatorCodes=[
            FakeOperatorCode(builtinCode=builtin_code)
            for builtin_code in operator_codes
        ],
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


def _make_single_data_document(
    builtin_code: int,
    *,
    source_shape: tuple[int, ...],
    region_output_shape: tuple[int, ...],
    source_output_shape: tuple[int, ...],
    constants: tuple[_ConstantSpec, ...],
    builtin_options: Any = None,
    quantized: bool = False,
) -> tuple[CircleDocument, int, int, tuple[int, ...]]:
    """Create one single-data axis-remap op between inverse Transpose nodes."""

    tensor_type, qparam = _encoding(quantized=quantized)
    region_shape = tuple(source_shape[index] for index in _SOURCE_TO_REGION)
    tensors = [
        _tensor(
            "source_nhwc",
            source_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        FakeTensor("to_nchw", buffer=1, shape=[4], type=_INT32_TENSOR_TYPE),
        _tensor(
            "region_input_nchw",
            region_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
    ]
    buffers = [FakeBuffer(), _permutation_buffer(_SOURCE_TO_REGION)]
    constant_indices: list[int] = []
    for constant in constants:
        buffer_index = len(buffers)
        buffers.append(FakeBuffer(data=constant.payload))
        constant_indices.append(len(tensors))
        tensors.append(
            FakeTensor(
                constant.name,
                buffer=buffer_index,
                shape=list(constant.shape),
                shapeSignature=list(constant.shape),
                type=constant.tensor_type,
            )
        )

    region_output_index = len(tensors)
    tensors.append(
        _tensor(
            "region_output_nchw",
            region_output_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        )
    )
    inverse_buffer_index = len(buffers)
    buffers.append(_permutation_buffer(_REGION_TO_SOURCE))
    inverse_tensor_index = len(tensors)
    tensors.append(
        FakeTensor(
            "to_nhwc",
            buffer=inverse_buffer_index,
            shape=[4],
            type=_INT32_TENSOR_TYPE,
        )
    )
    final_output_index = len(tensors)
    tensors.append(
        _tensor(
            "output_nhwc",
            source_output_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        )
    )

    target_operator_index = 1
    operators = [
        FakeOperator(opcodeIndex=0, inputs=[0, 1], outputs=[2]),
        FakeOperator(
            opcodeIndex=1,
            inputs=[2, *constant_indices],
            outputs=[region_output_index],
            builtinOptions=builtin_options,
        ),
        FakeOperator(
            opcodeIndex=0,
            inputs=[region_output_index, inverse_tensor_index],
            outputs=[final_output_index],
        ),
    ]
    return (
        _document(
            tensors=tensors,
            operators=operators,
            buffers=buffers,
            operator_codes=[_TRANSPOSE_BUILTIN_CODE, builtin_code],
            inputs=[0],
            output=final_output_index,
        ),
        target_operator_index,
        region_output_index,
        tuple(constant_indices),
    )


def _make_concatenation_document(
    *,
    axis: int,
    second_source_shape: tuple[int, ...] = (1, 2, 3, 5),
) -> tuple[CircleDocument, int, int]:
    """Create one two-input CONCATENATION region in NCHW order."""

    first_source_shape = (1, 2, 3, 4)
    source_shapes = (first_source_shape, second_source_shape)
    region_shapes = tuple(
        tuple(shape[index] for index in _SOURCE_TO_REGION) for shape in source_shapes
    )
    normalized_axis = axis + 4 if axis < 0 else axis
    if 0 <= normalized_axis < 4:
        region_output_shape = list(region_shapes[0])
        region_output_shape[normalized_axis] = sum(
            shape[normalized_axis] for shape in region_shapes
        )
    else:
        region_output_shape = list(region_shapes[0])
    source_output_shape = tuple(
        region_output_shape[index] for index in _REGION_TO_SOURCE
    )

    tensors = [
        _tensor(
            "first_nhwc",
            first_source_shape,
            tensor_type=_FLOAT32_TENSOR_TYPE,
            quantization=None,
        ),
        _tensor(
            "second_nhwc",
            second_source_shape,
            tensor_type=_FLOAT32_TENSOR_TYPE,
            quantization=None,
        ),
        FakeTensor("to_nchw", buffer=1, shape=[4], type=_INT32_TENSOR_TYPE),
        _tensor(
            "first_nchw",
            region_shapes[0],
            tensor_type=_FLOAT32_TENSOR_TYPE,
            quantization=None,
        ),
        _tensor(
            "second_nchw",
            region_shapes[1],
            tensor_type=_FLOAT32_TENSOR_TYPE,
            quantization=None,
        ),
        _tensor(
            "concat_nchw",
            tuple(region_output_shape),
            tensor_type=_FLOAT32_TENSOR_TYPE,
            quantization=None,
        ),
        FakeTensor("to_nhwc", buffer=2, shape=[4], type=_INT32_TENSOR_TYPE),
        _tensor(
            "output_nhwc",
            source_output_shape,
            tensor_type=_FLOAT32_TENSOR_TYPE,
            quantization=None,
        ),
    ]
    target_operator_index = 2
    operators = [
        FakeOperator(opcodeIndex=0, inputs=[0, 2], outputs=[3]),
        FakeOperator(opcodeIndex=0, inputs=[1, 2], outputs=[4]),
        FakeOperator(
            opcodeIndex=1,
            inputs=[3, 4],
            outputs=[5],
            builtinOptions=SimpleNamespace(
                axis=axis,
                fusedActivationFunction=0,
            ),
        ),
        FakeOperator(opcodeIndex=0, inputs=[5, 6], outputs=[7]),
    ]
    return (
        _document(
            tensors=tensors,
            operators=operators,
            buffers=[
                FakeBuffer(),
                _permutation_buffer(_SOURCE_TO_REGION),
                _permutation_buffer(_REGION_TO_SOURCE),
            ],
            operator_codes=[
                _TRANSPOSE_BUILTIN_CODE,
                rules._CONCATENATION_BUILTIN_CODE,
            ],
            inputs=[0, 1],
            output=7,
        ),
        target_operator_index,
        5,
    )


def _builtin_codes(document: CircleDocument) -> list[int]:
    """Return live operator builtin codes in graph order."""

    return [
        document.model.operatorCodes[operator.opcodeIndex].builtinCode
        for operator in document.subgraph().operators
    ]


class AxisRemapRuleRegistryTest(unittest.TestCase):
    """Verify rule registration for the axis-remapping operator family."""

    def test_axis_remap_builtins_use_expected_rules(self) -> None:
        """Resolve every new builtin to its dedicated axis-remapping rule."""

        expected = {
            "CONCATENATION": rules._ConcatenationRule,
            "MIRROR_PAD": rules._MirrorPadRule,
            "PADV2": rules._PadV2Rule,
            "SLICE": rules._SliceRule,
            "TILE": rules._TileRule,
        }
        for name, rule_type in expected.items():
            with self.subTest(name=name):
                builtin_code = rules._AXIS_REMAP_BUILTIN_CODES[name]
                self.assertIsInstance(
                    rules._rule_for_builtin_code(builtin_code),
                    rule_type,
                )

    def test_multi_output_axis_ops_remain_unregistered(self) -> None:
        """Keep SPLIT and SPLIT_V for the later multi-output extension."""

        for name in ("SPLIT", "SPLIT_V"):
            with self.subTest(name=name):
                builtin_code = rules._builtin_operator_value(name)
                self.assertIsNone(rules._rule_for_builtin_code(builtin_code))


class AxisRemapRegionPassTest(unittest.TestCase):
    """Verify bounded-region elimination for axis-remapping operators."""

    def test_rewrites_concatenation_and_remaps_channel_axis(self) -> None:
        """Move NCHW channel concatenation to the NHWC channel axis."""

        document, operator_index, output_index = _make_concatenation_document(axis=1)
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        self.assertEqual(result.changes, 3)
        operator = document.subgraph().operators[operator_index]
        self.assertEqual(operator.inputs, [0, 1])
        self.assertEqual(operator.builtinOptions.axis, 3)
        self.assertEqual(document.subgraph().outputs, [output_index])
        self.assertEqual(document.subgraph().tensors[output_index].shape, [1, 2, 3, 9])
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_normalizes_negative_concatenation_axis(self) -> None:
        """Normalize negative NCHW channel axis before remapping to NHWC."""

        document, operator_index, _ = _make_concatenation_document(axis=-3)
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        self.assertEqual(
            document.subgraph().operators[operator_index].builtinOptions.axis,
            3,
        )

    def test_rejects_invalid_concatenation_axis(self) -> None:
        """Reject CONCATENATION when its axis is outside the tensor rank."""

        document, operator_index, _ = _make_concatenation_document(axis=4)
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)
        self.assertEqual(
            document.subgraph().operators[operator_index].builtinOptions.axis,
            4,
        )

    def test_rewrites_padv2_and_preserves_scalar_value(self) -> None:
        """Remap PADV2 rows without changing its scalar padding value input."""

        (
            document,
            operator_index,
            output_index,
            constant_indices,
        ) = _make_single_data_document(
            rules._PADV2_BUILTIN_CODE,
            source_shape=(1, 2, 3, 4),
            region_output_shape=(1, 7, 2, 3),
            source_output_shape=(1, 2, 3, 7),
            constants=(
                _padding_constant(
                    "paddings",
                    ((0, 0), (1, 2), (0, 0), (0, 0)),
                ),
                _float_scalar("value", 1.5),
            ),
        )
        old_value_index = constant_indices[1]
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        operator = document.subgraph().operators[operator_index]
        self.assertEqual(operator.inputs[0], 0)
        self.assertNotEqual(operator.inputs[1], constant_indices[0])
        self.assertEqual(operator.inputs[2], old_value_index)
        new_padding = document.subgraph().tensors[operator.inputs[1]]
        self.assertEqual(
            _decode_i32_buffer(document.model.buffers[new_padding.buffer]),
            (0, 0, 0, 0, 0, 0, 1, 2),
        )
        self.assertEqual(document.subgraph().outputs, [output_index])
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_rewrites_mirror_pad_and_preserves_mode(self) -> None:
        """Remap MIRROR_PAD rows while preserving its reflection mode."""

        options = SimpleNamespace(mode=0)
        document, operator_index, _, constant_indices = _make_single_data_document(
            rules._MIRROR_PAD_BUILTIN_CODE,
            source_shape=(1, 2, 3, 4),
            region_output_shape=(1, 4, 4, 3),
            source_output_shape=(1, 4, 3, 4),
            constants=(
                _padding_constant(
                    "paddings",
                    ((0, 0), (0, 0), (1, 1), (0, 0)),
                ),
            ),
            builtin_options=options,
        )
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        operator = document.subgraph().operators[operator_index]
        self.assertEqual(operator.builtinOptions.mode, 0)
        self.assertNotEqual(operator.inputs[1], constant_indices[0])
        new_padding = document.subgraph().tensors[operator.inputs[1]]
        self.assertEqual(
            _decode_i32_buffer(document.model.buffers[new_padding.buffer]),
            (0, 0, 1, 1, 0, 0, 0, 0),
        )

    def test_rewrites_per_tensor_quantized_tile(self) -> None:
        """Remap TILE multiples while preserving per-tensor activation qparams."""

        (
            document,
            operator_index,
            output_index,
            constant_indices,
        ) = _make_single_data_document(
            rules._TILE_BUILTIN_CODE,
            source_shape=(1, 2, 3, 4),
            region_output_shape=(1, 8, 6, 3),
            source_output_shape=(1, 6, 3, 8),
            constants=(_i32_constant("multiples", (1, 2, 3, 1)),),
            quantized=True,
        )
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        operator = document.subgraph().operators[operator_index]
        self.assertNotEqual(operator.inputs[1], constant_indices[0])
        multiples = document.subgraph().tensors[operator.inputs[1]]
        self.assertEqual(
            _decode_i32_buffer(document.model.buffers[multiples.buffer]),
            (1, 3, 1, 2),
        )
        output = document.subgraph().tensors[output_index]
        self.assertEqual(output.quantization.scale, [0.125])
        self.assertEqual(output.quantization.zeroPoint, [127])
        self.assertEqual(output.quantization.quantizedDimension, 0)

    def test_rewrites_slice_begin_and_size_vectors(self) -> None:
        """Remap both static SLICE vectors and preserve size equal to minus one."""

        (
            document,
            operator_index,
            output_index,
            constant_indices,
        ) = _make_single_data_document(
            rules._SLICE_BUILTIN_CODE,
            source_shape=(1, 4, 5, 6),
            region_output_shape=(1, 3, 2, 3),
            source_output_shape=(1, 2, 3, 3),
            constants=(
                _i32_constant("begin", (0, 1, 1, 2)),
                _i32_constant("size", (1, 3, 2, -1)),
            ),
        )
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        operator = document.subgraph().operators[operator_index]
        self.assertNotEqual(operator.inputs[1], constant_indices[0])
        self.assertNotEqual(operator.inputs[2], constant_indices[1])
        begin = document.subgraph().tensors[operator.inputs[1]]
        size = document.subgraph().tensors[operator.inputs[2]]
        self.assertEqual(
            _decode_i32_buffer(document.model.buffers[begin.buffer]),
            (0, 1, 2, 1),
        )
        self.assertEqual(
            _decode_i32_buffer(document.model.buffers[size.buffer]),
            (1, 2, -1, 3),
        )
        self.assertEqual(document.subgraph().outputs, [output_index])
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_restart_pipeline_removes_tile_boundary_transposes(self) -> None:
        """Run axis remapping and cleanup until no TILE boundary remains."""

        document, _, _, _ = _make_single_data_document(
            rules._TILE_BUILTIN_CODE,
            source_shape=(1, 2, 3, 4),
            region_output_shape=(1, 8, 6, 3),
            source_output_shape=(1, 6, 3, 8),
            constants=(_i32_constant("multiples", (1, 2, 3, 1)),),
        )
        pipeline = CirclePassManager(
            [
                EliminateTransposeBoundedLayoutRegionPass(),
                RemoveRedundantLayoutOpsPass(),
                DeadCodeEliminationPass(),
                CompactIndicesPass(),
            ],
            strategy=CirclePassStrategy.RESTART,
        )
        result = pipeline.run(document, CirclePassContext())

        self.assertTrue(result.modified)
        self.assertEqual(_builtin_codes(document), [rules._TILE_BUILTIN_CODE])
        output_index = document.subgraph().outputs[0]
        self.assertEqual(
            document.subgraph().tensors[output_index].shape,
            [1, 6, 3, 8],
        )
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_rejects_nonconstant_tile_multiples(self) -> None:
        """Reject TILE when its multiples tensor has no inline constant buffer."""

        document, operator_index, _, constant_indices = _make_single_data_document(
            rules._TILE_BUILTIN_CODE,
            source_shape=(1, 2, 3, 4),
            region_output_shape=(1, 8, 6, 3),
            source_output_shape=(1, 6, 3, 8),
            constants=(_i32_constant("multiples", (1, 2, 3, 1)),),
        )
        document.subgraph().tensors[constant_indices[0]].buffer = 0
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)
        self.assertEqual(document.subgraph().operators[operator_index].inputs[0], 2)

    def test_rejects_invalid_slice_size(self) -> None:
        """Reject SLICE when its static size exceeds the input dimension."""

        document, operator_index, _, _ = _make_single_data_document(
            rules._SLICE_BUILTIN_CODE,
            source_shape=(1, 4, 5, 6),
            region_output_shape=(1, 6, 2, 3),
            source_output_shape=(1, 2, 3, 6),
            constants=(
                _i32_constant("begin", (0, 1, 1, 2)),
                _i32_constant("size", (1, 6, 2, -1)),
            ),
        )
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)
        self.assertEqual(document.subgraph().operators[operator_index].inputs[0], 2)

    def test_rejects_nonscalar_padv2_value(self) -> None:
        """Reject PADV2 when its padding value input contains multiple elements."""

        value = _ConstantSpec(
            name="value",
            shape=(2,),
            tensor_type=_FLOAT32_TENSOR_TYPE,
            payload=struct.pack("<2f", 1.0, 2.0),
        )
        document, operator_index, _, _ = _make_single_data_document(
            rules._PADV2_BUILTIN_CODE,
            source_shape=(1, 2, 3, 4),
            region_output_shape=(1, 7, 2, 3),
            source_output_shape=(1, 2, 3, 7),
            constants=(
                _padding_constant(
                    "paddings",
                    ((0, 0), (1, 2), (0, 0), (0, 0)),
                ),
                value,
            ),
        )
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)
        self.assertEqual(document.subgraph().operators[operator_index].inputs[0], 2)

    def test_rejects_mirror_pad_without_mode_option(self) -> None:
        """Reject MIRROR_PAD when its required mode option is unavailable."""

        document, operator_index, _, _ = _make_single_data_document(
            rules._MIRROR_PAD_BUILTIN_CODE,
            source_shape=(1, 2, 3, 4),
            region_output_shape=(1, 4, 4, 3),
            source_output_shape=(1, 4, 3, 4),
            constants=(
                _padding_constant(
                    "paddings",
                    ((0, 0), (0, 0), (1, 1), (0, 0)),
                ),
            ),
        )
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)
        self.assertEqual(document.subgraph().operators[operator_index].inputs[0], 2)


if __name__ == "__main__":
    unittest.main()

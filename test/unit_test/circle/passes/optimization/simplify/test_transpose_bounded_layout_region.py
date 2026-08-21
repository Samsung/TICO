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

from tico.circle.document import CircleDocument
from tico.circle.passes import (
    CirclePassContext,
    CirclePassManager,
    CirclePassStrategy,
    EliminateTransposeBoundedLayoutRegionPass,
    SimplifyViewOpsPass,
)
from tico.circle.passes.cleanup import CompactIndicesPass, DeadCodeEliminationPass
from tico.circle.passes.optimization.simplify._layout_utils import (
    _TRANSPOSE_BUILTIN_CODE,
)
from tico.circle.passes.optimization.simplify.transpose_region import (
    _ADD_BUILTIN_CODE,
    _PAD_BUILTIN_CODE,
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


_RELU_BUILTIN_CODE = 19
_FLOAT32_TENSOR_TYPE = 0
_INT32_TENSOR_TYPE = 2
_UINT8_TENSOR_TYPE = 3
_SOURCE_TO_REGION = (0, 3, 1, 2)
_REGION_TO_SOURCE = (0, 2, 3, 1)


@dataclass
class FakeQuantization:
    """Provide affine quantization fields used by region-pass tests."""

    scale: list[float]
    zeroPoint: list[int]
    quantizedDimension: int = 0


def _encoding(
    *,
    quantized: bool,
    scale: float = 0.125,
    zero_point: int = 127,
) -> tuple[int, FakeQuantization | None]:
    """Return a test tensor type and optional per-tensor quantization object."""

    if not quantized:
        return _FLOAT32_TENSOR_TYPE, None
    return _UINT8_TENSOR_TYPE, FakeQuantization([scale], [zero_point])


def _tensor(
    name: str,
    shape: list[int],
    *,
    tensor_type: int,
    quantization: FakeQuantization | None,
    buffer: int = 0,
) -> FakeTensor:
    """Create one fake data tensor with copied quantization metadata."""

    return FakeTensor(
        name,
        shape=shape,
        shapeSignature=list(shape),
        type=tensor_type,
        quantization=quantization,
        buffer=buffer,
    )


def _model(
    tensors: list[FakeTensor],
    operators: list[FakeOperator],
    buffers: list[FakeBuffer],
    *,
    inputs: list[int],
    outputs: list[int],
    signature_outputs: list[FakeTensorMap],
) -> CircleDocument:
    """Create one single-subgraph Circle document for a region test."""

    subgraph = FakeSubGraph(
        name="main",
        tensors=tensors,
        inputs=inputs,
        outputs=outputs,
        operators=operators,
    )
    model = FakeModel(
        subgraphs=[subgraph],
        buffers=buffers,
        operatorCodes=[
            FakeOperatorCode(builtinCode=_TRANSPOSE_BUILTIN_CODE),
            FakeOperatorCode(builtinCode=_ADD_BUILTIN_CODE),
            FakeOperatorCode(builtinCode=_PAD_BUILTIN_CODE),
            FakeOperatorCode(builtinCode=_RELU_BUILTIN_CODE),
        ],
        signatureDefs=[
            FakeSignatureDef(
                signatureKey="main",
                subgraphIndex=0,
                inputs=[
                    FakeTensorMap(f"input_{position}", index)
                    for position, index in enumerate(inputs)
                ],
                outputs=signature_outputs,
            )
        ],
    )
    return CircleDocument(model)


def _permutation_buffer(values: tuple[int, ...]) -> FakeBuffer:
    """Create one inline INT32 permutation buffer."""

    return FakeBuffer(data=struct.pack(f"<{len(values)}i", *values))


def _padding_buffer(rows: tuple[tuple[int, int], ...]) -> FakeBuffer:
    """Create one inline INT32 rank-by-two padding buffer."""

    values = [value for row in rows for value in row]
    return FakeBuffer(data=struct.pack(f"<{len(values)}i", *values))


def _decode_i32_buffer(buffer: FakeBuffer) -> tuple[int, ...]:
    """Decode every INT32 value from one fake inline buffer."""

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


def _make_simple_add_document(
    *,
    quantized: bool = False,
    rhs_source_shape: list[int] | None = None,
    shared_first_transpose: bool = False,
) -> CircleDocument:
    """Create one Transpose-bounded binary ADD component."""

    tensor_type, qparam = _encoding(quantized=quantized)
    source_shape = [1, 4, 5, 3]
    rhs_shape = rhs_source_shape or source_shape
    lhs_region_shape = [source_shape[index] for index in _SOURCE_TO_REGION]
    rhs_region_shape = [rhs_shape[index] for index in _SOURCE_TO_REGION]

    tensors = [
        _tensor(
            "lhs_nhwc",
            source_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        _tensor(
            "rhs_nhwc",
            rhs_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        FakeTensor("to_nchw", buffer=1, shape=[4], type=_INT32_TENSOR_TYPE),
        _tensor(
            "lhs_nchw",
            lhs_region_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        _tensor(
            "rhs_nchw",
            rhs_region_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        _tensor(
            "add_nchw",
            lhs_region_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        FakeTensor("to_nhwc", buffer=2, shape=[4], type=_INT32_TENSOR_TYPE),
        _tensor(
            "output_nhwc",
            source_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
    ]
    operators = [
        FakeOperator(opcodeIndex=0, inputs=[0, 2], outputs=[3]),
        FakeOperator(opcodeIndex=0, inputs=[1, 2], outputs=[4]),
        FakeOperator(opcodeIndex=1, inputs=[3, 4], outputs=[5]),
        FakeOperator(opcodeIndex=0, inputs=[5, 6], outputs=[7]),
    ]
    outputs = [7]
    signature_outputs = [FakeTensorMap("output", 7)]

    if shared_first_transpose:
        tensors.append(
            _tensor(
                "fanout_output",
                lhs_region_shape,
                tensor_type=tensor_type,
                quantization=qparam,
            )
        )
        operators.append(FakeOperator(opcodeIndex=3, inputs=[3], outputs=[8]))
        outputs.append(8)
        signature_outputs.append(FakeTensorMap("fanout", 8))

    return _model(
        tensors,
        operators,
        [
            FakeBuffer(),
            _permutation_buffer(_SOURCE_TO_REGION),
            _permutation_buffer(_REGION_TO_SOURCE),
        ],
        inputs=[0, 1],
        outputs=outputs,
        signature_outputs=signature_outputs,
    )


def _make_downsample_document(*, quantized: bool = False) -> CircleDocument:
    """Create one PAD-plus-ADD downsample layout region."""

    tensor_type, qparam = _encoding(quantized=quantized)
    main_source_shape = [1, 4, 5, 6]
    shortcut_source_shape = [1, 4, 5, 3]
    main_region_shape = [main_source_shape[index] for index in _SOURCE_TO_REGION]
    shortcut_region_shape = [
        shortcut_source_shape[index] for index in _SOURCE_TO_REGION
    ]
    padding_rows = ((0, 0), (0, 3), (0, 0), (0, 0))

    tensors = [
        _tensor(
            "main_nhwc",
            main_source_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        _tensor(
            "shortcut_nhwc",
            shortcut_source_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        FakeTensor("to_nchw", buffer=1, shape=[4], type=_INT32_TENSOR_TYPE),
        _tensor(
            "main_nchw",
            main_region_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        _tensor(
            "shortcut_nchw",
            shortcut_region_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        FakeTensor("paddings", buffer=2, shape=[4, 2], type=_INT32_TENSOR_TYPE),
        _tensor(
            "padded_nchw",
            main_region_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        _tensor(
            "add_nchw",
            main_region_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        FakeTensor("to_nhwc", buffer=3, shape=[4], type=_INT32_TENSOR_TYPE),
        _tensor(
            "output_nhwc",
            main_source_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
    ]
    operators = [
        FakeOperator(opcodeIndex=0, inputs=[0, 2], outputs=[3]),
        FakeOperator(opcodeIndex=0, inputs=[1, 2], outputs=[4]),
        FakeOperator(opcodeIndex=2, inputs=[4, 5], outputs=[6]),
        FakeOperator(opcodeIndex=1, inputs=[6, 3], outputs=[7]),
        FakeOperator(opcodeIndex=0, inputs=[7, 8], outputs=[9]),
    ]
    return _model(
        tensors,
        operators,
        [
            FakeBuffer(),
            _permutation_buffer(_SOURCE_TO_REGION),
            _padding_buffer(padding_rows),
            _permutation_buffer(_REGION_TO_SOURCE),
        ],
        inputs=[0, 1],
        outputs=[9],
        signature_outputs=[FakeTensorMap("output", 9)],
    )


def _make_decoder_document(*, quantized: bool = False) -> CircleDocument:
    """Create two connected ADD nodes with an NHWC side path between them."""

    tensor_type, qparam = _encoding(quantized=quantized)
    source_shape = [1, 4, 5, 3]
    region_shape = [source_shape[index] for index in _SOURCE_TO_REGION]

    tensors = [
        _tensor(
            "skip_nhwc",
            source_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        _tensor(
            "upsample_nhwc",
            source_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        FakeTensor("to_nchw", buffer=1, shape=[4], type=_INT32_TENSOR_TYPE),
        _tensor(
            "skip_nchw",
            region_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        _tensor(
            "upsample_nchw",
            region_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        _tensor(
            "merge_nchw",
            region_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        FakeTensor("to_nhwc", buffer=2, shape=[4], type=_INT32_TENSOR_TYPE),
        _tensor(
            "merge_nhwc",
            source_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        _tensor(
            "path_nhwc",
            source_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        _tensor(
            "path_nchw",
            region_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        _tensor(
            "residual_nchw",
            region_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
        _tensor(
            "output_nhwc",
            source_shape,
            tensor_type=tensor_type,
            quantization=qparam,
        ),
    ]
    operators = [
        FakeOperator(opcodeIndex=0, inputs=[0, 2], outputs=[3]),
        FakeOperator(opcodeIndex=0, inputs=[1, 2], outputs=[4]),
        FakeOperator(opcodeIndex=1, inputs=[3, 4], outputs=[5]),
        FakeOperator(opcodeIndex=0, inputs=[5, 6], outputs=[7]),
        FakeOperator(opcodeIndex=3, inputs=[7], outputs=[8]),
        FakeOperator(opcodeIndex=0, inputs=[8, 2], outputs=[9]),
        FakeOperator(opcodeIndex=1, inputs=[5, 9], outputs=[10]),
        FakeOperator(opcodeIndex=0, inputs=[10, 6], outputs=[11]),
    ]
    return _model(
        tensors,
        operators,
        [
            FakeBuffer(),
            _permutation_buffer(_SOURCE_TO_REGION),
            _permutation_buffer(_REGION_TO_SOURCE),
        ],
        inputs=[0, 1],
        outputs=[11],
        signature_outputs=[FakeTensorMap("output", 11)],
    )


class EliminateTransposeBoundedLayoutRegionPassTest(unittest.TestCase):
    """Test Circle-side conversion of Transpose-bounded ADD/PAD regions."""

    def test_rewrites_simple_float_add_region(self) -> None:
        """Rewrite one floating-point ADD and bypass three Transpose nodes."""

        document = _make_simple_add_document()
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        self.assertEqual(result.changes, 3)
        add = document.subgraph().operators[2]
        self.assertEqual(add.inputs, [0, 1])
        self.assertEqual(document.subgraph().outputs, [5])
        self.assertEqual(document.subgraph().tensors[5].shape, [1, 4, 5, 3])
        self.assertEqual(document.subgraph().tensors[5].name, "output_nhwc")
        self.assertTrue(
            document.subgraph().tensors[7].name.startswith("output_nhwc::dead_layout_")
        )
        names = [tensor.name for tensor in document.subgraph().tensors if tensor.name]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_rewrites_per_tensor_quantized_add_region(self) -> None:
        """Rewrite one UINT8 region while preserving one activation qparam."""

        document = _make_simple_add_document(quantized=True)
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        output = document.subgraph().tensors[5]
        self.assertEqual(output.type, _UINT8_TENSOR_TYPE)
        self.assertEqual(output.quantization.scale, [0.125])
        self.assertEqual(output.quantization.zeroPoint, [127])
        self.assertEqual(output.quantization.quantizedDimension, 0)

    def test_allows_distinct_per_tensor_qparams_between_add_operands(self) -> None:
        """Preserve independent branch qparams across removed Transpose nodes."""

        document = _make_simple_add_document(quantized=True)
        tensors = document.subgraph().tensors
        lhs = FakeQuantization([0.125], [127])
        rhs = FakeQuantization([0.25], [120])
        output = FakeQuantization([0.5], [111])
        tensors[0].quantization = lhs
        tensors[3].quantization = lhs
        tensors[1].quantization = rhs
        tensors[4].quantization = rhs
        tensors[5].quantization = output
        tensors[7].quantization = output

        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        self.assertEqual(document.subgraph().operators[2].inputs, [0, 1])
        self.assertEqual(document.subgraph().tensors[5].quantization, output)

    def test_rewrites_downsample_pad_add_region(self) -> None:
        """Move channel PAD and ADD to NHWC and remap the padding constant."""

        document = _make_downsample_document()
        old_padding_payload = document.model.buffers[2].data
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        self.assertEqual(result.changes, 3)
        subgraph = document.subgraph()
        pad = subgraph.operators[2]
        add = subgraph.operators[3]
        self.assertEqual(pad.inputs[0], 1)
        self.assertNotEqual(pad.inputs[1], 5)
        self.assertEqual(subgraph.tensors[pad.outputs[0]].shape, [1, 4, 5, 6])
        self.assertEqual(add.inputs, [pad.outputs[0], 0])
        self.assertEqual(subgraph.tensors[add.outputs[0]].shape, [1, 4, 5, 6])
        new_padding = subgraph.tensors[pad.inputs[1]]
        self.assertEqual(
            _decode_i32_buffer(document.model.buffers[new_padding.buffer]),
            (0, 0, 0, 0, 0, 0, 0, 3),
        )
        self.assertEqual(document.model.buffers[2].data, old_padding_payload)
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_rewrites_decoder_region_with_fanout_and_side_path(self) -> None:
        """Rewrite two connected ADD nodes and bypass five Transpose nodes."""

        document = _make_decoder_document()
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertTrue(result.modified)
        self.assertEqual(result.changes, 5)
        subgraph = document.subgraph()
        first_add = subgraph.operators[2]
        path = subgraph.operators[4]
        second_add = subgraph.operators[6]
        self.assertEqual(first_add.inputs, [0, 1])
        self.assertEqual(path.inputs, [5])
        self.assertEqual(second_add.inputs, [5, 8])
        self.assertEqual(subgraph.tensors[5].shape, [1, 4, 5, 3])
        self.assertEqual(subgraph.tensors[10].shape, [1, 4, 5, 3])
        self.assertEqual(subgraph.outputs, [10])
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_restart_pipeline_removes_all_dead_transposes(self) -> None:
        """Run the production pipeline until a decoder graph has no Transpose."""

        document = _make_decoder_document(quantized=True)
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
            [_ADD_BUILTIN_CODE, _RELU_BUILTIN_CODE, _ADD_BUILTIN_CODE],
        )
        self.assertEqual(document.subgraph().outputs, [4])
        self.assertEqual(document.subgraph().tensors[4].shape, [1, 4, 5, 3])
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_preserves_shared_input_transpose(self) -> None:
        """Keep an input Transpose that still feeds another live operator."""

        document = _make_simple_add_document(shared_first_transpose=True)
        context = CirclePassContext()
        EliminateTransposeBoundedLayoutRegionPass().run(document, context)
        DeadCodeEliminationPass().run(document, context)

        self.assertEqual(
            _builtin_codes(document),
            [_TRANSPOSE_BUILTIN_CODE, _ADD_BUILTIN_CODE, _RELU_BUILTIN_CODE],
        )
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_rejects_broadcasting_add(self) -> None:
        """Do not rewrite an ADD whose input shapes require broadcasting."""

        document = _make_simple_add_document(rhs_source_shape=[1, 1, 1, 3])
        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)
        self.assertEqual(document.subgraph().operators[2].inputs, [3, 4])

    def test_rejects_mismatched_transpose_qparams(self) -> None:
        """Do not cross a Transpose whose input and output qparams differ."""

        document = _make_simple_add_document(quantized=True)
        document.subgraph().tensors[3].quantization = FakeQuantization([0.25], [127])

        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)
        self.assertEqual(document.subgraph().outputs, [7])

    def test_rejects_per_channel_activation_qparams(self) -> None:
        """Reject activation metadata that contains more than one scale."""

        document = _make_simple_add_document(quantized=True)
        per_channel = FakeQuantization(
            [0.125, 0.25, 0.5],
            [127, 127, 127],
            quantizedDimension=3,
        )
        document.subgraph().tensors[0].quantization = per_channel
        document.subgraph().tensors[3].quantization = per_channel

        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)
        self.assertEqual(document.subgraph().outputs, [7])

    def test_rejects_mismatched_boundary_shape_signature(self) -> None:
        """Reject a Transpose boundary with inconsistent shape signatures."""

        document = _make_simple_add_document()
        document.subgraph().tensors[3].shapeSignature = [1, 3, 5, 4]

        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)
        self.assertEqual(document.subgraph().outputs, [7])

    def test_rejects_negative_padding_even_when_total_padding_matches(self) -> None:
        """Reject PAD constants that use unsupported negative padding values."""

        document = _make_downsample_document()
        document.model.buffers[2] = _padding_buffer(((0, 0), (-1, 4), (0, 0), (0, 0)))

        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)
        self.assertEqual(document.subgraph().operators[2].inputs, [4, 5])

    def test_rejects_dead_output_transpose_boundary(self) -> None:
        """Do not count a dead Transpose output as a visible region boundary."""

        document = _make_simple_add_document()
        document.subgraph().outputs = []
        document.model.signatureDefs[0].outputs = []

        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)
        self.assertEqual(document.subgraph().operators[2].inputs, [3, 4])

    def test_rejects_region_tensor_exposed_as_graph_output(self) -> None:
        """Do not change the layout of an already exposed region tensor."""

        document = _make_simple_add_document()
        subgraph = document.subgraph()
        subgraph.outputs.append(5)
        document.model.signatureDefs[0].outputs.append(
            FakeTensorMap("region_output", 5)
        )

        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)
        self.assertEqual(subgraph.operators[2].inputs, [3, 4])

    def test_rejects_unsupported_direct_region_consumer(self) -> None:
        """Reject a region output that directly feeds a non-Transpose operator."""

        document = _make_simple_add_document()
        subgraph = document.subgraph()
        subgraph.tensors.append(
            _tensor(
                "direct_consumer_output",
                [1, 3, 4, 5],
                tensor_type=_FLOAT32_TENSOR_TYPE,
                quantization=None,
            )
        )
        subgraph.operators.append(FakeOperator(opcodeIndex=3, inputs=[5], outputs=[8]))
        subgraph.outputs.append(8)
        document.model.signatureDefs[0].outputs.append(FakeTensorMap("direct", 8))

        result = EliminateTransposeBoundedLayoutRegionPass().run(
            document,
            CirclePassContext(),
        )

        self.assertFalse(result.modified)
        self.assertEqual(subgraph.operators[2].inputs, [3, 4])


if __name__ == "__main__":
    unittest.main()

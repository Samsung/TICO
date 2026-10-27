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

"""Exercise MX grouping through the public PyTorch-to-Circle conversion path."""

import unittest

import tico
import torch
from circle_schema import circle
from tico.utils.errors import NotYetSupportedError
from tico.utils.mx.mx_ops import quantize_mx


class _MXIdentity(torch.nn.Module):
    def __init__(self, axis=-1, elem_format="int8"):
        super().__init__()
        self.axis = axis
        self.elem_format = elem_format

    def forward(self, value):
        return quantize_mx(value, self.elem_format, self.axis)


class _MXTranspose(_MXIdentity):
    def __init__(self, axis=3, elem_format="int8", output_axis=None):
        super().__init__(axis, elem_format)
        self.output_axis = output_axis

    def forward(self, value):
        value = super().forward(value).transpose(-2, -1)
        if self.output_axis is not None:
            value = quantize_mx(value, self.elem_format, self.output_axis)
        return value


class _MXHeadSplit(_MXIdentity):
    def __init__(self, head_dim=64):
        super().__init__(axis=2)
        self.head_dim = head_dim

    def forward(self, value):
        value = super().forward(value)
        return value.reshape(1, 3, -1, self.head_dim).transpose(1, 2)


class _MXHeadMerge(_MXIdentity):
    def __init__(self):
        super().__init__(axis=3)

    def forward(self, value):
        value = super().forward(value)
        return value.transpose(1, 2).reshape(1, 3, 128)


class _MXAttentionProjection(torch.nn.Module):
    """Two 64-element heads, with a distinct sequence axis as in Q/K attention."""

    weight: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.linspace(-1, 1, 128 * 64).reshape(128, 64))

    def forward(self, value):
        value = torch.nn.functional.linear(value, self.weight)
        value = quantize_mx(value, "int8", -1)
        heads = value.reshape(1, 5, 2, 64).transpose(1, 2)
        return heads, heads.transpose(-2, -1)


class _MixedMXAffine(torch.nn.Module):
    def forward(self, value):
        value = torch.fake_quantize_per_tensor_affine(value, 0.25, 0, -32768, 32767)
        value = quantize_mx(value, "int8", -1).transpose(-2, -1)
        return torch.fake_quantize_per_tensor_affine(value, 0.5, 0, -32768, 32767)


class _MXSharedDomains(torch.nn.Module):
    def forward(self, value):
        value = quantize_mx(value, "int8", -1)
        return value, quantize_mx(value.transpose(-2, -1), "int8", -1)


class _MXSplitToAffine(torch.nn.Module):
    def forward(self, value):
        value = quantize_mx(value, "int8", -1)
        first, _ = value.split([32, 32], dim=-1)
        return torch.fake_quantize_per_tensor_affine(first, 0.5, 0, -32768, 32767)


class _MXPadToAffine(torch.nn.Module):
    def forward(self, value):
        value = quantize_mx(value, "int8", -1)
        value = torch.nn.functional.pad(value, (0, 32))
        return torch.fake_quantize_per_tensor_affine(value, 0.5, 0, -32768, 32767)


def _export(module, value, optimize):
    converted = tico.convert(
        module.eval(),
        (value,),
        config=tico.CompileConfigV1(circle_optimize=optimize),
    )
    return circle.Model.ModelT.InitFromObj(
        circle.Model.Model.GetRootAsModel(converted.circle_binary, 0)
    )


def _quantize_operators(model):
    return [
        op
        for op in model.subgraphs[0].operators
        if model.operatorCodes[op.opcodeIndex].builtinCode
        == circle.BuiltinOperator.BuiltinOperator.QUANTIZE
    ]


class MXAxisExportTest(unittest.TestCase):
    def _assert_tensor(self, tensor, shape, axis, dtype=None):
        self.assertEqual(list(tensor.shape), list(shape))
        self.assertIsNotNone(tensor.quantization)
        self.assertEqual(tensor.quantization.quantizedDimension, axis)
        self.assertGreaterEqual(axis, 0)
        self.assertLess(axis, len(shape))
        if dtype is not None:
            self.assertEqual(tensor.type, dtype)

    def _assert_io(self, model, input_shape, input_axis, output_shape, output_axis):
        graph = model.subgraphs[0]
        self._assert_tensor(graph.tensors[graph.inputs[0]], input_shape, input_axis)
        self._assert_tensor(graph.tensors[graph.outputs[0]], output_shape, output_axis)

    def test_negative_axis_is_normalized_without_a_layout_change(self):
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                model = _export(_MXIdentity(), torch.ones(1, 2, 3, 64), optimize)
                self._assert_io(model, (1, 2, 3, 64), 3, (1, 2, 3, 64), 3)

    def test_negative_axis_follows_transpose(self):
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                model = _export(
                    _MXTranspose(axis=-1), torch.ones(1, 2, 3, 64), optimize
                )
                self._assert_io(model, (1, 2, 3, 64), 3, (1, 2, 64, 3), 2)

    def test_positive_axis_follows_transpose_for_supported_formats(self):
        formats = (
            ("int8", circle.TensorType.TensorType.MXINT8),
            ("fp4", circle.TensorType.TensorType.MXFP4),
            ("fp4_e2m1", circle.TensorType.TensorType.MXFP4),
        )
        for elem_format, dtype in formats:
            for optimize in (False, True):
                with self.subTest(elem_format=elem_format, optimize=optimize):
                    model = _export(
                        _MXTranspose(elem_format=elem_format),
                        torch.ones(1, 2, 3, 64),
                        optimize,
                    )
                    self._assert_io(model, (1, 2, 3, 64), 3, (1, 2, 64, 3), 2)
                    graph = model.subgraphs[0]
                    self._assert_tensor(
                        graph.tensors[graph.outputs[0]], (1, 2, 64, 3), 2, dtype
                    )

    def test_aligned_head_split_moves_channel_axis(self):
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                model = _export(_MXHeadSplit(), torch.ones(1, 3, 128), optimize)
                self._assert_io(model, (1, 3, 128), 2, (1, 2, 3, 64), 3)

    def test_aligned_head_merge_reduces_axis_without_losing_grouping(self):
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                model = _export(_MXHeadMerge(), torch.ones(1, 2, 3, 64), optimize)
                self._assert_io(model, (1, 2, 3, 64), 3, (1, 3, 128), 2)

    def test_matching_observer_after_transpose_keeps_the_mapped_axis(self):
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                model = _export(
                    _MXTranspose(output_axis=2), torch.ones(1, 2, 3, 64), optimize
                )
                self._assert_io(model, (1, 2, 3, 64), 3, (1, 2, 64, 3), 2)
                self.assertEqual(_quantize_operators(model), [])

    def test_different_grouping_after_transpose_retains_quantize(self):
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                model = _export(
                    _MXTranspose(output_axis=3), torch.ones(1, 2, 3, 64), optimize
                )
                self._assert_io(model, (1, 2, 3, 64), 3, (1, 2, 64, 3), 3)
                quantizes = _quantize_operators(model)
                self.assertEqual(len(quantizes), 1)
                graph = model.subgraphs[0]
                self._assert_tensor(
                    graph.tensors[quantizes[0].inputs[0]], (1, 2, 64, 3), 2
                )
                self._assert_tensor(
                    graph.tensors[quantizes[0].outputs[0]], (1, 2, 64, 3), 3
                )

    def test_reshape_that_splits_mx_blocks_is_rejected(self):
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                with self.assertRaisesRegex(
                    NotYetSupportedError, "MX.*reshape|reshape.*MX"
                ):
                    _export(_MXHeadSplit(head_dim=16), torch.ones(1, 3, 64), optimize)

    def test_out_of_range_axes_are_rejected(self):
        value = torch.ones(1, 2, 3, 64)
        for axis in (-5, 4):
            with self.subTest(axis=axis):
                with self.assertRaisesRegex(ValueError, "MX axis.*rank"):
                    _MXIdentity(axis=axis)(value)

    def test_mixed_affine_and_mx_boundaries_preserve_the_transposed_axis(self):
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                model = _export(_MixedMXAffine(), torch.ones(1, 2, 3, 64), optimize)
                graph = model.subgraphs[0]
                quantizes = _quantize_operators(model)
                self.assertEqual(len(quantizes), 2)
                transitions = [
                    (graph.tensors[op.inputs[0]], graph.tensors[op.outputs[0]])
                    for op in quantizes
                ]
                into_mx = next(
                    pair
                    for pair in transitions
                    if pair[1].type == circle.TensorType.TensorType.MXINT8
                )
                from_mx = next(
                    pair
                    for pair in transitions
                    if pair[0].type == circle.TensorType.TensorType.MXINT8
                )
                self.assertEqual(into_mx[0].type, circle.TensorType.TensorType.INT16)
                self._assert_tensor(into_mx[1], (1, 2, 3, 64), 3)
                self._assert_tensor(from_mx[0], (1, 2, 64, 3), 2)
                self.assertEqual(from_mx[1].type, circle.TensorType.TensorType.INT16)

    def test_shared_quantized_input_requantizes_after_transpose(self):
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                model = _export(_MXSharedDomains(), torch.ones(1, 3, 64), optimize)
                graph = model.subgraphs[0]
                self.assertEqual(graph.inputs[0], graph.outputs[0])
                self._assert_tensor(
                    graph.tensors[graph.outputs[0]],
                    (1, 3, 64),
                    2,
                    circle.TensorType.TensorType.MXINT8,
                )
                transposes = [
                    op
                    for op in graph.operators
                    if model.operatorCodes[op.opcodeIndex].builtinCode
                    == circle.BuiltinOperator.BuiltinOperator.TRANSPOSE
                ]
                self.assertEqual(len(transposes), 1)
                self._assert_tensor(
                    graph.tensors[transposes[0].inputs[0]],
                    (1, 3, 64),
                    2,
                    circle.TensorType.TensorType.MXINT8,
                )
                self._assert_tensor(
                    graph.tensors[transposes[0].outputs[0]],
                    (1, 64, 3),
                    1,
                    circle.TensorType.TensorType.MXINT8,
                )
                quantizes = _quantize_operators(model)
                self.assertEqual(len(quantizes), 1)
                self.assertEqual(quantizes[0].inputs[0], transposes[0].outputs[0])
                self._assert_tensor(
                    graph.tensors[graph.outputs[1]],
                    (1, 64, 3),
                    2,
                    circle.TensorType.TensorType.MXINT8,
                )

    def test_split_preserves_mx_on_unused_output_and_affine_boundary(self):
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                model = _export(_MXSplitToAffine(), torch.ones(1, 3, 64), optimize)
                graph = model.subgraphs[0]
                split_ops = [
                    op
                    for op in graph.operators
                    if model.operatorCodes[op.opcodeIndex].builtinCode
                    in (
                        circle.BuiltinOperator.BuiltinOperator.SPLIT,
                        circle.BuiltinOperator.BuiltinOperator.SPLIT_V,
                    )
                ]
                self.assertEqual(len(split_ops), 1)
                self.assertEqual(len(split_ops[0].outputs), 2)
                for index in split_ops[0].outputs:
                    self._assert_tensor(
                        graph.tensors[index],
                        (1, 3, 32),
                        2,
                        circle.TensorType.TensorType.MXINT8,
                    )
                quantizes = _quantize_operators(model)
                self.assertEqual(len(quantizes), 1)
                self.assertEqual(quantizes[0].inputs[0], split_ops[0].outputs[0])
                self.assertEqual(
                    graph.tensors[graph.outputs[0]].type,
                    circle.TensorType.TensorType.INT16,
                )

    def test_aligned_padding_preserves_mx_before_affine_boundary(self):
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                model = _export(_MXPadToAffine(), torch.ones(1, 3, 64), optimize)
                graph = model.subgraphs[0]
                pad_ops = [
                    op
                    for op in graph.operators
                    if model.operatorCodes[op.opcodeIndex].builtinCode
                    in (
                        circle.BuiltinOperator.BuiltinOperator.PAD,
                        circle.BuiltinOperator.BuiltinOperator.PADV2,
                    )
                ]
                self.assertEqual(len(pad_ops), 1)
                self._assert_tensor(
                    graph.tensors[pad_ops[0].outputs[0]],
                    (1, 3, 96),
                    2,
                    circle.TensorType.TensorType.MXINT8,
                )
                quantizes = _quantize_operators(model)
                self.assertEqual(len(quantizes), 1)
                self.assertEqual(quantizes[0].inputs[0], pad_ops[0].outputs[0])
                self.assertEqual(
                    graph.tensors[graph.outputs[0]].type,
                    circle.TensorType.TensorType.INT16,
                )

    def test_projection_head_reshape_and_key_transpose_keep_feature_axis(self):
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                model = _export(
                    _MXAttentionProjection(), torch.ones(1, 5, 64), optimize
                )
                graph = model.subgraphs[0]
                self.assertEqual(len(graph.outputs), 2)
                self._assert_tensor(graph.tensors[graph.outputs[0]], (1, 2, 5, 64), 3)
                self._assert_tensor(graph.tensors[graph.outputs[1]], (1, 2, 64, 5), 2)

    def test_aligned_layout_changes_preserve_eager_mx_values(self):
        # Different token magnitudes make a stale sequence axis numerically visible.
        generator = torch.Generator().manual_seed(0)
        value = torch.randn(1, 2, 3, 64, generator=generator)
        value = value * torch.tensor([0.01, 1.0, 100.0]).reshape(1, 1, 3, 1)
        quantized = quantize_mx(value, "int8", -1)
        transposed = value.transpose(-2, -1)
        self.assertTrue(
            torch.equal(
                quantized.transpose(-2, -1), quantize_mx(transposed, "int8", -2)
            )
        )
        self.assertFalse(
            torch.equal(
                quantized.transpose(-2, -1), quantize_mx(transposed, "int8", -1)
            )
        )
        merged = value.transpose(1, 2).reshape(1, 3, 128)
        self.assertTrue(
            torch.equal(
                quantized.transpose(1, 2).reshape(1, 3, 128),
                quantize_mx(merged, "int8", -1),
            )
        )


if __name__ == "__main__":
    unittest.main()

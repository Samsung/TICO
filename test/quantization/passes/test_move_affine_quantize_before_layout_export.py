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

"""Public export policy, serialized contracts, and supported reference suffix."""

import unittest

import numpy as np

import tico
import torch
from circle_schema import circle
from tico.circle import CircleDocument, extract_by_operator_indices
from tico.circle.runtime import (
    CircleReferenceRuntime,
    ExecutionMode,
    UnsupportedCircleOperatorError,
)
from tico.utils.errors import NotYetSupportedError
from tico.utils.mx.mx_ops import quantize_mx

_SCALE = 0.03125
_ZERO_POINT = 3
_TYPES = circle.TensorType.TensorType
_OPS = circle.BuiltinOperator.BuiltinOperator


class _AttentionProjection(torch.nn.Module):
    """Share the original MX projection with an ordered graph output."""

    weight: torch.Tensor

    def __init__(self, head_dim=32):
        super().__init__()
        self.register_buffer("weight", torch.eye(64))
        self.head_dim = head_dim

    def forward(self, value):
        source = quantize_mx(torch.nn.functional.linear(value, self.weight), "int8", -1)
        heads = source.reshape(1, 3, 64 // self.head_dim, self.head_dim).transpose(1, 2)
        result = torch.fake_quantize_per_tensor_affine(
            heads, _SCALE, _ZERO_POINT, -32768, 32767
        )
        return result, source


class _SharedProjection(_AttentionProjection):
    def forward(self, value):
        result, source = super().forward(value)
        return result, source.transpose(1, 2), source


def _operator_code(document, operator):
    return document.model.operatorCodes[operator.opcodeIndex].builtinCode


def _export(module, value, optimize, enabled=None):
    options = {"circle_optimize": optimize}
    if enabled is not None:
        options["move_affine_quantize_before_layout"] = enabled
    converted = tico.convert(
        module.eval(), (value,), config=tico.CompileConfigV1(**options)
    )
    document = CircleDocument.from_bytes(converted.circle_binary)
    document.verify(raise_on_error=True)
    return document


class MoveAffineQuantizeBeforeLayoutExportTest(unittest.TestCase):
    def _assert_affine(self, tensor):
        self.assertEqual(tensor.type, _TYPES.INT16)
        self.assertIsNotNone(tensor.quantization)
        np.testing.assert_array_equal(tensor.quantization.scale, [_SCALE])
        np.testing.assert_array_equal(tensor.quantization.zeroPoint, [_ZERO_POINT])
        self.assertEqual(tensor.quantization.quantizedDimension, 0)

    def _assert_contracts(self, document, head_dim, moved):
        graph = document.model.subgraphs[0]
        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(len(graph.outputs), 2)
        inp = graph.tensors[graph.inputs[0]]
        self.assertEqual(list(inp.shape), [1, 3, 64])
        self.assertEqual(inp.type, _TYPES.FLOAT32)
        self.assertIsNone(inp.quantization)
        result = graph.tensors[graph.outputs[0]]
        self.assertEqual(list(result.shape), [1, 64 // head_dim, 3, head_dim])
        self._assert_affine(result)
        source = graph.tensors[graph.outputs[1]]
        self.assertEqual(list(source.shape), [1, 3, 64])
        self.assertEqual(source.type, _TYPES.MXINT8)
        self.assertEqual(source.quantization.quantizedDimension, 2)
        self.assertIsNone(source.quantization.scale)
        self.assertIsNone(source.quantization.zeroPoint)

        layouts = [
            (index, operator)
            for index, operator in enumerate(graph.operators)
            if _operator_code(document, operator) in (_OPS.RESHAPE, _OPS.TRANSPOSE)
        ]
        self.assertEqual(len(layouts), 2)
        for _, layout in layouts:
            for index in (layout.inputs[0], layout.outputs[0]):
                tensor = graph.tensors[index]
                if moved:
                    self._assert_affine(tensor)
                else:
                    self.assertEqual(tensor.type, _TYPES.MXINT8)
        quantizers = [
            operator
            for operator in graph.operators
            if _operator_code(document, operator) == _OPS.QUANTIZE
        ]
        self.assertEqual(len(quantizers), 1)
        conversion = quantizers[0]
        self.assertEqual(graph.tensors[conversion.inputs[0]].type, _TYPES.MXINT8)
        self._assert_affine(graph.tensors[conversion.outputs[0]])
        if moved:
            self.assertEqual(conversion.inputs[0], graph.outputs[1])
            self.assertEqual(conversion.outputs[0], layouts[0][1].inputs[0])
            self.assertEqual(
                list(graph.tensors[conversion.outputs[0]].shape), [1, 3, 64]
            )
        else:
            self.assertEqual(conversion.inputs[0], layouts[-1][1].outputs[0])
        return layouts

    def test_public_option_preserves_io_and_moves_only_target_layout(self):
        value = torch.linspace(-3, 3, 192).reshape(1, 3, 64)
        for optimize in (False, True):
            for enabled in (None, False, True):
                with self.subTest(optimize=optimize, enabled=enabled):
                    document = _export(_AttentionProjection(), value, optimize, enabled)
                    self._assert_contracts(document, 32, moved=enabled is not False)

    def test_omitted_compile_config_uses_default_layout_policy(self):
        value = torch.linspace(-3, 3, 192).reshape(1, 3, 64)
        converted = tico.convert(_AttentionProjection(head_dim=16).eval(), (value,))
        document = CircleDocument.from_bytes(converted.circle_binary)
        document.verify(raise_on_error=True)
        self._assert_contracts(document, 16, moved=True)

    def test_shared_projection_keeps_unrelated_mx_layout_and_output(self):
        value = torch.linspace(-3.01, 3.03, 192).reshape(1, 3, 64)
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                document = _export(
                    _SharedProjection(head_dim=16), value, optimize, True
                )
                graph = document.model.subgraphs[0]
                self.assertEqual(len(graph.inputs), 1)
                self.assertEqual(len(graph.outputs), 3)
                inp = graph.tensors[graph.inputs[0]]
                self.assertEqual(list(inp.shape), [1, 3, 64])
                self.assertEqual(inp.type, _TYPES.FLOAT32)
                self.assertIsNone(inp.quantization)
                target, unrelated, source = [graph.tensors[i] for i in graph.outputs]
                self.assertEqual(list(target.shape), [1, 4, 3, 16])
                self._assert_affine(target)
                for tensor, shape, axis in (
                    (unrelated, [1, 64, 3], 1),
                    (source, [1, 3, 64], 2),
                ):
                    self.assertEqual(list(tensor.shape), shape)
                    self.assertEqual(tensor.type, _TYPES.MXINT8)
                    self.assertEqual(tensor.quantization.quantizedDimension, axis)
                    self.assertIsNone(tensor.quantization.scale)
                    self.assertIsNone(tensor.quantization.zeroPoint)

                producers = {
                    index: operator
                    for operator in graph.operators
                    for index in operator.outputs
                }
                # Follow only the selected output's layout chain. The unrelated
                # MX transpose is intentionally retained elsewhere in this graph.
                current = graph.outputs[0]
                target_layouts = []
                while _operator_code(document, producers[current]) in (
                    _OPS.RESHAPE,
                    _OPS.TRANSPOSE,
                ):
                    layout = producers[current]
                    target_layouts.append(layout)
                    for index in (layout.inputs[0], layout.outputs[0]):
                        self._assert_affine(graph.tensors[index])
                    current = layout.inputs[0]
                self.assertEqual(len(target_layouts), 2)
                conversion = producers[current]
                self.assertEqual(_operator_code(document, conversion), _OPS.QUANTIZE)
                self.assertEqual(conversion.inputs[0], graph.outputs[2])
                self._assert_affine(graph.tensors[conversion.outputs[0]])
                other_layout = producers[graph.outputs[1]]
                self.assertEqual(_operator_code(document, other_layout), _OPS.TRANSPOSE)
                self.assertEqual(other_layout.inputs[0], graph.outputs[2])

    @torch.no_grad()
    def test_regrouping_export_and_affine_suffix_reference_values(self):
        # A 16-element head splits the source's 32-element MX blocks. The existing
        # INT16 boundary permits this layout after conversion without regrouping MX.
        value = torch.linspace(-3.01, 3.03, 192).reshape(1, 3, 64)
        value[0, 0, :2] = torch.tensor([-2049.125, 2051.25])
        module = _AttentionProjection(head_dim=16).eval()
        expected, source = module(value)
        self.assertFalse(torch.equal(source, value))  # MX rounding is observable.
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                with self.assertRaisesRegex(
                    NotYetSupportedError, "MX.*reshape|reshape.*MX"
                ):
                    _export(module, value, optimize, False)
                document = _export(module, value, optimize, True)
                layouts = self._assert_contracts(document, 16, moved=True)
                # Full-model MX execution is an explicit runtime limitation,
                # separate from successful structural verification above.
                with self.assertRaisesRegex(
                    UnsupportedCircleOperatorError, "tensor type -7.*no value codec"
                ):
                    CircleReferenceRuntime(document)

                # Execute the actual exported INT16 layout region with the existing
                # boundary's codes. This checks both serialized layout contracts
                # and exact value preservation in the supported reference path.
                suffix = extract_by_operator_indices(
                    document, [index for index, _ in layouts]
                ).document
                suffix.verify(raise_on_error=True)
                runtime = CircleReferenceRuntime(suffix)
                boundary = torch.fake_quantize_per_tensor_affine(
                    source, _SCALE, _ZERO_POINT, -32768, 32767
                )
                codes = torch.round(boundary / _SCALE + _ZERO_POINT).to(torch.int16)
                self.assertEqual(codes.min().item(), -32768)
                self.assertEqual(codes.max().item(), 32767)
                native = runtime.run([codes.numpy()], mode=ExecutionMode.NATIVE)
                fake = runtime.run([boundary.numpy()], mode=ExecutionMode.FAKE_QUANTIZE)
                expected_codes = torch.round(expected / _SCALE + _ZERO_POINT).to(
                    torch.int16
                )
                np.testing.assert_array_equal(native.outputs[0], expected_codes.numpy())
                np.testing.assert_array_equal(fake.outputs[0], expected.numpy())
                self.assertEqual(native.outputs[0].dtype, np.int16)
                self.assertEqual(fake.outputs[0].dtype, np.float32)
                self.assertEqual(native.outputs[0].shape, (1, 4, 3, 16))


if __name__ == "__main__":
    unittest.main()

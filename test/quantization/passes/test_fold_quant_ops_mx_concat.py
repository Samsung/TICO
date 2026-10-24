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

"""Keep independent affine values intact before full-sequence MX quantization."""

import unittest

import tico
import torch
from circle_schema import circle
from tico.circle.document import CircleDocument
from tico.circle.io import model_from_bytes
from tico.circle.operations import extract_by_tensor_indices
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.serialize.quant_param import QPARAM_KEY
from tico.utils.errors import NotYetSupportedError
from tico.utils.mx.mx_ops import quantize_mx


class _AffineReadThenMXConcat(torch.nn.Module):
    def __init__(
        self, read_scale=0.5, read_zp=3, read_uint8=False, view_before_mx=False
    ):
        super().__init__()
        self.read_scale = read_scale
        self.read_zp = read_zp
        self.read_uint8 = read_uint8
        self.view_before_mx = view_before_mx

    def forward(self, current, past):
        new = torch.fake_quantize_per_tensor_affine(current, 0.125, 0, -32768, 32767)
        storage_past = torch.fake_quantize_per_tensor_affine(
            past, 0.125, 0, -32768, 32767
        )
        read = torch.fake_quantize_per_tensor_affine(
            storage_past,
            self.read_scale,
            self.read_zp,
            0 if self.read_uint8 else -32768,
            255 if self.read_uint8 else 32767,
        )
        full = torch.cat((read, new), dim=-2)
        if self.view_before_mx:
            full = full.transpose(-2, -1).reshape(1, 32, 32)
        compute = quantize_mx(
            full, elem_format="int8", axis=-1 if self.view_before_mx else -2
        )
        storage = torch.cat((storage_past, new), dim=-2)
        return compute, new, storage, read


def _builtin(model, operator):
    code = model.operatorCodes[operator.opcodeIndex]
    return code.builtinCode or code.deprecatedBuiltinCode


class FoldQuantOpsMXConcatTest(unittest.TestCase):
    @staticmethod
    def _inputs():
        current = torch.full((1, 1, 1, 32), 0.375)
        current[..., 16:] = 100.0
        past = torch.full((1, 1, 31, 32), 0.375)
        return current, past

    def _assert_affine(self, tensor, *, dtype, scale, zero_point):
        self.assertEqual(tensor.type, dtype)
        self.assertEqual(tuple(tensor.quantization.scale), (scale,))
        self.assertEqual(tuple(tensor.quantization.zeroPoint), (zero_point,))

    def _assert_float_concat_reference(self, model, concat, args, expected_read):
        # MX reference execution is unsupported. Execute the serialized affine
        # conversions and FLOAT32 concat prefix using the existing extractor.
        graph = model.subgraphs[0]
        output_index = int(concat.outputs[0])
        prefix = extract_by_tensor_indices(
            CircleDocument(model),
            from_tensors=graph.inputs,
            to_tensors=(output_index,),
        )
        inputs = dict(zip(map(int, graph.inputs), args, strict=True))
        result = CircleReferenceRuntime(prefix.document).run(
            tuple(inputs[index].numpy() for index in prefix.source_boundary.inputs),
            mode=ExecutionMode.FAKE_QUANTIZE,
        )
        position = prefix.source_boundary.outputs.index(output_index)
        torch.testing.assert_close(
            torch.from_numpy(result.outputs[position]),
            torch.cat((expected_read, args[0]), dim=-2),
            rtol=0,
            atol=0,
        )

    def test_different_affine_grids_preserve_values_and_storage(self):
        T = circle.TensorType.TensorType
        B = circle.BuiltinOperator.BuiltinOperator
        for read_uint8, view_before_mx in (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ):
            with self.subTest(read_uint8=read_uint8, view_before_mx=view_before_mx):
                read_zp = 127 if read_uint8 else 3
                module = _AffineReadThenMXConcat(
                    read_zp=read_zp,
                    read_uint8=read_uint8,
                    view_before_mx=view_before_mx,
                ).eval()
                args = self._inputs()
                original = tuple(value.clone() for value in args)
                eager = module(*args)
                expected_read = torch.full_like(args[1], 0.5)
                expected = quantize_mx(
                    torch.cat((expected_read, args[0]), dim=-2),
                    elem_format="int8",
                    axis=-2,
                )
                if view_before_mx:
                    expected = expected.transpose(-2, -1).reshape(1, 32, 32)
                torch.testing.assert_close(eager[0], expected, rtol=0, atol=0)
                torch.testing.assert_close(eager[1], args[0], rtol=0, atol=0)
                torch.testing.assert_close(
                    eager[2], torch.cat((args[1], args[0]), dim=-2), rtol=0, atol=0
                )
                torch.testing.assert_close(eager[3], expected_read, rtol=0, atol=0)
                # Mapping current through the past grid adds rounding, and
                # UINT8 also clips the outlier to 64. Neither is intended here.
                aligned_current = torch.fake_quantize_per_tensor_affine(
                    args[0],
                    0.5,
                    read_zp,
                    0 if read_uint8 else -32768,
                    255 if read_uint8 else 32767,
                )
                wrong_read_grid = quantize_mx(
                    torch.cat((expected_read, aligned_current), dim=-2),
                    elem_format="int8",
                    axis=-2,
                )
                if view_before_mx:
                    wrong_read_grid = wrong_read_grid.transpose(-2, -1).reshape(
                        1, 32, 32
                    )
                self.assertFalse(torch.equal(expected, wrong_read_grid))
                per_part_mx = torch.cat(
                    (
                        quantize_mx(expected_read, "int8", -2),
                        quantize_mx(args[0], "int8", -2),
                    ),
                    dim=-2,
                )
                if view_before_mx:
                    per_part_mx = per_part_mx.transpose(-2, -1).reshape(1, 32, 32)
                self.assertFalse(torch.equal(expected, per_part_mx))
                ep = torch.export.export(module, args)
                exported = ep.module()(*args)
                torch.testing.assert_close(exported, eager, rtol=0, atol=0)
                for value, before in zip(args, original, strict=True):
                    torch.testing.assert_close(value, before, rtol=0, atol=0)
                eager[0].zero_()
                torch.testing.assert_close(eager[1], args[0], rtol=0, atol=0)
                torch.testing.assert_close(eager[3], expected_read, rtol=0, atol=0)

                model = model_from_bytes(tico.convert(module, args).circle_binary)
                graph = model.subgraphs[0]
                for index in (*graph.inputs, graph.outputs[1], graph.outputs[2]):
                    self._assert_affine(
                        graph.tensors[index], dtype=T.INT16, scale=0.125, zero_point=0
                    )
                read_dtype = T.UINT8 if read_uint8 else T.INT16
                self._assert_affine(
                    graph.tensors[graph.outputs[3]],
                    dtype=read_dtype,
                    scale=0.5,
                    zero_point=read_zp,
                )
                mx_output = graph.tensors[graph.outputs[0]]
                self.assertEqual(mx_output.type, T.MXINT8)
                self.assertEqual(mx_output.quantization.quantizedDimension, 2)
                quantize = next(
                    op
                    for op in graph.operators
                    if _builtin(model, op) == B.QUANTIZE
                    and op.outputs[0] == graph.outputs[0]
                )
                index = quantize.inputs[0]
                view_codes = []
                while True:
                    concat = next(op for op in graph.operators if index in op.outputs)
                    if _builtin(model, concat) not in (B.RESHAPE, B.TRANSPOSE):
                        break
                    view_codes.append(_builtin(model, concat))
                    self.assertEqual(graph.tensors[index].type, T.FLOAT32)
                    self.assertIsNone(graph.tensors[index].quantization)
                    index = concat.inputs[0]
                self.assertEqual(
                    set(view_codes),
                    {B.RESHAPE, B.TRANSPOSE} if view_before_mx else set(),
                )
                self.assertEqual(_builtin(model, concat), B.CONCATENATION)
                self.assertEqual(graph.tensors[concat.outputs[0]].type, T.FLOAT32)
                self.assertEqual(
                    list(graph.tensors[concat.outputs[0]].shape), [1, 1, 32, 32]
                )
                for index, dtype, scale, zp in zip(
                    concat.inputs,
                    (read_dtype, T.INT16),
                    (0.5, 0.125),
                    (read_zp, 0),
                    strict=True,
                ):
                    value = graph.tensors[index]
                    self.assertEqual(value.type, T.FLOAT32)
                    self.assertIsNone(value.quantization)
                    dequantize = next(
                        op for op in graph.operators if index in op.outputs
                    )
                    self.assertEqual(_builtin(model, dequantize), B.DEQUANTIZE)
                    self._assert_affine(
                        graph.tensors[dequantize.inputs[0]],
                        dtype=dtype,
                        scale=scale,
                        zero_point=zp,
                    )

                self._assert_float_concat_reference(model, concat, args, expected_read)

    def test_equal_affine_grids_keep_integer_concat_without_bridges(self):
        module = _AffineReadThenMXConcat(read_scale=0.125, read_zp=0).eval()
        ep = torch.export.export(module, self._inputs())
        DecomposeFakeQuantize().call(ep)
        FoldQuantOps().call(ep)
        quantize = next(
            node
            for node in ep.graph.nodes
            if node.target == torch.ops.circle_custom.quantize_mx.default
        )
        concat = quantize.args[0]
        self.assertEqual(concat.target, torch.ops.aten.cat.default)
        self.assertEqual(concat.meta[QPARAM_KEY].dtype, "int16")
        self.assertEqual(concat.meta[QPARAM_KEY].scale, [0.125])
        self.assertEqual(concat.meta[QPARAM_KEY].zero_point, [0])
        for source in concat.args[0]:
            self.assertEqual(source.meta[QPARAM_KEY], concat.meta[QPARAM_KEY])
            self.assertNotEqual(
                source.target,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            )
        self.assertEqual(quantize.meta[QPARAM_KEY].dtype, "mxint8")
        self.assertEqual(quantize.meta[QPARAM_KEY].quantized_dimension, 2)

    def test_float_concat_keeps_float_inputs_without_extra_dequantize(self):
        class FloatConcat(torch.nn.Module):
            def forward(self, current, past):
                return quantize_mx(torch.cat((past, current), dim=-2), "int8", -2)

        args = self._inputs()
        ep = torch.export.export(FloatConcat(), args)
        DecomposeFakeQuantize().call(ep)
        FoldQuantOps().call(ep)
        concat = next(
            node for node in ep.graph.nodes if node.target == torch.ops.aten.cat.default
        )
        self.assertNotIn(QPARAM_KEY, concat.meta)
        self.assertTrue(all(QPARAM_KEY not in source.meta for source in concat.args[0]))
        self.assertFalse(
            any(
                node.target
                == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                for node in ep.graph.nodes
            )
        )

    def test_existing_mx_input_is_not_silently_dequantized_for_float_concat(self):
        class MixedMXConcat(torch.nn.Module):
            def __init__(self, mx_first):
                super().__init__()
                self.mx_first = mx_first

            def forward(self, current, past):
                past = quantize_mx(past, "int8", -2)
                pair = (past, current) if self.mx_first else (current, past)
                return quantize_mx(torch.cat(pair, dim=-2), "int8", -2)

        for mx_first in (False, True):
            with self.subTest(mx_first=mx_first), self.assertRaisesRegex(
                NotYetSupportedError, "per-tensor affine|MX blocks"
            ):
                tico.convert(MixedMXConcat(mx_first).eval(), self._inputs())

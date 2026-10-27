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

"""Assemble affine tensor domains before one full-sequence MX conversion."""

import unittest

import tico
import torch
from circle_schema import circle
from tico.circle.document import CircleDocument
from tico.circle.io import model_from_bytes
from tico.circle.operations import extract_by_tensor_indices
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.circle.runtime.errors import UnsupportedCircleOperatorError
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.serialize.quant_param import QPARAM_KEY
from tico.utils.errors import NotYetSupportedError
from tico.utils.mx.mx_ops import quantize_mx


class _StorageThenMXConcat(torch.nn.Module):
    def forward(self, current, past):
        delta = torch.fake_quantize_per_tensor_affine(current, 0.5, 0, -32768, 32767)
        stored_past = torch.fake_quantize_per_tensor_affine(past, 0.5, 0, -32768, 32767)
        compute = quantize_mx(torch.cat((stored_past, current), dim=-2), "int8", -2)
        storage = torch.cat((stored_past, delta), dim=-2)
        return compute, delta, storage


class _AffineComputeThenMXConcat(torch.nn.Module):
    def __init__(self, view_before_mx):
        super().__init__()
        self.view_before_mx = view_before_mx

    def forward(self, current, past):
        current = torch.fake_quantize_per_tensor_affine(current, 0.25, 0, -32768, 32767)
        delta = torch.fake_quantize_per_tensor_affine(current, 0.125, 0, -32768, 32767)
        stored_past = torch.fake_quantize_per_tensor_affine(
            past, 0.125, 0, -32768, 32767
        )
        compute_past = torch.fake_quantize_per_tensor_affine(
            stored_past, 1.0, 3, -32768, 32767
        )
        compute_current = torch.fake_quantize_per_tensor_affine(
            current, 1.0, 3, -32768, 32767
        )
        full = torch.cat((compute_past, compute_current), dim=-2)
        if self.view_before_mx:
            full = full.transpose(-2, -1).reshape(1, 8, 32)
        compute = quantize_mx(full, "int8", -1 if self.view_before_mx else -2)
        storage = torch.cat((stored_past, delta), dim=-2)
        return compute, delta, storage


def _builtin(model, operator):
    code = model.operatorCodes[operator.opcodeIndex]
    return code.builtinCode or code.deprecatedBuiltinCode


class FoldQuantOpsMXConcatTest(unittest.TestCase):
    @staticmethod
    def _inputs():
        current = torch.full((1, 1, 1, 8), 0.22)
        current[..., 4:] = 100.375
        past = torch.full((1, 1, 31, 8), 0.375)
        return current, past

    def _assert_affine(self, tensor, *, scale, zero_point):
        self.assertEqual(tensor.type, circle.TensorType.TensorType.INT16)
        self.assertEqual(tuple(tensor.quantization.scale), (scale,))
        self.assertEqual(tuple(tensor.quantization.zeroPoint), (zero_point,))

    def _export(self, module, args, optimize):
        eager = module(*args)
        ep = torch.export.export(module, args)
        torch.testing.assert_close(ep.module()(*args), eager, atol=0, rtol=0)
        converted = tico.convert_from_exported_program(
            ep, config=tico.CompileConfigV1(circle_optimize=optimize)
        )
        model = model_from_bytes(converted.circle_binary)
        document = CircleDocument(model)
        document.verify()
        with self.assertRaisesRegex(UnsupportedCircleOperatorError, "no value codec"):
            CircleReferenceRuntime(document)
        return model

    def _assert_quantized_data(self, model):
        graph = model.subgraphs[0]
        B = circle.BuiltinOperator.BuiltinOperator
        semantic_inputs = {
            int(index)
            for operator in graph.operators
            if _builtin(model, operator) in (B.RESHAPE, B.TRANSPOSE)
            for index in operator.inputs[1:]
        }
        for index, tensor in enumerate(graph.tensors):
            if index in semantic_inputs:
                self.assertEqual(tensor.type, circle.TensorType.TensorType.INT32)
                self.assertIsNone(tensor.quantization)
                continue
            self.assertNotEqual(tensor.type, circle.TensorType.TensorType.FLOAT32)
            self.assertIsNotNone(tensor.quantization)
        self.assertFalse(
            any(
                _builtin(model, operator)
                == circle.BuiltinOperator.BuiltinOperator.DEQUANTIZE
                for operator in model.subgraphs[0].operators
            )
        )

    def _mx_source_concat(self, model, *, view_before_mx=False):
        graph = model.subgraphs[0]
        B = circle.BuiltinOperator.BuiltinOperator
        T = circle.TensorType.TensorType
        quantizes = [
            operator
            for operator in graph.operators
            if _builtin(model, operator) == B.QUANTIZE
            and graph.tensors[operator.outputs[0]].type == T.MXINT8
        ]
        self.assertEqual(len(quantizes), 1)
        mx_quantize = quantizes[0]
        self.assertEqual(mx_quantize.outputs[0], graph.outputs[0])
        mx_output = graph.tensors[mx_quantize.outputs[0]]
        self.assertEqual(mx_output.quantization.quantizedDimension, 2)
        self.assertEqual(
            list(mx_output.shape), [1, 8, 32] if view_before_mx else [1, 1, 32, 8]
        )
        producers = {
            int(index): operator
            for operator in graph.operators
            for index in operator.outputs
        }
        index = int(mx_quantize.inputs[0])
        views = []
        while _builtin(model, producers[index]) in (B.RESHAPE, B.TRANSPOSE):
            operator = producers[index]
            views.append(_builtin(model, operator))
            index = int(operator.inputs[0])
        self.assertEqual(
            set(views), {B.RESHAPE, B.TRANSPOSE} if view_before_mx else set()
        )
        concat = producers[index]
        self.assertEqual(_builtin(model, concat), B.CONCATENATION)
        self.assertEqual(
            [int(graph.tensors[index].shape[-2]) for index in concat.inputs], [31, 1]
        )
        self.assertEqual(list(graph.tensors[concat.outputs[0]].shape), [1, 1, 32, 8])
        return concat, producers

    def _assert_affine_prefix(self, model, concat, encoded_inputs, expected):
        graph = model.subgraphs[0]
        prefix = extract_by_tensor_indices(
            CircleDocument(model),
            from_tensors=graph.inputs,
            to_tensors=concat.outputs,
        )
        inputs = dict(zip(map(int, graph.inputs), encoded_inputs, strict=True))
        actual = CircleReferenceRuntime(prefix.document).run(
            tuple(inputs[index].numpy() for index in prefix.source_boundary.inputs),
            mode=ExecutionMode.FAKE_QUANTIZE,
        )
        torch.testing.assert_close(
            torch.from_numpy(actual.outputs[0]), expected, atol=0, rtol=0
        )

    def test_first_storage_marker_defines_input_abi_before_full_mx(self):
        module = _StorageThenMXConcat().eval()
        args = self._inputs()
        current, past = args
        eager = module(*args)
        stored_current = torch.round(current / 0.5).clamp(-32768, 32767) * 0.5
        stored_past = torch.round(past / 0.5).clamp(-32768, 32767) * 0.5
        full_raw = torch.cat((stored_past, current), dim=-2)
        expected_eager = quantize_mx(full_raw, "int8", -2)
        torch.testing.assert_close(eager[0], expected_eager, atol=0, rtol=0)
        self.assertEqual(float(eager[0][0, 0, -1, 0]), 0.21875)
        self.assertEqual(float(eager[0][0, 0, 0, -1]), 1.0)
        self.assertEqual(float(eager[0][0, 0, -1, -1]), 100.0)
        self.assertFalse(
            torch.equal(
                expected_eager,
                quantize_mx(
                    torch.cat((stored_past, stored_current), dim=-2), "int8", -2
                ),
            )
        )
        self.assertFalse(
            torch.equal(
                expected_eager,
                torch.cat(
                    (
                        quantize_mx(stored_past, "int8", -2),
                        quantize_mx(current, "int8", -2),
                    ),
                    dim=-2,
                ),
            )
        )
        # Eager keeps raw branches. In this boundary-only Circle graph, the
        # first S marker annotates the shared current input and its delta ABI.
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                model = self._export(module, args, optimize)
                graph = model.subgraphs[0]
                self._assert_quantized_data(model)
                current_index, past_index = map(int, graph.inputs)
                self.assertEqual(current_index, graph.outputs[1])
                concat, _ = self._mx_source_concat(model)
                self.assertEqual(
                    list(map(int, concat.inputs)), [past_index, current_index]
                )
                for tensor in graph.tensors:
                    if tensor.type != circle.TensorType.TensorType.MXINT8:
                        self._assert_affine(tensor, scale=0.5, zero_point=0)
                self._assert_affine_prefix(
                    model,
                    concat,
                    (stored_current, stored_past),
                    torch.cat((stored_past, stored_current), dim=-2),
                )

    def test_concat_requantizes_edges_without_changing_established_source_domains(self):
        class EstablishedAffineConcat(torch.nn.Module):
            def __init__(
                self,
                current_domain: tuple[float, int],
                past_domain: tuple[float, int],
                view_before_mx: bool,
            ):
                super().__init__()
                self.current_domain = current_domain
                self.past_domain = past_domain
                self.view_before_mx = view_before_mx

            def forward(self, current, past):
                current = torch.fake_quantize_per_tensor_affine(
                    current,
                    self.current_domain[0],
                    self.current_domain[1],
                    -32768,
                    32767,
                )
                past = torch.fake_quantize_per_tensor_affine(
                    past, self.past_domain[0], self.past_domain[1], -32768, 32767
                )
                full = torch.cat((past, current), dim=-2)
                if self.view_before_mx:
                    full = full.transpose(-2, -1).reshape(1, 8, 32)
                result = quantize_mx(full, "int8", -1 if self.view_before_mx else -2)
                return result, current, past

        args = self._inputs()
        args[0][..., :4] = 0.37
        B = circle.BuiltinOperator.BuiltinOperator
        for current_domain, past_domain in (
            ((0.5, -7), (0.125, 3)),
            ((0.125, 3), (0.5, -7)),
        ):
            current_scale, current_zp = current_domain
            past_scale, past_zp = past_domain
            compute_scale, compute_zp = max((current_domain, past_domain))
            encoded_current = torch.round(args[0] / current_scale) * current_scale
            encoded_past = torch.round(args[1] / past_scale) * past_scale
            expected = torch.cat(
                (
                    torch.round(encoded_past / compute_scale) * compute_scale,
                    torch.round(encoded_current / compute_scale) * compute_scale,
                ),
                dim=-2,
            )
            for view_before_mx in (False, True):
                for optimize in (False, True):
                    with self.subTest(
                        current=current_domain,
                        past=past_domain,
                        view_before_mx=view_before_mx,
                        optimize=optimize,
                    ):
                        module = EstablishedAffineConcat(
                            current_domain, past_domain, view_before_mx
                        ).eval()
                        model = self._export(module, args, optimize)
                        graph = model.subgraphs[0]
                        self._assert_quantized_data(model)
                        current_index, past_index = map(int, graph.inputs)
                        self.assertEqual(graph.outputs[1], current_index)
                        self.assertEqual(graph.outputs[2], past_index)
                        self._assert_affine(
                            graph.tensors[current_index],
                            scale=current_scale,
                            zero_point=current_zp,
                        )
                        self._assert_affine(
                            graph.tensors[past_index],
                            scale=past_scale,
                            zero_point=past_zp,
                        )
                        concat, producers = self._mx_source_concat(
                            model, view_before_mx=view_before_mx
                        )
                        for index in (*concat.inputs, *concat.outputs):
                            self._assert_affine(
                                graph.tensors[index],
                                scale=compute_scale,
                                zero_point=compute_zp,
                            )
                        for source, input_index, domain in zip(
                            (past_index, current_index),
                            concat.inputs,
                            (past_domain, current_domain),
                            strict=True,
                        ):
                            if domain == (compute_scale, compute_zp):
                                self.assertEqual(input_index, source)
                            else:
                                quantize = producers[int(input_index)]
                                self.assertEqual(_builtin(model, quantize), B.QUANTIZE)
                                self.assertEqual(quantize.inputs[0], source)
                        self._assert_affine_prefix(
                            model, concat, (encoded_current, encoded_past), expected
                        )

    def test_affine_requantization_and_views_precede_one_full_mx_conversion(self):
        args = self._inputs()
        # Avoid half-grid rounding ties in the separate reference prefix check.
        args[0][..., 4:] = 100.22
        encoded_current = torch.round(args[0] / 0.25).clamp(-32768, 32767) * 0.25
        encoded_past = torch.round(args[1] / 0.125).clamp(-32768, 32767) * 0.125
        compute_current = (torch.round(encoded_current) + 3).clamp(-32768, 32767).sub(3)
        compute_past = (torch.round(encoded_past) + 3).clamp(-32768, 32767).sub(3)
        for view_before_mx in (False, True):
            for optimize in (False, True):
                with self.subTest(view_before_mx=view_before_mx, optimize=optimize):
                    model = self._export(
                        _AffineComputeThenMXConcat(view_before_mx).eval(),
                        args,
                        optimize,
                    )
                    graph = model.subgraphs[0]
                    self._assert_quantized_data(model)
                    self._assert_affine(
                        graph.tensors[graph.inputs[0]], scale=0.25, zero_point=0
                    )
                    self._assert_affine(
                        graph.tensors[graph.inputs[1]], scale=0.125, zero_point=0
                    )
                    for index in graph.outputs[1:]:
                        self._assert_affine(
                            graph.tensors[index], scale=0.125, zero_point=0
                        )
                    concat, producers = self._mx_source_concat(
                        model, view_before_mx=view_before_mx
                    )
                    for index in (*concat.inputs, *concat.outputs):
                        self._assert_affine(
                            graph.tensors[index], scale=1.0, zero_point=3
                        )
                    for index, source in zip(
                        concat.inputs, reversed(graph.inputs), strict=True
                    ):
                        quantize = producers[int(index)]
                        self.assertEqual(
                            _builtin(model, quantize),
                            circle.BuiltinOperator.BuiltinOperator.QUANTIZE,
                        )
                        self.assertEqual(quantize.inputs[0], source)
                    self._assert_affine_prefix(
                        model,
                        concat,
                        (encoded_current, encoded_past),
                        torch.cat((compute_past, compute_current), dim=-2),
                    )

    def test_same_affine_domain_is_proven_during_folding(self):
        ep = torch.export.export(_StorageThenMXConcat().eval(), self._inputs())
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
        self.assertEqual(concat.meta[QPARAM_KEY].scale, [0.5])
        for source in concat.args[0]:
            self.assertEqual(source.meta[QPARAM_KEY], concat.meta[QPARAM_KEY])
        self.assertEqual(quantize.meta[QPARAM_KEY].dtype, "mxint8")
        self.assertEqual(quantize.meta[QPARAM_KEY].quantized_dimension, 2)
        self.assertFalse(
            any(
                node.target
                == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                for node in ep.graph.nodes
            )
        )

    def test_partial_mx_blocks_cannot_be_concatenated_as_one_existing_domain(self):
        class PartialMXConcat(torch.nn.Module):
            def forward(self, current, past):
                current = quantize_mx(current, "int8", -2)
                past = quantize_mx(past, "int8", -2)
                return quantize_mx(torch.cat((past, current), dim=-2), "int8", -2)

        with self.assertRaisesRegex(NotYetSupportedError, "incomplete MX blocks"):
            tico.convert(PartialMXConcat().eval(), self._inputs())

    def test_mixed_mx_and_affine_concat_is_rejected_in_either_order(self):
        class MixedMXConcat(torch.nn.Module):
            def __init__(self, mx_first):
                super().__init__()
                self.mx_first = mx_first

            def forward(self, current, past):
                current = torch.fake_quantize_per_tensor_affine(
                    current, 0.5, 0, -32768, 32767
                )
                past = quantize_mx(past, "int8", -2)
                pair = (past, current) if self.mx_first else (current, past)
                return quantize_mx(torch.cat(pair, dim=-2), "int8", -2)

        for mx_first in (False, True):
            with (
                self.subTest(mx_first=mx_first),
                self.assertRaisesRegex(NotYetSupportedError, "different MX domains"),
            ):
                args = (torch.ones(1, 1, 32, 8), torch.ones(1, 1, 32, 8))
                tico.convert(MixedMXConcat(mx_first).eval(), args)

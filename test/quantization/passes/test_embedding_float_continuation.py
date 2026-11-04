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

"""Structural CPU continuations, with nearby normal quantized projections."""

import copy
import unittest

import numpy as np

import tico
import torch
import torch.ao.quantization.fx._decomposed
from tico.circle import CircleDocument
from tico.circle.runtime import (
    CircleReferenceRuntime,
    ExecutionMode,
    UnsupportedCircleOperatorError,
)
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.quantization.passes.preserve_embedding_quantization import (
    EXPLICIT_EMBEDDING,
    FLOAT_EMBEDDING,
    PreserveEmbeddingQuantization,
)
from tico.serialize.quant_param import QPARAM_KEY
from tico.utils.convert import _prepare_exported_program


class _Table(torch.nn.Module):
    table: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "table",
            torch.tensor(
                [[0, 3, 7, 15], [4, 8, 12, 16], [8, 16, 24, 32], [250, 200, 150, 100]],
                dtype=torch.uint8,
            ),
        )
        self.register_buffer("scale", torch.tensor([0.125, 0.25, 0.5, 0.0625]))
        self.register_buffer("zero", torch.tensor([3, 8, 16, 200], dtype=torch.int32))

    def weight(self):
        return torch.ops.quantized_decomposed.dequantize_per_channel.default(
            self.table, self.scale, self.zero, 0, 0, 255, torch.uint8
        )

    def lookup(self, ids):
        return torch.fake_quantize_per_tensor_affine(
            torch.nn.functional.embedding(ids, self.weight()), 0.25, 0, -32768, 32767
        )


class _FloatSideConcat(_Table):
    def forward(self, ids, side):
        value = self.lookup(ids)
        value = torch.ops.aten.slice_copy.Tensor(value, 2, 0, 2)
        side = side.permute(0, 2, 1)
        value = torch.cat((value, side), dim=2)
        return torch.fake_quantize_per_tensor_affine(value, 0.125, 0, -32768, 32767)


class _SharedSideConcat(_FloatSideConcat):
    def forward(self, ids, side):
        combined = super().forward(ids, side)
        separately_quantized = torch.fake_quantize_per_tensor_affine(
            side, 0.25, 0, -32768, 32767
        )
        return combined, separately_quantized


class _IntegerSelection(_Table):
    def __init__(self, sliced):
        super().__init__()
        self.sliced = sliced

    def forward(self, ids):
        value = torch.nn.functional.embedding(ids, self.weight())
        codes = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            value, 0.25, 0, -32768, 32767, torch.int16
        )
        if self.sliced:
            codes = torch.ops.aten.slice.Tensor(codes, 2, 0, 2)
        return torch.cat((codes, codes), dim=2)


class _MixedFloatQuantizedConcat(_Table):
    def forward(self, ids, image, independent):
        value = self.lookup(ids)
        independent = torch.fake_quantize_per_tensor_affine(
            independent, 0.25, 0, -32768, 32767
        )
        combined = torch.cat((value, image, independent), dim=2)
        return torch.fake_quantize_per_tensor_affine(combined, 0.125, 0, -32768, 32767)


class _SiblingConcat(_Table):
    def __init__(self, right_first):
        super().__init__()
        self.right_first = right_first

    def forward(self, ids):
        value = self.lookup(ids)
        if self.right_first:
            right = torch.ops.aten.slice_copy.Tensor(value, 2, 2, 4)
            left = torch.ops.aten.slice_copy.Tensor(value, 2, 0, 2)
        else:
            left = torch.ops.aten.slice_copy.Tensor(value, 2, 0, 2)
            right = torch.ops.aten.slice_copy.Tensor(value, 2, 2, 4)
        return torch.fake_quantize_per_tensor_affine(
            torch.cat((right, left), dim=2), 0.125, 0, -32768, 32767
        )


class _Projection(_Table):
    rotation: torch.Tensor
    projection: torch.Tensor
    projection_scale: torch.Tensor
    projection_zero: torch.Tensor
    bias: torch.Tensor

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.register_buffer(
            "rotation",
            torch.tensor(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, -1.0, 1.0, -1.0],
                    [1.0, 1.0, -1.0, -1.0],
                    [1.0, -1.0, -1.0, 1.0],
                ]
            )
            / 2,
        )
        rows = 3 if kind == "nonsquare" else 4
        self.register_buffer(
            "projection", torch.arange(rows * 4, dtype=torch.uint8).reshape(rows, 4)
        )
        self.register_buffer("projection_scale", torch.full((rows,), 0.125))
        self.register_buffer(
            "projection_zero", torch.full((rows,), 3, dtype=torch.int32)
        )
        self.register_buffer("bias", torch.full((rows,), 0.25))

    def forward(self, ids):
        value = self.lookup(ids)
        if self.kind == "square":
            weight = self.rotation
        elif self.kind == "shared":
            weight = self.weight()
        else:
            weight = torch.ops.quantized_decomposed.dequantize_per_channel.default(
                self.projection,
                self.projection_scale,
                self.projection_zero,
                0,
                0,
                255,
                torch.uint8,
            )
        value = torch.ops.aten.linear.default(
            value, weight, self.bias if self.kind == "biased" else None
        )
        return torch.fake_quantize_per_tensor_affine(value, 0.125, 0, -32768, 32767)


def _q(value, scale):
    value = np.asarray(value, np.float32) / np.float32(scale)
    rounded = np.copysign(np.floor(np.abs(value.astype(np.float64)) + 0.5), value)
    return np.clip(rounded, -32768, 32767).astype(np.int16)


def _lookup_float(module, ids):
    row = module.table.numpy()[ids].astype(np.float32)
    row -= module.zero.numpy()[ids, None].astype(np.float32)
    row *= module.scale.numpy()[ids, None]
    return _q(row, 0.25).astype(np.float32) * np.float32(0.25)


class EmbeddingFloatContinuationTest(unittest.TestCase):
    def test_three_way_float_concat_rejects_independent_quantized_side(self):
        inputs = (
            torch.tensor([[3, 0]]),
            torch.arange(8, dtype=torch.float32).reshape(1, 2, 4) / 16,
            torch.linspace(-1, 1, 8).reshape(1, 2, 4),
        )
        exported = torch.export.export(_MixedFloatQuantizedConcat().eval(), inputs)
        with self.assertRaisesRegex(
            RuntimeError, "Unsupported explicit embedding float side operand"
        ):
            tico.convert_from_exported_program(exported)

    def _prepared(self, module, inputs):
        return _prepare_exported_program(torch.export.export(module.eval(), inputs))

    def _runtime(self, module, inputs):
        converted = tico.convert_from_exported_program(
            torch.export.export(module.eval(), inputs)
        )
        document = CircleDocument.from_bytes(converted.circle_binary)
        self.assertTrue(document.verify().ok)
        return CircleReferenceRuntime(CircleDocument.from_bytes(document.to_bytes()))

    def test_genuine_float_side_concat_keeps_float_layout_and_native_boundaries(self):
        module = _FloatSideConcat().eval()
        ids = torch.tensor([[3, 0]])
        side = torch.tensor([[[0.0625, -0.1875], [0.3125, -0.4375]]])
        prepared = self._prepared(module, (ids, side))
        concat = next(
            node
            for node in prepared.graph.nodes
            if node.target == torch.ops.aten.cat.default
        )
        self.assertTrue(concat.meta.get(FLOAT_EMBEDDING))
        self.assertNotIn(QPARAM_KEY, concat.meta)
        for operand in concat.all_input_nodes:
            self.assertTrue(operand.meta.get(FLOAT_EMBEDDING))
            self.assertNotIn(QPARAM_KEY, operand.meta)
        runtime = self._runtime(module, (ids, side))
        concat_op = next(
            op for op in runtime.program.operators if op.name == "CONCATENATION"
        )
        self.assertEqual(runtime.program.tensor(concat_op.outputs[0]).dtype, np.float32)
        expected = _q(
            np.concatenate(
                (
                    _lookup_float(module, ids.numpy())[..., :2],
                    side.numpy().transpose(0, 2, 1),
                ),
                axis=2,
            ),
            0.125,
        )
        np.testing.assert_array_equal(
            runtime.run((ids.numpy(), side.numpy()), mode=ExecutionMode.NATIVE).outputs[
                0
            ],
            expected,
        )
        self.assertTrue(runtime.rowwise_gather_plans)

    def test_all_island_siblings_ignore_creation_order_and_pass_is_idempotent(self):
        ids = torch.tensor([[3, 0]])
        results = []
        for right_first in (False, True):
            with self.subTest(right_first=right_first):
                module = _SiblingConcat(right_first).eval()
                exported = torch.export.export(module, (ids,))
                DecomposeFakeQuantize().call(exported)
                policy = PreserveEmbeddingQuantization()
                policy.call(exported)
                original_graph = str(exported.graph)
                original_meta = {
                    node.name: (
                        node.meta.get(EXPLICIT_EMBEDDING),
                        node.meta.get(FLOAT_EMBEDDING),
                        copy.deepcopy(node.meta.get(QPARAM_KEY)),
                    )
                    for node in exported.graph.nodes
                }
                policy.call(exported)
                self.assertEqual(str(exported.graph), original_graph)
                self.assertEqual(
                    {
                        node.name: (
                            node.meta.get(EXPLICIT_EMBEDDING),
                            node.meta.get(FLOAT_EMBEDDING),
                            node.meta.get(QPARAM_KEY),
                        )
                        for node in exported.graph.nodes
                    },
                    original_meta,
                )
                concat = next(
                    node
                    for node in exported.graph.nodes
                    if node.target == torch.ops.aten.cat.default
                )
                self.assertTrue(concat.meta.get(FLOAT_EMBEDDING))
                runtime = self._runtime(module, (ids,))
                selected = _lookup_float(module, ids.numpy())
                expected = _q(
                    np.concatenate((selected[..., 2:], selected[..., :2]), axis=2),
                    0.125,
                )
                actual = runtime.run((ids.numpy(),), mode=ExecutionMode.NATIVE).outputs[
                    0
                ]
                np.testing.assert_array_equal(actual, expected)
                results.append(actual)
        np.testing.assert_array_equal(results[0], results[1])

    def test_independent_bias_free_square_transform_is_float_and_native(self):
        module = _Projection("square").eval()
        ids = torch.tensor([[3, 0]])
        prepared = self._prepared(module, (ids,))
        linear = next(
            node
            for node in prepared.graph.nodes
            if node.target == torch.ops.aten.linear.default
        )
        self.assertTrue(linear.meta.get(FLOAT_EMBEDDING))
        self.assertNotIn(QPARAM_KEY, linear.meta)
        runtime = self._runtime(module, (ids,))
        fc = next(
            op for op in runtime.program.operators if op.name == "FULLY_CONNECTED"
        )
        self.assertEqual(runtime.program.tensor(fc.outputs[0]).dtype, np.float32)
        expected = _q(
            _lookup_float(module, ids.numpy()) @ module.rotation.numpy().T, 0.125
        )
        np.testing.assert_array_equal(
            runtime.run((ids.numpy(),), mode=ExecutionMode.NATIVE).outputs[0], expected
        )

    def test_float_side_shared_quantizer_preserves_both_output_contracts(self):
        module = _SharedSideConcat().eval()
        ids = torch.tensor([[3, 0]])
        side = torch.tensor([[[0.0625, -0.1875], [0.3125, -0.4375]]])
        prepared = self._prepared(module, (ids, side))
        side_node = next(
            node
            for node in prepared.graph.nodes
            if node.op == "placeholder"
            and node.name in prepared.graph_signature.user_inputs
            and node.meta["val"].dtype == torch.float32
        )
        self.assertTrue(side_node.meta.get(FLOAT_EMBEDDING))
        self.assertNotIn(QPARAM_KEY, side_node.meta)
        separate_q = next(
            node
            for node in prepared.graph.nodes
            if node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
            and node.args[0] is side_node
        )
        self.assertTrue(separate_q.meta.get(EXPLICIT_EMBEDDING))
        self.assertEqual(separate_q.args[1:5], (0.25, 0, -32768, 32767))
        self.assertEqual(separate_q.meta[QPARAM_KEY].dtype, "int16")
        runtime = self._runtime(module, (ids, side))
        self.assertEqual(runtime.input_tensors[1].dtype, np.float32)
        self.assertFalse(runtime.input_tensors[1].is_quantized)
        result = runtime.run((ids.numpy(), side.numpy()), mode=ExecutionMode.NATIVE)
        combined = np.concatenate(
            (
                _lookup_float(module, ids.numpy())[..., :2],
                side.numpy().transpose(0, 2, 1),
            ),
            axis=2,
        )
        np.testing.assert_array_equal(result.outputs[0], _q(combined, 0.125))
        np.testing.assert_array_equal(result.outputs[1], _q(side.numpy(), 0.25))
        self.assertEqual(
            [value.dtype for value in result.outputs], [np.int16, np.int16]
        )

    def test_direct_integer_quantize_slice_and_concat_are_integer_selections(self):
        ids = torch.tensor([[3, 0]])
        for sliced in (False, True):
            with self.subTest(sliced=sliced):
                module = _IntegerSelection(sliced).eval()
                prepared = self._prepared(module, (ids,))
                for node in prepared.graph.nodes:
                    if node.target in (
                        torch.ops.aten.cat.default,
                        torch.ops.aten.slice.Tensor,
                        torch.ops.aten.slice_copy.Tensor,
                    ):
                        self.assertFalse(node.meta.get(FLOAT_EMBEDDING, False))
                        self.assertIn(QPARAM_KEY, node.meta)
                        self.assertEqual(node.meta[QPARAM_KEY].dtype, "int16")
                runtime = self._runtime(module, (ids,))
                self.assertEqual(
                    sum(op.name == "QUANTIZE" for op in runtime.program.operators), 1
                )
                concat = next(
                    op for op in runtime.program.operators if op.name == "CONCATENATION"
                )
                self.assertEqual(
                    runtime.program.tensor(concat.outputs[0]).dtype, np.int16
                )
                codes = (_lookup_float(module, ids.numpy()) / np.float32(0.25)).astype(
                    np.int16
                )
                if sliced:
                    codes = codes[..., :2]
                expected = np.concatenate((codes, codes), axis=2)
                np.testing.assert_array_equal(
                    runtime.run((ids.numpy(),), mode=ExecutionMode.NATIVE).outputs[0],
                    expected,
                )

    def test_keyword_bias_does_not_admit_square_linear(self):
        exported = torch.export.export(
            _Projection("biased").eval(), (torch.tensor([[3, 0]]),)
        )
        DecomposeFakeQuantize().call(exported)
        linear = next(
            node
            for node in exported.graph.nodes
            if node.target == torch.ops.aten.linear.default
        )
        self.assertEqual(len(linear.args), 3)
        bias = linear.args[2]
        linear.args = linear.args[:2]
        linear.kwargs = {"bias": bias}
        PreserveEmbeddingQuantization().call(exported)
        self.assertIs(linear.kwargs["bias"], bias)
        self.assertFalse(linear.meta.get(EXPLICIT_EMBEDDING, False))
        self.assertFalse(linear.meta.get(FLOAT_EMBEDDING, False))

    def test_nonsquare_biased_and_shared_square_projections_keep_normal_policy(self):
        ids = torch.tensor([[3, 0]])
        for kind in ("nonsquare", "biased", "shared"):
            with self.subTest(kind=kind):
                module = _Projection(kind).eval()
                prepared = self._prepared(module, (ids,))
                linear = next(
                    node
                    for node in prepared.graph.nodes
                    if node.target == torch.ops.aten.linear.default
                )
                self.assertFalse(linear.meta.get(EXPLICIT_EMBEDDING, False))
                self.assertFalse(linear.meta.get(FLOAT_EMBEDDING, False))
                self.assertIn(QPARAM_KEY, linear.meta)
                runtime = self._runtime(module, (ids,))
                fc = next(
                    op
                    for op in runtime.program.operators
                    if op.name == "FULLY_CONNECTED"
                )
                self.assertTrue(
                    runtime.program.tensor(fc.outputs[0]).is_quantized_integer
                )
                # These ordinary quantized FCs retain their existing CPU native
                # diagnostic, rather than being mistaken for float rotations.
                with self.assertRaisesRegex(
                    UnsupportedCircleOperatorError, "backend-defined integer arithmetic"
                ):
                    runtime.run((ids.numpy(),), mode=ExecutionMode.NATIVE)
                if kind == "shared":
                    dq = next(
                        op
                        for op in runtime.program.operators
                        if op.name == "DEQUANTIZE"
                    )
                    self.assertEqual(fc.inputs[1], dq.inputs[0])

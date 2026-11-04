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

"""Preserve integer consumer contracts after a float embedding layout."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

import tico
import torch
from tico.circle import CircleDocument
from tico.circle.operations import extract_by_tensor_indices
from tico.circle.runtime import (
    CircleReferenceRuntime,
    ExecutionMode,
    UnsupportedCircleOperatorError,
)
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.passes.fill_meta_val import FillMetaVal
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.quantization.passes.preserve_embedding_quantization import (
    EXPLICIT_EMBEDDING,
    FLOAT_EMBEDDING,
    PreserveEmbeddingQuantization,
)
from tico.serialize.quant_param import QPARAM_KEY
from tico.utils.convert import _prepare_exported_program

from test.quantization.passes.test_embedding_float_continuation import (
    _lookup_float,
    _Projection,
    _q,
)


class _ProjectionAfterLayout(_Projection):
    def __init__(self, kind, path="reshape"):
        super().__init__(kind)
        self.path = path

    def lookup(self, ids):
        value = super().lookup(ids)
        if self.path == "reshape":
            return value.reshape(-1, 4)
        if self.path == "permute":
            return value.permute(1, 0, 2)
        if self.path == "chain":
            return value.permute(1, 0, 2).reshape(-1, 4)
        return value


class _SharedLayoutProjection(_ProjectionAfterLayout):
    def __init__(self, kind, fc_first):
        super().__init__(kind, "chain")
        self.fc_first = fc_first

    def project(self, value):
        if self.kind == "shared":
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
        projected = torch.ops.aten.linear.default(
            value, weight, self.bias if self.kind == "biased" else None
        )
        return torch.fake_quantize_per_tensor_affine(projected, 0.125, 0, -32768, 32767)

    def float_branch(self, value):
        combined = torch.cat((value, value), dim=-1)
        return torch.fake_quantize_per_tensor_affine(combined, 0.125, 0, -32768, 32767)

    def forward(self, ids):
        value = self.lookup(ids)
        if self.fc_first:
            projected = self.project(value)
            combined = self.float_branch(value)
        else:
            combined = self.float_branch(value)
            projected = self.project(value)
        return projected, combined


class _ReducedRangeProjection(_ProjectionAfterLayout):
    def __init__(self):
        super().__init__("nonsquare")
        self.scale[2] = 2048

    def lookup(self, ids):
        value = torch.nn.functional.embedding(ids, self.weight())
        return torch.fake_quantize_per_tensor_affine(
            value, 0.25, 0, -32767, 32767
        ).reshape(-1, 4)


class _FloatProjectionAfterLayout(_ProjectionAfterLayout):
    float_projection: torch.Tensor

    def __init__(self):
        super().__init__("nonsquare")
        self.register_buffer(
            "float_projection",
            torch.arange(12, dtype=torch.float32).reshape(3, 4) / 8,
        )

    def forward(self, ids):
        return torch.ops.aten.linear.default(
            self.lookup(ids), self.float_projection, None
        )


def _policy_signature(exported):
    return (
        str(exported.graph),
        {
            node.name: (
                node.meta.get(EXPLICIT_EMBEDDING),
                node.meta.get(FLOAT_EMBEDDING),
                copy.deepcopy(node.meta.get(QPARAM_KEY)),
            )
            for node in exported.graph.nodes
        },
        sum(
            node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
            for node in exported.graph.nodes
        ),
        sum(
            bool(node.meta.get("_embedding_range_clamp_"))
            for node in exported.graph.nodes
        ),
    )


def _selected_after_layout(module, ids):
    # Retain the original embedding Q_A/DQ_A before applying any layout.
    value = _lookup_float(module, ids)
    if module.path == "reshape":
        return value.reshape(-1, 4)
    if module.path == "permute":
        return value.transpose(1, 0, 2)
    if module.path == "chain":
        return value.transpose(1, 0, 2).reshape(-1, 4)
    return value


def _projection_oracle(module, ids):
    value = _selected_after_layout(module, ids)
    if module.kind == "shared":
        weight = module.table.numpy().astype(np.float32)
        weight -= module.zero.numpy()[:, None].astype(np.float32)
        weight *= module.scale.numpy()[:, None]
    else:
        weight = module.projection.numpy().astype(np.float32)
        weight -= module.projection_zero.numpy()[:, None].astype(np.float32)
        weight *= module.projection_scale.numpy()[:, None]
    # All factors are small binary fractions, making this independent scalar
    # product exact in FP32 without relying on a kernel-specific matmul helper.
    result = np.sum(value[..., None, :] * weight, axis=-1, dtype=np.float32)
    if module.kind == "biased":
        result += module.bias.numpy()
    return _q(result, 0.125).astype(np.float32) * np.float32(0.125)


class EmbeddingLayoutConsumerBoundaryTest(unittest.TestCase):
    def _runtime(self, module, inputs):
        exported = torch.export.export(module.eval(), inputs)
        converted = tico.convert_from_exported_program(exported)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "embedding_projection.circle"
            converted.save(destination)
            document = CircleDocument.load(destination)
        self.assertTrue(document.verify().ok)
        return CircleReferenceRuntime(document)

    def _assert_projection_contract(self, runtime, module):
        fc = next(
            op for op in runtime.program.operators if op.name == "FULLY_CONNECTED"
        )
        value = runtime.program.tensor(fc.inputs[0])
        self.assertEqual(value.dtype, np.int16)
        self.assertIsNotNone(value.quantization)
        assert value.quantization is not None
        self.assertEqual(value.quantization.scale, (0.25,))
        self.assertEqual(value.quantization.zero_point, (0,))
        weight = runtime.program.tensor(fc.inputs[1])
        self.assertEqual(weight.dtype, np.uint8)
        self.assertTrue(weight.is_constant)
        assert weight.quantization is not None
        self.assertEqual(weight.quantization.quantized_dimension, 0)
        if module.kind == "shared":
            expected_codes, scales, zeros = module.table, module.scale, module.zero
        else:
            expected_codes = module.projection
            scales, zeros = module.projection_scale, module.projection_zero
        np.testing.assert_array_equal(weight.constant, expected_codes.numpy())
        self.assertEqual(weight.quantization.scale, tuple(scales.tolist()))
        self.assertEqual(weight.quantization.zero_point, tuple(zeros.tolist()))
        output = runtime.program.tensor(fc.outputs[0])
        self.assertEqual(output.dtype, np.int16)
        assert output.quantization is not None
        self.assertEqual(output.quantization.scale, (0.125,))
        self.assertEqual(output.quantization.zero_point, (0,))
        if module.kind == "shared":
            table_dq = next(
                op for op in runtime.program.operators if op.name == "DEQUANTIZE"
            )
            self.assertEqual(fc.inputs[1], table_dq.inputs[0])

    def _assert_numerical_contract(self, runtime, module, ids):
        fake = runtime.run((ids.numpy(),), mode=ExecutionMode.FAKE_QUANTIZE)
        np.testing.assert_array_equal(
            fake.outputs[0], _projection_oracle(module, ids.numpy())
        )
        self.assertEqual(fake.outputs[0].dtype, np.float32)
        # FAKE_QUANTIZE supplies dequantized reference arithmetic for FC.
        # Native integer FC remains an explicit unsupported backend contract.
        with self.assertRaisesRegex(
            UnsupportedCircleOperatorError, "backend-defined integer arithmetic"
        ) as raised:
            runtime.run((ids.numpy(),), mode=ExecutionMode.NATIVE)
        self.assertIn("FULLY_CONNECTED", str(raised.exception))
        self.assertTrue(runtime.rowwise_gather_plans)
        return fake

    def test_reshape_nonsquare_projection_keeps_original_integer_input_grid(self):
        module = _ProjectionAfterLayout("nonsquare").eval()
        ids = torch.tensor([[3, 0]])
        runtime = self._runtime(module, (ids,))
        self._assert_projection_contract(runtime, module)
        self._assert_numerical_contract(runtime, module, ids)

    def test_genuine_float_nonsquare_projection_keeps_native_float_contract(self):
        module = _FloatProjectionAfterLayout().eval()
        ids = torch.tensor([[3, 0], [1, 2]])
        runtime = self._runtime(module, (ids,))
        fc = next(
            op for op in runtime.program.operators if op.name == "FULLY_CONNECTED"
        )
        for index in (*fc.inputs[:2], *fc.outputs):
            tensor = runtime.program.tensor(index)
            self.assertEqual(tensor.dtype, np.float32)
            self.assertIsNone(tensor.quantization)
        layouts = [op for op in runtime.program.operators if op.name == "RESHAPE"]
        self.assertEqual(len(layouts), 1)
        self.assertEqual(
            runtime.program.tensor(layouts[0].outputs[0]).dtype, np.float32
        )
        selected = _selected_after_layout(module, ids.numpy())
        expected = np.sum(
            selected[..., None, :] * module.float_projection.numpy(),
            axis=-1,
            dtype=np.float32,
        )
        actual = runtime.run((ids.numpy(),), mode=ExecutionMode.NATIVE).outputs[0]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(runtime.rowwise_gather_plans)

    def test_layout_and_ordinary_projection_matrix_including_direct_controls(self):
        ids = torch.tensor([[3, 0], [1, 2]])
        for path in ("direct", "reshape", "permute", "chain"):
            for kind in ("nonsquare", "biased", "shared"):
                with self.subTest(path=path, kind=kind):
                    module = _ProjectionAfterLayout(kind, path).eval()
                    original = {
                        name: value.clone() for name, value in module.named_buffers()
                    }
                    runtime = self._runtime(module, (ids,))
                    self._assert_projection_contract(runtime, module)
                    self._assert_numerical_contract(runtime, module, ids)
                    for name, value in module.named_buffers():
                        self.assertTrue(torch.equal(value, original[name]))

    def test_shared_float_layout_preserves_edge_grid_in_both_consumer_orders(self):
        ids = torch.tensor([[3, 0], [1, 2]])
        for kind in ("nonsquare", "biased", "shared"):
            outputs = []
            for fc_first in (False, True):
                with self.subTest(kind=kind, fc_first=fc_first):
                    module = _SharedLayoutProjection(kind, fc_first).eval()
                    prepared = _prepare_exported_program(
                        torch.export.export(module, (ids,))
                    )
                    linear = next(
                        node
                        for node in prepared.graph.nodes
                        if node.target == torch.ops.aten.linear.default
                    )
                    self.assertFalse(linear.meta.get(FLOAT_EMBEDDING, False))
                    self.assertIn(QPARAM_KEY, linear.args[0].meta)
                    self.assertEqual(linear.args[0].meta[QPARAM_KEY].scale, [0.25])
                    concat = next(
                        node
                        for node in prepared.graph.nodes
                        if node.target == torch.ops.aten.cat.default
                    )
                    for node in (concat, *concat.all_input_nodes):
                        self.assertTrue(node.meta.get(FLOAT_EMBEDDING))
                        self.assertNotIn(QPARAM_KEY, node.meta)
                    runtime = self._runtime(module, (ids,))
                    self._assert_projection_contract(runtime, module)
                    fake = self._assert_numerical_contract(runtime, module, ids)
                    selected = _selected_after_layout(module, ids.numpy())
                    expected = _q(np.concatenate((selected, selected), axis=-1), 0.125)
                    np.testing.assert_array_equal(
                        fake.outputs[1], expected.astype(np.float32) * np.float32(0.125)
                    )
                    # The float branch can execute natively, while this full
                    # graph still diagnoses its separate quantized FC branch.
                    concat_op = next(
                        op
                        for op in runtime.program.operators
                        if op.name == "CONCATENATION"
                    )
                    for index in (*concat_op.inputs, *concat_op.outputs):
                        tensor = runtime.program.tensor(index)
                        self.assertEqual(tensor.dtype, np.float32)
                        self.assertIsNone(tensor.quantization)
                    selected_output = runtime.program.outputs[1]
                    partial = extract_by_tensor_indices(
                        runtime.program.document, to_tensors=(selected_output,)
                    )
                    branch = CircleReferenceRuntime(
                        CircleDocument.from_bytes(partial.document.to_bytes())
                    )
                    position = partial.source_boundary.outputs.index(selected_output)
                    np.testing.assert_array_equal(
                        branch.run((ids.numpy(),), mode=ExecutionMode.NATIVE).outputs[
                            position
                        ],
                        expected,
                    )
                    outputs.append(fake.outputs)
            with self.subTest(kind=kind, comparison="consumer-order"):
                self.assertEqual(len(outputs), 2)
                for left, right in zip(outputs[0], outputs[1]):
                    np.testing.assert_array_equal(left, right)

    def test_consumer_boundary_pass_is_idempotent(self):
        ids = torch.tensor([[3, 0], [1, 2]])
        for kind in ("nonsquare", "biased", "shared"):
            for fc_first in (False, True):
                with self.subTest(kind=kind, fc_first=fc_first):
                    exported = torch.export.export(
                        _SharedLayoutProjection(kind, fc_first).eval(), (ids,)
                    )
                    DecomposeFakeQuantize().call(exported)
                    FillMetaVal().call(exported)
                    policy = PreserveEmbeddingQuantization()
                    policy.call(exported)
                    original = _policy_signature(exported)
                    policy.call(exported)
                    self.assertEqual(_policy_signature(exported), original)
                    FoldQuantOps().call(exported)
                    folded = _policy_signature(exported)
                    policy.call(exported)
                    self.assertEqual(_policy_signature(exported), folded)

    def test_reduced_range_clamp_is_not_duplicated_or_lost_on_integer_layout(self):
        module = _ReducedRangeProjection().eval()
        ids = torch.tensor([[2, 0]])
        exported = torch.export.export(module, (ids,))
        DecomposeFakeQuantize().call(exported)
        FillMetaVal().call(exported)
        policy = PreserveEmbeddingQuantization()
        policy.call(exported)
        self.assertEqual(_policy_signature(exported)[3], 1)
        FoldQuantOps().call(exported)
        folded = _policy_signature(exported)
        policy.call(exported)
        self.assertEqual(_policy_signature(exported), folded)
        self.assertEqual(_policy_signature(exported)[3], 1)

        runtime = self._runtime(module, (ids,))
        self._assert_projection_contract(runtime, module)
        self.assertEqual(
            sum(op.name == "MAXIMUM" for op in runtime.program.operators), 1
        )
        self.assertEqual(
            sum(op.name == "QUANTIZE" for op in runtime.program.operators), 1
        )
        fc = next(
            op for op in runtime.program.operators if op.name == "FULLY_CONNECTED"
        )
        partial = extract_by_tensor_indices(
            runtime.program.document, to_tensors=(fc.inputs[0],)
        )
        branch = CircleReferenceRuntime(
            CircleDocument.from_bytes(partial.document.to_bytes())
        )
        position = partial.source_boundary.outputs.index(fc.inputs[0])
        actual = branch.run((ids.numpy(),), mode=ExecutionMode.NATIVE).outputs[position]
        selected = module.table.numpy()[ids.numpy()].astype(np.float32)
        selected -= module.zero.numpy()[ids.numpy(), None].astype(np.float32)
        selected *= module.scale.numpy()[ids.numpy(), None]
        expected = np.maximum(_q(selected, 0.25), np.int16(-32767)).reshape(-1, 4)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.int16)
        self.assertIn(-32767, actual)
        with self.assertRaisesRegex(
            UnsupportedCircleOperatorError, "backend-defined integer arithmetic"
        ):
            runtime.run((ids.numpy(),), mode=ExecutionMode.NATIVE)


if __name__ == "__main__":
    unittest.main()

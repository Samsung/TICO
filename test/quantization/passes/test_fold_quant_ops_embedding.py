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

"""Embedding table and activation domains belong to serialized quantized tensors."""

import unittest

import numpy as np

import tico
import torch
from circle_schema import circle
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.quantization import convert as quantize, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.quantization.passes.remove_weight_dequant_op import RemoveWeightDequantOp
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme
from tico.serialize.quant_param import QPARAM_KEY


class _EmbeddingWeight(torch.nn.Module):
    def __init__(self, *, per_channel: bool = True):
        super().__init__()
        self.per_channel = per_channel
        self.register_buffer(
            "weight", torch.tensor([[1, 3, 7, -1], [1, 2, 5, -4]], dtype=torch.int8)
        )
        self.register_buffer("scales", torch.tensor([0.125, 0.25]))
        self.register_buffer("zero_points", torch.tensor([0, 0], dtype=torch.int32))

    def dequantized_weight(self):
        if self.per_channel:
            return torch.ops.quantized_decomposed.dequantize_per_channel.default(
                self.weight, self.scales, self.zero_points, 0, -128, 127, torch.int8
            )
        return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            self.weight, 0.125, 0, -128, 127, torch.int8
        )


class _QuantizedEmbedding(_EmbeddingWeight):
    def forward(self, input_ids):
        embedded = torch.nn.functional.embedding(input_ids, self.dequantized_weight())
        return torch.fake_quantize_per_tensor_affine(embedded, 0.3, 128, 0, 255)


class _SharedEmbeddingWeight(_EmbeddingWeight):
    def forward(self, input_ids, hidden_states):
        weight = self.dequantized_weight()
        embedded = torch.nn.functional.embedding(input_ids, weight)
        embedded = torch.fake_quantize_per_tensor_affine(embedded, 0.3, 128, 0, 255)
        projected = torch.nn.functional.linear(hidden_states, weight)
        projected = torch.fake_quantize_per_tensor_affine(projected, 0.2, 127, 0, 255)
        return embedded, projected


def _nodes(program, target):
    return [
        node
        for node in program.graph.nodes
        if node.op == "call_function" and node.target == target
    ]


class FoldQuantOpsEmbeddingTest(unittest.TestCase):
    def test_folds_activation_domain_into_embedding_output(self):
        tokens = torch.tensor([[0, 1]], dtype=torch.int64)
        program = torch.export.export(_QuantizedEmbedding().eval(), (tokens,))
        DecomposeFakeQuantize().call(program)
        FoldQuantOps().call(program)
        embedding = _nodes(program, torch.ops.aten.embedding.default)[0]
        qparam = embedding.meta[QPARAM_KEY]
        self.assertEqual(qparam.dtype, "uint8")
        self.assertEqual(qparam.scale, [0.3])
        self.assertEqual(qparam.zero_point, [128])

        RemoveWeightDequantOp().call(program)
        weight = embedding.args[0]
        weight_qparam = weight.meta[QPARAM_KEY]
        self.assertEqual(weight_qparam.dtype, "int8")
        self.assertEqual(weight_qparam.scale, [0.125, 0.25])
        self.assertEqual(weight_qparam.zero_point, [0, 0])
        self.assertEqual(weight_qparam.quantized_dimension, 0)
        # Tensor annotations are stable when the passes run again.
        FoldQuantOps().call(program)
        RemoveWeightDequantOp().call(program)
        self.assertIs(embedding.args[0], weight)
        self.assertEqual(embedding.meta[QPARAM_KEY], qparam)
        for target in (
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
        ):
            self.assertFalse(_nodes(program, target))

    def test_shared_weight_owns_domain_for_embedding_and_linear(self):
        module = _SharedEmbeddingWeight().eval()
        inputs = (torch.tensor([[0, 1]]), torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
        program = torch.export.export(module, inputs)
        DecomposeFakeQuantize().call(program)
        FoldQuantOps().call(program)
        RemoveWeightDequantOp().call(program)
        embedding = _nodes(program, torch.ops.aten.embedding.default)[0]
        linear = _nodes(program, torch.ops.aten.linear.default)[0]
        self.assertIs(embedding.args[0], linear.args[1])
        self.assertEqual(embedding.args[0].meta[QPARAM_KEY].scale, [0.125, 0.25])
        self.assertEqual(embedding.meta[QPARAM_KEY].scale, [0.3])
        self.assertEqual(linear.meta[QPARAM_KEY].scale, [0.2])
        self.assertFalse(
            _nodes(
                program, torch.ops.quantized_decomposed.dequantize_per_channel.default
            )
        )

    def test_mismatched_qdq_is_not_folded(self):
        program = torch.export.export(
            _QuantizedEmbedding().eval(), (torch.tensor([[0, 1]]),)
        )
        DecomposeFakeQuantize().call(program)
        target = torch.ops.quantized_decomposed.dequantize_per_tensor.default
        dequantizer = _nodes(program, target)[0]
        dequantizer.args = (dequantizer.args[0], 0.4, *dequantizer.args[2:])
        FoldQuantOps().call(program)
        self.assertEqual(len(_nodes(program, target)), 1)
        embedding = _nodes(program, torch.ops.aten.embedding.default)[0]
        quantizer = _nodes(
            program, torch.ops.quantized_decomposed.quantize_per_tensor.default
        )[0]
        self.assertNotIn(QPARAM_KEY, embedding.meta)
        self.assertNotIn(QPARAM_KEY, quantizer.meta)

    def _assert_quantized_embedding_circle(self, converted):
        document = CircleDocument.from_bytes(converted.circle_binary)
        self.assertTrue(document.verify().ok)
        program = CircleReferenceRuntime(document).program
        self.assertEqual([operator.name for operator in program.operators], ["GATHER"])
        gather = program.operators[0]
        weight = program.tensor(gather.inputs[0])
        indices = program.tensor(gather.inputs[1])
        output = program.tensor(gather.outputs[0])
        self.assertEqual(indices.tensor_type, circle.TensorType.TensorType.INT64)
        self.assertFalse(indices.is_quantized)
        self.assertFalse(indices.is_constant)
        self.assertTrue(weight.is_constant)
        self.assertFalse(output.is_constant)

        # This minimal NPU graph has only table/activation data and semantic
        # GATHER indices. Exclude that operand by role, not by integer dtype:
        # the quantized data tensors are integer tensors too.
        for tensor in program.tensors:
            if tensor.index == indices.index:
                continue
            with self.subTest(tensor=tensor.name):
                self.assertTrue(tensor.is_quantized_integer, tensor.describe())
                self.assertIsNotNone(tensor.quantization)
                quantization = tensor.quantization
                assert quantization is not None
                self.assertTrue(quantization.scale)
                self.assertTrue(all(np.isfinite(quantization.scale)))
                self.assertTrue(all(scale > 0 for scale in quantization.scale))
                self.assertEqual(len(quantization.zero_point), len(quantization.scale))
        return weight, output

    def test_npu_circle_data_invariant_with_independent_embedding_domains(self):
        tokens = torch.tensor([[0, 1]], dtype=torch.int64)
        for per_channel in (True, False):
            with self.subTest(per_channel=per_channel):
                module = _QuantizedEmbedding(per_channel=per_channel).eval()
                weight, output = self._assert_quantized_embedding_circle(
                    tico.convert(module, (tokens,))
                )
                self.assertEqual(weight.tensor_type, circle.TensorType.TensorType.INT8)
                self.assertEqual(output.tensor_type, circle.TensorType.TensorType.UINT8)
                np.testing.assert_array_equal(
                    weight.quantization.scale,
                    [0.125, 0.25] if per_channel else [0.125],
                )
                np.testing.assert_array_equal(
                    weight.quantization.zero_point, [0, 0] if per_channel else [0]
                )
                self.assertEqual(weight.quantization.quantized_dimension, 0)
                np.testing.assert_array_equal(
                    output.quantization.scale, [np.float32(0.3)]
                )
                self.assertEqual(output.quantization.zero_point, (128,))

    def test_wrapq_embedding_export_owns_calibrated_tensor_domains(self):
        tokens = torch.tensor([[0, 1]], dtype=torch.int64)
        module = torch.nn.Embedding.from_pretrained(
            torch.tensor([[0.125, 0.375, 0.875, -0.125], [0.25, 0.5, 1.25, -1.0]])
        ).eval()
        config = PTQConfig(
            activation=affine(DType.uint(8), qscheme=QScheme.PER_TENSOR_ASYMM),
            weight=affine(DType.uint(8), qscheme=QScheme.PER_CHANNEL_ASYMM),
        )
        prepared = prepare(module, config)
        with torch.no_grad():
            prepared(tokens)
        quantized = quantize(prepared).eval()
        weight_scale, weight_zero = quantized.wrapped.weight_obs.compute_qparams()
        act_scale, act_zero = quantized.wrapped.act_out_obs.compute_qparams()
        weight, output = self._assert_quantized_embedding_circle(
            tico.convert(quantized, (tokens,))
        )
        self.assertEqual(weight.tensor_type, circle.TensorType.TensorType.UINT8)
        self.assertEqual(output.tensor_type, circle.TensorType.TensorType.UINT8)
        np.testing.assert_array_equal(weight.quantization.scale, weight_scale.tolist())
        np.testing.assert_array_equal(
            weight.quantization.zero_point, weight_zero.tolist()
        )
        np.testing.assert_array_equal(
            output.quantization.scale, act_scale.reshape(-1).tolist()
        )
        np.testing.assert_array_equal(
            output.quantization.zero_point, act_zero.reshape(-1).tolist()
        )


if __name__ == "__main__":
    unittest.main()

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

"""Keep the activation quantizer separate from value-preserving embedding lookup."""

import unittest

import numpy as np

import tico
import torch
from circle_schema import circle
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.quantization.passes.remove_weight_dequant_op import RemoveWeightDequantOp
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
    def test_keeps_explicit_activation_quantizer_and_float_lookup(self):
        tokens = torch.tensor([[0, 1]], dtype=torch.int64)
        program = torch.export.export(_QuantizedEmbedding().eval(), (tokens,))
        DecomposeFakeQuantize().call(program)
        FoldQuantOps().call(program)
        embedding = _nodes(program, torch.ops.aten.embedding.default)[0]
        quantizers = _nodes(
            program, torch.ops.quantized_decomposed.quantize_per_tensor.default
        )
        self.assertEqual(len(quantizers), 1)
        self.assertIs(quantizers[0].args[0], embedding)
        self.assertNotIn(QPARAM_KEY, embedding.meta)
        qparam = quantizers[0].meta[QPARAM_KEY]
        self.assertEqual(qparam.dtype, "uint8")
        self.assertEqual(qparam.scale, [0.3])
        self.assertEqual(qparam.zero_point, [128])
        self.assertFalse(
            _nodes(
                program, torch.ops.quantized_decomposed.dequantize_per_tensor.default
            )
        )

        RemoveWeightDequantOp().call(program)
        dequantizers = _nodes(
            program, torch.ops.quantized_decomposed.dequantize_per_channel.default
        )
        self.assertEqual(len(dequantizers), 1)
        self.assertIs(embedding.args[0], dequantizers[0])
        self.assertEqual(dequantizers[0].args[0].meta[QPARAM_KEY].scale, [0.125, 0.25])
        # Both passes must preserve this boundary on a second invocation.
        FoldQuantOps().call(program)
        RemoveWeightDequantOp().call(program)
        self.assertEqual(
            len(
                _nodes(
                    program, torch.ops.quantized_decomposed.quantize_per_tensor.default
                )
            ),
            1,
        )
        self.assertIs(embedding.args[0], dequantizers[0])

    def test_shared_weight_keeps_dequantize_only_for_float_embedding(self):
        module = _SharedEmbeddingWeight().eval()
        inputs = (torch.tensor([[0, 1]]), torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
        program = torch.export.export(module, inputs)
        DecomposeFakeQuantize().call(program)
        FoldQuantOps().call(program)
        RemoveWeightDequantOp().call(program)
        embedding = _nodes(program, torch.ops.aten.embedding.default)[0]
        linear = _nodes(program, torch.ops.aten.linear.default)[0]
        dequantizer = _nodes(
            program, torch.ops.quantized_decomposed.dequantize_per_channel.default
        )[0]
        self.assertIs(embedding.args[0], dequantizer)
        self.assertIs(linear.args[1], dequantizer.args[0])
        self.assertNotIn(QPARAM_KEY, embedding.meta)
        self.assertEqual(linear.meta[QPARAM_KEY].scale, [0.2])
        self.assertEqual(set(dequantizer.users), {embedding})

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

    def test_public_conversion_preserves_per_row_and_per_tensor_weight_domains(self):
        tokens = torch.tensor([[0, 1]], dtype=torch.int64)
        for per_channel in (True, False):
            with self.subTest(per_channel=per_channel):
                module = _QuantizedEmbedding(per_channel=per_channel).eval()
                converted = tico.convert(module, (tokens,))
                document = CircleDocument.from_bytes(converted.circle_binary)
                document.verify()
                runtime = CircleReferenceRuntime(document)
                names = [operator.name for operator in runtime.program.operators]
                self.assertEqual(names, ["DEQUANTIZE", "GATHER", "QUANTIZE"])
                gather = runtime.program.operators[1]
                for index in (gather.inputs[0], gather.outputs[0]):
                    self.assertEqual(
                        runtime.program.tensor(index).tensor_type,
                        circle.TensorType.TensorType.FLOAT32,
                    )
                # All table values avoid half-grid ties, making both reference
                # rounding conventions agree with PyTorch fake quantization.
                expected = module(tokens).numpy()
                actual = runtime.run(
                    [tokens.numpy()], mode=ExecutionMode.FAKE_QUANTIZE
                ).outputs[0]
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
                raw = runtime.run([tokens.numpy()]).outputs[0]
                expected_codes = np.rint(expected / np.float32(0.3) + 128).astype(
                    np.uint8
                )
                np.testing.assert_array_equal(raw, expected_codes)


if __name__ == "__main__":
    unittest.main()

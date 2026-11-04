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

import unittest

import torch
import torch.ao.quantization.fx._decomposed

from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.quantization.passes.preserve_embedding_quantization import (
    PreserveEmbeddingQuantization,
)
from tico.quantization.passes.remove_weight_dequant_op import RemoveWeightDequantOp
from tico.quantization.wrapq.utils.check_missing_qparam import check_missing_qparam
from tico.serialize.quant_param import QPARAM_KEY
from tico.utils.convert import _prepare_exported_program


class _Embedding(torch.nn.Module):
    def __init__(self, *, axis=0, shared=False, quant_min=-32768):
        super().__init__()
        self.axis, self.shared, self.quant_min = axis, shared, quant_min
        self.register_buffer(
            "table", torch.tensor([[0, 10], [20, 255]], dtype=torch.uint8)
        )
        self.register_buffer("scales", torch.tensor([0.125, 0.5]))
        self.register_buffer("zeros", torch.tensor([1, 200], dtype=torch.int64))

    def forward(self, ids):
        table = torch.ops.quantized_decomposed.dequantize_per_channel.default(
            self.table, self.scales, self.zeros, self.axis, 0, 255, torch.uint8
        )
        emb = torch.nn.functional.embedding(ids, table)
        emb = torch.fake_quantize_per_tensor_affine(
            emb, 0.125, 0, self.quant_min, 32767
        )
        scaled = torch.fake_quantize_per_tensor_affine(emb * 2, 0.25, 0, -32768, 32767)
        return (scaled, emb) if self.shared else scaled


class _DynamicScaleEmbedding(_Embedding):
    def forward(self, ids, scale=None):
        emb = super().forward(ids)
        scale = torch.fake_quantize_per_tensor_affine(scale, 0.25, 0, -32768, 32767)
        return torch.fake_quantize_per_tensor_affine(
            emb * scale, 0.25, 0, -32768, 32767
        )


class PreserveEmbeddingQuantizationTest(unittest.TestCase):
    def test_dynamic_quantized_scale_has_explicit_unsupported_diagnostic(self):
        program = torch.export.export(
            _DynamicScaleEmbedding().eval(),
            (torch.tensor([[0, 1]]), torch.tensor(1.7)),
        )
        with self.assertRaisesRegex(
            RuntimeError, "dynamic quantized MUL operand.*static scale required"
        ):
            _prepare_exported_program(program)

    def test_strict_checker_accepts_only_explicit_cpu_contract(self):
        ep = _prepare_exported_program(
            torch.export.export(_Embedding().eval(), (torch.tensor([[0, 1]]),))
        )
        check_missing_qparam(ep, strict=True)
        emb = next(
            n for n in ep.graph.nodes if n.target == torch.ops.aten.embedding.default
        )
        self.assertNotIn(QPARAM_KEY, emb.meta)
        # An unrelated floating NPU tensor is still rejected.
        emb.meta.pop("_float_embedding_tensor_")
        with self.assertRaisesRegex(RuntimeError, "without qparam"):
            check_missing_qparam(ep, strict=True)

    def test_shared_activation_integer_boundary_and_idempotence(self):
        ep = torch.export.export(
            _Embedding(shared=True, quant_min=-32767).eval(), (torch.tensor([[0, 1]]),)
        )
        DecomposeFakeQuantize().call(ep)
        for _ in range(2):
            PreserveEmbeddingQuantization().call(ep)
            FoldQuantOps().call(ep)
            RemoveWeightDequantOp().call(ep)
        output = next(n for n in ep.graph.nodes if n.op == "output")
        self.assertEqual(
            output.args[0][1].target,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        )
        self.assertEqual(
            sum(n.target == torch.ops.aten.clamp.default for n in ep.graph.nodes), 1
        )
        mul = next(n for n in ep.graph.nodes if n.target == torch.ops.aten.mul.Tensor)
        self.assertEqual(
            mul.args[0].target,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        )

    def test_other_axis_keeps_existing_policy(self):
        ep = torch.export.export(_Embedding(axis=1).eval(), (torch.tensor([[0, 1]]),))
        DecomposeFakeQuantize().call(ep)
        PreserveEmbeddingQuantization().call(ep)
        FoldQuantOps().call(ep)
        RemoveWeightDequantOp().call(ep)
        emb = next(
            n for n in ep.graph.nodes if n.target == torch.ops.aten.embedding.default
        )
        self.assertIn(QPARAM_KEY, emb.meta)
        self.assertEqual(emb.args[0].op, "placeholder")


if __name__ == "__main__":
    unittest.main()

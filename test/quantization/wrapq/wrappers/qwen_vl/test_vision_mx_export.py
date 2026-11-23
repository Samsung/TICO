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

"""Tiny CPU vision contracts; no downloaded model or target NPU is required."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tico
import torch
from tico.circle import CircleDocument
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, mx
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.quantization.passes.legalize_mx_affine_domains import run_target_reference
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_attention import (
    QuantQwen3VLVisionAttention,
)
from tico.utils.errors import NotYetSupportedError
from tico.utils.mx.mx_ops import quantize_mx

from test.support.quantization.mx_layout import assert_qwen_mx_layout


class _Attention(torch.nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.config = SimpleNamespace()
        self.num_heads = 2
        self.head_dim = head_dim
        hidden = head_dim * self.num_heads
        self.qkv = torch.nn.Linear(hidden, hidden * 3)
        self.proj = torch.nn.Linear(hidden, hidden)


class _StaticAttention(torch.nn.Module):
    def __init__(self, attention, split_sizes):
        super().__init__()
        self.attention = attention
        self.split_sizes = split_sizes

    def forward(self, hidden, cos, sin):
        return self.attention(
            hidden,
            # The static profile owns chunk sizes; this tensor is unused.
            torch.tensor([0, hidden.shape[0]], dtype=torch.int32),
            position_embeddings=(cos, sin),
            attention_split_sizes=self.split_sizes,
        )


def _fixture(head_dim, split_sizes, *, mx_qkv_output=False):
    torch.manual_seed(123)
    cfg = PTQConfig(
        activation=affine(DType.int(16)),
        weight=affine(DType.uint(8)),
        overrides={
            "qkv.act_in": mx(axis=-1),
            "qkv.act_out": mx(axis=-1) if mx_qkv_output else affine(DType.int(16)),
            # Explicit affine qkv.act_out uses the existing observer before
            # split/RoPE; dimension 32 must not split an MX block in half.
            "proj.act_in": mx(axis=-1),
            "proj.act_out": mx(axis=-1),
            "logits": mx(axis=-1),
            "softmax": mx(axis=-1),
        },
    )
    attention = QuantQwen3VLVisionAttention(_Attention(head_dim), qcfg=cfg).eval()
    seq = sum(split_sizes)
    hidden = torch.randn(seq, head_dim * 2)
    angles = torch.randn(seq, head_dim)
    inputs = (hidden, angles.cos(), angles.sin())
    module = _StaticAttention(attention, split_sizes).eval()
    attention.enable_calibration()
    module(*inputs)
    attention.freeze_qparams()
    return module, inputs


def _reference(module, inputs, *, target):
    """Old rank-4 head/chunk structure, with explicit target rounding.

    Source observers keep their exact frozen grids. Native target LINEAR and
    PV output rounding is additional to those source grids; PV's RHS MX axis
    is the reduction dimension, not the value's feature dimension.
    """
    attention = module.attention
    hidden, cos, sin = inputs
    fq = attention._fq

    def linear(x, wrapped):
        layer = wrapped.wrapped
        if not target:
            return layer(x)
        x = fq(x, layer.obs_act_in)
        weight = fq(layer.module.weight, layer.obs_weight)
        result = torch.nn.functional.linear(x, weight, layer.module.bias)
        return fq(quantize_mx(result, "int8", -1), layer.obs_act_out)

    hidden = fq(hidden, attention.obs_hidden)
    qkv = linear(hidden, attention.qkv)
    q, k, v = qkv.reshape(hidden.shape[0], 3, 2, -1).permute(1, 0, 2, 3).unbind(0)
    cos = fq(cos, attention.obs_cos).unsqueeze(-2)
    sin = fq(sin, attention.obs_sin).unsqueeze(-2)

    def rope(x, prefix):
        first, second = x.chunk(2, dim=-1)
        first = fq(first, getattr(attention, f"obs_{prefix}_x1"))
        second = fq(second, getattr(attention, f"obs_{prefix}_x2"))
        negative = fq(-second, getattr(attention, f"obs_{prefix}_neg"))
        rotated = fq(
            torch.cat((negative, first), -1), getattr(attention, f"obs_{prefix}_cat")
        )
        cosine = fq(x * cos, getattr(attention, f"obs_{prefix}_cos"))
        sine = fq(rotated * sin, getattr(attention, f"obs_{prefix}_sin"))
        return fq(cosine + sine, getattr(attention, f"obs_{prefix}_rot"))

    q, k = rope(q, "q"), rope(k, "k")
    q, k, v = [x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
    q_chunks, k_chunks, v_chunks = [
        x.split(module.split_sizes, dim=2) for x in (q, k, v)
    ]
    chunks = []
    for qc, kc, vc in zip(q_chunks, k_chunks, v_chunks):
        heads = []
        for head in range(2):
            qi, ki, vi = [x[:, head : head + 1] for x in (qc, kc, vc)]
            if target:
                qi = quantize_mx(qi, "int8", -1)
                ki = quantize_mx(ki, "int8", -1)
            logits = fq(qi @ ki.transpose(-2, -1), attention.obs_logits)
            probabilities = fq(logits.softmax(-1), attention.obs_softmax)
            if target:
                # Rank-4 reference axis 2 corresponds to rank-3 PV axis 1.
                vi = quantize_mx(vi, "int8", 2)
            result = probabilities @ vi
            if target:
                result = quantize_mx(result, "int8", -1)
            heads.append(fq(result, attention.obs_attn_out))
        merged = torch.cat(heads, dim=1).transpose(1, 2).contiguous()
        chunks.append(merged.reshape(merged.shape[1], -1))
    return linear(torch.cat(chunks, dim=0), attention.proj)


class TestVisionMxExport(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)

    def test_rank3_bmm_preserves_frozen_head_and_chunk_computation(self):
        for head_dim, splits in ((32, (4,)), (64, (3, 5))):
            with self.subTest(head_dim=head_dim, chunks=splits):
                module, inputs = _fixture(head_dim, splits)
                expected = _reference(module, inputs, target=False)
                operands = []
                bmm = torch.bmm

                def record(lhs, rhs):
                    operands.append((lhs.shape, rhs.shape))
                    return bmm(lhs, rhs)

                with patch.object(torch, "bmm", record):
                    actual = module(*inputs)
                self.assertEqual(len(operands), 4 * len(splits))
                self.assertTrue(all(len(lhs) == len(rhs) == 3 for lhs, rhs in operands))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.shape, inputs[0].shape)
                self.assertEqual(module.attention.obs_softmax.axis, -1)

    def test_default_and_rank_sensitive_observers_preserve_legacy_values(self):
        torch.manual_seed(91)
        scalar = affine(DType.int(16))
        configs = (
            PTQConfig(activation=scalar, weight=affine(DType.uint(8))),
            PTQConfig(
                activation=affine(
                    DType.int(16),
                    qscheme=QScheme.PER_CHANNEL_ASYMM,
                    channel_axis=1,
                ),
                weight=affine(DType.uint(8)),
            ),
            PTQConfig(
                activation=scalar,
                weight=affine(DType.uint(8)),
                overrides={
                    site: mx(axis=3) for site in ("logits", "softmax", "attn_out")
                },
            ),
        )
        for index, config in enumerate(configs):
            with self.subTest(policy=index):
                attention = QuantQwen3VLVisionAttention(
                    _Attention(8), qcfg=config
                ).eval()
                module = _StaticAttention(attention, (3, 5)).eval()
                angles = torch.randn(8, 8)
                inputs = (torch.randn(8, 16), angles.cos(), angles.sin())
                torch.testing.assert_close(
                    module(*inputs),
                    _reference(module, inputs, target=False),
                    rtol=0,
                    atol=0,
                )
                attention.enable_calibration()
                module(*inputs)
                attention.freeze_qparams()
                torch.testing.assert_close(
                    module(*inputs),
                    _reference(module, inputs, target=False),
                    rtol=0,
                    atol=0,
                )
                self.assertEqual(attention._rank3_attention(), index == 0)
                # A frozen checkpoint retains the original rank-specific grid.
                if index == 1:
                    self.assertEqual(
                        attention.obs_softmax.require_frozen().observed_shape,
                        (1, 1, 5, 5),
                    )
                if index == 2:
                    self.assertEqual(attention.obs_softmax.axis, 3)

    def test_frozen_mx_qkv_is_not_replaced_with_an_uncalibrated_affine_grid(self):
        module, inputs = _fixture(32, (4,), mx_qkv_output=True)
        observer = module.attention.qkv.wrapped.obs_act_out
        policy = observer.require_frozen().policy
        with self.assertRaisesRegex(
            NotYetSupportedError, "cuts or regroups an MX block"
        ):
            tico.convert(
                module,
                inputs,
                config=tico.CompileConfigV1(legalize_mx_affine_domains=True),
            )
        self.assertIs(module.attention.qkv.wrapped.obs_act_out, observer)
        self.assertEqual(observer.require_frozen().policy, policy)

    def test_circle_domains_and_pre_fold_target_reference(self):
        for head_dim, splits in ((32, (4,)), (64, (3, 5))):
            module, inputs = _fixture(head_dim, splits)
            expected = _reference(module, inputs, target=True)
            for optimize in (False, True):
                with self.subTest(head_dim=head_dim, chunks=splits, o1=optimize):
                    snapshots: list[torch.export.ExportedProgram] = []
                    fold = FoldQuantOps.call

                    def capture(pass_self, ep):
                        if not snapshots:
                            snapshots.append(copy.deepcopy(ep))
                        return fold(pass_self, ep)

                    with patch.object(FoldQuantOps, "call", capture):
                        converted = tico.convert(
                            module,
                            inputs,
                            config=tico.CompileConfigV1(
                                move_affine_quantize_before_layout=True,
                                legalize_mx_affine_domains=True,
                                circle_optimize=optimize,
                            ),
                        )
                    self.assertEqual(len(snapshots), 1)
                    pre_fold = snapshots[0]
                    self.assertTrue(
                        any(
                            n.target == torch.ops.circle_custom.dequantize_mx.default
                            for n in pre_fold.graph.nodes
                        )
                    )
                    actual = run_target_reference(pre_fold, *inputs)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    document = CircleDocument.from_bytes(converted.circle_binary)
                    document.verify(raise_on_error=True)
                    counts = assert_qwen_mx_layout(self, document)
                    self.assertGreaterEqual(counts["BATCH_MATMUL"], 4 * len(splits))
                    bmm_nodes = [
                        n
                        for n in pre_fold.graph.nodes
                        if n.target == torch.ops.aten.bmm.default
                    ]
                    self.assertEqual(len(bmm_nodes), 4 * len(splits))
                    for node in bmm_nodes:
                        lhs, rhs = node.args
                        self.assertEqual(lhs.meta["val"].ndim, 3)
                        self.assertEqual(rhs.meta["val"].ndim, 3)
                        if lhs.target == torch.ops.circle_custom.dequantize_mx.default:
                            self.assertEqual(lhs.args[2] % 3, 2)
                            self.assertEqual(rhs.args[2] % 3, 1)


if __name__ == "__main__":
    unittest.main()

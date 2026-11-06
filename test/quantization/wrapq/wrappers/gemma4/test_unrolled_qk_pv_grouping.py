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

"""Regression contracts for the actual grouped Gemma4 attention wrapper core."""

import unittest
from typing import Any
from unittest.mock import patch

import torch
from tico.quantization.wrapq.dtypes import UINT8
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.observers.ema import EMAObserver
from tico.quantization.wrapq.observers.identity import IdentityObserver
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
    QuantGemma4TextAttention,
)
from torch import nn

from test.quantization.wrapq.wrappers.gemma4.test_quant_text_attention import (
    _has_gemma4,
    _make_text_config,
    _SKIP_MSG,
)


class _CoreAdapter(nn.Module):
    """Expose the real core without Q/K/V projections obscuring graph counts."""

    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, query, key, value, mask):
        """Call the product implementation under test."""
        return self.attention._forward_unrolled(
            query_states=query,
            key_states=key,
            value_states=value,
            attention_mask=mask,
        )


@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestUnrolledQKPVGrouping(unittest.TestCase):
    """Protect row order, shared mask semantics, and rank-3 grouping."""

    @staticmethod
    def _attention(heads=24, kv_heads=8):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        cfg = _make_text_config(
            hidden_size=heads * 4,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            num_global_key_value_heads=kv_heads,
        )
        attention = QuantGemma4TextAttention(Gemma4TextAttention(cfg, layer_idx=0))
        return attention.eval()

    @staticmethod
    def _inputs(batch, heads, kv_heads, tokens, noncontiguous):
        generator = torch.Generator().manual_seed(987)
        tensors = []
        for shape in ((batch, heads, tokens, 4), (batch, kv_heads, 7, 4)):
            tensor = torch.randn(shape, generator=generator)
            if noncontiguous:
                tensor = tensor.transpose(-1, -2).contiguous().transpose(-1, -2)
                assert not tensor.is_contiguous()
            tensors.append(tensor)
        value = torch.arange(batch * kv_heads * 7 * 4, dtype=torch.float32)
        value = value.reshape(batch, kv_heads, 7, 4) / 13
        if noncontiguous:
            value = value.transpose(-1, -2).contiguous().transpose(-1, -2)
            assert not value.is_contiguous()
        return *tensors, value

    def test_masks_noncontiguous_and_head_token_order(self):
        """Shared masks broadcast correctly with multiple batches and query rows."""
        for heads, kv_heads, batch, tokens, noncontiguous in (
            (24, 8, 2, 3, True),
            (4, 2, 1, 4, False),
            (4, 4, 2, 3, True),
            (3, 1, 2, 3, True),
            (24, 8, 1, 1, False),
        ):
            attention = self._attention(heads, kv_heads)
            query, key, value = self._inputs(
                batch, heads, kv_heads, tokens, noncontiguous
            )
            for mask_batch in (1, batch):
                for mask_tokens in (1, tokens):
                    for scaling in (1.0, 0.375):
                        with self.subTest(
                            heads=heads,
                            kv_heads=kv_heads,
                            batch=batch,
                            tokens=tokens,
                            mask_batch=mask_batch,
                            mask_tokens=mask_tokens,
                            scaling=scaling,
                        ):
                            attention.scaling = scaling
                            mask = torch.arange(mask_batch * mask_tokens * 7)
                            mask = mask.reshape(mask_batch, 1, mask_tokens, 7)
                            mask = (mask.remainder(5) == 0).float() * -1000
                            args = dict(
                                query_states=query,
                                key_states=key,
                                value_states=value,
                                attention_mask=mask,
                            )
                            actual = attention._forward_unrolled(**args)
                            expected = attention._forward_batched(**args)
                            self.assertEqual(
                                actual[0].shape, (batch, tokens, heads * 4)
                            )
                            self.assertEqual(actual[1].shape, (batch, heads, tokens, 7))
                            for left, right in zip(actual, expected):
                                torch.testing.assert_close(
                                    left, right, atol=2e-5, rtol=1e-5
                                )

    def test_explicit_head_token_pattern(self):
        """Distinct one-hot heads and token queries expose row permutation errors."""
        attention = self._attention(6, 2)
        query = torch.zeros(1, 6, 3, 4)
        key = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
        for head in range(6):
            for token in range(3):
                query[0, head, token, (head + token) % 4] = 20
        value = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
        output, weights = attention._forward_unrolled(
            query_states=query,
            key_states=key,
            value_states=value,
            attention_mask=torch.zeros(1, 1, 3, 4),
        )
        expected = torch.empty(1, 3, 6, 4)
        for head in range(6):
            for token in range(3):
                expected[0, token, head] = value[0, head // 3, (head + token) % 4]
        torch.testing.assert_close(
            output, attention.o_proj(expected.reshape(1, 3, 24)), atol=2e-6, rtol=1e-6
        )
        self.assertEqual(
            weights.argmax(-1).tolist(),
            [[[(h + t) % 4 for t in range(3)] for h in range(6)]],
        )

    def test_observer_shapes_and_return_only_weights_boundary(self):
        """Observe grouped PV and keep the return-only weight boundary off PV."""
        attention = self._attention(6, 2)
        query, key, value = self._inputs(1, 6, 2, 3, False)
        with patch.object(attention, "_fq", wraps=attention._fq) as calls:
            _CoreAdapter(attention)(query, key, value, torch.zeros(1, 1, 3, 7))
        for name, expected_shapes in (
            ("obs_logits_raw", [(1, 9, 7)] * 2),
            ("obs_logits", [(1, 9, 7)] * 2),
            # Only the mask add keeps the [B, rep, T, S] broadcast view.
            ("obs_mask_add", [(1, 3, 3, 7)] * 2),
            ("obs_softmax", [(1, 9, 7)] * 2),
            ("obs_attn_out", [(1, 9, 4)] * 2),
            ("obs_attn_weights", [(1, 6, 3, 7)]),
            ("obs_attn_out_h", [(1, 3, 6, 4)]),
        ):
            actual_shapes = [
                tuple(call.args[0].shape)
                for call in calls.call_args_list
                if call.args[1] is getattr(attention, name)
            ]
            self.assertEqual(actual_shapes, expected_shapes, name)

    def test_frozen_scalar_pv_boundary_keeps_one_fq_per_element(self):
        """A frozen group marker preserves scalar FQ while giving PV a domain."""
        attention = self._attention(6, 2)
        attention.enable_calibration()
        hidden = torch.randn(1, 3, attention.config.hidden_size)
        rope = (torch.ones(1, 3, 4), torch.zeros(1, 3, 4))
        attention(hidden, rope, attention_mask=torch.zeros(1, 1, 3, 3))
        attention.freeze_qparams()
        observer = attention.obs_attn_out
        self.assertTrue(observer._frozen)
        self.assertFalse(observer.enabled)
        query, key, value = self._inputs(1, 6, 2, 3, False)
        with patch.object(attention, "_fq", wraps=attention._fq) as calls:
            _CoreAdapter(attention)(query, key, value, torch.zeros(1, 1, 3, 7))
        pv_inputs = [
            call.args[0] for call in calls.call_args_list if call.args[1] is observer
        ]
        self.assertEqual([tuple(t.shape) for t in pv_inputs], [(1, 9, 4)] * 2)
        for group in pv_inputs:
            headwise = torch.cat(
                [observer.fake_quant(head) for head in group.split(3, dim=1)], dim=1
            )
            torch.testing.assert_close(
                observer.fake_quant(group), headwise, atol=0, rtol=0
            )

    def test_pv_boundary_is_grouped_for_all_modes_and_policies(self):
        """CALIB and QUANT use the same rank-3 PV boundary, including g=1."""
        affine = {"dtype": UINT8, "qscheme": QScheme.PER_TENSOR_ASYMM}
        policies: tuple[tuple[str, type[ObserverBase], dict[str, Any]], ...] = (
            ("minmax", MinMaxObserver, affine),
            ("ema", EMAObserver, {**affine, "momentum": 0.9}),
            ("identity", IdentityObserver, {}),
            ("mx_feature", MXObserver, {"axis": -1}),
            ("mx_row", MXObserver, {"axis": 1}),
            (
                "per_channel_feature",
                MinMaxObserver,
                {
                    "dtype": UINT8,
                    "qscheme": QScheme.PER_CHANNEL_ASYMM,
                    "channel_axis": -1,
                },
            ),
            (
                "per_channel_row",
                MinMaxObserver,
                {
                    "dtype": UINT8,
                    "qscheme": QScheme.PER_CHANNEL_ASYMM,
                    "channel_axis": 1,
                },
            ),
        )
        for heads, kv_heads in ((4, 4), (6, 2)):
            for name, observer_type, kwargs in policies:
                with self.subTest(heads=heads, kv_heads=kv_heads, policy=name):
                    torch.manual_seed(708)
                    attention = self._attention(heads, kv_heads)
                    observer = observer_type(name="attn_out", **kwargs)
                    attention.obs_attn_out = observer
                    attention.enable_calibration()
                    hidden = torch.randn(1, 3, attention.config.hidden_size)
                    rope = (torch.ones(1, 3, 4), torch.zeros(1, 3, 4))
                    attention(hidden, rope, attention_mask=torch.zeros(1, 1, 3, 3))
                    args = (
                        *self._inputs(1, heads, kv_heads, 3, False),
                        torch.zeros(1, 1, 3, 7),
                    )
                    expected = [(1, heads // kv_heads * 3, 4)] * kv_heads
                    for quantized in (False, True):
                        with self.subTest(quantized=quantized):
                            if quantized:
                                attention.freeze_qparams()
                                self.assertFalse(observer.enabled)
                            with patch.object(
                                attention, "_fq", wraps=attention._fq
                            ) as calls:
                                output, weights = _CoreAdapter(attention)(*args)
                            shapes = [
                                tuple(call.args[0].shape)
                                for call in calls.call_args_list
                                if call.args[1] is observer
                            ]
                            self.assertEqual(shapes, expected)
                            self.assertTrue(torch.isfinite(output).all())
                            self.assertTrue(torch.isfinite(weights).all())

    def test_pv_ema_updates_once_per_group(self):
        """EMA tracks group extrema rather than replaying head-wise updates."""
        torch.manual_seed(709)
        attention = self._attention(6, 2)
        observer = EMAObserver(
            name="attn_out",
            dtype=UINT8,
            qscheme=QScheme.PER_TENSOR_ASYMM,
            momentum=0.9,
        )
        attention.obs_attn_out = observer
        attention.enable_calibration()
        query = torch.zeros(1, 6, 3, 4)
        key = torch.zeros(1, 2, 7, 4)
        value = torch.tensor([1.0, 10.0]).reshape(1, 2, 1, 1).expand(1, 2, 7, 4)
        with patch.object(observer, "collect", wraps=observer.collect) as calls:
            _CoreAdapter(attention)(query, key, value, torch.zeros(1, 1, 3, 7))
        groups = [call.args[0] for call in calls.call_args_list]
        self.assertEqual([tuple(group.shape) for group in groups], [(1, 9, 4)] * 2)
        expected_min, expected_max = groups[0].min(), groups[0].max()
        momentum = observer.momentum
        for group in groups[1:]:
            expected_min = momentum * expected_min + (1 - momentum) * group.min()
            expected_max = momentum * expected_max + (1 - momentum) * group.max()
        torch.testing.assert_close(observer.min_val, expected_min, atol=0, rtol=0)
        torch.testing.assert_close(observer.max_val, expected_max, atol=0, rtol=0)

    def test_export_grouped_matmul_shapes(self):
        """24/8 has eight rank-3 QK and eight PV operations, without K/V repeats."""
        attention = self._attention()
        args = (*self._inputs(1, 24, 8, 3, False), torch.zeros(1, 1, 3, 7))
        adapter = _CoreAdapter(attention).eval()
        program = torch.export.export(adapter, args)
        matmuls = [
            n for n in program.graph.nodes if n.target == torch.ops.aten.matmul.default
        ]
        self.assertEqual(len(matmuls), 16)
        shapes = [tuple(n.meta["val"].shape) for n in matmuls]
        self.assertEqual(shapes.count((1, 9, 7)), 8)
        self.assertEqual(shapes.count((1, 9, 4)), 8)
        for node in matmuls:
            for operand in node.args[:2]:
                self.assertEqual(operand.meta["val"].dim(), 3)
        self.assertFalse(any("repeat" in str(n.target) for n in program.graph.nodes))
        for actual, expected in zip(program.module()(*args), adapter(*args)):
            torch.testing.assert_close(actual, expected)

    def test_export_grouping_avoids_mask_and_query_concat(self):
        """Tm=T masks broadcast over a head view; grouped queries are a reshape."""
        attention = self._attention(6, 2)
        args = (*self._inputs(1, 6, 2, 3, False), torch.zeros(1, 1, 3, 7))
        adapter = _CoreAdapter(attention).eval()
        program = torch.export.export(adapter, args)
        cats = [
            n for n in program.graph.nodes if n.target == torch.ops.aten.cat.default
        ]
        # Only the per-head output/weight stacks may concatenate; the mask and
        # the grouped query rows must not be materialized by a cat.
        for node in cats:
            shape = tuple(node.meta["val"].shape)
            self.assertNotIn(shape, [(1, 9, 7), (1, 9, 4)], node.format_node())
        adds = [n for n in program.graph.nodes if n.target == torch.ops.aten.add.Tensor]
        mask_adds = [
            n for n in adds if any(getattr(a, "name", "") == "mask" for a in n.args)
        ]
        self.assertEqual(len(mask_adds), 2)
        for node in mask_adds:
            self.assertEqual(tuple(node.meta["val"].shape), (1, 3, 3, 7))
        softmaxes = [
            n
            for n in program.graph.nodes
            if n.target in (torch.ops.aten._softmax.default, torch.ops.aten.softmax.int)
        ]
        self.assertEqual(
            [tuple(n.meta["val"].shape) for n in softmaxes], [(1, 9, 7)] * 2
        )
        for actual, expected in zip(program.module()(*args), adapter(*args)):
            torch.testing.assert_close(actual, expected)

    def test_g1_avoids_group_concat(self):
        """No group concat is introduced with one query per KV."""
        attention = self._attention(4, 4)
        args = (*self._inputs(1, 4, 4, 3, False), torch.zeros(1, 1, 3, 7))
        program = torch.export.export(_CoreAdapter(attention).eval(), args)
        # Output stack lowers separately; no explicit cat is used by the core.
        self.assertFalse(
            any(n.target == torch.ops.aten.cat.default for n in program.graph.nodes)
        )

    def test_rank_three_softmax_for_g1_and_decode(self):
        """Both g=1 and g>1 keep decode/prefill softmax on grouped rank-3 rows."""
        for heads, kv_heads in ((4, 4), (6, 2)):
            for tokens in (1, 3):
                with self.subTest(heads=heads, kv_heads=kv_heads, tokens=tokens):
                    attention = self._attention(heads, kv_heads)
                    args = (
                        *self._inputs(1, heads, kv_heads, tokens, False),
                        torch.zeros(1, 1, tokens, 7),
                    )
                    adapter = _CoreAdapter(attention).eval()
                    program = torch.export.export(adapter, args)
                    softmaxes = [
                        node
                        for node in program.graph.nodes
                        if node.target
                        in (
                            torch.ops.aten._softmax.default,
                            torch.ops.aten.softmax.int,
                        )
                    ]
                    expected = (1, heads // kv_heads * tokens, 7)
                    self.assertEqual(len(softmaxes), kv_heads)
                    for node in softmaxes:
                        self.assertEqual(
                            tuple(node.args[0].meta["val"].shape), expected
                        )
                        self.assertEqual(tuple(node.meta["val"].shape), expected)
                    for actual, expected_output in zip(
                        program.module()(*args), adapter(*args)
                    ):
                        torch.testing.assert_close(actual, expected_output)

    def test_training_dropout_remains_supported(self):
        """Grouped call shapes may change samples, but training still executes gradients."""
        attention = self._attention(6, 2).train()
        attention.attention_dropout = 0.25
        query, key, value = self._inputs(2, 6, 2, 3, True)
        query.requires_grad_()
        output, weights = attention._forward_unrolled(
            query_states=query,
            key_states=key,
            value_states=value,
            attention_mask=torch.zeros(1, 1, 1, 7),
        )
        self.assertTrue(torch.isfinite(weights).all())
        output.sum().backward()
        self.assertIsNotNone(query.grad)
        self.assertTrue(torch.isfinite(query.grad).all())

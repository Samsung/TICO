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

"""Checkpoint-free E2B/E4B key-axis and actual persistent storage accounting."""

import unittest
from types import SimpleNamespace
from typing import Any

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.recipes.debug.static_gemma4_runtime import StaticGemma4Runtime
from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
    QuantGemma4TextAttention,
)

from test.quantization.wrapq.wrappers.gemma4.test_quant_text_attention import (
    _has_gemma4,
    _make_text_config,
)


class TestGemma4CompactGeometry(unittest.TestCase):
    def test_real_backing_bytes_for_dense_variants(self):
        # These published geometries are fixtures, not implementation constants.
        for period, producers, shared, kv_heads, expected_bytes in (
            (5, 15, 20, 1, 37_711_872),
            (6, 24, 18, 2, 108_937_216),
        ):
            with self.subTest(kv_heads=kv_heads):
                count = producers + shared
                types = [
                    "full_attention" if (i + 1) % period == 0 else "sliding_attention"
                    for i in range(count)
                ]
                source = {
                    kind: max(i for i in range(producers) if types[i] == kind)
                    for kind in set(types)
                }
                runtime: Any = SimpleNamespace(
                    text_config=SimpleNamespace(
                        num_hidden_layers=count,
                        layer_types=types,
                        sliding_window=512,
                        head_dim=256,
                        global_head_dim=512,
                        num_key_value_heads=kv_heads,
                    ),
                    shared_kv_producers={
                        i: source[types[i]] for i in range(producers, count)
                    },
                    layout=SimpleNamespace(max_seq=2048),
                    device=torch.device("cpu"),
                )
                caches = StaticGemma4Runtime._allocate_empty_cache(
                    runtime, 1, torch.float32
                )
                actual = sum(
                    tensor.untyped_storage().nbytes()
                    for cache in caches
                    for tensor in (cache.past_k, cache.past_v)
                )
                self.assertEqual(actual, expected_bytes)
                for i, cache in enumerate(caches):
                    self.assertEqual(
                        cache.past_k.shape[2],
                        (
                            0
                            if i >= producers
                            else 2047 if types[i] == "full_attention" else 511
                        ),
                    )
                    # No sliced view retaining a larger storage allocation.
                    for tensor in (cache.past_k, cache.past_v):
                        self.assertEqual(
                            tensor.untyped_storage().nbytes(), tensor.numel() * 4
                        )

    def test_backing_capacity_context_window_edges(self):
        for context, window, sliding_past in (
            (3, 5, 2),
            (4, 4, 3),
            (5, 3, 2),
            (5, 1, 0),
        ):
            with self.subTest(context=context, window=window):
                runtime: Any = SimpleNamespace(
                    text_config=SimpleNamespace(
                        num_hidden_layers=2,
                        layer_types=["sliding_attention", "full_attention"],
                        sliding_window=window,
                        head_dim=4,
                        global_head_dim=4,
                        num_key_value_heads=1,
                    ),
                    layout=SimpleNamespace(max_seq=context),
                    device=torch.device("cpu"),
                )
                caches = StaticGemma4Runtime._allocate_empty_cache(
                    runtime, 1, torch.float32
                )
                self.assertEqual(caches[0].past_k.shape, (1, 1, sliding_past, 4))
                self.assertEqual(caches[1].past_k.shape, (1, 1, context - 1, 4))
                actual_bytes = sum(
                    value.untyped_storage().nbytes()
                    for cache in caches
                    for value in (cache.past_k, cache.past_v)
                )
                self.assertEqual(actual_bytes, 2 * 4 * 4 * (sliding_past + context - 1))

    @unittest.skipUnless(_has_gemma4(), "Transformers Gemma4 is required")
    def test_exported_qk_softmax_pv_use_window_with_real_head_geometry(self):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        torch.manual_seed(31)
        for kv_heads in (1, 2):
            config = _make_text_config(
                num_hidden_layers=2,
                num_attention_heads=8,
                num_key_value_heads=kv_heads,
                head_dim=256,
                global_head_dim=512,
                max_position_embeddings=2048,
                sliding_window=512,
                layer_types=["sliding_attention", "full_attention"],
            )
            for index, capacity, width in ((0, 512, 256), (1, 2048, 512)):
                with self.subTest(kv_heads=kv_heads, capacity=capacity):
                    wrapped = QuantGemma4TextAttention(
                        Gemma4TextAttention(config, layer_idx=index).eval(),
                        qcfg=PTQConfig(model_args={"text": {"max_seq": 2048}}),
                    ).eval()
                    module = wrapped.as_export_module("decode")
                    past = torch.zeros(1, kv_heads, capacity - 1, width)
                    with torch.no_grad():
                        program = torch.export.export(
                            module,
                            (torch.zeros(1, 1, 16),),
                            kwargs={
                                "attention_mask": torch.zeros(1, 1, capacity),
                                "position_embeddings": (
                                    torch.ones(1, 1, width),
                                    torch.zeros(1, 1, width),
                                ),
                                "past_key_value": (past, past.clone()),
                            },
                            strict=False,
                        )
                    matmuls = [
                        node
                        for node in program.graph.nodes
                        if node.op == "call_function"
                        and node.target == torch.ops.aten.matmul.default
                    ]
                    self.assertEqual(len(matmuls), 2 * kv_heads)
                    for qk, pv in zip(matmuls[::2], matmuls[1::2]):
                        self.assertEqual(qk.meta["val"].shape[-1], capacity)
                        self.assertEqual(pv.args[0].meta["val"].shape[-1], capacity)
                        self.assertEqual(pv.args[1].meta["val"].shape[-2], capacity)
                    softmaxes = [
                        node
                        for node in program.graph.nodes
                        if node.op == "call_function" and "softmax" in str(node.target)
                    ]
                    self.assertEqual(len(softmaxes), kv_heads)
                    self.assertTrue(
                        all(
                            node.meta["val"].shape[-1] == capacity for node in softmaxes
                        )
                    )

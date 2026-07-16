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

import importlib.util
import unittest
from types import SimpleNamespace
from typing import Optional

import torch

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


class _FixedDeltaAppendLayer(torch.nn.Module):
    def __init__(self, new_k: torch.Tensor, new_v: torch.Tensor):
        super().__init__()
        self.register_buffer("new_k", new_k)
        self.register_buffer("new_v", new_v)
        self.past_key_value: Optional[tuple[torch.Tensor, ...]] = None
        self.attention_mask: Optional[torch.Tensor] = None
        self.position_embeddings: Optional[tuple[torch.Tensor, ...]] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ):
        self.past_key_value = tuple(tensor.clone() for tensor in past_key_value)
        self.attention_mask = attention_mask.clone()
        self.position_embeddings = tuple(
            tensor.clone() for tensor in position_embeddings
        )
        return hidden_states, self.new_k, self.new_v


@unittest.skipUnless(
    HAS_TRANSFORMERS, "transformers is required for static runtime helpers"
)
class TestStaticLlamaRuntimeHelpers(unittest.TestCase):
    def test_static_llama_padding_and_position_helpers(self):
        """Static LLaMA runtime helpers should handle right-padded prompt layouts."""
        from tico.quantization.recipes.debug.static_llama_runtime import (
            _build_position_ids_from_valid_token_mask,
            _normalize_valid_token_mask,
            _validate_padding_layout,
        )

        input_ids = torch.tensor([[4, 5, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 0, 0]])

        valid = _normalize_valid_token_mask(
            input_ids,
            attention_mask,
            pad_token_id=0,
            device=torch.device("cpu"),
        )
        _validate_padding_layout(valid, "right")
        position_ids = _build_position_ids_from_valid_token_mask(valid)

        self.assertEqual(valid.tolist(), [[True, True, False, False]])
        self.assertEqual(position_ids.tolist(), [[0, 1, 0, 0]])

    def test_static_llama_gather_last_token_logits(self):
        """Logit gathering should select the last real prompt token for each batch row."""
        from tico.quantization.recipes.debug.static_llama_runtime import (
            _gather_last_token_logits,
        )

        logits = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        valid = torch.tensor(
            [
                [True, True, False, False],
                [False, False, True, True],
            ]
        )

        gathered = _gather_last_token_logits(logits, valid)

        self.assertTrue(torch.equal(gathered[0], logits[0, 1]))
        self.assertTrue(torch.equal(gathered[1], logits[1, 3]))

    def test_append_prefill_attention_mask_layout(self):
        """Append mask should expose valid past and causal tail slots only."""
        from tico.quantization.recipes.debug.static_llama_runtime import (
            _build_append_prefill_attention_mask,
        )

        mask_value = -120.0
        mask = _build_append_prefill_attention_mask(
            batch_size=1,
            past_len=2,
            actual_len=2,
            bucket_size=3,
            max_seq=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            mask_value=mask_value,
        )

        expected = torch.full((1, 3, 8), mask_value, dtype=torch.float32)
        expected[0, 0, [0, 1, 5]] = 0.0
        expected[0, 1, [0, 1, 5, 6]] = 0.0

        torch.testing.assert_close(mask, expected)

    def test_append_prefill_writes_only_valid_delta_kv(self):
        """Only the valid prefix of a bucket's delta KV should enter the cache."""
        from tico.quantization.recipes.debug.static_llama_runtime import (
            LayerCache,
            StaticLlamaLayerRuntime,
        )

        runtime = StaticLlamaLayerRuntime.__new__(StaticLlamaLayerRuntime)
        runtime.device = torch.device("cpu")
        runtime.padding_side = "right"
        runtime.tokenizer = SimpleNamespace(pad_token_id=0)
        runtime.max_seq = 8
        runtime.num_hidden_layers = 1
        runtime.past_len = 2
        runtime.embed_tokens = torch.nn.Embedding(32, 2)
        runtime.final_norm = torch.nn.Identity()
        runtime.lm_head = torch.nn.Identity()
        runtime.rope_cos = torch.zeros(1, runtime.max_seq, 1)
        runtime.rope_sin = torch.zeros_like(runtime.rope_cos)

        past_k = torch.full((1, 1, runtime.max_seq - 1, 1), -1.0)
        past_v = torch.full_like(past_k, -2.0)
        past_k[0, 0, :2, 0] = torch.tensor([10.0, 11.0])
        past_v[0, 0, :2, 0] = torch.tensor([20.0, 21.0])
        before_k = past_k.clone()
        before_v = past_v.clone()
        cache = LayerCache(past_k=past_k, past_v=past_v)
        runtime.layer_caches = [cache]

        new_k = torch.tensor([[[[30.0], [31.0], [99.0]]]])
        new_v = torch.tensor([[[[40.0], [41.0], [98.0]]]])
        append_layer = _FixedDeltaAppendLayer(new_k, new_v)
        runtime.append_prefill_layers = torch.nn.ModuleList([append_layer])

        logits = runtime.append_prefill(
            torch.tensor([[5, 6]]),
            torch.ones(1, 2, dtype=torch.long),
            bucket_size=3,
        )

        self.assertEqual(runtime.past_len, 4)
        self.assertEqual(tuple(logits.shape), (1, 2))
        torch.testing.assert_close(cache.past_k[:, :, :2, :], before_k[:, :, :2, :])
        torch.testing.assert_close(cache.past_v[:, :, :2, :], before_v[:, :, :2, :])
        torch.testing.assert_close(cache.past_k[:, :, 2:4, :], new_k[:, :, :2, :])
        torch.testing.assert_close(cache.past_v[:, :, 2:4, :], new_v[:, :, :2, :])
        torch.testing.assert_close(cache.past_k[:, :, 4:, :], before_k[:, :, 4:, :])
        torch.testing.assert_close(cache.past_v[:, :, 4:, :], before_v[:, :, 4:, :])

        assert append_layer.past_key_value is not None
        self.assertEqual(tuple(append_layer.past_key_value[0].shape), (1, 1, 5, 1))
        assert append_layer.attention_mask is not None
        self.assertEqual(tuple(append_layer.attention_mask.shape), (1, 3, 8))
        assert append_layer.position_embeddings is not None
        self.assertEqual(tuple(append_layer.position_embeddings[0].shape), (1, 3, 1))
        self.assertEqual(tuple(append_layer.position_embeddings[1].shape), (1, 3, 1))

    def test_append_prefill_chunked_keeps_bucket_for_short_final_chunk(self):
        """The final short chunk should still invoke the fixed-size bucket graph."""
        from tico.quantization.recipes.debug.static_llama_runtime import (
            StaticLlamaLayerRuntime,
        )

        runtime = StaticLlamaLayerRuntime.__new__(StaticLlamaLayerRuntime)
        runtime.device = torch.device("cpu")
        runtime.padding_side = "right"
        runtime.tokenizer = SimpleNamespace(pad_token_id=0)

        calls = []

        def fake_append_prefill(
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            *,
            bucket_size: int,
        ) -> torch.Tensor:
            calls.append((input_ids.clone(), attention_mask.clone(), int(bucket_size)))
            return torch.tensor([[float(len(calls))]])

        runtime.append_prefill = fake_append_prefill
        logits = runtime.append_prefill_chunked(
            torch.tensor([[1, 2, 3, 4, 5]]),
            torch.ones(1, 5, dtype=torch.long),
            bucket_size=3,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0].tolist(), [[1, 2, 3]])
        self.assertEqual(calls[1][0].tolist(), [[4, 5]])
        self.assertEqual(calls[0][1].tolist(), [[1, 1, 1]])
        self.assertEqual(calls[1][1].tolist(), [[1, 1]])
        self.assertEqual(calls[0][2], 3)
        self.assertEqual(calls[1][2], 3)
        torch.testing.assert_close(logits, torch.tensor([[2.0]]))

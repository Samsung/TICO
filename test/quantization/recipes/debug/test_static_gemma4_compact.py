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

"""Compact caches versus independent full-capacity causal attention."""

import unittest
from unittest import mock

import torch

from tico.quantization.wrapq.wrappers.gemma4.rope import prepare_gemma4_rope_sin

from test.quantization.recipes.debug import test_static_gemma4_finalize as fixture
from test.quantization.wrapq.wrappers.gemma4.test_quant_text_model import _has_gemma4


def make_compact_runtime(*, quantized=False, multiple_consumers=True, window=3):
    """Reuse real random-weight stages with two consumers per producer."""
    make_config = fixture._make_text_config

    def config(**kwargs):
        kwargs["sliding_window"] = window
        if multiple_consumers:
            kwargs.update(
                layer_types=["sliding_attention", "full_attention"] * 3,
                num_kv_shared_layers=4,
            )
        return make_config(**kwargs)

    with mock.patch.object(fixture, "_make_text_config", side_effect=config):
        runtime = fixture.make_runtime(quantized=quantized)
    runtime.decode_kv_policy = "compact_sliding_v1"
    return runtime


class FullCapacityReference:
    """Explicit S-slot attention and absolute masks, without runtime cache helpers.

    Past slots retain every valid token at its absolute index. The current token
    occupies slot S-1. Sliding visibility follows k <= q and k > q-W, so the
    reference never crops, rolls or remaps keys.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.position = 0
        self.keys = {}

    def masks_and_rope(self, length, *, decode):
        runtime = self.runtime
        size = runtime.layout.max_seq
        if decode:
            query = torch.tensor([self.position])
            key = torch.arange(size)
            key[-1] = self.position
            occupied = torch.arange(size) < self.position
            occupied[-1] = True
        else:
            query = torch.arange(size)
            key = torch.arange(size)
            occupied = key < length
        masks, rope = {}, {}
        rotary = runtime.model.model.language_model.rotary_emb
        for kind in set(runtime.text_config.layer_types):
            keep = (key[None, :] <= query[:, None]) & occupied[None, :]
            if kind == "sliding_attention":
                keep &= (
                    key[None, :] > query[:, None] - runtime.text_config.sliding_window
                )
            mask = torch.where(keep, 0.0, -120.0)[None]
            masks[kind] = mask if decode else mask.unsqueeze(1)
            cos, sin = rotary(torch.zeros(1, len(query), 1), query[None], kind)
            rope[kind] = (cos, prepare_gemma4_rope_sin(sin, runtime.rope_convention))
        return masks, rope

    def run_layers(self, hidden, ple, masks, rope, *, decode):
        runtime = self.runtime
        shared: dict[int, tuple[torch.Tensor, ...]] = {}
        layers = runtime.decode_layers if decode else runtime.prefill_layers
        for index, layer in enumerate(layers):
            kind = runtime.text_config.layer_types[index]
            producer = runtime.shared_kv_producers.get(index)
            past = None
            if decode and producer is None:
                past = self.keys[index]
            kwargs = dict(
                hidden_states=hidden,
                attention_mask=masks[kind],
                position_embeddings=rope[kind],
                per_layer_input=ple[:, :, index],
                shared_key_value=shared.get(producer),
            )
            if decode:
                kwargs["past_key_value"] = past
            result = layer(**kwargs)
            if producer is not None:
                hidden = result
                continue
            hidden, key, value = result
            if decode:
                assert past is not None
                # Snapshot before independently writing into the full S-1 cache.
                shared[index] = tuple(
                    torch.cat((old, new), 2) for old, new in zip(past, (key, value))
                )
                for old, new in zip(past, (key, value)):
                    old[:, :, self.position : self.position + 1].copy_(new)
            else:
                shared[index] = (key, value)
                self.keys[index] = tuple(
                    torch.zeros_like(tensor[:, :, :-1]) for tensor in (key, value)
                )
                for old, new in zip(self.keys[index], (key, value)):
                    old[:, :, : self.position].copy_(new[:, :, : self.position])
        return hidden

    def prefill(self, batch):
        runtime = self.runtime
        self.position = int(batch["valid_length"])
        hidden = runtime.token_embedding(batch["llm_input_ids"])
        ple = runtime._build_per_layer_inputs(batch["llm_input_ids"], hidden)
        masks, rope = self.masks_and_rope(self.position, decode=False)
        hidden = self.run_layers(hidden, ple, masks, rope, decode=False)
        last = hidden[:, self.position - 1 : self.position]
        return last, runtime._apply_final_logit_softcapping(
            runtime.lm_head(last)[:, -1]
        )

    def decode(self, token):
        runtime = self.runtime
        hidden = runtime.token_embedding(token)
        ple = runtime._build_per_layer_inputs(token, hidden)
        masks, rope = self.masks_and_rope(1, decode=True)
        hidden = self.run_layers(hidden, ple, masks, rope, decode=True)
        self.position += 1
        return hidden, runtime._apply_final_logit_softcapping(
            runtime.lm_head(hidden)[:, -1]
        )


@unittest.skipUnless(_has_gemma4(), "transformers Gemma4 modules are required")
class TestStaticGemma4Compact(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)

    def assert_cache_contract(self, runtime, reference):
        for index, cache in enumerate(runtime.layer_caches):
            if index in runtime.shared_kv_producers:
                self.assertEqual(cache.past_k.numel(), 0)
                self.assertEqual(cache.past_v.numel(), 0)
                continue
            window = (
                runtime.text_config.sliding_window
                if runtime.text_config.layer_types[index] == "sliding_attention"
                else runtime.layout.max_seq
            )
            capacity = min(runtime.layout.max_seq, window) - 1
            retained = min(reference.position, capacity)
            self.assertEqual(cache.past_k.shape[2], capacity)
            self.assertEqual(cache.retained_len, retained)
            for actual, expected in zip(
                (cache.past_k, cache.past_v), reference.keys[index]
            ):
                # Storage must own only C-1 tokens, rather than a view of S-1.
                self.assertEqual(
                    actual.untyped_storage().nbytes(),
                    actual.numel() * actual.element_size(),
                )
                torch.testing.assert_close(
                    actual[:, :, :retained],
                    expected[:, :, reference.position - retained : reference.position],
                    atol=1e-5,
                    rtol=1e-5,
                )

    def _exercise(
        self, *, length, quantized=False, finalize=False, split=False, window=3
    ):
        runtime = make_compact_runtime(quantized=quantized, window=window)
        reference = FullCapacityReference(runtime)
        batch = fixture.make_batch(length)
        observed = []
        handles = [
            layer.register_forward_pre_hook(
                lambda module, args, kwargs, index=index: observed.append(
                    (index, kwargs)
                ),
                with_kwargs=True,
            )
            for index, layer in enumerate(runtime.decode_layers)
        ]
        last_hidden = []
        head_hook = runtime.lm_head.register_forward_pre_hook(
            lambda module, args: last_hidden.append(args[0].clone())
        )
        with torch.no_grad():
            expected_hidden, expected_logits = reference.prefill(batch)
            logits = runtime.prefill(
                batch, last_token_only=finalize, ple_prefill_split=split
            )
            # Same FP32 weights/grids, differing attention reduction lengths.
            torch.testing.assert_close(logits, expected_logits, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(
                last_hidden[-1], expected_hidden, atol=1e-5, rtol=1e-5
            )
            self.assert_cache_contract(runtime, reference)
            observed.clear()
            while reference.position < 9:
                token = torch.tensor([[17 + reference.position]])
                expected_hidden, expected_logits = reference.decode(token)
                observed.clear()
                logits = runtime.decode_one(token)
                torch.testing.assert_close(
                    logits, expected_logits, atol=1e-5, rtol=1e-5
                )
                torch.testing.assert_close(
                    last_hidden[-1], expected_hidden, atol=1e-5, rtol=1e-5
                )
                self.assertEqual(runtime.past_len, reference.position)
                self.assert_cache_contract(runtime, reference)
                self.assertEqual(len(observed), 6)
                for index, inputs in observed:
                    capacity = min(10, window) if index % 2 == 0 else 10
                    self.assertEqual(inputs["attention_mask"].shape, (1, 1, capacity))
                    key = (
                        inputs["shared_key_value"][0]
                        if index in runtime.shared_kv_producers
                        else inputs["past_key_value"][0]
                    )
                    self.assertEqual(
                        key.shape[2], capacity if index >= 2 else capacity - 1
                    )
                # Two consumers observe the exact same invocation-owned snapshot.
                self.assertIs(
                    observed[2][1]["shared_key_value"],
                    observed[4][1]["shared_key_value"],
                )
                self.assertIs(
                    observed[3][1]["shared_key_value"],
                    observed[5][1]["shared_key_value"],
                )
            with self.assertRaisesRegex(ValueError, "full"):
                runtime.decode_one(torch.tensor([[1]]))
        head_hook.remove()
        for handle in handles:
            handle.remove()

    def test_full_reference_across_window_boundaries_and_multiple_evictions(self):
        for length in (1, 2, 3, 4):
            with self.subTest(length=length):
                self._exercise(length=length)

    def test_finalize_and_split_ple_match_full_reference(self):
        for length in (1, 3, 4):
            for split in (False, True):
                with self.subTest(length=length, split=split):
                    self._exercise(length=length, finalize=True, split=split)

    def test_frozen_affine_cache_keeps_same_grids_during_eviction(self):
        self._exercise(length=1, quantized=True, finalize=True)

    def test_finalize_preserves_bias_and_rejects_visible_discarded_key(self):
        runtime = make_compact_runtime()
        batch = fixture.make_batch(4)
        ids = batch["llm_input_ids"]
        with torch.no_grad():
            hidden = runtime.token_embedding(ids)
            ple = runtime._build_per_layer_inputs(ids, hidden)
            masks, rope = runtime.build_prefill_masks_and_rope(
                ids, batch["attention_mask"]
            )
            # The final query sees original columns 1, 2, 3. Preserve both an
            # extra restriction and a nonzero additive bias on retained keys.
            masks["sliding_attention"][0, 0, 3, 1] = -120.0
            masks["sliding_attention"][0, 0, 3, 2] = -0.75
            full = runtime.prefill_decoder(
                hidden, ple, masks, rope, 4, last_token_only=False
            )
            observed = []
            handle = runtime.decode_layers[2].register_forward_pre_hook(
                lambda module, args, kwargs: observed.append(
                    kwargs["attention_mask"].clone()
                ),
                with_kwargs=True,
            )
            self.addCleanup(handle.remove)
            single = runtime.prefill_decoder(
                hidden, ple, masks, rope, 4, last_token_only=True
            )
            torch.testing.assert_close(single, full, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(
                observed[0], torch.tensor([[[-120.0, -0.75, 0.0]]]), atol=0, rtol=0
            )
            masks["sliding_attention"][0, 0, 3, 0] = -1.0
            with self.assertRaisesRegex(ValueError, "outside the retained window"):
                runtime.prefill_decoder(
                    hidden, ple, masks, rope, 4, last_token_only=True
                )

    def test_window_one_has_empty_persistent_past(self):
        self._exercise(length=2, finalize=True, split=True, window=1)


if __name__ == "__main__":
    unittest.main()

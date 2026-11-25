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

"""Small real Gemma4 regressions; no checkpoint, processor, accelerator or network."""

import unittest
from types import SimpleNamespace
from typing import Any, cast

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.recipes.debug.export_regression import _state_contract
from tico.quantization.recipes.debug.static_gemma4_runtime import StaticGemma4Runtime
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    Gemma4MMFusionExportAdapter,
    Gemma4PLEEmbeddingExportAdapter,
    Gemma4PLEProjectionExportAdapter,
    Gemma4TokenEmbeddingExportAdapter,
)
from tico.quantization.wrapq.wrappers.gemma4.quant_text_decoder_layer import (
    QuantGemma4TextDecoderLayer,
)
from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
    QuantGemma4TextModel,
)
from tico.quantization.wrapq.wrappers.gemma4.utils import (
    resolve_gemma4_shared_kv_suffix,
)

from test.quantization.wrapq.wrappers.gemma4.test_quant_text_model import (
    _has_gemma4,
    _make_text_config,
    _make_text_model,
)


def make_runtime(*, quantized=False, ple=True, shared=True, visual_tokens=0):
    """Use production wrappers and stage boundaries around a tiny random model."""
    torch.manual_seed(136)
    config = _make_text_config(
        layer_types=["sliding_attention", "full_attention"] * 2,
        num_kv_shared_layers=2 if shared else 0,
        hidden_size_per_layer_input=8 if ple else 0,
        vocab_size_per_layer_input=128,
        sliding_window=3,
        global_head_dim=8,
        max_position_embeddings=16,
    )
    original = _make_text_model(config)
    text = QuantGemma4TextModel(
        original, qcfg=PTQConfig(model_args={"text": {"max_seq": 10}})
    ).eval()
    if quantized:
        # Collect full prefill statistics, including cache storage. The comparison
        # uses these same frozen grids in both adapters; no decode recalibration.
        text.enable_calibration()
        with torch.no_grad():
            text(input_ids=torch.arange(1, 10).reshape(1, 9), use_cache=True)
        text.freeze_qparams()
    runtime: Any = object.__new__(StaticGemma4Runtime)
    runtime.model = original
    original.model = SimpleNamespace(language_model=original)
    runtime.text_model = text
    runtime.text_config = config
    runtime.device = torch.device("cpu")
    runtime.layout = SimpleNamespace(max_seq=10, padding_side="right")
    runtime.rope_convention = "pre_negated_sin"
    runtime.attention_mask_fill_value = -120.0
    runtime.last_token_only_prefill = False
    runtime.shared_kv_suffix_start, runtime.shared_kv_producers = (
        resolve_gemma4_shared_kv_suffix(text.layers)
    )
    runtime.token_embedding = Gemma4TokenEmbeddingExportAdapter(text)
    runtime.ple_embedding = Gemma4PLEEmbeddingExportAdapter(text) if ple else None
    runtime.ple_projection = Gemma4PLEProjectionExportAdapter(text) if ple else None
    runtime.mm_fusion = Gemma4MMFusionExportAdapter(
        visual_start_idx=1, num_visual_tokens=visual_tokens
    )
    # Only the expensive vision frontend is replaced. Its output still traverses
    # the real fixed-slot fusion and full-sequence PLE projection.
    runtime._get_or_create_vision_prefill = lambda _: torch.nn.Identity()
    runtime.lm_head = torch.nn.Sequential(text.norm, torch.nn.Linear(16, 128))
    observer = text.obs_per_layer_inputs if ple else None
    runtime.prefill_layers = torch.nn.ModuleList(
        cast(QuantGemma4TextDecoderLayer, layer.wrapped).as_export_module(
            "prefill", return_kv=True, per_layer_input_observer=observer
        )
        for layer in text.layers
    )
    runtime.decode_layers = torch.nn.ModuleList(
        cast(QuantGemma4TextDecoderLayer, layer.wrapped).as_export_module(
            "decode", return_kv=True, per_layer_input_observer=observer
        )
        for layer in text.layers
    )
    runtime.layer_caches = []
    runtime.past_len = 0
    return runtime


def make_batch(length, *, visual_tokens=0):
    """Right padding is physical padding after fusion, not a shorter graph."""
    ids = torch.zeros(1, 10, dtype=torch.long)
    ids[:, :length] = torch.arange(1, length + 1)
    valid = torch.arange(10).unsqueeze(0) < length
    return {
        "llm_input_ids": ids,
        "attention_mask": valid.long(),
        "valid_length": torch.tensor(length),
        "pixel_values": torch.arange(visual_tokens * 16).reshape(visual_tokens, 16)
        / 20.0,
        "image_position_ids": torch.zeros(1, 1, 2, dtype=torch.long),
    }


def cache_snapshot(runtime):
    """Capture physical persistent slots, including the unused zero padding."""
    return [
        (cache.past_k.clone(), cache.past_v.clone()) for cache in runtime.layer_caches
    ]


@unittest.skipUnless(_has_gemma4(), "transformers Gemma4 modules are required")
class TestStaticGemma4Finalize(unittest.TestCase):
    def assert_caches_equal(self, runtime, expected):
        for cache, (key, value) in zip(runtime.layer_caches, expected):
            torch.testing.assert_close(cache.past_k, key, atol=0, rtol=0)
            torch.testing.assert_close(cache.past_v, value, atol=0, rtol=0)

    def _parity(self, *, quantized=False, length=4, ple=True, visual_tokens=0):
        runtime = make_runtime(
            quantized=quantized, ple=ple, visual_tokens=visual_tokens
        )
        batch = make_batch(length, visual_tokens=visual_tokens)
        calls = []
        last_hidden = []
        ple_lengths = []
        handles = []
        if ple:
            handles.append(
                runtime.ple_projection.register_forward_pre_hook(
                    lambda module, args: ple_lengths.append(args[0].shape[1])
                )
            )
        for index, layer in enumerate(runtime.prefill_layers):
            handles.append(
                layer.register_forward_pre_hook(
                    lambda module, args, kwargs, i=index: calls.append(
                        ("prefill", i, kwargs["hidden_states"].shape[1])
                    ),
                    with_kwargs=True,
                )
            )
        for index, layer in enumerate(runtime.decode_layers):
            handles.append(
                layer.register_forward_pre_hook(
                    lambda module, args, kwargs, i=index: calls.append(
                        ("decode", i, kwargs["hidden_states"].shape[1])
                    ),
                    with_kwargs=True,
                )
            )
        handles.append(
            runtime.lm_head.register_forward_pre_hook(
                lambda module, args: last_hidden.append(args[0].clone())
            )
        )
        frozen_before = _state_contract(runtime.text_model) if quantized else None
        with torch.no_grad():
            baseline = runtime.prefill(batch, last_token_only=False)
            baseline_hidden = last_hidden[-1]
            prefill_caches = cache_snapshot(runtime)
            self.assertEqual(runtime.past_len, length)
            reference = []
            for token in (17, 8, 29)[: min(3, 9 - length)]:
                logits = runtime.decode_one(torch.tensor([[token]]))
                reference.append(
                    (logits.clone(), cache_snapshot(runtime), runtime.past_len)
                )
            calls.clear()
            optimized = runtime.prefill(batch, last_token_only=True)
            optimized_hidden = last_hidden[-1]
            self.assertEqual(runtime.past_len, length)
            self.assert_caches_equal(runtime, prefill_caches)
            self.assertEqual(
                calls,
                [
                    ("prefill", 0, 10),
                    ("prefill", 1, 10),
                    ("decode", 2, 1),
                    ("decode", 3, 1),
                ],
            )
            # Float32 GEMM can change reduction scheduling for S=10 vs S=1.
            # 1e-5 covers roundoff, far below a uint8 activation grid step.
            torch.testing.assert_close(
                optimized_hidden, baseline_hidden, atol=1e-5, rtol=1e-5
            )
            torch.testing.assert_close(optimized, baseline, atol=1e-5, rtol=1e-5)
            for token, (expected_logits, expected_cache, expected_len) in zip(
                (17, 8, 29), reference
            ):
                logits = runtime.decode_one(torch.tensor([[token]]))
                torch.testing.assert_close(
                    logits, expected_logits, atol=1e-5, rtol=1e-5
                )
                self.assert_caches_equal(runtime, expected_cache)
                self.assertEqual(runtime.past_len, expected_len)
        if ple:
            self.assertEqual(
                ple_lengths, [10] + [1] * len(reference) + [10] + [1] * len(reference)
            )
        if quantized:
            self.assertEqual(_state_contract(runtime.text_model), frozen_before)
        for handle in handles:
            handle.remove()

    def test_fp_right_padding_window_and_capacity_boundaries(self):
        # Includes L=1, both sides of W=3, and the S-1 persistent output limit.
        for length in (1, 2, 3, 4, 8, 9):
            with self.subTest(length=length):
                self._parity(length=length)

    def test_frozen_quantized_prefill_and_three_decode_steps(self):
        for length in (1, 4, 6):
            with self.subTest(length=length):
                self._parity(quantized=True, length=length)

    def test_fused_visual_inputs_keep_decoder_positions_and_ple(self):
        self._parity(length=5, visual_tokens=2)

    def test_without_ple(self):
        self._parity(length=4, ple=False)

    def test_disabled_and_non_shared_models_keep_full_prefill(self):
        runtime = make_runtime(shared=False)
        batch = make_batch(3)
        self.assertEqual(runtime.shared_kv_suffix_start, 4)
        with torch.no_grad():
            full = runtime.prefill(batch, last_token_only=False)
            expected_cache = cache_snapshot(runtime)
            runtime.last_token_only_prefill = True
            fallback = runtime.prefill(batch)
        torch.testing.assert_close(fallback, full, atol=0, rtol=0)
        self.assert_caches_equal(runtime, expected_cache)

    def test_invalid_decoder_padding_and_length_fail_before_execution(self):
        runtime = make_runtime()
        for mask, valid_length, error in (
            ([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], 2, "Right padding"),
            ([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 2, "valid_length disagrees"),
            ([0] * 10, 0, "at least one valid token"),
            ([1] * 10, 10, "reserve one decode slot"),
        ):
            with self.subTest(mask=mask, valid_length=valid_length):
                batch = make_batch(3)
                batch["attention_mask"] = torch.tensor([mask])
                batch["valid_length"] = torch.tensor(valid_length)
                with self.assertRaisesRegex(ValueError, error):
                    runtime.prefill(batch, last_token_only=True)
                self.assertEqual(runtime.past_len, 0)
                self.assertEqual(runtime.layer_caches, [])

    def test_suffix_receives_original_query_mask_rope_and_full_invocation_kv(self):
        runtime = make_runtime()
        batch = make_batch(4)
        ids = batch["llm_input_ids"]
        masks, rope = runtime.build_prefill_masks_and_rope(ids, batch["attention_mask"])
        # An external visibility restriction must survive query selection exactly.
        masks["full_attention"][:, :, 3, 1] = -120.0
        hidden = runtime.token_embedding(ids)
        ple = runtime._build_per_layer_inputs(ids, hidden)
        suffix_inputs = []
        handles = [
            layer.register_forward_pre_hook(
                lambda module, args, kwargs: suffix_inputs.append(kwargs.copy()),
                with_kwargs=True,
            )
            for layer in runtime.decode_layers[2:]
        ]
        with torch.no_grad():
            runtime.layer_caches = runtime._allocate_empty_cache(1, hidden.dtype)
            full = runtime._prefill_decoder(
                hidden, ple, masks, rope, 4, last_token_only=False
            )
            runtime.layer_caches = runtime._allocate_empty_cache(1, hidden.dtype)
            single = runtime._prefill_decoder(
                hidden, ple, masks, rope, 4, last_token_only=True
            )
        torch.testing.assert_close(single, full[:, 3:4], atol=1e-5, rtol=1e-5)
        for i, actual in enumerate(suffix_inputs, start=2):
            kind = runtime.text_config.layer_types[i]
            torch.testing.assert_close(
                actual["attention_mask"].unsqueeze(1),
                masks[kind][:, :, 3:4],
                atol=0,
                rtol=0,
            )
            for actual_table, original_table in zip(
                actual["position_embeddings"], rope[kind]
            ):
                torch.testing.assert_close(
                    actual_table, original_table[:, 3:4], atol=0, rtol=0
                )
            torch.testing.assert_close(
                actual["per_layer_input"], ple[:, 3:4, i], atol=0, rtol=0
            )
            self.assertEqual(actual["shared_key_value"][0].shape[2], 10)
            self.assertGreater(
                torch.count_nonzero(actual["shared_key_value"][0][:, :, 3:4]).item(), 0
            )
            self.assertIsNone(actual.get("past_key_value"))
        self.assertEqual(len(suffix_inputs), 2)
        # Full and sliding consumers read distinct actual producers.
        self.assertNotEqual(
            suffix_inputs[0]["shared_key_value"][0].data_ptr(),
            suffix_inputs[1]["shared_key_value"][0].data_ptr(),
        )
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    unittest.main()

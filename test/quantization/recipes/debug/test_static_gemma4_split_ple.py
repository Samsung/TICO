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

"""Initial-prefill PLE range computation through real Gemma4 decoder stages."""

import unittest
import weakref
from unittest import mock

import torch

from tico.quantization.recipes.debug.export_regression import _state_contract
from tico.quantization.recipes.debug.static_gemma4_runtime import (
    StaticGemma4RuntimeConfig,
)

from test.quantization.recipes.debug import test_static_gemma4_finalize as fixture
from test.quantization.wrapq.wrappers.gemma4.test_quant_text_model import _has_gemma4


@unittest.skipUnless(_has_gemma4(), "transformers Gemma4 modules are required")
class TestStaticGemma4SplitPLE(unittest.TestCase):
    def assert_caches_close(self, runtime, expected):
        for cache, (key, value) in zip(runtime.layer_caches, expected):
            # The producer prefix always has the original S-token execution.
            torch.testing.assert_close(cache.past_k, key, atol=0, rtol=0)
            torch.testing.assert_close(cache.past_v, value, atol=0, rtol=0)

    def _compare_three_paths(self, *, length, quantized=False, visual_tokens=0):
        runtime = fixture.make_runtime(quantized=quantized, visual_tokens=visual_tokens)
        batch = fixture.make_batch(length, visual_tokens=visual_tokens)
        before = _state_contract(runtime.text_model) if quantized else None
        last_hidden = []
        handle = runtime.lm_head.register_forward_pre_hook(
            lambda module, args: last_hidden.append(args[0].clone())
        )
        with torch.no_grad():
            full_logits = runtime.prefill(batch, last_token_only=False)
            full_hidden = last_hidden[-1]
            expected_cache = fixture.cache_snapshot(runtime)
            self.assertEqual(runtime.past_len, length)
            continuation = []
            for token in (17, 8, 29)[: min(3, 9 - length)]:
                result = runtime.decode_one(torch.tensor([[token]]))
                continuation.append(
                    (
                        token,
                        result.clone(),
                        fixture.cache_snapshot(runtime),
                        runtime.past_len,
                    )
                )
            for split in (False, True):
                logits = runtime.prefill(
                    batch, last_token_only=True, ple_prefill_split=split
                )
                # The float S-token versus one-token GEMM reduction can differ
                # by roundoff. This is the existing finalize regression bound.
                torch.testing.assert_close(logits, full_logits, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(
                    last_hidden[-1], full_hidden, atol=1e-5, rtol=1e-5
                )
                self.assert_caches_close(runtime, expected_cache)
                self.assertEqual(runtime.past_len, length)
                for token, expected_logits, cache, past_len in continuation:
                    actual = runtime.decode_one(torch.tensor([[token]]))
                    torch.testing.assert_close(
                        actual, expected_logits, atol=1e-5, rtol=1e-5
                    )
                    self.assert_caches_close(runtime, cache)
                    self.assertEqual(runtime.past_len, past_len)
        handle.remove()
        if quantized:
            self.assertEqual(_state_contract(runtime.text_model), before)

    def test_fp_and_frozen_quantized_three_paths_with_continuation(self):
        for quantized in (False, True):
            for length in (1, 4, 9):
                with self.subTest(quantized=quantized, length=length):
                    self._compare_three_paths(length=length, quantized=quantized)

    def test_multimodal_fused_h0_three_path_parity(self):
        # p=2 falls inside the visual span, so replacing H0 with the text lookup
        # or the post-prefix decoder state changes the suffix projection input.
        for quantized in (False, True):
            with self.subTest(quantized=quantized):
                self._compare_three_paths(
                    length=3, quantized=quantized, visual_tokens=2
                )

    def test_selected_inputs_shapes_storage_and_normal_decode(self):
        runtime = fixture.make_runtime(visual_tokens=2)
        batch = fixture.make_batch(3, visual_tokens=2)
        stages = runtime._get_split_ple_stages()
        prefix_lookup, prefix_projection, suffix_lookup, suffix_projection = stages
        h0 = []
        inputs: dict[str, tuple[torch.Tensor, ...]] = {}
        outputs: dict[str, tuple[int, ...]] = {}
        calls = []
        handles = [
            runtime.mm_fusion.register_forward_hook(
                lambda module, args, result: h0.append(result.clone())
            )
        ]
        for name, module in zip(
            (
                "prefix_lookup",
                "prefix_projection",
                "suffix_lookup",
                "suffix_projection",
            ),
            stages,
        ):
            handles.append(
                module.register_forward_pre_hook(
                    lambda module, args, name=name: inputs.setdefault(
                        name, tuple(arg.clone() for arg in args)
                    )
                    and None
                )
            )
            handles.append(
                module.register_forward_hook(
                    lambda module, args, result, name=name: outputs.setdefault(
                        name, tuple(result.shape)
                    )
                    and None
                )
            )
        for phase, layers in (
            ("prefill", runtime.prefill_layers),
            ("decode", runtime.decode_layers),
        ):
            for index, layer in enumerate(layers):
                handles.append(
                    layer.register_forward_pre_hook(
                        lambda module, args, kwargs, phase=phase, index=index: calls.append(
                            (phase, index, kwargs["hidden_states"].shape[1])
                        ),
                        with_kwargs=True,
                    )
                )
        full_projection_shapes = []
        handles.append(
            runtime.ple_projection.register_forward_hook(
                lambda module, args, result: full_projection_shapes.append(
                    tuple(result.shape)
                )
            )
        )
        with torch.no_grad():
            runtime.prefill(batch, last_token_only=True, ple_prefill_split=True)
            self.assertEqual(full_projection_shapes, [])
            self.assertEqual(
                outputs,
                {
                    "prefix_lookup": (1, 10, 2, 8),
                    "prefix_projection": (1, 10, 2, 8),
                    "suffix_lookup": (1, 1, 2, 8),
                    "suffix_projection": (1, 1, 2, 8),
                },
            )
            torch.testing.assert_close(
                inputs["prefix_projection"][0], h0[0], atol=0, rtol=0
            )
            torch.testing.assert_close(
                inputs["suffix_projection"][0], h0[0][:, 2:3], atol=0, rtol=0
            )
            torch.testing.assert_close(
                inputs["suffix_lookup"][0],
                batch["llm_input_ids"][:, 2:3],
                atol=0,
                rtol=0,
            )
            self.assertFalse(torch.equal(inputs["suffix_projection"][0], h0[0][:, -1:]))
            text = runtime.token_embedding(batch["llm_input_ids"])
            self.assertFalse(torch.equal(inputs["suffix_projection"][0], text[:, 2:3]))
            self.assertEqual(
                calls,
                [
                    ("prefill", 0, 10),
                    ("prefill", 1, 10),
                    ("decode", 2, 1),
                    ("decode", 3, 1),
                ],
            )
            self.assertIs(
                prefix_lookup.embed_tokens_per_layer,
                runtime.ple_embedding.embed_tokens_per_layer,
            )
            self.assertIs(
                suffix_lookup.embed_tokens_per_layer,
                runtime.ple_embedding.embed_tokens_per_layer,
            )
            self.assertFalse(
                any(
                    "embed_tokens" in name
                    for name, _ in prefix_projection.named_modules()
                )
            )
            self.assertFalse(
                any(
                    "embed_tokens" in name
                    for name, _ in suffix_projection.named_modules()
                )
            )
            calls.clear()
            runtime.decode_one(torch.tensor([[17]]))
            self.assertEqual(calls, [("decode", i, 1) for i in range(4)])
            self.assertEqual(full_projection_shapes, [(1, 1, 4, 8)])
        for handle in handles:
            handle.remove()

    def test_prefix_buffer_released_before_suffix_decoder(self):
        runtime = fixture.make_runtime()
        stages = runtime._get_split_ple_stages()
        prefix = []
        lifetime = []
        handles = [
            stages[1].register_forward_hook(
                lambda module, args, result: prefix.append(weakref.ref(result))
            ),
            runtime.decode_layers[2].register_forward_pre_hook(
                lambda module, args, kwargs: lifetime.append(prefix[0]() is None),
                with_kwargs=True,
            ),
        ]
        with torch.no_grad():
            runtime.prefill(
                fixture.make_batch(4), last_token_only=True, ple_prefill_split=True
            )
        self.assertEqual(lifetime, [True])
        for handle in handles:
            handle.remove()

    def test_no_ple_and_no_suffix_are_noops(self):
        for options in ({"ple": False}, {"shared": False}):
            with self.subTest(options=options):
                runtime = fixture.make_runtime(**options)
                batch = fixture.make_batch(4)
                with torch.no_grad():
                    expected = runtime.prefill(batch, last_token_only=False)
                    cache = fixture.cache_snapshot(runtime)
                    actual = runtime.prefill(
                        batch, last_token_only=True, ple_prefill_split=True
                    )
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                self.assert_caches_close(runtime, cache)
                self.assertIsNone(getattr(runtime, "_split_ple_stages", None))

    def test_non_e2b_topology_uses_resolved_layer_mapping(self):
        make_config = fixture._make_text_config

        def alternate_config(**kwargs):
            kwargs.update(
                layer_types=["sliding_attention", "full_attention"] * 3,
                num_kv_shared_layers=2,
            )
            return make_config(**kwargs)

        with mock.patch.object(
            fixture, "_make_text_config", side_effect=alternate_config
        ):
            runtime = fixture.make_runtime()
        self.assertEqual(runtime.shared_kv_suffix_start, 4)
        batch = fixture.make_batch(4)
        with torch.no_grad():
            expected = runtime.prefill(batch, last_token_only=False)
            cache = fixture.cache_snapshot(runtime)
            actual = runtime.prefill(
                batch, last_token_only=True, ple_prefill_split=True
            )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assert_caches_close(runtime, cache)
        prefix_lookup, _, suffix_lookup, _ = runtime._get_split_ple_stages()
        self.assertEqual((prefix_lookup.layer_start, prefix_lookup.layer_end), (0, 4))
        self.assertEqual((suffix_lookup.layer_start, suffix_lookup.layer_end), (4, 6))

    def test_config_default_reference_override_and_type_diagnostics(self):
        self.assertFalse(StaticGemma4RuntimeConfig().ple_prefill_split)
        self.assertTrue(
            StaticGemma4RuntimeConfig(
                last_token_only_prefill=True, ple_prefill_split=True
            ).ple_prefill_split
        )
        with self.assertRaisesRegex(ValueError, "requires last_token_only_prefill"):
            StaticGemma4RuntimeConfig(ple_prefill_split=True)
        with self.assertRaisesRegex(ValueError, "ple_prefill_split must be a boolean"):
            StaticGemma4RuntimeConfig(ple_prefill_split="true")  # type: ignore[arg-type]
        runtime = fixture.make_runtime()
        runtime.last_token_only_prefill = True
        runtime.ple_prefill_split = True
        with (
            torch.no_grad(),
            mock.patch.object(
                runtime,
                "_build_split_per_layer_inputs",
                side_effect=AssertionError("split called"),
            ),
        ):
            runtime.prefill(fixture.make_batch(3), last_token_only=False)
        with self.assertRaisesRegex(ValueError, "must be booleans"):
            runtime.prefill(fixture.make_batch(3), ple_prefill_split="true")


if __name__ == "__main__":
    unittest.main()
